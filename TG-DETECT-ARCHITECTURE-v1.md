# TG-Detect Architecture

> As-built architecture reference for the repository at commit `88fa442`
> (`graph representation`). Audited 2026-09-05.

This document describes what is implemented in the repository today, how the
pieces exchange data, which files own each responsibility, and where the
current design should be strengthened. It deliberately separates the working
Python/ML product core from the TypeScript web scaffold, because they are not
connected by an API at present.

## 1. Executive Summary

TG-Detect is a hybrid repository with two mostly independent systems:

1. **The implemented detection pipeline (Python):** a streaming ETL and
   temporal graph pipeline that converts security logs into a typed graph,
   reconstructs attack chains, creates time-windowed snapshots, trains a
   temporal graph neural network (TGNN), and emits evaluation/visualization
   artifacts.
2. **A presentation scaffold (TypeScript):** a Lovable-generated TanStack
   Start + React application with SSR/error handling and a large set of
   shadcn/Radix UI primitives. It currently renders a blank placeholder at
   `/` and has no route, API, database, inference endpoint, or data client for
   the Python artifacts.

The implemented Python path is:

```text
raw logs
  -> dataset parser generators
  -> TGEvent (shared event contract)
  -> normalization and canonical IDs/timestamps
  -> optional heuristic labeling
  -> graph statistics/node state + malicious-event collection
  -> Parquet graph tables and attack-chain exports
  -> loaded graph dataset
  -> sliding temporal snapshots
  -> TemporalGNN training/evaluation
  -> checkpoints, predictions, metrics, and reports
```

The web path is currently separate:

```text
HTTP request -> src/server.ts -> TanStack Start server entry
             -> generated route tree -> root shell -> / placeholder
```

There is no path from the web request to `data/`, `models/`, `results/`, or
`reports/`.

### Current state at a glance

| Area | State | Evidence |
| --- | --- | --- |
| Graph construction | Implemented batch/streaming CLI | `graph_builder/`, `scripts/build_graph.py` |
| Synthetic input | Implemented | `parse_synthetic_jsonl` |
| Mordor/Security-Datasets input | Implemented adapter and metadata join | `parse_mordor_jsonl` |
| Labeling | Parser/forced/heuristic modes | `build_graph.py`, `labeler.py` |
| Attack-chain reconstruction | Three strategies | `attack_tracker.py` |
| Parquet handoff | Implemented | `exporters.py`, `loader.py` |
| Snapshot generation | Implemented | `temporal.py`, `build_snapshots.py` |
| TGNN training/evaluation | Implemented baseline scripts | `models/tgnn.py`, `train_tgnn.py`, `evaluate_tgnn.py` |
| Cloud execution | Modal orchestration | `modal_train.py` |
| Offline reports | Implemented PNG/HTML generators and checked-in examples | `scripts/visualize_*.py`, `reports/` |
| Product web UI | Scaffold only | `src/routes/index.tsx` |
| Online serving/API | Not implemented | No server functions or API routes |
| Automated tests/CI | Minimal smoke script; no CI | `scripts/smoke_test.py`, no test config |

## 2. Repository Map

```text
TG-DETECT/
|-- graph_builder/             Python domain library: events -> graph -> chains
|-- scripts/                   Command-line pipeline, ML, validation, reports
|-- models/                    TGNN implementation and committed checkpoints
|-- modal_train.py             Modal image, volume, and remote job orchestration
|-- data/                      Runtime data roots (raw/processed/snapshots ignored)
|-- results/                   Downloaded/derived run outputs
|-- reports/                   Static and interactive visualization artifacts
|-- lib/                      Vendored browser assets used by report HTML
|-- src/                      TanStack Start/React web scaffold
|-- public/                   Favicon and robots file
|-- package.json               Frontend scripts/dependencies
|-- bun.lock, bunfig.toml      Bun lock/configuration
|-- requirements-graph.txt    Python graph-layer dependencies
|-- requirements-ml.txt       Python ML-layer dependencies
|-- vite.config.ts             Lovable/TanStack/Vite configuration
|-- tsconfig.json              TypeScript compiler configuration
|-- eslint.config.js           ESLint flat configuration
|-- components.json            shadcn component configuration
|-- .lovable/                  Lovable metadata and design plan
|-- README.md                  Generic Lovable starter README
`-- AGENTS.md                  Lovable history-safety instructions
```

### Tracked versus runtime-only data

`data/.gitignore` ignores all contents under `data/` (including the expected
`raw`, `processed`, and `snapshots` roots). The repository therefore contains
the code and some example outputs, but not the raw security captures or the
normal intermediate graph/snapshot datasets.
The tracked binary/generated material includes Python bytecode, model
checkpoints, prediction Parquet files, PNG/HTML reports, and vendored
JavaScript/CSS. These are useful evidence of completed runs, but they are not
the source code of the system.

## 3. End-to-End Data Flow

### 3.1 Ingestion and event normalization

1. `scripts/build_graph.py` resolves a parser from the `PARSERS` registry in
   `graph_builder/parsers.py`.
2. The parser yields one or more `TGEvent` objects per source record. All
   readers are generators and can consume plain files, `.gz`, `.zip`, and
   tar-family archives.
3. `normalize_stream` validates required IDs/timestamps, infers endpoint
   types, canonicalizes IDs, normalizes relation names, and records rejected
   records in `NormalizationStats`.
4. In `heuristic` label mode, `HeuristicLabeler.process` runs after
   normalization and before graph construction.
5. The same event object flows through the graph builder, attack tracker
   observer, and Parquet exporter.

The actual nesting in `build_graph.py` is:

```text
parser
  -> normalize_stream
  -> (optional) HeuristicLabeler.process
  -> StreamingGraphBuilder.add_events
  -> AttackTracker.observe_stream
  -> GraphExporter.stream_events
```

After the stream is exhausted, the script writes the node table, reconstructs
chains from the malicious events retained by `AttackTracker`, writes summaries
and subgraphs, and writes `graph_stats.json`.

### 3.2 Graph representation

The graph is directed, heterogeneous, temporal, and multi-relational:

- Each normalized `TGEvent` represents one timestamped directed edge.
- Node identity is canonicalized to `type:value`, for example
  `process:{GUID}`, `host:server-a`, or `ip:10.0.0.5`.
- Endpoint types are one of `USER`, `HOST`, `PROCESS`, `FILE`, `IP`,
  `DOMAIN`, `SOCKET`, or `UNKNOWN`.
- Relations are one of the 14 values defined in `RelationType`, including
  `EXECUTES`, `CONNECTS_TO`, `LOGON`, `AUTHENTICATES_TO`, `WRITES`,
  `LATERAL_MOVE`, `EXFILTRATE`, and `GENERIC`.
- Event attributes remain extensible because `attrs` is serialized as a JSON
  string in Parquet.

There are two builders:

- `StreamingGraphBuilder` retains a node aggregate in memory (first/last seen,
  in/out degree, malicious-event count) and leaves edge/event rows to the
  exporter. Its memory is proportional to distinct entities, not event count.
- `TemporalGraphBuilder` extends the streaming builder with a NetworkX
  `MultiDiGraph`. It is intended for small/debug runs and is explicitly not the
  scalable path.

### 3.3 Storage and ML handoff

`GraphExporter` writes row groups using `pyarrow.parquet.ParquetWriter` with
Snappy compression. The resulting directory is loaded by
`graph_builder.loader.load_graph`, which currently materializes all three
Parquet tables into pandas DataFrames and builds integer vocabularies for node
types and relations.

`SnapshotBuilder` then:

- sorts the edge stream by timestamp;
- creates an event-driven sliding window grid;
- locally re-indexes nodes touched by each window;
- creates node features (type one-hot plus degree/in-degree/out-degree and
  malicious ratio by default);
- creates edge features (relation one-hot plus normalized in-window time by
  default); and
- derives edge labels, node labels, and a window-level label.

`scripts/build_snapshots.py` serializes each snapshot as a pickle and writes a
`meta.json` containing dimensions and categorical maps. The snapshots are then
consumed by the training and evaluation scripts.

## 4. Python Domain Layer (`graph_builder/`)

| File | Responsibility | Important public objects/behavior |
| --- | --- | --- |
| `schema.py` | Shared data contract | `TGEvent`, `NodeType`, `RelationType`, Arrow schemas |
| `parsers.py` | Dataset adapters and archive readers | `iter_lines`, `parse_synthetic_jsonl`, `parse_mordor_jsonl`, `PARSERS` |
| `normalizer.py` | Type inference, canonical IDs, timestamp coercion, validation | `infer_node_type`, `canonical_id`, `coerce_ts`, `normalize_stream` |
| `labeler.py` | Rule-based labels for mixed/unlabeled captures | `detect_indicator`, `HeuristicLabeler`, `label_stream` |
| `builder.py` | Graph/node state and statistics | `GraphStats`, `StreamingGraphBuilder`, `TemporalGraphBuilder` |
| `attack_tracker.py` | Attack-chain/subgraph reconstruction | `AttackChain`, three trackers, `AttackTracker` |
| `exporters.py` | Incremental output writing | `ParquetStreamWriter`, `GraphExporter`, chain/stats writers |
| `loader.py` | Parquet -> in-memory dataset bridge | `TemporalGraphDataset`, `load_graph`, `inspect_graph` |
| `temporal.py` | Graph tables -> temporal ML windows | `TemporalSnapshot`, `SnapshotBuilder` |
| `__init__.py` | Package facade | Re-exports schemas, builders, trackers, exporters, and normalizer helpers |
| `README.md` | Graph-layer usage and design notes | Installation, examples, output contract, extension guidance |

### 4.1 `TGEvent` contract

`TGEvent` in `schema.py` is the only event shape downstream code should need:

| Field | Type | Meaning |
| --- | --- | --- |
| `event_id` | `str` | Stable source/fan-out event identifier |
| `ts` | `float` | Epoch seconds after normalization |
| `src_id`, `dst_id` | `str` | Endpoint IDs, canonicalized downstream |
| `src_type`, `dst_type` | `str` | Endpoint type hints or inferred types |
| `relation` | `str` | Directed relationship/action |
| `label` | `int` | Binary benign (`0`) or malicious (`1`) label |
| `tactics` | `list[str]` | ATT&CK/synthetic tactic labels |
| `apt_stage` | `str | None` | Synthetic stage when available |
| `source_tag` | `str` | Dataset/source identifier |
| `chain_id` | `str | None` | Ground-truth or inferred chain grouping |
| `causal_parent` | `str | None` | Parent event ID when causal metadata exists |
| `attrs` | `dict[str, Any]` | Dataset-specific fields and derived values |

`to_row()` expands the stable columns and JSON-serializes `attrs`; `edge_row()`
produces the smaller edge projection.

### 4.2 Parser behavior

#### Generic reader

`iter_lines` handles plain files, gzip, ZIP members, and tar archives without
extracting the whole container. `iter_json_records` parses one JSON object per
line and skips blank or malformed lines. The malformed-line skip is currently
silent; rejected-record statistics only cover normalization failures.

#### Synthetic adapter

`parse_synthetic_jsonl` consumes the project's generated event format. It
derives the relation from the first tactic using `TACTIC_TO_RELATION`, preserves
`label`, `apt_stage`, `chain_id`, and `causal_parent`, and copies `ood_type` into
`attrs`. Endpoint types are intentionally left for the normalizer to infer from
ID prefixes such as `apt_actor_`, `internal_target_`, and `external_ip_`.

#### Mordor / OTRF adapter

`load_mordor_metadata` reads `_metadata/*.yaml`, extracts scenario IDs, ATT&CK
tactics/techniques, linked filenames, and a scenario-level label. The parser
fans one Windows Sysmon/Security record into typed events:

| Source event | Produced relationship(s) |
| --- | --- |
| Sysmon 1 | parent process `EXECUTES` child; child `GENERIC` to host; user `EXECUTES` child when known |
| Sysmon 3 | process `CONNECTS_TO` destination IP/domain |
| Sysmon 2, 11, 15, 23, 26 | process `WRITES` or `DELETES` file |
| Sysmon 22 | process `CONNECTS_TO` queried domain |
| Security 4624, 4625, 4634, 4672 | user `LOGON` host and optional source-IP edge |
| Security 4648, 4768, 4769, 4776 | host/user `AUTHENTICATES_TO` host |
| Other/unmapped records | host-scoped user `GENERIC` fallback |

Process nodes prefer a process GUID and otherwise use a host-qualified image
name so equal image names on different machines do not collapse by accident.
Directories are walked recursively and each file can receive a file-derived
chain ID. Metadata matching currently uses the basename of the log file. In
directory mode the file-derived chain ID is already supplied before metadata
lookup, so it takes precedence over a YAML scenario ID.

### 4.3 Normalization

`normalizer.py` performs these transformations:

- validates non-empty event ID, source ID, destination ID, and parseable time;
- infers types from explicit hints, canonical prefixes, IP/domain/path syntax,
  and known synthetic prefixes;
- canonicalizes both endpoints to `type:value`;
- converts epoch seconds, milliseconds, microseconds, Windows FILETIME, and
  ISO-8601 strings to epoch seconds;
- uppercases/normalizes relation spelling while allowing dataset-specific
  relation names; and
- coerces labels and optional metadata to stable scalar/string forms.

The function mutates the `TGEvent` object in place. It does not sort events;
out-of-order input is counted by `GraphStats`, and order-sensitive heuristic
label propagation therefore depends on source order.

### 4.4 Labels and attack chains

`build_graph.py` supports three label modes:

- `parser`: use labels supplied by the adapter or Mordor metadata;
- `force`: request one label for every parsed event (the current CLI wiring
  applies this override in the Mordor branch; synthetic events retain parser
  labels unless the adapter is changed); and
- `heuristic`: start benign and promote indicator hits plus nearby related
  events.

The heuristic labeler detects suspicious tools, LOLBins with remote/network
signals, command-line patterns, suspicious Windows event IDs, and suspicious
relations. It keeps a bounded event buffer and a hot-entity expiry map. Host,
domain, and unknown nodes are excluded from propagation because they are too
broad in lab captures.

`AttackTracker` retains malicious events (default cap: two million) and applies
strategies in the configured priority order. Events grouped by an earlier
strategy are removed before the next strategy runs:

1. `ChainIdTracker`: groups explicit `chain_id` values.
2. `CausalParentTracker`: union-find over `causal_parent` links and records
   dangling references.
3. `EntityTimeTracker`: expands malicious events through shared endpoints within
   a time window and hop limit.

`AttackChain` can become both a row in `chains_summary.parquet` and a JSON
subgraph containing typed nodes and ordered edges.

## 5. Output Contracts

For a processed graph directory, the normal output set is:

| File | Contents | Primary consumer |
| --- | --- | --- |
| `events.parquet` | Full one-row-per-`TGEvent` table, including tactics and JSON attrs | Loader, validation, analysis |
| `edges.parquet` | Thin edge projection with IDs, relation, time, label, chain metadata | Loader, snapshot builder, visualizers |
| `nodes.parquet` | Node type, first/last seen, in/out degree, malicious event count | Loader, features, visualizers |
| `chains_summary.parquet` | Chain counts, time span, tactic/stage/relation sequences, nodes/event IDs | Chain analysis/reporting |
| `graph_stats.json` | Normalization, label, graph, attack, and output statistics | Operations/validation |
| `data/graphs/chains/<id>.json` | Per-chain typed node/edge subgraphs | Chain visualizations/consumers |

The Arrow schemas are defined centrally in `schema.py`. The Lovable design plan
mentions optional day partitioning for very large output, but the current
`GraphExporter` writes unpartitioned files.

### Snapshot contract

Each `snapshot_*.pkl` dictionary contains:

```text
ts_start, ts_end
node_ids, node_types
x                  # node feature matrix [N, F]
edge_index         # local endpoint indices [2, E]
edge_attr          # edge features [E, R]
edge_labels
node_labels
snapshot_label
num_events
```

With the default modes, `F = number_of_node_types + 4` and
`R = number_of_relations + 1`. `meta.json` records these dimensions, feature
modes, and the categorical maps needed to reconstruct a model.

## 6. TemporalGNN and ML Workflow

### 6.1 Model

`models/tgnn.py` defines `TemporalGNN`:

1. For each snapshot, a stack of GraphSAGE (`SAGEConv`) layers produces node
   embeddings.
2. If edge features are present, a linear edge encoder creates messages that
   are mean-aggregated onto destination nodes.
3. Embeddings are aligned to the node IDs appearing in the final snapshot.
4. A GRU is intended to aggregate the aligned embeddings across snapshots.
5. A node classifier emits one maliciousness logit per final-snapshot node; a
   second classifier emits a snapshot-level logit.

### 6.2 Training

`scripts/train_tgnn.py`:

- loads every snapshot pickle and `meta.json` into memory;
- forms sliding sequences of `--window-size` snapshots;
- supports chronological or interleaved-block train/validation/test splits;
- computes an optional positive-class BCE weight;
- trains with AdamW, a ReduceLROnPlateau scheduler, gradient clipping, and
  optional early stopping;
- tunes a validation decision threshold unless disabled;
- selects `best_model.pt` by AUC-PR, F1, or ROC-AUC (AUC-PR by default); and
- writes `best_model.pt`, `final_model.pt`, `history.json`, and optional
  `test_metrics.json`.

The declared `--batch-size` option is currently not used: the loop processes
one sequence at a time. The training loss uses final-snapshot node labels only;
the snapshot classifier is not part of the loss.

### 6.3 Evaluation

`scripts/evaluate_tgnn.py` reloads the checkpoint and snapshot metadata, builds
sequences, evaluates node probabilities, and writes:

- `predictions_<split>.parquet` with sequence, node ID, probability, prediction,
  ground truth, and snapshot label; and
- `metrics_<split>.json` with threshold, counts, accuracy, precision, recall,
  F1, ROC-AUC/PR-AUC when defined, and a confusion matrix.

Chain-level metrics are explicitly a placeholder because snapshots do not carry
chain IDs. Standalone evaluation always uses a chronological split, even when
training used block splitting. The evaluator loads the checkpoint dictionary;
the bare `final_model.pt` state dict is not directly compatible with its current
loader without an adapter.

### 6.4 Important implementation observations

These are implementation facts to address before treating the baseline as a
production temporal detector:

- The GRU is declared with `batch_first=True`, but the model supplies a tensor
  shaped `[time, nodes, hidden]`. With that setting, the first dimension is
  interpreted as batch, so the current tensor layout should be verified/fixed
  before relying on temporal semantics.
- Default node features include whole-graph `malicious_events` and
  `malicious_ratio`. Those are label-derived features and can leak target
  information into training/evaluation.
- `--batch-size` is only a CLI declaration; no mini-batch collation is present.
- The snapshot-level head is defined but not trained or reported.
- Evaluation split behavior is not guaranteed to match block-mode training.
- Prediction rows lack chain IDs, so chain detection metrics cannot yet be
  computed.
- `SnapshotBuilder` and `TemporalGraphDataset` are memory-bound after the
  Parquet boundary even though graph construction itself streams.
- Mordor parser mode defaults to `label=1` when no metadata or explicit label is
  supplied, and the `force` override is not currently applied to synthetic
  input. Label mode should therefore be made explicit in reproducible runs.

## 7. Command-Line Surface

| Script | Role |
| --- | --- |
| `scripts/inspect_dataset.py` | Inventory a directory/archive and infer timestamp/entity/action/label fields before writing an adapter |
| `scripts/build_graph.py` | Parse, normalize, label, build, export, track chains, and write stats |
| `scripts/validate_graph.py` | Batch-read Parquet and report counts, ranges, duplicates, unknowns, causal references, and chain sizes |
| `scripts/load_graph.py` | Load a processed graph, print summaries/heads, or write a JSON summary |
| `scripts/merge_graphs.py` | Merge processed captures, optionally override labels/rebase time, and recompute nodes |
| `scripts/build_snapshots.py` | Convert processed graph tables to snapshot pickles and metadata |
| `scripts/train_tgnn.py` | Train the baseline TGNN and save checkpoints/history/metrics |
| `scripts/evaluate_tgnn.py` | Evaluate a checkpoint and save prediction/metric Parquet/JSON |
| `scripts/visualize_graph.py` | Draw overview, chain, timeline, distribution, and optional pyvis reports |
| `scripts/visualize_severity_graph.py` | Create severity-colored vis-network HTML (CDN-dependent) and optional PNG |
| `scripts/plot_training.py` | Plot loss/AUC-PR/F1 from `history.json` |
| `scripts/plot_evaluation.py` | Plot confusion matrix, score distribution, ROC, and PR curves |
| `scripts/smoke_test.py` | Small graph -> Parquet -> loader -> snapshot shape check |

Typical local execution is:

```bash
python scripts/inspect_dataset.py path/to/capture.zip
python scripts/build_graph.py --dataset mordor --input path/to/log.json \
  --metadata-dir path/to/_metadata --out data/processed/run
python scripts/validate_graph.py --processed data/processed/run \
  --track-duplicates
python scripts/build_snapshots.py --data data/processed/run \
  --out data/snapshots/run --window-size 60 --stride 30
python scripts/train_tgnn.py --snapshots data/snapshots/run \
  --out models/checkpoints/run
python scripts/evaluate_tgnn.py --checkpoint models/checkpoints/run/best_model.pt \
  --snapshots data/snapshots/run --out results/run
```

## 8. Modal Execution Architecture

`modal_train.py` is the cloud execution wrapper, not a model implementation.
It builds a Debian slim image with Python, PyTorch/PyG, graph dependencies, and
reporting libraries; mounts the code at `/root/tgdetect`; and mounts a persistent
Modal volume named `tgdetect-data` at `/data`.

The volume layout is:

```text
/data/raw/<name>         uploaded raw captures
/data/processed/<name>   graph Parquet and stats
/data/snapshots/<name>   snapshot pickles/meta
/data/checkpoints/<name> model checkpoints and evaluation outputs
```

Local entrypoints upload/download data and invoke remote functions for:

- graph building on CPU (`_build`);
- graph merging (`_merge`);
- snapshot generation (`_snapshots`);
- A10G GPU training (`_train`); and
- A10G evaluation (`_evaluate`).

The `all` entrypoint chains snapshots, training, and test evaluation. Each stage
commits the volume. This provides a reproducible batch workflow, but there is
no job database, status API, retry policy beyond Modal behavior, or online
inference service.

## 9. Reporting and Checked-In Artifacts

The report generators are offline consumers of processed tables and predictions:

- `reports/mordor_full/` contains graph overview/chain/timeline/distribution
  PNGs and a pyvis interactive HTML report.
- `reports/mordor_mixed/` contains a severity-colored attack graph HTML/PNG,
  training curves, and evaluation plots. The generated HTML embeds graph data
  but loads vis-network from a public CDN at runtime.
- `models/checkpoints/` contains best/final model files and histories, plus
  evaluation outputs for `mordor_mixed` and `mordor_test`.
- `results/mordor_mixed/` contains a downloaded/derived copy of a mixed run.
- `lib/vis-9.1.2/`, `lib/tom-select/`, and `lib/bindings/` are vendored browser
  assets, not React-route modules. Generated report output references the
  vis/bindings family; the checked-in reports do not currently reference the
  tom-select files.

The committed `mordor_test` evaluation is single-class, and the `mordor_full`
history also shows single-class behavior. Perfect F1/accuracy and undefined AUC
are therefore not evidence of generalization. Mixed-run metrics are the more
informative examples, but they still represent a particular captured run rather
than a validated production benchmark.

## 10. TypeScript Web Architecture

### 10.1 Build and runtime

The web project uses React 19, TypeScript, TanStack Router/Start, React Query,
Tailwind CSS v4, and a Lovable Vite configuration wrapper. There is no
conventional `src/main.tsx` or checked-in `index.html`; the TanStack Start/Vite
plugin supplies the client/server entry wiring.

`vite.config.ts` delegates to `@lovable.dev/vite-tanstack-config`, which bundles
the React, TanStack Start, Tailwind, TypeScript-path, Nitro, and environment
injection plugins. The configured TanStack Start server entry is `src/server.ts`.

### 10.2 Request lifecycle

```text
Request
  -> src/server.ts fetch wrapper
  -> lazy @tanstack/react-start/server-entry
  -> startInstance middleware (error handling + CSRF for future serverFns)
  -> generated routeTree.gen.ts
  -> __root.tsx shell and QueryClientProvider
  -> route component (currently only /)
```

`src/server.ts` catches catastrophic failures and converts h3-swallowed JSON
500 responses to the static HTML from `src/lib/error-page.ts`. `src/start.ts`
adds an error middleware and CSRF middleware filtered to server functions.
`src/routes/__root.tsx` owns global metadata, stylesheet/favicon links, the
`<Outlet />`, the 404 component, and the React error boundary. Error details are
captured/expanded by `src/lib/error-capture.ts` and optionally sent to Lovable
telemetry by `src/lib/lovable-error-reporting.ts`.

### 10.3 Routes and state

`src/routeTree.gen.ts` registers only:

```text
__root__
`-- /
```

`src/routes/index.tsx` renders an externally hosted blank-page SVG and includes
an explicit `REMOVE_THIS` marker. There are no graph, alert, chain, timeline,
model, or settings routes. A `QueryClient` is created by `src/router.tsx` and
provided by the root route, but no query, mutation, loader, API client, domain
store, persistence, or server function uses it.

### 10.4 UI/design system inventory

`src/components/ui/` is a generated shadcn-style primitive library. It wraps
Radix primitives and supporting packages such as Recharts, Embla, react-day-
picker, react-resizable-panels, Vaul, cmdk, Sonner, and input-otp. The set
includes:

```text
accordion, alert-dialog, alert, aspect-ratio, avatar, badge, breadcrumb,
button, calendar, card, carousel, chart, checkbox, collapsible, command,
context-menu, dialog, drawer, dropdown-menu, form, hover-card, input,
input-otp, label, menubar, navigation-menu, pagination, popover, progress,
radio-group, resizable, scroll-area, select, separator, sheet, sidebar,
skeleton, slider, sonner, switch, table, tabs, textarea, toggle-group,
toggle, tooltip
```

These components mostly import one another and are currently not imported by
the route or root shell. `src/hooks/use-mobile.tsx` supports the sidebar's
responsive behavior, and `src/lib/utils.ts` provides the `cn` class-merging
helper. `src/styles.css` defines Tailwind v4 generation, semantic light/dark
tokens, chart colors, sidebar tokens, and base body styling; no product-specific
visual language is present yet.

## 11. Configuration and Dependency Boundaries

### Python

- `requirements-graph.txt` intentionally excludes Torch and contains pandas,
  PyArrow, NetworkX, tqdm, orjson, PyYAML, matplotlib, and optional pyvis.
- `requirements-ml.txt` adds Torch, PyTorch Geometric, scikit-learn, matplotlib,
  and seaborn.
- There is no `pyproject.toml` or installed Python package entry point; scripts
  add the repository root to `sys.path`.

### TypeScript

- `package.json` provides `dev`, `build`, `build:dev`, `preview`, `lint`, and
  `format` scripts, but no test or typecheck script.
- `bun.lock` and `bunfig.toml` are committed, while the README instructs
  `npm i`; no npm lockfile is present. A CI/deployment choice should be made
  explicit to avoid divergent dependency resolution.
- `tsconfig.json` enables strict TypeScript settings and the `@/*` alias.
- `eslint.config.js` uses the flat ESLint configuration with React Hooks and
  Prettier integration.
- `components.json` defines the shadcn New York style, Lucide icons, Tailwind
  source, and aliases.

## 12. Current Limitations and Recommended Improvement Path

The following roadmap follows the boundaries already present in the code.

### Priority 0: make the product boundary real

1. Define a typed serving API (for example, a Python FastAPI/worker service or
   a well-defined artifact service) for graph summaries, nodes/edges, chains,
   timelines, predictions, and job status.
2. Add a frontend data-access layer with stable query keys and route loaders;
   connect dashboard, graph explorer, chain detail, timeline, and evaluation
   views to that API.
3. Decide whether reports are immutable downloadable artifacts or whether graph
   data should be indexed in a query store. Do not make browser code read local
   Parquet or pickle files directly.

### Priority 1: make data and ML contracts reliable

1. Add a versioned schema/config manifest covering node/relation vocabularies,
   parser version, labeling mode, feature modes, and model checkpoint.
2. Remove label-derived node features from evaluation (or compute them using
   only information available before the prediction horizon).
3. Fix and test the GRU tensor layout so the time dimension is actually the
   recurrent sequence dimension.
4. Make training and evaluation share one split/config implementation; either
   implement real batching or remove the unused `--batch-size` option.
5. Train/report the snapshot head or remove it; add chain IDs/time ranges to
   snapshots and prediction rows to enable chain-level metrics.
6. Provide a loader/iterator that streams Parquet and snapshots for large
   captures. Keep NetworkX and full pandas loading as explicit debug modes.
7. Make merge operations rewrite `causal_parent` references consistently and
   regenerate chain summaries/subgraphs.

### Priority 2: operations, security, and maintainability

1. Add unit tests for timestamp coercion, type inference, parser fan-out,
   metadata labeling, chain strategies, schema round trips, and snapshot shapes.
2. Add an end-to-end CI job that installs pinned graph dependencies, runs the
   smoke test, compiles Python, and runs frontend typecheck/build/lint.
3. Record parse errors and skipped JSON lines in explicit counters/logs.
4. Add job manifests, atomic output directories, resumability, and structured
   logging for local and Modal runs.
5. Treat security logs, command lines, model checkpoints, and pickles as
   sensitive/untrusted data; add access controls, retention rules, and safe
   deserialization policies before serving them.
6. Stop tracking generated `__pycache__` and decide whether large checkpoints
   and reports belong in Git, release storage, or an artifact registry.
7. Replace generic Lovable metadata and the remote blank placeholder with
   TG-Detect branding and product routes; add deployment/environment validation
   and an explicit Cloudflare/Nitro deployment manifest if that target remains
   the plan.

### Suggested target architecture

```text
                         +-----------------------------+
                         | React/TanStack web client   |
                         | dashboard / graph / chains  |
                         +--------------+--------------+
                                        |
                              typed HTTPS API
                                        |
               +------------------------+------------------------+
               | artifact/query service + job/status store        |
               | summaries, graph slices, predictions, metadata   |
               +-----------+--------------------+------------------+
                           |                    |
                    object/column store       worker queue
                           |                    |
                    Parquet/checkpoints   graph -> snapshot -> TGNN
                           ^                    |
                           +---- offline ingest/Modal ----------+
```

This preserves the current strong offline batch pipeline while adding a clear,
testable boundary for an actual user-facing detection product.

## 13. Verification Notes

The repository was inspected as source plus tracked artifacts. The following
read-only checks were performed during this audit:

- Python source compilation with `python -m compileall -q graph_builder models
  scripts modal_train.py` succeeded.
- The graph smoke test could not execute in the current environment because
  `pyarrow`, pandas, NetworkX, Torch, and PyG are not installed here. This is an
  environment prerequisite, not a behavioral pass.
- Frontend dependencies are not installed (`node_modules` is absent), so a
  frontend build/typecheck/lint run was not available in this environment.

The architecture above is therefore based on source inspection, import/entry
point tracing, artifact inspection, and static compilation rather than a fresh
full-data pipeline run.

## 14. Source Reference Index

The most important implementation references are:

- [graph_builder/schema.py](graph_builder/schema.py) - event and Arrow contracts
- [graph_builder/parsers.py](graph_builder/parsers.py) - input adapters
- [graph_builder/normalizer.py](graph_builder/normalizer.py) - canonicalization
- [graph_builder/labeler.py](graph_builder/labeler.py) - heuristic labels
- [graph_builder/builder.py](graph_builder/builder.py) - graph state
- [graph_builder/attack_tracker.py](graph_builder/attack_tracker.py) - chains
- [graph_builder/exporters.py](graph_builder/exporters.py) - output writers
- [graph_builder/loader.py](graph_builder/loader.py) - Parquet bridge
- [graph_builder/temporal.py](graph_builder/temporal.py) - snapshots
- [models/tgnn.py](models/tgnn.py) - model
- [scripts/build_graph.py](scripts/build_graph.py) - main graph orchestration
- [scripts/train_tgnn.py](scripts/train_tgnn.py) - training loop
- [scripts/evaluate_tgnn.py](scripts/evaluate_tgnn.py) - evaluation loop
- [modal_train.py](modal_train.py) - Modal workflow
- [src/routes/index.tsx](src/routes/index.tsx) - current web home route
- [src/routes/__root.tsx](src/routes/__root.tsx) - web shell/error boundaries
- [src/server.ts](src/server.ts) and [src/start.ts](src/start.ts) - SSR/middleware
- [graph_builder/README.md](graph_builder/README.md) - graph-layer usage notes

## Appendix A. Complete File Inventory by Role

This appendix groups every tracked file in the working tree so that generated
assets are not mistaken for runtime modules.

### Root and project configuration

```text
AGENTS.md
README.md
.gitignore
.prettierignore
.prettierrc
bun.lock
bunfig.toml
components.json
data/.gitignore
eslint.config.js
package.json
requirements-graph.txt
requirements-ml.txt
tsconfig.json
vite.config.ts
.lovable/project.json
.lovable/plan/tg-detect-temporal-heterogeneous-graph-construction-pipeline-2026-08-16.md
```

### Graph library

```text
graph_builder/__init__.py
graph_builder/README.md
graph_builder/schema.py
graph_builder/parsers.py
graph_builder/normalizer.py
graph_builder/labeler.py
graph_builder/builder.py
graph_builder/attack_tracker.py
graph_builder/exporters.py
graph_builder/loader.py
graph_builder/temporal.py
```

The `graph_builder/__pycache__/` files are CPython bytecode generated from
these modules and are not architectural components.

### Workflow and report scripts

```text
scripts/inspect_dataset.py
scripts/build_graph.py
scripts/validate_graph.py
scripts/load_graph.py
scripts/merge_graphs.py
scripts/build_snapshots.py
scripts/train_tgnn.py
scripts/evaluate_tgnn.py
scripts/visualize_graph.py
scripts/visualize_severity_graph.py
scripts/plot_training.py
scripts/plot_evaluation.py
scripts/smoke_test.py
```

### Model and orchestration

```text
models/tgnn.py
modal_train.py
models/checkpoints/mordor_full/{best_model.pt,final_model.pt,history.json}
models/checkpoints/mordor_mixed/{best_model.pt,final_model.pt,history.json}
models/checkpoints/mordor_mixed/eval/metrics_test.json
models/checkpoints/mordor_mixed/eval/predictions_test.parquet
models/checkpoints/mordor_test/{best_model.pt,final_model.pt,history.json}
models/checkpoints/mordor_test/eval/metrics_test.json
models/checkpoints/mordor_test/eval/predictions_test.parquet
results/mordor_mixed/checkpoints/mordor_mixed/{best_model.pt,final_model.pt,history.json,test_metrics.json}
results/mordor_mixed/checkpoints/mordor_mixed/eval_test/{metrics_test.json,predictions_test.parquet}
```

The `models/__pycache__/` and root `__pycache__/modal_train...pyc` files are
bytecode artifacts.

### Web source

```text
src/router.tsx
src/start.ts
src/server.ts
src/routeTree.gen.ts
src/styles.css
src/hooks/use-mobile.tsx
src/lib/error-capture.ts
src/lib/error-page.ts
src/lib/lovable-error-reporting.ts
src/lib/utils.ts
src/routes/README.md
src/routes/__root.tsx
src/routes/index.tsx
```

The 46 generated UI primitives under `src/components/ui/` are:

```text
accordion.tsx, alert-dialog.tsx, alert.tsx, aspect-ratio.tsx, avatar.tsx,
badge.tsx, breadcrumb.tsx, button.tsx, calendar.tsx, card.tsx, carousel.tsx,
chart.tsx, checkbox.tsx, collapsible.tsx, command.tsx, context-menu.tsx,
dialog.tsx, drawer.tsx, dropdown-menu.tsx, form.tsx, hover-card.tsx,
input-otp.tsx, input.tsx, label.tsx, menubar.tsx, navigation-menu.tsx,
pagination.tsx, popover.tsx, progress.tsx, radio-group.tsx, resizable.tsx,
scroll-area.tsx, select.tsx, separator.tsx, sheet.tsx, sidebar.tsx,
skeleton.tsx, slider.tsx, sonner.tsx, switch.tsx, table.tsx, tabs.tsx,
textarea.tsx, toggle-group.tsx, toggle.tsx, tooltip.tsx
```

### Public, vendor, and report assets

```text
public/favicon.ico
public/robots.txt
lib/bindings/utils.js
lib/tom-select/tom-select.complete.min.js
lib/tom-select/tom-select.css
lib/vis-9.1.2/vis-network.css
lib/vis-9.1.2/vis-network.min.js
reports/mordor_full/{graph_overview.png,timeline.png,distributions.png,
  graph_interactive.html,chain_*.png}
reports/mordor_mixed/{attack_graph.html,attack_graph.png,
  training_curves.png,evaluation_plots.png}
```

The report HTML is generated data, not a route in the TypeScript application;
the interactive pages may load visualization libraries from a CDN when opened.
