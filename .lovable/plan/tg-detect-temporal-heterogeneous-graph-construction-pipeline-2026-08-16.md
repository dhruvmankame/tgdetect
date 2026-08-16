# TG-Detect — Temporal Heterogeneous Graph Construction Pipeline

## What I looked at

The uploaded `TG-Detect.zip` contains two Python files plus specification/report markdown:

- `data_gen_v6.py` — Modal job that writes `/data/v6_ood_formal.jsonl` (1M events, 14-day timeline). Each line has `event_id, ts, src_id, dst_id, label, source_tag, tactics, apt_stage, chain_id, ood_type, causal_parent, attrs{frequency, temporal_burst, rarity}`. Benign events have no `chain_id` key at all and `apt_stage = -1`.
- `modal_train_v16.py` — the TGNN training job (Universal Encoder + GRL + CausalHTAConv + MemBank + rehearsal), reading the same JSONL directly.

There is no existing graph-construction layer, so this is additive: nothing in the current codebase gets replaced.

Two facts that shape the design:

1. Entity IDs in the synthetic data are untyped strings (`apt_actor_1234`, `internal_target_501`, `user_88`, `sys_412`, `critical_server_57`, `external_ip_233`). Node types must be inferred from the ID prefix, and the relation must be inferred from tactic + stage — the generator emits neither.
2. The real dataset is Mordor / OTRF **Security-Datasets**: per-scenario Windows Sysmon + Security event logs as JSON (one JSON object per line, often gzipped), with `_metadata/*.yaml` files carrying the ATT&CK technique/tactic mapping per scenario. There is no per-row `label` column — labels come from the scenario metadata, so the plan treats labeling as a metadata-join step, and I will only finalize field names after running the inspector against your local copy.

## Proposed structure

Added at the repo root, next to the existing scripts:

```text
graph_builder/
  __init__.py
  schema.py         TGEvent dataclass, NodeType/RelationType enums, parquet arrow schemas
  parsers.py        parse_synthetic_jsonl(), parse_mordor_jsonl(), parser registry
  normalizer.py     entity typing, ID canonicalization, ts coercion, validation
  builder.py        TemporalGraphBuilder (NetworkX MultiDiGraph) + StreamingGraphBuilder
  attack_tracker.py chain_id / causal_parent / entity-time chain reconstruction
  exporters.py      streaming parquet writers for events/nodes/edges + graph snapshots
scripts/
  inspect_dataset.py
  build_graph.py
  validate_graph.py
data/
  raw/  processed/  graphs/          (gitignored)
requirements-graph.txt
```

## Module behaviour

**schema.py** — `TGEvent` exactly as specified. Enums for node types (USER, HOST, PROCESS, FILE, IP, DOMAIN, SOCKET, UNKNOWN) and relations (LOGON, EXECUTES, READS, WRITES, DELETES, CONNECTS_TO, AUTHENTICATES_TO, NETWORK_FLOW, EXPLOIT, LATERAL_MOVE, EXFILTRATE, DISCOVER, IMPACT, GENERIC). `to_row()` flattens for parquet (`tactics` as list<string>, `attrs` as a JSON string so heterogeneous keys never break the arrow schema).

**parsers.py** — one generator function per dataset, each yielding `TGEvent`, all streaming line-by-line with `orjson`, never materializing the file.

- `parse_synthetic_jsonl`: maps prefixes to types (`apt_actor_*`/`user_*` → USER, `internal_target_*`/`sys_*`/`critical_server_*` → HOST, `external_ip_*` → IP), derives the relation from the first tactic (Initial_Access → EXPLOIT, Lateral_Movement → LATERAL_MOVE, Exfiltration → EXFILTRATE, Discovery → DISCOVER, Impact → IMPACT, benign → GENERIC), passes `ood_type` through into `attrs`, and preserves `chain_id`/`causal_parent`/`apt_stage` verbatim.
- `parse_mordor_jsonl`: written after inspection. Expected shape — one Sysmon/WinEvent record per line, fan-out into multiple TGEvents per record keyed on EventID (process create → `process --EXECUTES--> process`, network connect → `process --CONNECTS_TO--> ip`, file create → `process --WRITES--> file`, 4624/4625 logon → `user --LOGON--> host`, 4648/4768 → `host --AUTHENTICATES_TO--> host`). Process nodes keyed by GUID where available so the same name on different hosts stays distinct. Labels and `chain_id` derived from the scenario's `_metadata` YAML (attack scenario → label 1, `chain_id` = scenario ID, tactics = the YAML's ATT&CK tactics); benign/known-good sets → label 0.
- Registry `PARSERS = {"synthetic": ..., "mordor": ...}` so `build_graph.py --dataset mordor` picks the adapter; adding LANL/DARPA/CICIDS later is one function plus one registry entry.

**normalizer.py** — canonical `type:value` IDs, timestamp coercion (epoch float, ISO-8601, Windows filetime), required-field validation with a rejected-record counter, and a `normalize_stream(events)` generator that drops nothing silently but records reject reasons.

**builder.py** — `TemporalGraphBuilder` wrapping `nx.MultiDiGraph` for correctness work and small runs, plus `StreamingGraphBuilder` for the real dataset: keeps only a node table (`node_id → type, first_seen_ts, last_seen_ts, degree`) in memory and flushes edges to parquet every N events, so RAM stays proportional to distinct entities rather than event count. Both track the same statistics object (counts by node type, relation type, label, ts range).

**attack_tracker.py** — three pluggable strategies over the malicious-event stream, applied in priority order and combinable:

- `ChainIdTracker` — groups by ground-truth `chain_id`.
- `CausalParentTracker` — walks `causal_parent` links into ordered causal paths and detects broken/dangling parents.
- `EntityTimeTracker` — for datasets without ground truth: seeds on malicious events, expands through shared entities within a configurable time window (default 24h) and hop limit (default 2), producing attack subgraphs.

Output per chain: `chain_id`, ordered event list, stage/tactic sequence, participating nodes, start/end ts, duration, node/edge counts. Subgraphs are exported as `data/graphs/chains/<chain_id>.json` plus a `chains_summary.parquet` index.

**exporters.py** — incremental `pyarrow.parquet.ParquetWriter` for `events.parquet`, `nodes.parquet`, `edges.parquet` with row-group flushing, optional partitioning by day for very large runs, and a `graph_stats.json` sidecar.

## Scripts

- `inspect_dataset.py <path>` — walks a directory, zip, or tar.gz without full extraction; reports file inventory by extension and size, then for each candidate log file samples the first N records and prints inferred field names, types, example values, timestamp-field candidates, entity-field candidates, and label-field candidates. This is what I run against your local Security-Datasets copy before writing the Mordor parser.
- `build_graph.py --dataset synthetic --input data/raw/v6_ood_formal.jsonl --out data/processed --limit 100000` — runs parse → normalize → build → track → export, with `tqdm` progress, `--limit` for sampling, `--chunk-size` for flush cadence, and `--networkx` to additionally hold the in-memory graph for debugging.
- `validate_graph.py --processed data/processed` — reads the parquet outputs with pyarrow (batched, not fully loaded) and prints the full validation report you specified: totals, per-node-type counts, per-relation counts, benign/malicious split, chain count and events-per-chain distribution, timestamp range and monotonicity, plus integrity checks (duplicate event_ids, unparseable timestamps, empty src/dst, unknown types, dangling `causal_parent` references, `chain_id` groups of size 1).

## Order of work

1. Scaffold `graph_builder/` + `scripts/` with `schema.py`, `normalizer.py`, `exporters.py`, `builder.py`.
2. Synthetic parser + `build_graph.py`, validated on a 100k-line slice of `v6_ood_formal.jsonl`.
3. `attack_tracker.py` with all three strategies; verify chains reconstruct against generator ground truth.
4. `validate_graph.py` and a full 1M-event run of Path A.
5. `inspect_dataset.py`, then run it against your local dataset and report the actual Mordor schema.
6. Write `parse_mordor_jsonl` from the observed schema, including the `_metadata` YAML label join, and scale-test.

TGNN training stays untouched — the parquet outputs are the handoff point for a later `to_pyg.py` converter.

## Notes

- `requirements-graph.txt`: `pandas, pyarrow, networkx, tqdm, orjson, pyyaml`. Torch stays out of this layer.
- All commands are runnable locally on your machine; the dataset never needs to be uploaded.
- This repo's web preview is unrelated to the pipeline and stays as-is.
