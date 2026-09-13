"""TG-Detect training on Modal (GPU/CPU).

The local machine only uploads code. Heavy work (build, snapshots, train, eval)
runs in Modal containers. The CTU-13 source lives in the user's `mega-10gb-dataset`
volume, mounted READ-ONLY at /raw and never committed.

CTU-13 workflow:
    modal run modal_train.py::smoke                    # cents CPU end-to-end (dir-52)
    modal run modal_train.py::build_ctu13_all          # build all 13 scenarios (CPU)
    modal run modal_train.py::snapshots_ctu13_all      # flow snapshots for all
    modal run modal_train.py::train_ho \
        --train-scenarios ctu13_c48,ctu13_c51,ctu13_c53 \
        --test-scenarios ctu13_c47 --out-name ho_c47   # GPU held-out-family train
    modal run modal_train.py::evaluate_ho --out-name ho_c47

Legacy Mordor workflow (upload/build/merge/snapshots/train/evaluate/all) is kept.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "tgdetect"
REMOTE_CODE = "/root/tgdetect"
REMOTE_DATA = "/data"   # writable output volume
RAW_DATA = "/raw"       # read-only CTU-13 source volume

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "torch-geometric>=2.6",
        "numpy>=1.26",
        "pandas>=2.1",
        "pyarrow>=15",
        "scikit-learn>=1.4",
        "orjson>=3.9",
        "networkx>=3.2",
        "tqdm>=4.66",
        "matplotlib>=3.8",
        "pyvis>=0.3.2",
    )
    .add_local_dir(
        ".",
        REMOTE_CODE,
        ignore=[
            "data/**", "node_modules/**", ".git/**", "**/__pycache__/**",
            "reports/**", "models/checkpoints/**", "*.zip", "venv/**", "*.pdf",
        ],
    )
)

app = modal.App(APP_NAME, image=image)

# Writable output volume: processed graphs, snapshots, checkpoints, metrics.
volume = modal.Volume.from_name("tgdetect-data", create_if_missing=True)

# Read-only mount of the user-uploaded CTU-13 source (13 scenario dirs). NEVER
# committed — the raw 10 GB dataset is treated as immutable input.
raw_vol = modal.Volume.from_name("mega-10gb-dataset")
try:
    RAW_MOUNT = raw_vol.read_only()
except Exception:  # older client without read_only(): mount rw but never commit it
    RAW_MOUNT = raw_vol


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REMOTE_CODE, env={**os.environ, "PYTHONPATH": REMOTE_CODE})
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


# ---------------------------------------------------------------- data movement
@app.local_entrypoint()
def upload(local_dir: str, remote_name: str) -> None:
    """Upload a processed graph directory (parquet files) into the volume."""
    src = Path(local_dir)
    if not src.exists():
        raise SystemExit(f"not found: {src}")
    with volume.batch_upload(force=True) as batch:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                batch.put_file(f, f"/processed/{remote_name}/{f.relative_to(src).as_posix()}")
                print("uploaded", f.name)
    print(f"done -> volume:/processed/{remote_name}")


@app.local_entrypoint()
def download(name: str, local_dir: str = "results") -> None:
    """Download checkpoints + metrics back to the local machine."""
    out = Path(local_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    for entry in volume.iterdir(f"/checkpoints/{name}", recursive=True):
        if entry.type.name != "FILE":
            continue
        rel = entry.path.split(f"/checkpoints/{name}/", 1)[-1]
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            for chunk in volume.read_file(entry.path):
                fh.write(chunk)
        print("downloaded", rel)
    print(f"done -> {out}")


@app.local_entrypoint()
def download_graph(name: str, local_dir: str = "data/processed") -> None:
    """Download a processed graph (parquet) built on Modal back to local."""
    out = Path(local_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    for entry in volume.iterdir(f"/processed/{name}", recursive=True):
        if entry.type.name != "FILE":
            continue
        rel = entry.path.split(f"/processed/{name}/", 1)[-1]
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            for chunk in volume.read_file(entry.path):
                fh.write(chunk)
        print("downloaded", rel)
    print(f"done -> {out}")


# ------------------------------------------------------------------ remote fns
@app.function(volumes={REMOTE_DATA: volume}, timeout=60 * 60 * 8, cpu=16.0, memory=65536)
def _build(input_rel: str, out_name: str, source_tag: str, label_mode: str,
           label: int, label_window: int, chunk_size: int, extra: str) -> None:
    """Legacy Mordor build against raw data already in the writable volume."""
    cmd = [
        "python", "scripts/build_graph.py", "--dataset", "mordor",
        "--input", f"{REMOTE_DATA}/raw/{input_rel}",
        "--out", f"{REMOTE_DATA}/processed/{out_name}",
        "--source-tag", source_tag,
        "--label-mode", label_mode,
        "--chunk-size", str(chunk_size),
    ]
    if label_mode == "force":
        cmd += ["--label", str(label)]
    if label_mode == "heuristic":
        cmd += ["--label-window", str(label_window)]
    cmd += [t for t in extra.split() if t]
    _run(cmd)
    volume.commit()


@app.function(volumes={REMOTE_DATA: volume, RAW_DATA: RAW_MOUNT},
              timeout=60 * 60 * 8, cpu=16.0, memory=65536)
def _build_ctu13(scenario_dir: str, out_name: str, source_tag: str,
                 background: str, limit: int, chunk_size: int) -> None:
    """Build one CTU-13 scenario graph from the READ-ONLY /raw source volume.

    Resolves the single `*.binetflow.xz` inside the scenario dir remotely (so
    the caller never needs the exact filename), then writes the processed graph
    to the writable output volume. Only the output volume is committed.
    """
    base = f"{RAW_DATA}/{scenario_dir}"
    matches = (
        sorted(glob.glob(f"{base}/*.binetflow.xz"))
        or sorted(glob.glob(f"{base}/**/*.binetflow.xz", recursive=True))
        or sorted(glob.glob(f"{base}/*.binetflow"))
        or sorted(glob.glob(f"{base}/**/*.binetflow", recursive=True))
    )
    if not matches:
        raise SystemExit(f"no *.binetflow[.xz] found under {base}")
    infile = matches[0]
    print(f"ctu13 build: {scenario_dir} -> {infile}", flush=True)
    cmd = [
        "python", "scripts/build_graph.py", "--dataset", "ctu13",
        "--input", infile,
        "--out", f"{REMOTE_DATA}/processed/{out_name}",
        "--source-tag", source_tag,
        "--label-mode", "parser",
        "--ctu13-background", background,
        "--chunk-size", str(chunk_size),
        # Chains are unused by training; ChainIdTracker on chain_id-less NetFlow
        # is O(malicious) and produces ~0 chains, avoiding the O(M^2) entity_time
        # blow-up on high-botnet-volume captures.
        "--strategies", "chain_id",
        "--max-subgraphs", "0",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    _run(cmd)
    volume.commit()  # OUTPUT volume only — raw_vol is never committed


@app.function(volumes={REMOTE_DATA: volume}, timeout=60 * 60 * 4, cpu=16.0, memory=65536)
def _merge(names: str, out_name: str) -> None:
    inputs = [f"{REMOTE_DATA}/processed/{n}" for n in names.split(",") if n]
    _run(["python", "scripts/merge_graphs.py", "--inputs", *inputs,
          "--out", f"{REMOTE_DATA}/processed/{out_name}"])
    volume.commit()


@app.function(volumes={REMOTE_DATA: volume}, timeout=60 * 60 * 4, cpu=8.0, memory=32768)
def _snapshots(name: str, window_size: float, stride: float, max_snapshots: int,
               node_feature_mode: str, edge_feature_mode: str, max_edges: int,
               out_name: str = "") -> None:
    # out_name lets a rebuild land beside the existing snapshots instead of
    # overwriting them (e.g. flow_agg -> ctu13_c47_flowagg).
    dest = out_name or name
    cmd = [
        "python", "scripts/build_snapshots.py",
        "--data", f"{REMOTE_DATA}/processed/{name}",
        "--out", f"{REMOTE_DATA}/snapshots/{dest}",
        "--window-size", str(window_size),
        "--stride", str(stride),
        "--node-feature-mode", node_feature_mode,
        "--edge-feature-mode", edge_feature_mode,
    ]
    if max_snapshots:
        cmd += ["--max-snapshots", str(max_snapshots)]
    if max_edges:
        cmd += ["--max-edges-per-snapshot", str(max_edges)]
    if dest != name:
        # Keep provenance pointing at the scenario, not the rebuild directory,
        # so held-out-family splits still match on scenario_id.
        cmd += ["--scenario-id", name]
    _run(cmd)
    volume.commit()


def _train_cmd(single_name: str, train_scen: str, test_scen: str, out_name: str,
               epochs: int, hidden: int, lr: float, target: str, extra: str,
               script: str = "train_tgnn.py") -> list[str]:
    """Build the training script argv for either single-dir or scenario-held-out."""
    cmd = [
        "python", f"scripts/{script}",
        "--out", f"{REMOTE_DATA}/checkpoints/{out_name}",
        "--epochs", str(epochs),
        "--hidden-channels", str(hidden),
        "--lr", str(lr),
        "--target", target,
    ]
    if train_scen or test_scen:
        cmd += ["--snapshots-root", f"{REMOTE_DATA}/snapshots"]
        if train_scen:
            cmd += ["--train-scenarios", train_scen]
        if test_scen:
            cmd += ["--test-scenarios", test_scen]
    else:
        cmd += ["--snapshots", f"{REMOTE_DATA}/snapshots/{single_name}"]
    cmd += [t for t in extra.split() if t]
    return cmd


@app.function(gpu="A10G", volumes={REMOTE_DATA: volume}, timeout=60 * 60 * 6, memory=32768)
def _train_gpu(single_name: str, train_scen: str, test_scen: str, out_name: str,
               epochs: int, hidden: int, lr: float, target: str, extra: str,
               script: str = "train_tgnn.py") -> None:
    _run(["nvidia-smi"])
    _run(_train_cmd(single_name, train_scen, test_scen, out_name, epochs, hidden, lr, target, extra, script))
    volume.commit()


@app.function(volumes={REMOTE_DATA: volume}, timeout=60 * 60 * 4, cpu=8.0, memory=32768)
def _train_cpu(single_name: str, train_scen: str, test_scen: str, out_name: str,
               epochs: int, hidden: int, lr: float, target: str, extra: str,
               script: str = "train_tgnn.py") -> None:
    _run(_train_cmd(single_name, train_scen, test_scen, out_name, epochs, hidden, lr, target, extra, script))
    volume.commit()


@app.function(volumes={REMOTE_DATA: volume}, timeout=60 * 60 * 2, cpu=8.0, memory=32768)
def _evaluate(ckpt_name: str, single_name: str, use_scenarios: bool, split: str) -> None:
    # evaluate_tgnn forces CPU + reconstructs the exact split from the saved
    # manifest, so no GPU is needed and train/eval always score the same rows.
    cmd = [
        "python", "scripts/evaluate_tgnn.py",
        "--checkpoint", f"{REMOTE_DATA}/checkpoints/{ckpt_name}/best_model.pt",
        "--out", f"{REMOTE_DATA}/checkpoints/{ckpt_name}/eval_{split}",
        "--split", split,
    ]
    if use_scenarios:
        cmd += ["--snapshots-root", f"{REMOTE_DATA}/snapshots"]
    else:
        cmd += ["--snapshots", f"{REMOTE_DATA}/snapshots/{single_name}"]
    _run(cmd)
    volume.commit()


@app.function(volumes={REMOTE_DATA: volume}, timeout=60 * 60 * 2, cpu=8.0, memory=32768)
def _evaluate_prometheus(ckpt_name: str, test_scenarios: str, snapshots_root: str,
                         window_size: int, seq_stride: int) -> None:
    """Evaluate a Prometheus checkpoint — evaluate_tgnn.py only handles
    edge-level TemporalGNN, so Prometheus needs its own evaluator. The
    readout (node-wise per paper Eq.4, or legacy graph-level pooling) is read
    back off the checkpoint, so this works for both."""
    out_dir = f"{REMOTE_DATA}/checkpoints/{ckpt_name}/eval_test"
    cmd = [
        "python", "scripts/eval_prometheus.py",
        "--checkpoint", f"{REMOTE_DATA}/checkpoints/{ckpt_name}/best_model.pt",
        "--snapshots-root", snapshots_root,
        "--test-scenarios", test_scenarios,
        "--out", out_dir,
        "--window-size", str(window_size),
        "--seq-stride", str(seq_stride),
    ]
    _run(cmd)
    volume.commit()


@app.local_entrypoint()
def evaluate_prometheus_ho(out_name: str = "ho", test_scenarios: str = "",
                           window_size: int = 10, seq_stride: int = 5) -> None:
    """Evaluate a held-out Prometheus checkpoint on its test scenarios."""
    _evaluate_prometheus.remote(out_name, test_scenarios or "ctu13_c47",
                               f"{REMOTE_DATA}/snapshots", window_size, seq_stride)


@app.function(volumes={REMOTE_DATA: volume}, timeout=60 * 10, cpu=2.0)
def _stats(prefix: str) -> list[dict]:
    """Read every /processed/<prefix>*/graph_stats.json and return the counts."""
    import json as _json

    volume.reload()
    root = Path(REMOTE_DATA) / "processed"
    rows: list[dict] = []
    if not root.exists():
        return rows
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        sp = d / "graph_stats.json"
        if not sp.exists():
            rows.append({"name": d.name, "missing": True})
            continue
        s = _json.loads(sp.read_text())
        g = s.get("graph", {})
        norm = s.get("normalization", {})
        rej = norm.get("rejected", {})
        rej_total = sum(rej.values()) if isinstance(rej, dict) else int(rej or 0)
        rows.append({
            "name": d.name,
            "events": int(g.get("total_events", 0)),
            "malicious": int(g.get("malicious_events", 0)),
            "benign": int(g.get("benign_events", 0)),
            "rejected": rej_total,
        })
    return rows


@app.function(volumes={REMOTE_DATA: volume}, timeout=60 * 30, cpu=8.0, memory=32768)
def _plots(ckpt_name: str, graph_name: str) -> None:
    """Render training/eval/graph figures on Modal (matplotlib/networkx/pyvis all present)."""
    volume.reload()
    ck = f"{REMOTE_DATA}/checkpoints/{ckpt_name}"
    outdir = f"{ck}/plots"
    os.makedirs(outdir, exist_ok=True)

    steps = [
        ("training_curves", ["python", "scripts/plot_training.py",
                             "--history", f"{ck}/history.json",
                             "--out", f"{outdir}/training_curves.png"]),
        ("evaluation_plots", ["python", "scripts/plot_evaluation.py",
                              "--predictions", f"{ck}/eval_test/predictions_test.parquet",
                              "--metrics", f"{ck}/eval_test/metrics_test.json",
                              "--out", f"{outdir}/evaluation_plots.png"]),
    ]
    if graph_name:
        steps.append(("graph_viz", ["python", "scripts/visualize_graph.py",
                                   "--data", f"{REMOTE_DATA}/processed/{graph_name}",
                                   "--out", outdir, "--interactive"]))

    ok, bad = [], []
    for name, cmd in steps:  # one figure failing must not discard the others
        try:
            _run(cmd); ok.append(name)
        except SystemExit as e:
            print(f"  FAILED {name}: exit {e.code}"); bad.append(name)
    volume.commit()
    print(f"plots done -> ok={ok} failed={bad}  (volume:/checkpoints/{ckpt_name}/plots)")


# --------------------------------------------------------- legacy CLI (Mordor)
@app.local_entrypoint()
def build(input_rel: str, out_name: str, source_tag: str = "mordor",
          label_mode: str = "heuristic", label: int = 1, label_window: int = 300,
          chunk_size: int = 50000, extra: str = "") -> None:
    _build.remote(input_rel, out_name, source_tag, label_mode, label,
                  label_window, chunk_size, extra)


@app.local_entrypoint()
def merge(names: str, out_name: str) -> None:
    _merge.remote(names, out_name)


@app.local_entrypoint()
def snapshots(name: str, window_size: float = 60, stride: float = 30,
              max_snapshots: int = 0, node_feature_mode: str = "type_degree",
              edge_feature_mode: str = "relation_time", max_edges: int = 200000) -> None:
    _snapshots.remote(name, window_size, stride, max_snapshots,
                      node_feature_mode, edge_feature_mode, max_edges)


@app.local_entrypoint()
def train(name: str, epochs: int = 50, hidden: int = 64, lr: float = 0.001,
          target: str = "node", extra: str = "") -> None:
    _train_gpu.remote(name, "", "", name, epochs, hidden, lr, target, extra)


@app.local_entrypoint()
def evaluate(name: str, split: str = "test") -> None:
    _evaluate.remote(name, name, False, split)


@app.local_entrypoint()
def all(name: str, window_size: float = 60, stride: float = 30, epochs: int = 50) -> None:
    _snapshots.remote(name, window_size, stride, 0, "type_degree", "relation_time", 200000)
    _train_gpu.remote(name, "", "", name, epochs, 64, 0.001, "node", "")
    _evaluate.remote(name, name, False, "test")


# ------------------------------------------------------------- CTU-13 CLI
def _discover_ctu13_dirs() -> list[str]:
    """Top-level `CTU-13.Dataset-NN` dirs in the read-only source volume."""
    dirs = []
    for entry in raw_vol.iterdir("/"):
        name = entry.path.rstrip("/").split("/")[-1]
        if name.startswith("CTU-13.Dataset-") and entry.type.name == "DIRECTORY":
            dirs.append(name)
    return sorted(dirs, key=lambda d: int(d.split("-")[-1]))


@app.local_entrypoint()
def build_ctu13_all(background: str = "benign", limit: int = 0,
                    chunk_size: int = 100000) -> None:
    """Build all 13 CTU-13 scenario graphs in parallel (CPU). One graph per dir.

    background='benign' keeps Background/Normal as label 0 (realistic imbalance,
    dirs named ctu13_cNN); 'drop' discards Background (clean, dirs ..._drop).
    """
    dirs = _discover_ctu13_dirs()
    if not dirs:
        raise SystemExit("no CTU-13.Dataset-* dirs in mega-10gb-dataset")
    suffix = "" if background == "benign" else f"_{background}"
    print(f"building {len(dirs)} scenarios (background={background}) ...")
    handles = []
    for d in dirs:
        cap = d.split("-")[-1]
        out_name = f"ctu13_c{cap}{suffix}"
        handles.append((out_name, _build_ctu13.spawn(d, out_name, cap, background, limit, chunk_size)))
    built, failed = [], []
    for out_name, h in handles:
        try:
            h.get()
            built.append(out_name)
            print("  built", out_name)
        except Exception as e:  # a dir with no *.binetflow (e.g. c51) must not abort the batch
            msg = (str(e).splitlines() or [""])[-1] or repr(e)
            failed.append((out_name, msg))
            print(f"  FAILED {out_name}: {msg}")
    print(f"done -> {len(built)} built, {len(failed)} failed  (volume:/processed/ctu13_c*)")
    if failed:
        print("failed:", ", ".join(n for n, _ in failed))


@app.local_entrypoint()
def stats(prefix: str = "ctu13_c") -> None:
    """Print authoritative per-scenario counts from the built graphs (no GPU)."""
    rows = _stats.remote(prefix)
    rows = sorted(rows, key=lambda r: r["name"])
    print(f"{'scenario':<16}{'events':>12}{'malicious':>12}{'benign':>14}{'mal%':>8}{'rej':>6}")
    print("-" * 68)
    tot_e = tot_m = 0
    for r in rows:
        if r.get("missing"):
            print(f"{r['name']:<16}{'(no graph_stats.json)':>52}")
            continue
        e, m, b = r["events"], r["malicious"], r["benign"]
        pct = (100.0 * m / e) if e else 0.0
        tot_e += e
        tot_m += m
        print(f"{r['name']:<16}{e:>12,}{m:>12,}{b:>14,}{pct:>7.2f}%{r['rejected']:>6}")
    print("-" * 68)
    print(f"{'TOTAL':<16}{tot_e:>12,}{tot_m:>12,}{'':>14}{(100.0*tot_m/tot_e if tot_e else 0):>7.2f}%")


@app.local_entrypoint()
def snapshots_ctu13(name: str, window_size: float = 60, stride: float = 30,
                    max_snapshots: int = 0, max_edges: int = 200000) -> None:
    """Flow snapshots for one CTU-13 graph (node=type_only, edge=flow)."""
    _snapshots.remote(name, window_size, stride, max_snapshots, "type_only", "flow", max_edges)


@app.local_entrypoint()
def snapshots_ctu13_all(window_size: float = 60, stride: float = 30,
                        max_snapshots: int = 0, max_edges: int = 200000,
                        suffix: str = "") -> None:
    """Flow snapshots for every ctu13_c* processed graph in the volume."""
    names = []
    for entry in volume.iterdir("/processed"):
        n = entry.path.rstrip("/").split("/")[-1]
        if n.startswith("ctu13_c") and entry.type.name == "DIRECTORY":
            if (suffix and n.endswith(suffix)) or (not suffix and not n.endswith("_drop")):
                names.append(n)
    names = sorted(names)
    if not names:
        raise SystemExit("no ctu13_c* processed graphs found; run build_ctu13_all first")
    handles = [(n, _snapshots.spawn(n, window_size, stride, max_snapshots, "type_only", "flow", max_edges))
               for n in names]
    for n, h in handles:
        h.get()
        print("  snapshots", n)


@app.local_entrypoint()
def snapshots_ctu13_subset(names: str, window_size: float = 60, stride: float = 30,
                           max_snapshots: int = 0, max_edges: int = 200000) -> None:
    """Flow snapshots for a comma-separated subset of ctu13_c* graphs (parallel CPU)."""
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    if not wanted:
        raise SystemExit("pass --names ctu13_c52,ctu13_c46,...")
    handles = [(n, _snapshots.spawn(n, window_size, stride, max_snapshots, "type_only", "flow", max_edges))
               for n in wanted]
    built, failed = [], []
    for n, h in handles:
        try:
            h.get()
            built.append(n)
            print("  snapshots", n)
        except Exception as e:
            msg = (str(e).splitlines() or [""])[-1] or repr(e)
            failed.append(n)
            print(f"  FAILED {n}: {msg}")
    print(f"done -> {len(built)} snapshotted, {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed))


@app.local_entrypoint()
def snapshots_ctu13_flowagg(names: str, window_size: float = 60, stride: float = 30,
                            max_snapshots: int = 0, max_edges: int = 200000,
                            suffix: str = "_flowagg") -> None:
    """Rebuild CTU-13 snapshots with node=flow_agg (edge=flow), writing to
    snapshots/<name><suffix> so the existing type_only snapshots survive.

    Needed because the paper's Eq. (3) aggregates node states only: with
    type_only the node feature is a constant column, every GAT attention weight
    is uniform, and every node collapses to the same representation
    (AUC-ROC == 0.5).  flow_agg gives each node leakage-free per-window
    topology + flow statistics so the paper's architecture is trainable.
    """
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    if not wanted:
        raise SystemExit("pass --names ctu13_c52,ctu13_c46,...")
    handles = [(n, _snapshots.spawn(n, window_size, stride, max_snapshots,
                                    "flow_agg", "flow", max_edges, f"{n}{suffix}"))
               for n in wanted]
    built, failed = [], []
    for n, h in handles:
        try:
            h.get()
            built.append(n)
            print(f"  snapshots {n} -> {n}{suffix}")
        except Exception as e:
            msg = (str(e).splitlines() or [""])[-1] or repr(e)
            failed.append(n)
            print(f"  FAILED {n}: {msg}")
    print(f"done -> {len(built)} snapshotted, {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed))


@app.function(volumes={REMOTE_DATA: volume}, timeout=60 * 30, cpu=4.0, memory=16384)
def _snapshot_density(names: str) -> None:
    """Report real per-window node/edge density for built snapshot dirs.

    Over-smoothing severity is governed by average node degree, so this is the
    measurement that says whether the paper's literal 3-layer depth is viable
    on CTU-13 or collapses the way it does on dense synthetic graphs.
    """
    import pickle
    import statistics
    from pathlib import Path

    for name in [n.strip() for n in names.split(",") if n.strip()]:
        d = Path(f"{REMOTE_DATA}/snapshots/{name}")
        files = sorted(d.glob("snapshot_*.pkl"))
        if not files:
            print(f"  {name}: NO SNAPSHOTS")
            continue
        step = max(1, len(files) // 200)  # sample up to ~200 windows
        nodes, edges, degs, posfrac = [], [], [], []
        for f in files[::step]:
            with open(f, "rb") as fh:
                s = pickle.load(fh)
            n = int(s["x"].shape[0])
            e = int(s["edge_index"].shape[1])
            nodes.append(n)
            edges.append(e)
            degs.append(2.0 * e / max(n, 1))
            nl = s.get("node_labels")
            if nl is not None and len(nl):
                posfrac.append(float((nl > 0).mean()))
        print(f"  {name}: windows={len(files)} sampled={len(nodes)}")
        print(f"    nodes/window   median={statistics.median(nodes):.0f} "
              f"mean={statistics.mean(nodes):.0f} max={max(nodes)}")
        print(f"    edges/window   median={statistics.median(edges):.0f} "
              f"mean={statistics.mean(edges):.0f} max={max(edges)}")
        print(f"    avg degree     median={statistics.median(degs):.2f} "
              f"mean={statistics.mean(degs):.2f} max={max(degs):.2f}")
        if posfrac:
            print(f"    positive nodes median={statistics.median(posfrac):.4f} "
                  f"mean={statistics.mean(posfrac):.4f}")


@app.local_entrypoint()
def snapshot_density(names: str) -> None:
    """CPU-only: print node/edge/degree density per snapshot directory."""
    _snapshot_density.remote(names)


@app.local_entrypoint()
def smoke(cap: str = "52", window_size: float = 30, stride: float = 10,
          epochs: int = 3, limit: int = 0, background: str = "benign",
          max_snapshots: int = 0, max_edges: int = 200000) -> None:
    """Cents-level CPU end-to-end on ONE scenario: build -> flow snapshots ->
    edge-level train (CPU) -> evaluate. Verifies the whole CTU-13 pipeline
    before any paid GPU run. Denser snapshots (30s/10s) + a small block size
    give a short capture enough sequences for a non-degenerate split."""
    d = f"CTU-13.Dataset-{cap}"
    name = f"ctu13_c{cap}"
    print(f"[smoke 1/4] build {d} -> {name}")
    _build_ctu13.remote(d, name, cap, background, limit, 100000)
    print(f"[smoke 2/4] snapshots {name} (flow / type_only)")
    _snapshots.remote(name, window_size, stride, max_snapshots, "type_only", "flow", max_edges)
    print(f"[smoke 3/4] train {name} (edge, CPU, {epochs} epochs)")
    _train_cpu.remote(name, "", "", name, epochs, 64, 0.001, "edge", "--block-size 5")
    print(f"[smoke 4/4] evaluate {name}")
    _evaluate.remote(name, name, False, "test")
    print(f"[smoke] done -> volume:/checkpoints/{name}")


@app.local_entrypoint()
def train_ho(train_scenarios: str, test_scenarios: str, out_name: str = "ho",
             epochs: int = 30, hidden: int = 64, lr: float = 0.001,
             target: str = "edge", extra: str = "", script: str = "train_tgnn.py") -> None:
    """GPU scenario/family-held-out training (edge-level by default)."""
    _train_gpu.remote("", train_scenarios, test_scenarios, out_name, epochs, hidden, lr, target, extra, script)


@app.local_entrypoint()
def evaluate_ho(out_name: str = "ho", split: str = "test") -> None:
    """Evaluate a held-out checkpoint on its manifest test scenarios."""
    _evaluate.remote(out_name, "", True, split)


@app.local_entrypoint()
def prometheus_paper(train_scenarios: str, test_scenarios: str,
                     out_name: str = "prometheus_paper",
                     epochs: int = 200, batch_size: int = 256,
                     seq_stride: int = 5, seeds: str = "",
                     cpu: bool = False, extra: str = "") -> None:
    """Paper-faithful Prometheus run (base paper Eq. 3-5).

    Defaults reproduce the published architecture exactly: node-wise softmax
    readout with no pooling, unweighted cross-entropy, no LayerNorm, no edge
    attributes in aggregation, no gradient clipping, Adam(1e-3, 1e-5),
    minibatch 256, 200 epochs, 10% reservoir rehearsal.

    Pass seeds="42,43,44,45,46" for the paper's 5-seed mean +/- std.
    Pass cpu=True for the smoke test before spending GPU time.
    """
    flags = f"--batch-size {batch_size} --seq-stride {seq_stride}"
    if seeds:
        flags += f" --seeds {seeds}"
    if extra:
        flags += f" {extra}"
    runner = _train_cpu if cpu else _train_gpu
    runner.remote("", train_scenarios, test_scenarios, out_name, epochs,
                  128, 0.001, "node", flags, "train_prometheus.py")


@app.local_entrypoint()
def plots(ckpt_name: str, graph_name: str = "") -> None:
    """Generate training curves, evaluation plots, and graph visualizations on Modal (CPU).

    All plotting deps (matplotlib, sklearn, networkx, pyvis) are in the Modal
    image. The outputs land in /checkpoints/<ckpt_name>/plots/ on the volume.
    Use `download` to pull them locally afterwards.
    """
    _plots.remote(ckpt_name, graph_name)


@app.local_entrypoint()
def download_plots(ckpt_name: str, local_dir: str = "reports") -> None:
    """Download rendered plots from a checkpoint's plots/ directory."""
    prefix = f"/checkpoints/{ckpt_name}/plots"
    out = Path(local_dir) / ckpt_name
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for entry in volume.iterdir(prefix, recursive=True):
        if entry.type.name != "FILE":
            continue
        rel = entry.path.split(f"{prefix}/", 1)[-1]
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            for chunk in volume.read_file(entry.path):
                fh.write(chunk)
        count += 1
        print("downloaded", rel)
    print(f"done -> {out}  ({count} files)")

# >>> PROMETHEUS_PAPER_ALIGNED_BEGIN >>>
@app.function(
    gpu="A10G",
    volumes={REMOTE_DATA: volume},
    timeout=60 * 60 * 24,
    memory=65536,
)
def _train_prometheus_paper(
    train_scenarios: str,
    test_scenarios: str,
    out_name: str,
    seeds: str,
    lambda_rehearsal: float,
) -> None:
    """Paper-configuration Prometheus run. All heavy work stays on Modal."""
    _run(["nvidia-smi"])
    cmd = [
        "python", "scripts/train_prometheus.py",
        "--snapshots-root", f"{REMOTE_DATA}/snapshots",
        "--train-scenarios", train_scenarios,
        "--test-scenarios", test_scenarios,
        "--out", f"{REMOTE_DATA}/checkpoints/{out_name}",
        "--epochs", "200",
        "--batch-size", "256",
        "--lr", "0.001",
        "--weight-decay", "0.00001",
        "--buffer-pct", "0.10",
        "--lambda-rehearsal", str(lambda_rehearsal),
        "--seeds", seeds,
        "--target", "node",
        "--hidden-channels", "128",
        "--gnn-layers", "3",
        "--dropout", "0.2",
        "--heads", "1",
    ]
    _run(cmd)
    volume.commit()


@app.local_entrypoint()
def train_prometheus_paper(
    train_scenarios: str,
    test_scenarios: str,
    out_name: str = "prometheus_paper",
    seeds: str = "42,43,44,45,46",
    lambda_rehearsal: float = 1.0,
) -> None:
    """Submit the strict paper configuration to an A10G cloud GPU."""
    _train_prometheus_paper.remote(
        train_scenarios,
        test_scenarios,
        out_name,
        seeds,
        lambda_rehearsal,
    )


@app.local_entrypoint()
def evaluate_prometheus_paper(
    out_name: str = "prometheus_paper",
    test_scenarios: str = "ctu13_c47",
) -> None:
    """Evaluate the compatibility checkpoint produced by the first seed."""
    _evaluate_prometheus.remote(
        out_name,
        test_scenarios,
        f"{REMOTE_DATA}/snapshots",
        1,
        1,
    )
# <<< PROMETHEUS_PAPER_ALIGNED_END <<<

# >>> PROMETHEUS_CTU13_PAPER_EXTRAS_BEGIN >>>
@app.function(
    gpu="A10G",
    volumes={REMOTE_DATA: volume},
    timeout=60 * 60 * 24,
    memory=65536,
)
def _prometheus_experiment_extra(
    experiment: str,
    train_scenarios: str,
    test_scenarios: str,
    out_name: str,
    seeds: str,
    epochs: int,
    extra: str,
) -> None:
    # Run one CTU-13 paper baseline/ablation on A10G.
    _run(["nvidia-smi"])
    cmd = [
        "python", "scripts/train_prometheus_experiment.py",
        "--snapshots-root", f"{REMOTE_DATA}/snapshots",
        "--train-scenarios", train_scenarios,
        "--test-scenarios", test_scenarios,
        "--out", f"{REMOTE_DATA}/checkpoints/{out_name}",
        "--experiment", experiment,
        "--epochs", str(epochs),
        "--batch-size", "256",
        "--lr", "0.001",
        "--weight-decay", "0.00001",
        "--buffer-pct", "0.10",
        "--lambda-rehearsal", "1.0",
        "--seeds", seeds,
    ]
    if extra:
        cmd += [x for x in extra.split() if x]
    _run(cmd)
    volume.commit()


@app.local_entrypoint()
def prometheus_experiment(
    experiment: str,
    train_scenarios: str,
    test_scenarios: str,
    out_name: str,
    seeds: str = "42,43,44,45,46",
    epochs: int = 200,
    extra: str = "",
) -> None:
    # Submit one paper comparison/ablation experiment to Modal A10G.
    # For paper-style CTU-13 Prometheus runs, pass *_flowagg snapshot names.
    _prometheus_experiment_extra.remote(
        experiment, train_scenarios, test_scenarios, out_name, seeds, epochs, extra
    )


@app.function(
    gpu="A10G",
    volumes={REMOTE_DATA: volume},
    timeout=60 * 60 * 2,
    memory=32768,
)
def _benchmark_prometheus_extra(checkpoint_name: str, scenarios: str) -> None:
    ckpt = f"{REMOTE_DATA}/checkpoints/{checkpoint_name}/best_model.pt"
    out = f"{REMOTE_DATA}/checkpoints/{checkpoint_name}/benchmark.json"
    _run([
        "python", "scripts/benchmark_prometheus.py",
        "--checkpoint", ckpt,
        "--snapshots-root", f"{REMOTE_DATA}/snapshots",
        "--scenarios", scenarios,
        "--out", out,
        "--batch-size", "1",
        "--warmup", "10",
        "--repeats", "100",
        "--max-graphs", "100",
    ])
    volume.commit()


@app.local_entrypoint()
def benchmark_prometheus_extra(
    checkpoint_name: str = "prometheus_paper",
    scenarios: str = "ctu13_c47_flowagg",
) -> None:
    # Measure Prometheus latency/memory on A10G with an explicit protocol.
    _benchmark_prometheus_extra.remote(checkpoint_name, scenarios)


@app.function(
    volumes={REMOTE_DATA: volume},
    timeout=60 * 60,
    cpu=4.0,
    memory=16384,
)
def _explain_prometheus_extra(checkpoint_name: str, scenario: str) -> None:
    ckpt = f"{REMOTE_DATA}/checkpoints/{checkpoint_name}/best_model.pt"
    out = f"{REMOTE_DATA}/checkpoints/{checkpoint_name}/explanations/{scenario}"
    _run([
        "python", "scripts/explain_prometheus.py",
        "--checkpoint", ckpt,
        "--snapshots", f"{REMOTE_DATA}/snapshots/{scenario}",
        "--out", out,
        "--top-nodes", "25",
        "--top-edges", "100",
        "--max-snapshots", "10",
    ])
    volume.commit()


@app.local_entrypoint()
def explain_prometheus_extra(
    checkpoint_name: str = "prometheus_paper",
    scenario: str = "ctu13_c47_flowagg",
) -> None:
    # Export attention-backed malicious-subgraph evidence.
    _explain_prometheus_extra.remote(checkpoint_name, scenario)


@app.function(
    volumes={REMOTE_DATA: volume},
    timeout=60 * 30,
    cpu=2.0,
    memory=8192,
)
def _report_prometheus_extra(root_name: str) -> None:
    root = f"{REMOTE_DATA}/checkpoints/{root_name}"
    _run([
        "python", "scripts/report_prometheus_paper.py",
        "--root", root,
        "--out", f"{root}/PROMETHEUS_CTU13_COMPARISON.md",
    ])
    volume.commit()


@app.local_entrypoint()
def report_prometheus_extra(root_name: str = "paper_suite") -> None:
    # Generate the CTU-13 paper-comparison Markdown report.
    _report_prometheus_extra.remote(root_name)
# <<< PROMETHEUS_CTU13_PAPER_EXTRAS_END <<<
