#!/usr/bin/env python3
"""Offline smoke test for the paper-faithful Prometheus retrofit.

Generates synthetic snapshots that match the CTU-13 schema produced by
`graph_builder.temporal.SnapshotBuilder.to_dict()` exactly (including the
`node_labels` array the node-wise readout consumes), then runs the full
train -> checkpoint -> standalone-eval path on CPU.

This validates tensor shapes, loss wiring, batching, checkpoint round-tripping
and the evaluator without touching Modal or any real data. It proves nothing
about accuracy — the labels are random.

Usage:
    venv/bin/python scripts/smoke_prometheus.py
"""
from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NODE_DIM = 1          # CTU-13 type_only: a single all-ones node-type column
EDGE_DIM = 37         # CTU-13 flow edge features


def make_scenario(out: Path, scenario_id: str, n_snapshots: int, rng: np.random.Generator,
                  mal_rate: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n_snapshots):
        n_nodes = int(rng.integers(8, 40))
        n_edges = int(rng.integers(n_nodes, n_nodes * 4))
        src = rng.integers(0, n_nodes, size=n_edges)
        dst = rng.integers(0, n_nodes, size=n_edges)
        edge_labels = (rng.random(n_edges) < mal_rate).astype(np.int64)

        # Same rule as temporal.py: a node is positive iff it SOURCES a
        # malicious edge in this window.
        node_labels = np.zeros(n_nodes, dtype=np.int64)
        np.maximum.at(node_labels, src, edge_labels)

        snap = {
            "ts_start": float(i * 30),
            "ts_end": float(i * 30 + 60),
            "node_ids": [f"10.0.0.{j}" for j in range(n_nodes)],
            "node_types": np.zeros(n_nodes, dtype=np.int64),
            "x": np.ones((n_nodes, NODE_DIM), dtype=np.float32),
            "edge_index": np.stack([src, dst]).astype(np.int64),
            "edge_attr": rng.random((n_edges, EDGE_DIM)).astype(np.float32),
            "edge_labels": edge_labels,
            "node_labels": node_labels,
            "snapshot_label": int(edge_labels.max()) if n_edges else 0,
            "num_events": n_edges,
            "scenario_id": scenario_id,
        }
        with open(out / f"snapshot_{i:06d}.pkl", "wb") as f:
            pickle.dump(snap, f)

    meta = {
        "source": "SYNTHETIC-SMOKE", "scenario_id": scenario_id,
        "num_snapshots": n_snapshots, "window_size_s": 60.0, "stride_s": 30.0,
        "node_feature_mode": "type_only", "edge_feature_mode": "flow",
        "node_feature_dim": NODE_DIM, "edge_feature_dim": EDGE_DIM,
        "num_node_types": 1, "num_relations": 1,
        "node_type_map": {"ip": 0}, "relation_map": {"flow": 0},
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*72}\n{label}\n{'='*72}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({label}): exit {r.returncode}")


def run_expect_fail(cmd: list[str], label: str) -> None:
    """Assert a command exits non-zero (used for the --paper-exact guard rail)."""
    print(f"\n{'='*72}\n{label}\n{'='*72}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode == 0:
        raise SystemExit(f"FAILED ({label}): expected a non-zero exit, got 0")
    print(f"  rejected as expected (exit {r.returncode})")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="prom_smoke_"))
    try:
        rng = np.random.default_rng(0)
        root = tmp / "snapshots"
        make_scenario(root / "ctu13_c52", "ctu13_c52", 60, rng, 0.15)
        make_scenario(root / "ctu13_c46", "ctu13_c46", 60, rng, 0.10)
        make_scenario(root / "ctu13_c47", "ctu13_c47", 40, rng, 0.20)
        print(f"synthetic snapshots -> {root}")

        py = sys.executable
        base = [
            py, "scripts/train_prometheus.py",
            "--snapshots-root", str(root),
            "--train-scenarios", "ctu13_c52,ctu13_c46",
            "--test-scenarios", "ctu13_c47",
            "--window-size", "5", "--seq-stride", "2",
            "--epochs", "3", "--batch-size", "16", "--patience", "3",
        ]

        # 1. Paper-faithful defaults: node-wise, no LN, no edge attrs, no clip.
        out_paper = tmp / "paper"
        run(base + ["--out", str(out_paper)], "1/7  paper-faithful (node-wise, Eq. 4)")

        # 2. Standalone evaluator against that checkpoint.
        run([py, "scripts/eval_prometheus.py",
             "--checkpoint", str(out_paper / "best_model.pt"),
             "--snapshots-root", str(root), "--test-scenarios", "ctu13_c47",
             "--out", str(out_paper / "eval_test"),
             "--window-size", "5", "--seq-stride", "2"],
            "2/7  standalone evaluator (node-wise checkpoint)")

        # 3. Superset ablation: every non-paper extra switched on at once.
        out_sup = tmp / "superset"
        run(base + ["--out", str(out_sup), "--use-layer-norm", "--use-edge-attr",
                    "--grad-clip", "1.0", "--class-weight", "--residual"],
            "3/7  superset ablation (LN + edge attrs + clip + weights + residual)")
        run([py, "scripts/eval_prometheus.py",
             "--checkpoint", str(out_sup / "best_model.pt"),
             "--snapshots-root", str(root), "--test-scenarios", "ctu13_c47",
             "--out", str(out_sup / "eval_test"),
             "--window-size", "5", "--seq-stride", "2"],
            "3b/7 standalone evaluator (superset checkpoint)")

        # 4. Legacy graph-level readout still works (backwards compatibility).
        out_graph = tmp / "graph"
        run(base + ["--out", str(out_graph), "--target", "graph"],
            "4/7  legacy graph-level readout")

        # 5. Multi-seed aggregation (paper reports mean +/- std over 5 seeds).
        out_seeds = tmp / "seeds"
        run(base + ["--out", str(out_seeds), "--seeds", "42,43"],
            "5/7  multi-seed mean +/- std")

        # 6. Paper-exact protocol: no schedule/early-stop, final-epoch model, argmax.
        out_pe = tmp / "paper_exact"
        run(base + ["--out", str(out_pe), "--paper-exact"],
            "6/7  paper-exact training protocol (final-epoch model, argmax)")
        run([py, "scripts/eval_prometheus.py",
             "--checkpoint", str(out_pe / "best_model.pt"),
             "--snapshots-root", str(root), "--test-scenarios", "ctu13_c47",
             "--out", str(out_pe / "eval_test"),
             "--window-size", "5", "--seq-stride", "2"],
            "6b/7 standalone evaluator (paper-exact checkpoint)")

        # 7. --paper-exact must REFUSE to run beside a non-paper architecture flag.
        run_expect_fail(base + ["--out", str(tmp / "reject"),
                                "--paper-exact", "--residual"],
                        "7/7  paper-exact rejects a non-paper flag (--residual)")

        print(f"\n{'='*72}\nARTEFACT CHECK\n{'='*72}")
        ok = True
        expect = [
            out_paper / "best_model.pt", out_paper / "history.json",
            out_paper / "test_metrics.json", out_paper / "eval_test" / "metrics_test.json",
            out_sup / "eval_test" / "metrics_test.json",
            out_graph / "test_metrics.json",
            out_seeds / "seed_summary.json",
            out_seeds / "seed_42" / "best_model.pt",
            out_seeds / "seed_43" / "best_model.pt",
            out_pe / "best_model.pt", out_pe / "eval_test" / "metrics_test.json",
        ]
        for p in expect:
            mark = "OK  " if p.exists() else "MISS"
            ok &= p.exists()
            print(f"  [{mark}] {p.relative_to(tmp)}")

        tm = json.loads((out_paper / "test_metrics.json").read_text())
        print(f"\n  paper-run target        : {tm['target']}")
        print(f"  paper-run test units    : {tm['overall']['num_samples']}")
        print(f"  paper-run positives     : {tm['overall']['num_positive']}")
        print(f"  confusion @argmax       : {tm['confusion_matrix_at_argmax']}")
        gm = json.loads((out_graph / "test_metrics.json").read_text())
        print(f"  graph-run test units    : {gm['overall']['num_samples']} "
              f"(one per snapshot — the old collapsed view)")
        ss = json.loads((out_seeds / "seed_summary.json").read_text())
        print(f"  multi-seed aggregate    : "
              f"f1@argmax {ss['aggregate']['f1_at_argmax']['mean']:.4f} "
              f"+/- {ss['aggregate']['f1_at_argmax']['std']:.4f}")

        if tm["target"] != "node":
            ok = False
            print("  !! paper run did not use the node-wise readout")
        if tm["overall"]["num_samples"] <= gm["overall"]["num_samples"]:
            ok = False
            print("  !! node-wise run produced no more units than the graph run")

        # Every opt-in deviation must survive the checkpoint -> evaluator trip.
        pa = json.loads((out_paper / "eval_test" / "metrics_test.json").read_text())["architecture"]
        sa = json.loads((out_sup / "eval_test" / "metrics_test.json").read_text())["architecture"]
        print(f"  paper-run architecture  : {pa}")
        print(f"  superset architecture   : {sa}")
        for k in ("use_layer_norm", "use_edge_attr", "residual"):
            if pa[k] is not False:
                ok = False
                print(f"  !! paper defaults enabled non-paper option {k}")
            if sa[k] is not True:
                ok = False
                print(f"  !! superset lost option {k} through the checkpoint")

        # Paper-exact run: architecture identical to the defaults, plus the
        # training-protocol flag recorded and argmax (0.5) used for decisions.
        pe = json.loads((out_pe / "eval_test" / "metrics_test.json").read_text())
        print(f"  paper-exact architecture: {pe['architecture']}")
        print(f"  paper-exact training    : {pe.get('training')}")
        if not pe.get("training", {}).get("paper_exact", False):
            ok = False
            print("  !! paper-exact checkpoint did not record paper_exact=True")
        if abs(float(pe["overall"].get("saved_threshold", -1.0)) - 0.5) > 1e-9:
            ok = False
            print("  !! paper-exact did not decide at argmax / threshold 0.5")
        for k in ("use_layer_norm", "use_edge_attr", "residual"):
            if pe["architecture"][k] is not False:
                ok = False
                print(f"  !! paper-exact enabled non-paper architecture option {k}")

        print(f"\n{'SMOKE TEST PASSED' if ok else 'SMOKE TEST FAILED'}")
        if not ok:
            raise SystemExit(1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
