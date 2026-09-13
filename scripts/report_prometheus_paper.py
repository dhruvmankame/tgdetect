#!/usr/bin/env python3
"""Build a compact CTU-13 paper-comparison report from experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    "cnn_bilstm": "CNN-BiLSTM + attention",
    "static_gnn": "Static GNN",
    "finetune": "Fine-tuned GNN (no replay)",
    "generative_replay": "Generative replay IDS (CTU-13 proxy)",
    "full_rehearsal": "Prometheus GNN + rehearsal CL",
    "cnn_bilstm_rehearsal": "Sequential CNN-BiLSTM + rehearsal",
}

PAPER_REPORTED = {
    "cnn_bilstm": {"precision": 0.935, "recall": 0.928, "f1": 0.931, "auc_roc": 0.94, "latency_ms": 2.4},
    "static_gnn": {"precision": 0.961, "recall": 0.949, "f1": 0.955, "auc_roc": 0.96, "latency_ms": 4.1},
    "finetune": {"precision": 0.967, "recall": 0.956, "f1": 0.960, "auc_roc": 0.97, "latency_ms": 5.0},
    "generative_replay": {"precision": 0.976, "recall": 0.963, "f1": 0.969, "auc_roc": 0.97, "latency_ms": 8.2},
    "full_rehearsal": {"precision": 0.985, "recall": 0.978, "f1": 0.981, "auc_roc": 0.98, "latency_ms": 6.7},
}


def fmt(mean, std, key):
    m = mean.get(key)
    s = std.get(key)
    if m is None:
        return "—"
    if isinstance(m, float) and m != m:
        return "—"
    return f"{m:.4f} ± {s:.4f}" if isinstance(s, (int, float)) and s == s else f"{m:.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="directory containing one subdir per experiment")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = []
    for exp, label in LABELS.items():
        path = args.root / exp / "summary.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append((exp, label, data))

    lines = [
        "# Prometheus CTU-13 Paper-Comparison Report", "",
        "> The IEEE paper used Wazuh/sandbox APT telemetry. These project results use CTU-13, so the numbers are **not a matched-dataset reproduction**. The paper values below are reference values only.", "",
        "## Main comparison", "",
        "| Model | CTU-13 Precision | CTU-13 Recall | CTU-13 F1 | CTU-13 AUC | Avg forgetting F1 | Paper F1 (reference) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for exp, label, data in rows:
        mean, std = data.get("mean", {}), data.get("std", {})
        paper = PAPER_REPORTED.get(exp, {})
        lines.append(
            f"| {label} | {fmt(mean,std,'precision')} | {fmt(mean,std,'recall')} | "
            f"{fmt(mean,std,'f1')} | {fmt(mean,std,'auc_roc')} | "
            f"{fmt(mean,std,'average_forgetting_f1')} | "
            f"{paper.get('f1', '—') if paper else '—'} |"
        )

    lines += ["", "## Ablation mapping", "",
              "| Paper ablation concept | Project experiment |",
              "|---|---|",
              "| GNN only, no rehearsal | `static_gnn` |",
              "| Sequential baseline + rehearsal | `cnn_bilstm_rehearsal` |",
              "| Fine-tuned GNN, no buffer | `finetune` |",
              "| Full GNN + rehearsal CL | `full_rehearsal` |",
              "", "## Interpretation rules", "",
              "- Use the five-seed mean ± sample standard deviation for claims.",
              "- Use average forgetting to support or reject the continual-learning claim.",
              "- Compare latency only when hardware and timing protocol are disclosed.",
              "- The `generative_replay` experiment is a transparent conditional-VAE proxy because the paper does not publish that baseline's architecture.",
              "- CTU-13 family/scenario breakdown replaces tactic-wise Wazuh reporting; do not fabricate MITRE tactic labels for CTU-13.",
              ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    json_out = args.out.with_suffix(".json")
    json_out.write_text(json.dumps({"experiments": [r[0] for r in rows]}, indent=2), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()

