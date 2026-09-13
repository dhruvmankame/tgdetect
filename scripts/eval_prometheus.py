#!/usr/bin/env python3
"""Evaluate a paper-aligned Prometheus checkpoint at node level."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.prometheus import Prometheus, snapshot_to_data


def load_graphs(path: Path) -> list:
    files = sorted(path.glob("snapshot_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No snapshot_*.pkl files in {path}")
    out = []
    for f in files:
        with open(f, "rb") as fh:
            out.append(snapshot_to_data(pickle.load(fh)))
    return out


@torch.no_grad()
def evaluate(model: Prometheus, graphs: list, device: torch.device, batch_size: int = 256):
    model.eval()
    ys, ps, scores = [], [], []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        pred = torch.argmax(logits, dim=-1)
        ys.extend(batch.y.cpu().numpy().astype(np.int64).tolist())
        ps.extend(pred.cpu().numpy().astype(np.int64).tolist())
        scores.extend(prob.cpu().numpy().astype(np.float64).tolist())

    y = np.asarray(ys, dtype=np.int64)
    p = np.asarray(ps, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    cm = confusion_matrix(y, p, labels=[0, 1])
    metrics = {
        "num_nodes": int(y.size),
        "num_positive": int(y.sum()),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "accuracy": float(accuracy_score(y, p)),
        "auc_roc": float(roc_auc_score(y, s)) if np.unique(y).size > 1 else float("nan"),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--snapshots-root", required=True)
    ap.add_argument("--test-scenarios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    # Accepted only so the repository's older Modal wrapper remains compatible.
    ap.add_argument("--window-size", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--seq-stride", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    in_channels = int(ckpt["meta"]["node_feature_dim"])
    model = Prometheus(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    all_graphs = []
    per_scenario = {}
    for name in [s.strip() for s in args.test_scenarios.split(",") if s.strip()]:
        graphs = load_graphs(Path(args.snapshots_root) / name)
        per_scenario[name] = evaluate(model, graphs, device, args.batch_size)
        all_graphs.extend(graphs)

    overall = evaluate(model, all_graphs, device, args.batch_size)
    out = {
        "target": "node",
        "overall": overall,
        "per_scenario": per_scenario,
        "paper_config": ckpt.get("paper_config", {}),
        "seed": ckpt.get("seed"),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("===== Prometheus paper-aligned evaluation (node-level) =====")
    print(json.dumps(overall, indent=2))
    print(f"saved -> {out_dir / 'metrics_test.json'}")


if __name__ == "__main__":
    main()
