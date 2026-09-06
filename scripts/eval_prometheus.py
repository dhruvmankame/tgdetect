#!/usr/bin/env python3
"""Standalone Prometheus evaluator — runs the same evaluate/predict/scored_metrics
as train_prometheus.py so the user gets the full metrics bundle without needing
evaluate_tgnn.py (which only handles edge-level TemporalGNN)."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/root/tgdetect")
import torch
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score,
)
from models.prometheus import Prometheus, snapshot_to_data
from models.rehearsal_buffer import RehearsalBuffer  # noqa: F401


def load_snapshots(d):
    files = sorted(Path(d).glob("snapshot_*.pkl"))
    snaps = [pickle.load(open(f, "rb")) for f in files]
    meta = json.load(open(Path(d) / "meta.json"))
    return snaps, meta


def build_sequences(snaps, ws, stride):
    return [snaps[i:i + ws] for i in range(0, len(snaps) - ws + 1, max(1, stride))]


def load_scenario_sequences(d, ws, stride):
    snaps, meta = load_snapshots(d)
    return build_sequences(snaps, ws, stride), meta, d.name


def recall_at_fpr(y_true, y_prob, fpr=0.01):
    neg, pos = y_prob[y_true == 0], y_prob[y_true == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    return float((pos >= np.quantile(neg, 1 - fpr)).mean())


def scored_metrics(y_true, y_prob, threshold):
    n = len(y_true)
    two = len(np.unique(y_true)) > 1
    def _pt(thr):
        p = (y_prob >= thr).astype(int)
        tp = int(((y_true == 1) & (p == 1)).sum())
        fp = int(((y_true == 0) & (p == 1)).sum())
        fn = int(((y_true == 1) & (p == 0)).sum())
        tn = int(((y_true == 0) & (p == 0)).sum())
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        re = tp / (tp + fn) if (tp + fn) else 0.0
        return {"f1": 2 * pr * re / (pr + re) if (pr + re) else 0.0,
                "precision": pr, "recall": re, "accuracy": (tp + tn) / n if n else 0.0}
    saved = _pt(threshold)
    out = {"num_samples": n, "num_positive": int(y_true.sum()),
           "recall_at_1pct_fpr": recall_at_fpr(y_true, y_prob),
           "auc_pr": float(average_precision_score(y_true, y_prob)) if two else float("nan"),
           "auc_roc": float(roc_auc_score(y_true, y_prob)) if two else float("nan"),
           "f1_at_saved_threshold": saved["f1"], "accuracy_at_saved_threshold": saved["accuracy"],
           "saved_threshold": float(threshold)}
    best_thr = threshold
    if two:
        pc, rc, tc = precision_recall_curve(y_true, y_prob)
        f1c = np.divide(2 * pc * rc, pc + rc, out=np.zeros_like(pc), where=(pc + rc) > 0)
        best_thr = float(tc[int(np.argmax(f1c[:-1]))])
    best = _pt(best_thr)
    out.update({"f1": best["f1"], "precision": best["precision"], "recall": best["recall"],
                "accuracy": best["accuracy"], "threshold_best": best_thr})
    if two:
        thr_fpr = float(np.quantile(y_prob[y_true == 0], 0.99))
        op = _pt(thr_fpr)
        out.update({"f1_at_1pct_fpr": op["f1"], "precision_at_1pct_fpr": op["precision"],
                    "accuracy_at_1pct_fpr": op["accuracy"], "threshold_at_1pct_fpr": thr_fpr})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--snapshots-root", required=True)
    ap.add_argument("--test-scenarios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--seq-stride", type=int, default=5)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    in_ch, edge_dim = meta["node_feature_dim"], meta["edge_feature_dim"]
    model = Prometheus(in_channels=in_ch, hidden_channels=ckpt["args"]["hidden_channels"],
                       num_gnn_layers=ckpt["args"]["gnn_layers"],
                       dropout=ckpt["args"]["dropout"], edge_dim=edge_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    seqs = []
    for name in [s.strip() for s in args.test_scenarios.split(",") if s.strip()]:
        s, _, _ = load_scenario_sequences(Path(args.snapshots_root) / name,
                                           args.window_size, args.seq_stride)
        seqs.extend(s)
    probs, labels = [], []
    with torch.no_grad():
        for seq in seqs:
            d = snapshot_to_data(seq[-1], device)
            logit = model(d.x, d.edge_index, torch.zeros(d.x.size(0), dtype=torch.long, device=device), d.edge_attr)
            probs.extend(torch.sigmoid(logit).cpu().numpy().ravel().tolist())
            labels.extend(d.y.cpu().numpy().ravel().tolist())
    y_true = np.array(labels, dtype=np.int64)
    y_prob = np.array(probs, dtype=np.float32)
    thr = float(ckpt.get("threshold", 0.5))
    overall = scored_metrics(y_true, y_prob, thr)
    cm = confusion_matrix(y_true, (y_prob >= thr).astype(int))
    print("\n===== Prometheus evaluation (graph-level) =====")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test scenarios: {args.test_scenarios}")
    print(f"Samples: {overall['num_samples']}, Positive: {overall['num_positive']}")
    print(f"\nThreshold (saved): {thr:.4f}")
    print(f"Confusion Matrix (saved threshold):")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n--- Metrics at saved threshold ({thr:.4f}) ---")
    print(f"  F1={overall['f1_at_saved_threshold']:.4f}  "
          f"Precision={overall.get('precision_at_saved_threshold', overall['precision']):.4f}  "
          f"Recall={overall.get('recall_at_saved_threshold', overall['recall']):.4f}  "
          f"Accuracy={overall['accuracy_at_saved_threshold']:.4f}")
    print(f"\n--- Metrics at best-F1 threshold ({overall['threshold_best']:.4f}) ---")
    print(f"  F1={overall['f1']:.4f}  Precision={overall['precision']:.4f}  "
          f"Recall={overall['recall']:.4f}  Accuracy={overall['accuracy']:.4f}")
    print(f"\n--- Metrics at 1%-FPR threshold ({overall.get('threshold_at_1pct_fpr', 'N/A'):.4f}) ---")
    print(f"  F1={overall.get('f1_at_1pct_fpr', float('nan')):.4f}  "
          f"Precision={overall.get('precision_at_1pct_fpr', float('nan')):.4f}  "
          f"Recall={overall['recall_at_1pct_fpr']:.4f}  "
          f"Accuracy={overall.get('accuracy_at_1pct_fpr', float('nan')):.4f}")
    print(f"\n--- Aggregate ---")
    print(f"  AUC-PR={overall['auc_pr']:.4f}  AUC-ROC={overall['auc_roc']:.4f}  "
          f"Recall@1%FPR={overall['recall_at_1pct_fpr']:.4f}")
    out = {"target": "graph", "overall": overall,
           "confusion_matrix_at_saved_threshold": cm.tolist(),
           "checkpoint_args": {k: str(v) for k, v in ckpt["args"].items()}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "metrics_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMetrics saved to {args.out}/metrics_test.json")


if __name__ == "__main__":
    main()
