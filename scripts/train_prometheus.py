#!/usr/bin/env python3
"""Train the Prometheus model (3-layer GAT + rehearsal continual learning) on CTU-13.

This is a graph-level classifier — one logit per snapshot (graph).  Unlike the
TGNN pipeline which does edge-level (per-flow) classification, Prometheus
classifies each temporal snapshot graph as a whole.

The pipeline:
  1. Build sequences of snapshots (as in the TGNN pipeline).
  2. For each step, take the LAST snapshot of the sequence as the graph.
  3. Train with L = L_new + λ * L_rehearsal(D_buf) where D_buf is a 10% reservoir.
  4. Evaluate with the same scored_metrics bundle (saved/best-F1/1%-FPR).

Usage:
    python scripts/train_prometheus.py \
        --snapshots-root data/snapshots \
        --train-scenarios ctu13_c52,ctu13_c46,ctu13_c53,ctu13_c48,ctu13_c45,ctu13_c49,ctu13_c54 \
        --test-scenarios ctu13_c47 \
        --epochs 5 --seq-stride 5 \
        --out models/checkpoints/prometheus_ho_c47

Operational constraints (from user):
  - mount mega-10gb-dataset READ-ONLY
  - NEVER use --label-mode heuristic for CTU-13
  - full Garcia split deferred until subset green + reconfirm
  - CPU-smoke-first-then-STOP-before-paid-GPU
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
    )
    from torch_geometric.data import Batch as PyGBatch
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Training requires the ML dependencies. "
        "Install them with: pip install -r requirements-ml.txt"
    ) from exc

from models.prometheus import Prometheus, snapshot_to_data
from models.rehearsal_buffer import RehearsalBuffer, buffer_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prometheus (3-layer GAT + rehearsal) on CTU-13")
    parser.add_argument("--snapshots", type=Path, default=None)
    parser.add_argument("--snapshots-root", type=Path, default=None)
    parser.add_argument("--train-scenarios", type=str, default=None)
    parser.add_argument("--test-scenarios", type=str, default=None)
    parser.add_argument("--target", type=str, default="graph")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--seq-stride", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-mode", choices=["time", "block"], default="block")
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--select-metric", choices=["auc_pr", "f1", "auc_roc"], default="auc_pr")
    parser.add_argument("--no-threshold-tuning", action="store_true")
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--pos-weight-cap", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    # Rehearsal params
    parser.add_argument("--buffer-pct", type=float, default=0.10,
                        help="Rehearsal buffer size as fraction of total training sequences")
    parser.add_argument("--lambda-rehearsal", type=float, default=1.0,
                        help="Weight of rehearsal loss relative to new-task loss")
    return parser.parse_args()


# ---- Reuse the TGNN harness helpers for snapshot/sequence loading ----

def load_snapshots(snapshots_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    files = sorted(snapshots_dir.glob("snapshot_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No snapshot_*.pkl files found in {snapshots_dir}")
    snapshots = []
    for f in files:
        with open(f, "rb") as fh:
            snapshots.append(pickle.load(fh))
    with open(snapshots_dir / "meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return snapshots, meta


def build_sequences(
    snapshots: List[Dict[str, Any]], window_size: int, seq_stride: int = 1
) -> List[List[Dict[str, Any]]]:
    sequences = []
    for i in range(0, len(snapshots) - window_size + 1, max(1, seq_stride)):
        sequences.append(snapshots[i : i + window_size])
    return sequences


def load_scenario_sequences(
    scen_dir: Path, window_size: int, seq_stride: int
) -> tuple[List[List[Dict[str, Any]]], Dict[str, Any], str]:
    snaps, meta = load_snapshots(scen_dir)
    sid = str(meta.get("scenario_id") or scen_dir.name)
    seqs = build_sequences(snaps, window_size, seq_stride)
    for seq in seqs:
        for snap in seq:
            snap.setdefault("scenario_id", sid)
    return seqs, meta, sid


def _scenario_of(seq: List[Dict[str, Any]]) -> str:
    return str(seq[-1].get("scenario_id", ""))


def split_by_block(
    sequences: List[List[Dict[str, Any]]],
    val_ratio: float,
    test_ratio: float,
    block_size: int = 10,
) -> tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    block_size = max(1, block_size)
    blocks = [sequences[i : i + block_size] for i in range(0, len(sequences), block_size)]
    train, val, test = [], [], []
    n_val = max(1, int(round(1 / max(val_ratio, 1e-6)))) if val_ratio > 0 else 0
    n_test = max(1, int(round(1 / max(test_ratio, 1e-6)))) if test_ratio > 0 else 0
    for i, block in enumerate(blocks):
        if n_test and i % n_test == n_test - 1:
            test.extend(block)
        elif n_val and i % n_val == n_val - 2:
            val.extend(block)
        else:
            train.extend(block)
    if not val:
        val = test[: max(1, len(test) // 2)]
    if (not test or not val) and len(sequences) >= 3:
        n = len(sequences)
        n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
        n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
        n_val = min(n_val, max(0, n - n_test - 1))
        cut_test = n - n_test
        cut_val = cut_test - n_val
        train = sequences[:cut_val]
        val = sequences[cut_val:cut_test]
        test = sequences[cut_test:]
    return train, val, test


def split_by_time(
    sequences: List[List[Dict[str, Any]]],
    val_ratio: float,
    test_ratio: float,
) -> tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    n = len(sequences)
    test_end = int(n * (1 - test_ratio))
    val_end = int(test_end * (1 - val_ratio))
    train = sequences[:val_end]
    val = sequences[val_end:test_end]
    test = sequences[test_end:]
    return train, val, test


def compute_pos_weight(
    sequences: List[List[Dict[str, Any]]],
    cap: float | None = None,
) -> torch.Tensor:
    """Compute pos_weight from snapshot_label (graph-level)."""
    labels = []
    for seq in sequences:
        labels.append(float(seq[-1].get("snapshot_label", 0)))
    labels = np.array(labels, dtype=np.int64)
    pos = int(labels.sum())
    neg = int((labels == 0).sum())
    if pos == 0:
        return torch.tensor(1.0)
    weight = max(1.0, neg / pos)
    if cap:
        weight = min(weight, float(cap))
    return torch.tensor(weight, dtype=torch.float32)


def _recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, fpr: float = 0.01) -> float:
    neg = y_prob[y_true == 0]
    pos = y_prob[y_true == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    thr = float(np.quantile(neg, 1.0 - fpr))
    return float((pos >= thr).mean())


def scored_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Same 3-operating-point bundle as the TGNN harness."""
    if y_true.size == 0:
        return {"num_samples": 0}
    n = int(y_true.size)
    two = len(np.unique(y_true)) > 1

    def _pt(thr: float) -> Dict[str, float]:
        preds = (y_prob >= thr).astype(int)
        tp = int(((y_true == 1) & (preds == 1)).sum())
        fp = int(((y_true == 0) & (preds == 1)).sum())
        fn = int(((y_true == 1) & (preds == 0)).sum())
        tn = int(((y_true == 0) & (preds == 0)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return {"f1": 2 * p * r / (p + r) if (p + r) else 0.0, "precision": p,
                "recall": r, "accuracy": (tp + tn) / n if n else 0.0}

    saved = _pt(threshold)
    out: Dict[str, float] = {
        "num_samples": n, "num_positive": int(y_true.sum()),
        "recall_at_1pct_fpr": _recall_at_fpr(y_true, y_prob, 0.01),
        "auc_pr": float(average_precision_score(y_true, y_prob)) if two else float("nan"),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if two else float("nan"),
        "f1_at_saved_threshold": saved["f1"],
        "accuracy_at_saved_threshold": saved["accuracy"],
        "saved_threshold": float(threshold),
    }
    best_thr = float(threshold)
    if two:
        prec_c, rec_c, thr_c = precision_recall_curve(y_true, y_prob)
        denom = prec_c + rec_c
        f1_c = np.divide(2 * prec_c * rec_c, denom, out=np.zeros_like(denom), where=denom > 0)
        if thr_c.size:
            best_thr = float(thr_c[int(np.argmax(f1_c[:-1]))])
    best = _pt(best_thr)
    out.update({"f1": best["f1"], "precision": best["precision"], "recall": best["recall"],
                "accuracy": best["accuracy"], "threshold_best": best_thr})
    if two:
        thr_fpr = float(np.quantile(y_prob[y_true == 0], 0.99))
        op = _pt(thr_fpr)
        out.update({"f1_at_1pct_fpr": op["f1"], "precision_at_1pct_fpr": op["precision"],
                    "accuracy_at_1pct_fpr": op["accuracy"], "threshold_at_1pct_fpr": thr_fpr})
    return out


def train_epoch(
    model: Prometheus,
    sequences: List[List[Dict[str, Any]]],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    buffer: RehearsalBuffer,
    grad_clip: float = 0.0,
    shuffle: bool = True,
    lambda_rehearsal: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    order = np.random.permutation(len(sequences)) if shuffle else np.arange(len(sequences))
    for idx in order:
        seq = sequences[int(idx)]
        optimizer.zero_grad()

        # Last snapshot of the sequence is the graph to classify
        snap = seq[-1]
        data = snapshot_to_data(snap, device)

        # L_new: loss on current graph
        logits = model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long, device=device),
                        data.edge_attr)
        loss = criterion(logits.squeeze(-1), data.y)

        # L_rehearsal: loss on buffer samples
        r_loss = buffer_loss(model, buffer, criterion, device,
                             batch_size=min(256, len(buffer)) if buffer else 1,
                             lambda_rehearsal=lambda_rehearsal)
        if r_loss is not None:
            loss = loss + r_loss

        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Add to reservoir buffer AFTER training on it (no leakage)
        buffer.add(data)

        total_loss += loss.item()
        count += 1
        if count % 100 == 0:
            print(f"    step {count}/{len(sequences)} loss={total_loss/count:.4f}", flush=True)

    return total_loss / max(count, 1)


def evaluate(
    model: Prometheus,
    sequences: List[List[Dict[str, Any]]],
    criterion: nn.Module,
    device: torch.device,
    threshold: float | None = None,
    tune_threshold: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for seq in sequences:
            snap = seq[-1]
            data = snapshot_to_data(snap, device)
            logits = model(data.x, data.edge_index,
                           torch.zeros(data.x.size(0), dtype=torch.long, device=device),
                           data.edge_attr)
            loss = criterion(logits.squeeze(-1), data.y)
            total_loss += loss.item()
            count += 1

            prob = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(prob.tolist())
            all_labels.extend(data.y.cpu().numpy().ravel().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if all_labels.size == 0:
        thr = 0.5 if threshold is None else float(threshold)
        return {"loss": float("nan"), "auc_roc": float("nan"),
                "auc_pr": float("nan"), "threshold": thr, "f1": float("nan"),
                "accuracy": float("nan"), "precision": float("nan"),
                "recall": float("nan")}

    metrics: Dict[str, float] = {"loss": total_loss / max(count, 1)}
    if len(np.unique(all_labels)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(all_labels, all_probs))
        metrics["auc_pr"] = float(average_precision_score(all_labels, all_probs))
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")

    thr = 0.5 if threshold is None else float(threshold)
    if tune_threshold and len(np.unique(all_labels)) > 1:
        best_thr, best_f1 = thr, -1.0
        for cand in np.unique(np.quantile(all_probs, np.linspace(0.01, 0.99, 99))):
            cand_f1 = f1_score(all_labels, (all_probs >= cand).astype(int), zero_division=0)
            if cand_f1 > best_f1:
                best_f1, best_thr = cand_f1, float(cand)
        thr = best_thr
    metrics["threshold"] = float(thr)
    preds = (all_probs >= thr).astype(int)
    metrics["f1"] = float(f1_score(all_labels, preds, zero_division=0))
    metrics["accuracy"] = float((preds == all_labels).mean())
    metrics["precision"] = float(
        (preds[all_labels == 1] == 1).sum() / max(preds.sum(), 1)
    )
    metrics["recall"] = float(
        (preds[all_labels == 1] == 1).sum() / max(all_labels.sum(), 1)
    )

    return metrics


def predict(
    model: Prometheus,
    sequences: List[List[Dict[str, Any]]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all, probs_all, scen_all = [], [], []
    with torch.no_grad():
        for seq in sequences:
            snap = seq[-1]
            data = snapshot_to_data(snap, device)
            logits = model(data.x, data.edge_index,
                           torch.zeros(data.x.size(0), dtype=torch.long, device=device),
                           data.edge_attr)
            prob = torch.sigmoid(logits).cpu().numpy().ravel()
            lab = data.y.cpu().numpy().ravel()
            probs_all.extend(prob.tolist())
            labels_all.extend(lab.tolist())
            scen_all.extend([_scenario_of(seq)] * len(lab))
    return (
        np.array(labels_all, dtype=np.int64),
        np.array(probs_all, dtype=np.float32),
        np.array(scen_all, dtype=object),
    )


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- data loading ----
    manifest: Dict[str, Any] = {"window_size": args.window_size,
                                "seq_stride": args.seq_stride,
                                "target": "graph"}
    if args.train_scenarios or args.test_scenarios:
        if args.snapshots_root is None:
            raise SystemExit("--train-scenarios/--test-scenarios need --snapshots-root")
        train_names = [s.strip() for s in (args.train_scenarios or "").split(",") if s.strip()]
        test_names = [s.strip() for s in (args.test_scenarios or "").split(",") if s.strip()]
        print(f"[1/4] scenario-held-out split: train={train_names} test={test_names}")
        train_seq, val_seq, test_seq, meta = [], [], [], None
        for name in train_names:
            seqs, m, _sid = load_scenario_sequences(
                args.snapshots_root / name, args.window_size, args.seq_stride)
            meta = meta or m
            k = max(1, int(len(seqs) * args.val_ratio)) if len(seqs) > 1 else 0
            train_seq.extend(seqs[: len(seqs) - k])
            val_seq.extend(seqs[len(seqs) - k:])
        for name in test_names:
            seqs, m, _sid = load_scenario_sequences(
                args.snapshots_root / name, args.window_size, args.seq_stride)
            meta = meta or m
            test_seq.extend(seqs)
        if meta is None:
            raise SystemExit("no scenarios loaded")
        if not val_seq:
            val_seq = test_seq[: max(1, len(test_seq) // 2)]
        manifest.update({"kind": "scenario", "train_scenarios": train_names,
                         "test_scenarios": test_names})
        print(f"  train={len(train_seq)} val={len(val_seq)} test={len(test_seq)}")
    else:
        if args.snapshots is None:
            raise SystemExit("provide --snapshots, or --snapshots-root with --*-scenarios")
        print(f"[1/4] loading snapshots from {args.snapshots}")
        snapshots, meta = load_snapshots(args.snapshots)
        print(f"  -> {len(snapshots)} snapshots")
        print("[2/4] building sequences")
        sequences = build_sequences(snapshots, args.window_size, args.seq_stride)
        if args.split_mode == "block":
            train_seq, val_seq, test_seq = split_by_block(
                sequences, args.val_ratio, args.test_ratio, args.block_size
            )
        else:
            train_seq, val_seq, test_seq = split_by_time(sequences, args.val_ratio, args.test_ratio)
        manifest.update({"kind": "single", "snapshots": str(args.snapshots),
                         "split_mode": args.split_mode, "val_ratio": args.val_ratio,
                         "test_ratio": args.test_ratio, "block_size": args.block_size})
        print(f"  split={args.split_mode} train={len(train_seq)} val={len(val_seq)} test={len(test_seq)}")

    in_channels = meta["node_feature_dim"]
    edge_dim = meta["edge_feature_dim"]

    print(f"[3/4] initialising Prometheus model")
    model = Prometheus(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        edge_dim=edge_dim,
        heads=args.heads,
    ).to(device)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    pos_weight = (
        torch.tensor(args.pos_weight, dtype=torch.float32)
        if args.pos_weight
        else compute_pos_weight(train_seq, cap=args.pos_weight_cap)
    )
    print(f"  pos_weight for BCE: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Rehearsal buffer: 10% of training sequences
    buffer_size = max(1, int(len(train_seq) * args.buffer_pct))
    buffer = RehearsalBuffer(max_size=buffer_size)
    print(f"  rehearsal buffer: {buffer_size} graphs ({args.buffer_pct*100:.0f}%)")

    print("[4/4] training")
    best_score = -1.0
    best_val_f1 = -1.0
    epochs_since_best = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_seq, optimizer, criterion, device, buffer,
            grad_clip=args.grad_clip, lambda_rehearsal=args.lambda_rehearsal,
        )
        val_metrics = evaluate(model, val_seq, criterion, device,
                               tune_threshold=not args.no_threshold_tuning)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = train_loss
        history.append(val_metrics)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc_pr={val_metrics['auc_pr']:.4f} | "
            f"val_auc_roc={val_metrics['auc_roc']:.4f} | "
            f"buffer={len(buffer)}",
            flush=True,
        )

        scheduler.step(val_metrics["loss"])

        score = val_metrics.get(args.select_metric, float("nan"))
        if score != score:
            score = val_metrics["f1"]
        if score != score:
            vloss = val_metrics.get("loss", float("nan"))
            score = -(vloss if vloss == vloss else train_loss)

        if score > best_score:
            best_score = score
            best_val_f1 = val_metrics["f1"]
            epochs_since_best = 0
            args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": args_dict,
                "meta": meta,
                "best_val_f1": best_val_f1,
                "best_score": float(best_score),
                "select_metric": args.select_metric,
                "threshold": float(val_metrics.get("threshold", 0.5)),
                "target": "graph",
                "split_manifest": manifest,
            }
            torch.save(checkpoint, args.out / "best_model.pt")
        else:
            epochs_since_best += 1
            if args.patience and epochs_since_best >= args.patience:
                print(f"Early stopping: no {args.select_metric} improvement "
                      f"for {args.patience} epochs.")
                break

    torch.save(model.state_dict(), args.out / "final_model.pt")
    with open(args.out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if test_seq:
        best = torch.load(args.out / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        thr = float(best.get("threshold", 0.5))
        y_true, y_prob, scen = predict(model, test_seq, device)

        overall = scored_metrics(y_true, y_prob, thr)
        overall["threshold"] = thr
        per_scenario: Dict[str, Any] = {}
        for sid in sorted(set(scen.tolist())):
            m = scen == sid
            per_scenario[sid] = scored_metrics(y_true[m], y_prob[m], thr)

        test_metrics = {"target": "graph", "overall": overall,
                        "per_scenario": per_scenario}
        print(f"\nTest (graph) @ threshold {thr:.3f}: " +
              ", ".join(f"{k}={v:.4f}" for k, v in overall.items()
                        if isinstance(v, float)))
        for sid, m in per_scenario.items():
            print(f"  [{sid}] auc_pr={m.get('auc_pr', float('nan')):.4f} "
                  f"recall@1%FPR={m.get('recall_at_1pct_fpr', float('nan')):.4f} "
                  f"f1={m.get('f1', float('nan')):.4f} (n={m.get('num_samples', 0)}, "
                  f"pos={m.get('num_positive', 0)})")
        with open(args.out / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    print(f"\nTraining complete. Best val {args.select_metric}: {best_score:.4f} "
          f"(val F1 at that epoch: {best_val_f1:.4f})")
    print(f"Checkpoint saved to {args.out / 'best_model.pt'}")


if __name__ == "__main__":
    main()
