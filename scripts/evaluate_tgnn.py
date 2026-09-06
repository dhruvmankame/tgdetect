#!/usr/bin/env python3
"""Evaluate a trained TGNN checkpoint on the SAME partition it was trained with.

Instead of re-deriving a split (which silently disagreed with training), this
loads the split manifest saved in the checkpoint and reconstructs the exact
train/val/test partition. Reports imbalance-aware, per-scenario metrics
(PR-AUC + recall@fixed-FPR) which are the meaningful numbers for CTU-13.

Example:
    python scripts/evaluate_tgnn.py \
        --checkpoint models/checkpoints/ctu13_c52/best_model.pt \
        --snapshots data/snapshots/ctu13_c52 \
        --out results/ctu13_c52
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Evaluation requires the ML dependencies. "
        "Install them with: pip install -r requirements-ml.txt"
    ) from exc

from models.tgnn import TemporalGNN

# Reuse the exact sequence/metric machinery training used, so the two agree.
import train_tgnn  # noqa: E402


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                     lr: float = 1e-2, max_iter: int = 300) -> float:
    """Learn a single scalar temperature T that minimises NLL on sigmoid(logits / T).

    Standard Platt-style temperature scaling (Guo et al., 2017). T > 1 flattens
    the sigmoid (the usual direction after over-confident training); T < 1
    sharpens. Fitting on the validation split guarantees the test metrics are
    not contaminated.
    """
    device = logits.device
    T = torch.ones(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    labels_f = labels.to(device).reshape(-1).float()

    def nll():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(-1) / T.clamp(min=1e-4), labels_f)
        loss.backward()
        return loss

    opt.step(nll)
    return float(T.detach().clamp(min=1e-4).cpu())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TGNN")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--snapshots", type=Path, default=None,
                        help="Single snapshot dir (single-dir checkpoints). "
                             "Defaults to the path saved in the manifest.")
    parser.add_argument("--snapshots-root", type=Path, default=None,
                        help="Parent dir of per-scenario snapshot subdirs "
                             "(scenario-held-out checkpoints).")
    parser.add_argument("--out", "--out-dir", dest="out", type=Path, required=True,
                        help="Output directory for predictions and metrics")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which partition to evaluate (single-dir only; "
                             "scenario checkpoints always score the held-out "
                             "test scenarios)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (defaults to the checkpoint's)")
    parser.add_argument("--gnn-aggregation", type=str, default=None,
                        help="GNN aggregation (sage/gat/gcn). Defaults to the "
                             "value saved in the checkpoint.")
    parser.add_argument("--temperature-scale", action="store_true",
                            help="Fit a temperature on the validation split and "
                                 "calibrate test probabilities before scoring.")
    return parser.parse_args()


def load_model(checkpoint: Dict[str, Any], meta: Dict[str, Any]) -> TemporalGNN:
    a = checkpoint.get("args", {})
    agg = a.get("gnn_aggregation", "sage")
    model = TemporalGNN(
        in_channels=meta["node_feature_dim"],
        edge_dim=meta["edge_feature_dim"],
        hidden_channels=int(a.get("hidden_channels", 64)),
        out_channels=int(a.get("out_channels", 64)),
        num_gnn_layers=int(a.get("gnn_layers", 2)),
        num_rnn_layers=int(a.get("rnn_layers", 1)),
        dropout=float(a.get("dropout", 0.3)),
        node_types=meta["num_node_types"],
        num_relations=meta["num_relations"],
        gnn_aggregation=str(agg),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def reconstruct_eval_sequences(
    manifest: Dict[str, Any], args: argparse.Namespace
) -> List[List[Dict[str, Any]]]:
    """Rebuild the requested partition exactly as training defined it."""
    window = int(manifest.get("window_size", 10))
    stride = int(manifest.get("seq_stride", 1))
    kind = manifest.get("kind", "single")

    if kind == "scenario":
        root = args.snapshots_root
        if root is None:
            raise SystemExit(
                "This checkpoint used a scenario-held-out split; pass "
                "--snapshots-root <parent dir of scenario subdirs>."
            )
        names = (manifest["test_scenarios"] if args.split == "test"
                 else manifest["train_scenarios"])
        seqs: List[List[Dict[str, Any]]] = []
        for name in names:
            s, _m, _sid = train_tgnn.load_scenario_sequences(root / name, window, stride)
            seqs.extend(s)
        print(f"  scenario split={args.split} scenarios={names} sequences={len(seqs)}")
        return seqs

    # single-dir: rebuild + apply the saved split, then pick the partition.
    snaps_dir = args.snapshots or Path(manifest.get("snapshots", ""))
    snapshots, _meta = train_tgnn.load_snapshots(snaps_dir)
    sequences = train_tgnn.build_sequences(snapshots, window, stride)
    if manifest.get("split_mode", "block") == "block":
        tr, va, te = train_tgnn.split_by_block(
            sequences, manifest.get("val_ratio", 0.15),
            manifest.get("test_ratio", 0.15), manifest.get("block_size", 10))
    else:
        tr, va, te = train_tgnn.split_by_time(
            sequences, manifest.get("val_ratio", 0.15), manifest.get("test_ratio", 0.15))
    part = {"train": tr, "val": va, "test": te}[args.split]
    print(f"  single-dir split={args.split} sequences={len(part)} (of {len(sequences)})")
    return part


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print(f"[1/4] loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = checkpoint["meta"]
    target = checkpoint.get("target", "node")
    manifest = checkpoint.get("split_manifest", {"kind": "single",
                                                 "window_size": int(checkpoint.get("args", {}).get("window_size", 10)),
                                                 "seq_stride": 1})
    # Allow the CLI to override the checkpoint's aggregation (e.g. when
    # re-evaluating a legacy checkpoint whose args didn't record it).
    if args.gnn_aggregation:
        cargs = dict(checkpoint.get("args", {}))
        cargs["gnn_aggregation"] = args.gnn_aggregation
        checkpoint["args"] = cargs
    model = load_model(checkpoint, meta).to(device)
    print(f"  target={target}  split_kind={manifest.get('kind')}")

    print("[2/4] reconstructing evaluation partition")
    eval_seq = reconstruct_eval_sequences(manifest, args)
    if not eval_seq:
        raise SystemExit("empty evaluation partition")

    thr = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.5))
    print(f"[3/4] evaluating ({target}) @ threshold {thr:.3f}")
    y_true, y_prob, scen = train_tgnn.predict(model, eval_seq, target, device)

    if args.temperature_scale and manifest.get("kind") == "scenario":
        # Calibrate on the validation tail of the training scenarios, then
        # apply T to the held-out test scenarios.
        val_names = manifest.get("train_scenarios", [])
        if val_names:
            val_root = args.snapshots_root
            if val_root is None:
                raise SystemExit("--temperature-scale needs --snapshots-root for scenario splits")
            val_logits, val_labels = [], []
            model.eval()
            with torch.no_grad():
                for name in val_names:
                    vseqs, _, _ = train_tgnn.load_scenario_sequences(
                        val_root / name, manifest.get("window_size", 10),
                        manifest.get("seq_stride", 1))
                    # Use the val tail the same way training carved it (last val_ratio).
                    # Approximate: take the last 15% as the calibration set.
                    k = max(1, int(len(vseqs) * 0.15)) if len(vseqs) > 1 else len(vseqs)
                    for seq in vseqs[len(vseqs) - k:]:
                        node_logits, _, edge_logits = model(seq)
                        logits, labels = (edge_logits, torch.from_numpy(seq[-1]["edge_labels"]).float()) \
                            if target == "edge" else (node_logits, torch.from_numpy(seq[-1]["node_labels"]).float())
                        if logits.numel() > 0:
                            val_logits.append(logits)
                            val_labels.append(labels.reshape(-1, 1))
            if val_logits:
                v_logits = torch.cat(val_logits)
                v_labels = torch.cat(val_labels)
                T = _fit_temperature(v_logits, v_labels)
                print(f"  temperature T={T:.4f} (fit on val tail of train scenarios)")
                # Re-score test set with calibrated probabilities.
                calibrated_probs = []
                model.eval()
                with torch.no_grad():
                    for seq in eval_seq:
                        node_logits, _, edge_logits = model(seq)
                        logits = edge_logits if target == "edge" else node_logits
                        if logits.numel() > 0:
                            calibrated_probs.append(torch.sigmoid(logits / T).cpu().numpy().ravel())
                        else:
                            calibrated_probs.append(np.array([], dtype=np.float32))
                y_prob = np.concatenate(calibrated_probs).astype(np.float32)
                # Align y_true to the same flattening as y_prob.
                y_true_acc, _, scen_acc = train_tgnn.predict(model, eval_seq, target, device)
                y_true = y_true_acc
                scen = scen_acc

    overall = train_tgnn.scored_metrics(y_true, y_prob, thr)
    overall["threshold"] = thr  # saved (checkpoint) threshold, kept for the plot's degeneracy check
    per_scenario: Dict[str, Any] = {}
    for sid in sorted(set(scen.tolist())):
        m = scen == sid
        per_scenario[sid] = train_tgnn.scored_metrics(y_true[m], y_prob[m], thr)

    metrics = {"target": target, "split": args.split,
               "overall": overall, "per_scenario": per_scenario}
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    # Headline f1/precision/recall/accuracy are reported at the best-F1 operating
    # point; label the saved parquet predictions there too so they agree.
    op_thr = float(overall.get("threshold_best", thr))
    predictions = pd.DataFrame({
        "scenario": scen.astype(str),
        "probability": y_prob,
        "prediction": (y_prob >= op_thr).astype(int),
        "ground_truth": y_true,
    })
    predictions.to_parquet(args.out / f"predictions_{args.split}.parquet", index=False)
    with open(args.out / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPredictions saved to {args.out / f'predictions_{args.split}.parquet'}")
    print(f"Metrics saved to {args.out / f'metrics_{args.split}.json'}")


if __name__ == "__main__":
    main()
