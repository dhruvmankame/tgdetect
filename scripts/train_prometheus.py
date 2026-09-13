#!/usr/bin/env python3
"""Train Prometheus using the base paper's explicitly reported configuration.

Heavy training is intended for Modal/cloud execution. This script itself is
ordinary Python and can also be invoked by ``modal_train.py``.

Paper-aligned fixed choices:
  - node-wise classification (2 classes, softmax semantics)
  - 3 attention GNN layers, hidden=128
  - ReLU + dropout=0.2
  - Adam(lr=1e-3, weight_decay=1e-5)
  - batch size 256
  - 200 epochs PER SCENARIO
  - rehearsal buffer capacity = 10% of unique past graph samples
  - reservoir sampling
  - five independent random-seed runs

Paper ambiguity retained explicitly:
  - exact attention operator/head count is not reported -> minimal 1-head GAT
  - lambda in L_new + lambda*L_rehearsal is not numerically reported -> CLI
    defaults to 1.0 and records the value in every artifact
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.loader import DataLoader

from models.prometheus import (
    PAPER_ATTENTION_HEADS,
    PAPER_DROPOUT,
    PAPER_GNN_LAYERS,
    PAPER_HIDDEN_DIM,
    Prometheus,
    snapshot_to_data,
)
from models.rehearsal_buffer import RehearsalBuffer

PAPER_BATCH_SIZE = 256
PAPER_EPOCHS_PER_SCENARIO = 200
PAPER_LR = 1e-3
PAPER_WEIGHT_DECAY = 1e-5
PAPER_BUFFER_PCT = 0.10
DEFAULT_SEEDS = "42,43,44,45,46"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-faithful Prometheus continual-learning trainer")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--snapshots", type=Path, help="single snapshot directory (one scenario/task)")
    src.add_argument("--snapshots-root", type=Path, help="root containing scenario snapshot directories")
    p.add_argument("--train-scenarios", default="", help="comma-separated task order under --snapshots-root")
    p.add_argument("--test-scenarios", default="", help="comma-separated held-out scenarios")
    p.add_argument("--out", type=Path, required=True)

    p.add_argument("--epochs", type=int, default=PAPER_EPOCHS_PER_SCENARIO)
    p.add_argument("--batch-size", type=int, default=PAPER_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=PAPER_LR)
    p.add_argument("--weight-decay", type=float, default=PAPER_WEIGHT_DECAY)
    p.add_argument("--buffer-pct", type=float, default=PAPER_BUFFER_PCT)
    p.add_argument("--lambda-rehearsal", type=float, default=1.0)
    p.add_argument("--seeds", default=DEFAULT_SEEDS, help="exactly five comma-separated seeds for a paper run")
    p.add_argument("--smoke", action="store_true", help="allow reduced epochs/seeds for a cheap code-path smoke test")

    # Compatibility with older tgdetect invocations. Values are checked, not used
    # to silently change the paper architecture.
    p.add_argument("--target", default="node")
    p.add_argument("--hidden-channels", type=int, default=PAPER_HIDDEN_DIM)
    p.add_argument("--gnn-layers", type=int, default=PAPER_GNN_LAYERS)
    p.add_argument("--dropout", type=float, default=PAPER_DROPOUT)
    p.add_argument("--heads", type=int, default=PAPER_ATTENTION_HEADS)
    p.add_argument("--window-size", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--seq-stride", type=int, default=1, help=argparse.SUPPRESS)
    return p.parse_args()


def comma_list(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    return seeds


def validate_paper_config(args: argparse.Namespace, seeds: list[int]) -> None:
    fixed = {
        "epochs": (args.epochs, PAPER_EPOCHS_PER_SCENARIO),
        "batch_size": (args.batch_size, PAPER_BATCH_SIZE),
        "lr": (args.lr, PAPER_LR),
        "weight_decay": (args.weight_decay, PAPER_WEIGHT_DECAY),
        "buffer_pct": (args.buffer_pct, PAPER_BUFFER_PCT),
        "hidden_channels": (args.hidden_channels, PAPER_HIDDEN_DIM),
        "gnn_layers": (args.gnn_layers, PAPER_GNN_LAYERS),
        "dropout": (args.dropout, PAPER_DROPOUT),
        "heads": (args.heads, PAPER_ATTENTION_HEADS),
    }
    mismatches = [f"{k}={got!r} (paper path expects {want!r})" for k, (got, want) in fixed.items() if got != want]
    if args.target != "node":
        mismatches.append(f"target={args.target!r} (paper equation is node-wise classification)")
    if not args.smoke and len(seeds) != 5:
        mismatches.append(f"seeds={len(seeds)} runs (paper reports five runs)")
    if mismatches and not args.smoke:
        raise SystemExit("Paper-strict configuration mismatch:\n  - " + "\n  - ".join(mismatches))
    if mismatches:
        print("SMOKE MODE: non-paper runtime settings allowed:\n  - " + "\n  - ".join(mismatches), flush=True)


def load_snapshot_dir(path: Path) -> tuple[list, dict[str, Any]]:
    # Local import for the annotation-free runtime path.
    files = sorted(path.glob("snapshot_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No snapshot_*.pkl files in {path}")
    graphs = []
    for f in files:
        with open(f, "rb") as fh:
            snap = pickle.load(fh)
        graphs.append(snapshot_to_data(snap))
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if "node_feature_dim" not in meta:
        meta["node_feature_dim"] = int(graphs[0].x.shape[1])
    return graphs, meta


def scenario_paths(args: argparse.Namespace) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    if args.snapshots is not None:
        return [(args.snapshots.name, args.snapshots)], []
    train_names = comma_list(args.train_scenarios)
    if not train_names:
        raise SystemExit("--snapshots-root requires --train-scenarios in the intended continual-learning order")
    test_names = comma_list(args.test_scenarios)
    train = [(n, args.snapshots_root / n) for n in train_names]
    test = [(n, args.snapshots_root / n) for n in test_names]
    return train, test


def count_graphs(tasks: list[tuple[str, Path]]) -> int:
    return sum(len(list(path.glob("snapshot_*.pkl"))) for _, path in tasks)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_scenario(
    model: Prometheus,
    graphs: list,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    buffer: RehearsalBuffer,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lambda_rehearsal: float,
    scenario_name: str,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_new = 0.0
        total_replay = 0.0
        steps = 0

        for current in loader:
            current = current.to(device)
            optimizer.zero_grad(set_to_none=True)

            current_logits = model(current.x, current.edge_index)
            new_loss = criterion(current_logits, current.y.long())
            loss = new_loss
            replay_value = 0.0

            replay = buffer.sample_batch(batch_size, device)
            if replay is not None:
                replay_logits = model(replay.x, replay.edge_index)
                replay_loss = criterion(replay_logits, replay.y.long())
                loss = loss + float(lambda_rehearsal) * replay_loss
                replay_value = float(replay_loss.detach().item())

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            total_new += float(new_loss.detach().item())
            total_replay += replay_value
            steps += 1

        row = {
            "scenario": scenario_name,
            "epoch": epoch,
            "loss": total_loss / max(steps, 1),
            "new_loss": total_new / max(steps, 1),
            "replay_loss": total_replay / max(steps, 1),
            "buffer_size": len(buffer),
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  {scenario_name} epoch {epoch:03d}/{epochs} "
                f"loss={row['loss']:.5f} new={row['new_loss']:.5f} "
                f"replay={row['replay_loss']:.5f} buffer={len(buffer)}",
                flush=True,
            )

    # Reservoir update ONCE per unique sample, after the scenario has finished.
    # This prevents 200-epoch duplicate insertion and makes the replay set a
    # sample of previously learned scenarios when the next scenario arrives.
    buffer.add_many(graphs)
    return history


@torch.no_grad()
def evaluate(model: Prometheus, graphs: list, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    labels: list[int] = []
    preds: list[int] = []
    probs: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        probability = torch.softmax(logits, dim=-1)[:, 1]
        prediction = torch.argmax(logits, dim=-1)
        labels.extend(batch.y.detach().cpu().numpy().astype(np.int64).tolist())
        preds.extend(prediction.detach().cpu().numpy().astype(np.int64).tolist())
        probs.extend(probability.detach().cpu().numpy().astype(np.float64).tolist())

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(preds, dtype=np.int64)
    score = np.asarray(probs, dtype=np.float64)
    if y.size == 0:
        return {"num_nodes": 0, "precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "accuracy": float("nan"), "auc_roc": float("nan")}

    out = {
        "num_nodes": int(y.size),
        "num_positive": int(y.sum()),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "accuracy": float(accuracy_score(y, p)),
        "auc_roc": float(roc_auc_score(y, score)) if np.unique(y).size > 1 else float("nan"),
    }
    return out


def aggregate_seed_metrics(per_seed: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = ["precision", "recall", "f1", "accuracy", "auc_roc"]
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in keys:
        vals = np.asarray([row["overall"][key] for row in per_seed], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        mean[key] = float(vals.mean()) if vals.size else float("nan")
        std[key] = float(vals.std(ddof=1)) if vals.size > 1 else (0.0 if vals.size == 1 else float("nan"))
    return mean, std


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    validate_paper_config(args, seeds)
    train_tasks, test_tasks = scenario_paths(args)
    args.out.mkdir(parents=True, exist_ok=True)

    total_unique_train = count_graphs(train_tasks)
    if total_unique_train < 1:
        raise SystemExit("No training snapshots found")
    buffer_capacity = max(1, int(math.ceil(PAPER_BUFFER_PCT * total_unique_train)))

    # Determine feature width from first training scenario and verify all tasks.
    first_graphs, first_meta = load_snapshot_dir(train_tasks[0][1])
    in_channels = int(first_meta.get("node_feature_dim", first_graphs[0].x.shape[1]))
    del first_graphs

    paper_config = {
        "task": "node-wise classification",
        "gnn_layers": PAPER_GNN_LAYERS,
        "hidden_dim": PAPER_HIDDEN_DIM,
        "activation": "ReLU",
        "dropout": PAPER_DROPOUT,
        "aggregation": "attention-based; implemented as single-head GATConv because operator/head count are not specified in the paper",
        "pooling": None,
        "output": "Linear(128,2) + softmax probabilities",
        "loss": "CrossEntropyLoss",
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_per_scenario": args.epochs,
        "rehearsal_buffer_fraction": args.buffer_pct,
        "rehearsal_buffer_capacity_graphs": buffer_capacity,
        "reservoir_sampling": True,
        "lambda_rehearsal": args.lambda_rehearsal,
        "lambda_note": "numeric lambda is not reported by the paper; this run records the chosen value",
        "num_seed_runs": len(seeds),
        "seeds": seeds,
    }
    (args.out / "paper_config.json").write_text(json.dumps(paper_config, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | unique_train_graphs={total_unique_train} | reservoir_capacity={buffer_capacity}")

    per_seed: list[dict[str, Any]] = []
    all_histories: dict[str, Any] = {}

    for run_index, seed in enumerate(seeds, start=1):
        print(f"\n===== paper run {run_index}/{len(seeds)} | seed={seed} =====", flush=True)
        set_seed(seed)
        model = Prometheus(in_channels=in_channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        buffer = RehearsalBuffer(max_size=buffer_capacity, seed=seed)
        seed_history: list[dict[str, Any]] = []

        for scenario_name, scenario_path in train_tasks:
            graphs, meta = load_snapshot_dir(scenario_path)
            dim = int(meta.get("node_feature_dim", graphs[0].x.shape[1]))
            if dim != in_channels:
                raise ValueError(f"Feature width mismatch: {scenario_name} has {dim}, expected {in_channels}")
            print(f"\n[seed {seed}] scenario={scenario_name} graphs={len(graphs)}", flush=True)
            seed_history.extend(
                train_scenario(
                    model=model,
                    graphs=graphs,
                    optimizer=optimizer,
                    criterion=criterion,
                    buffer=buffer,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lambda_rehearsal=args.lambda_rehearsal,
                    scenario_name=scenario_name,
                )
            )
            del graphs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        seed_dir = args.out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "meta": {"node_feature_dim": in_channels},
            "paper_config": paper_config,
            "train_scenarios": [name for name, _ in train_tasks],
            "test_scenarios": [name for name, _ in test_tasks],
            "buffer_total_seen": buffer.total_seen,
            "buffer_size": len(buffer),
        }
        torch.save(checkpoint, seed_dir / "model.pt")
        all_histories[str(seed)] = seed_history

        test_graphs_all = []
        per_scenario: dict[str, Any] = {}
        for scenario_name, scenario_path in test_tasks:
            graphs, _ = load_snapshot_dir(scenario_path)
            per_scenario[scenario_name] = evaluate(model, graphs, device, args.batch_size)
            test_graphs_all.extend(graphs)

        overall = evaluate(model, test_graphs_all, device, args.batch_size) if test_graphs_all else {
            "num_nodes": 0,
            "num_positive": 0,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
            "auc_roc": float("nan"),
        }
        seed_result = {"seed": seed, "overall": overall, "per_scenario": per_scenario}
        per_seed.append(seed_result)
        (seed_dir / "test_metrics.json").write_text(json.dumps(seed_result, indent=2), encoding="utf-8")
        print(f"[seed {seed}] test metrics: {overall}", flush=True)

    mean, std = aggregate_seed_metrics(per_seed)
    result = {
        "target": "node",
        "paper_config": paper_config,
        "per_seed": per_seed,
        "mean": mean,
        "std": std,
        "reporting": "mean +/- sample standard deviation over independent seed runs",
    }
    (args.out / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "paper_history.json").write_text(json.dumps(all_histories, indent=2), encoding="utf-8")

    # Backward-compatible alias only. It is NOT selected using test performance.
    first_ckpt = args.out / f"seed_{seeds[0]}" / "model.pt"
    shutil.copy2(first_ckpt, args.out / "best_model.pt")

    print("\n===== aggregate over seeds =====")
    for key in sorted(mean):
        print(f"{key}: {mean[key]:.6f} +/- {std[key]:.6f}")
    print(f"Artifacts: {args.out}")
    print("NOTE: best_model.pt is a compatibility alias for the first seed, not a test-selected model.")


if __name__ == "__main__":
    main()
