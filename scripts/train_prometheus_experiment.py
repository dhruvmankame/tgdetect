#!/usr/bin/env python3
"""CTU-13 experimental suite around the paper-faithful Prometheus model.

This script does NOT replace scripts/train_prometheus.py.  It implements the
paper's comparison/ablation protocols that were missing from the repository,
while keeping CTU-13 as the dataset requested by the project.

Experiments
-----------
full_rehearsal
    Prometheus GAT, sequential tasks, 10% reservoir replay.
finetune
    Same Prometheus GAT, sequential tasks, no replay.
static_gnn
    Same Prometheus GAT, all training scenarios pooled, no continual updates.
cnn_bilstm
    Two Conv1d layers + BiLSTM + attention, pooled training.
cnn_bilstm_rehearsal
    Same sequential baseline with 10% reservoir replay.
generative_replay
    Conditional-VAE node-feature replay + node MLP.  This is a transparent
    proxy because the paper does not specify the generative replay model.

Every continual experiment records an after-each-task evaluation matrix and
catastrophic-forgetting statistics, which directly test the purpose of the
paper's rehearsal method.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prometheus import Prometheus, snapshot_to_data
from models.prometheus_baselines import CNNBiLSTMAttention, ConditionalVAE, NodeMLP, vae_loss
from models.rehearsal_buffer import RehearsalBuffer

CTU13_FAMILY = {
    "42": "Neris", "43": "Neris", "44": "Rbot", "45": "Rbot",
    "46": "fast-flux/Virut", "47": "donbot", "48": "Sogou",
    "49": "qvod/Murlo", "50": "Neris", "52": "Rbot",
    "53": "NSIS.ay", "54": "fast-flux/Virut",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots-root", type=Path, required=True)
    p.add_argument("--train-scenarios", required=True)
    p.add_argument("--test-scenarios", default="")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--experiment", choices=[
        "full_rehearsal", "finetune", "static_gnn", "cnn_bilstm",
        "cnn_bilstm_rehearsal", "generative_replay",
    ], required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--buffer-pct", type=float, default=0.10)
    p.add_argument("--lambda-rehearsal", type=float, default=1.0)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--generator-epochs", type=int, default=25)
    p.add_argument("--generator-max-current-nodes", type=int, default=100000)
    p.add_argument("--generator-replay-nodes", type=int, default=50000)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def scenario_key(name: str) -> str:
    import re
    m = re.search(r"ctu13_c(\d+)", name.lower())
    return m.group(1) if m else name


def family(name: str) -> str:
    return CTU13_FAMILY.get(scenario_key(name), "unknown")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graphs(path: Path) -> tuple[list, dict[str, Any]]:
    files = sorted(path.glob("snapshot_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No snapshot_*.pkl files in {path}")
    graphs = []
    for fp in files:
        with open(fp, "rb") as fh:
            graphs.append(snapshot_to_data(pickle.load(fh)))
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.setdefault("node_feature_dim", int(graphs[0].x.shape[1]))
    return graphs, meta


def graph_count(root: Path, names: Iterable[str]) -> int:
    return sum(len(list((root / n).glob("snapshot_*.pkl"))) for n in names)


def forward_graph_model(model: nn.Module, batch, kind: str) -> torch.Tensor:
    if kind == "gnn":
        return model(batch.x, batch.edge_index)
    if kind == "cnn":
        return model(batch.x, batch.batch)
    raise KeyError(kind)


@torch.no_grad()
def evaluate_graph_model(
    model: nn.Module,
    graphs: list,
    device: torch.device,
    batch_size: int,
    kind: str,
) -> dict[str, Any]:
    model.eval()
    ys: list[int] = []
    ps: list[int] = []
    scores: list[float] = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        logits = forward_graph_model(model, batch, kind)
        prob = torch.softmax(logits, dim=-1)[:, 1]
        pred = torch.argmax(logits, dim=-1)
        ys.extend(batch.y.detach().cpu().numpy().astype(np.int64).tolist())
        ps.extend(pred.detach().cpu().numpy().astype(np.int64).tolist())
        scores.extend(prob.detach().cpu().numpy().astype(np.float64).tolist())
    return metrics_from_arrays(ys, ps, scores)


@torch.no_grad()
def evaluate_mlp(model: NodeMLP, graphs: list, device: torch.device, batch_nodes: int = 65536) -> dict[str, Any]:
    model.eval()
    ys: list[int] = []
    ps: list[int] = []
    scores: list[float] = []
    for graph in graphs:
        x = graph.x
        y = graph.y
        for start in range(0, x.size(0), batch_nodes):
            xb = x[start:start + batch_nodes].to(device)
            yb = y[start:start + batch_nodes]
            logits = model(xb)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            pred = torch.argmax(logits, dim=-1)
            ys.extend(yb.numpy().astype(np.int64).tolist())
            ps.extend(pred.cpu().numpy().astype(np.int64).tolist())
            scores.extend(prob.cpu().numpy().astype(np.float64).tolist())
    return metrics_from_arrays(ys, ps, scores)


def metrics_from_arrays(ys, ps, scores) -> dict[str, Any]:
    y = np.asarray(ys, dtype=np.int64)
    p = np.asarray(ps, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0:
        return {"num_nodes": 0, "num_positive": 0, "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan"),
                "accuracy": float("nan"), "auc_roc": float("nan")}
    return {
        "num_nodes": int(y.size),
        "num_positive": int(y.sum()),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "accuracy": float(accuracy_score(y, p)),
        "auc_roc": float(roc_auc_score(y, s)) if np.unique(y).size > 1 else float("nan"),
    }


def train_graph_epochs(
    model: nn.Module,
    graphs: list,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    batch_size: int,
    kind: str,
    buffer: RehearsalBuffer | None = None,
    lambda_rehearsal: float = 1.0,
    label: str = "",
) -> list[dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    history = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = new_total = replay_total = 0.0
        steps = 0
        for current in loader:
            current = current.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_graph_model(model, current, kind)
            new_loss = criterion(logits, current.y.long())
            loss = new_loss
            replay_val = 0.0
            if buffer is not None and len(buffer):
                replay = buffer.sample_batch(batch_size, device)
                if replay is not None:
                    replay_logits = forward_graph_model(model, replay, kind)
                    replay_loss = criterion(replay_logits, replay.y.long())
                    loss = loss + float(lambda_rehearsal) * replay_loss
                    replay_val = float(replay_loss.detach().item())
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item())
            new_total += float(new_loss.detach().item())
            replay_total += replay_val
            steps += 1
        row = {"epoch": epoch, "loss": total / max(steps, 1),
               "new_loss": new_total / max(steps, 1),
               "replay_loss": replay_total / max(steps, 1)}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  {label} epoch {epoch:03d}/{epochs} loss={row['loss']:.5f} "
                  f"new={row['new_loss']:.5f} replay={row['replay_loss']:.5f}", flush=True)
    return history


def train_joint_scenarios(
    model: nn.Module, root: Path, names: list[str], optimizer: optim.Optimizer,
    device: torch.device, epochs: int, batch_size: int, kind: str, seed: int,
) -> list[dict[str, float]]:
    """Joint/static training without loading all CTU-13 scenarios into RAM."""
    criterion = nn.CrossEntropyLoss()
    rng = random.Random(seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = steps = 0
        order = list(names)
        rng.shuffle(order)
        for name in order:
            graphs, _ = load_graphs(root / name)
            loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
            for current in loader:
                current = current.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(forward_graph_model(model, current, kind), current.y.long())
                loss.backward()
                optimizer.step()
                total += float(loss.detach().item())
                steps += 1
            del graphs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row = {"epoch": epoch, "loss": total / max(steps, 1)}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  joint epoch {epoch:03d}/{epochs} loss={row['loss']:.5f}", flush=True)
    return history


def sample_current_nodes(graphs: list, max_nodes: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform node sample using global indices (fast even for millions of nodes)."""
    counts = np.asarray([int(g.x.size(0)) for g in graphs], dtype=np.int64)
    total = int(counts.sum())
    if total < 1:
        raise ValueError("scenario has no nodes")
    k = min(int(max_nodes), total)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(total, size=k, replace=False))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    xs, ys = [], []
    for gi, graph in enumerate(graphs):
        lo, hi = int(offsets[gi]), int(offsets[gi + 1])
        left = int(np.searchsorted(chosen, lo, side="left"))
        right = int(np.searchsorted(chosen, hi, side="left"))
        if right <= left:
            continue
        local = torch.from_numpy((chosen[left:right] - lo).astype(np.int64))
        xs.append(graph.x.index_select(0, local).cpu())
        ys.append(graph.y.index_select(0, local).long().cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_mlp_tensor(
    model: NodeMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    batch_size: int,
    label: str,
) -> list[dict[str, float]]:
    ds = TensorDataset(x, y)
    criterion = nn.CrossEntropyLoss()
    hist = []
    for epoch in range(1, epochs + 1):
        loader = TorchDataLoader(ds, batch_size=batch_size, shuffle=True)
        model.train()
        total = steps = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item())
            steps += 1
        row = {"epoch": epoch, "loss": total / max(steps, 1)}
        hist.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  {label} epoch {epoch:03d}/{epochs} loss={row['loss']:.5f}", flush=True)
    return hist


def train_vae_tensor(
    vae: ConditionalVAE,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    ds = TensorDataset(x, y)
    opt = optim.Adam(vae.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        vae.train()
        loader = TorchDataLoader(ds, batch_size=batch_size, shuffle=True)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = vae(xb, yb)
            loss = vae_loss(recon, xb, mu, logvar)
            loss.backward()
            opt.step()


def evaluate_named_scenarios(
    model: nn.Module,
    model_kind: str,
    root: Path,
    names: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    out = {}
    for name in names:
        graphs, _ = load_graphs(root / name)
        if model_kind == "mlp":
            m = evaluate_mlp(model, graphs, device)
        else:
            m = evaluate_graph_model(model, graphs, device, batch_size, model_kind)
        m["family"] = family(name)
        out[name] = m
        del graphs
    return out


def aggregate_family_metrics(
    model: nn.Module,
    model_kind: str,
    root: Path,
    names: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    groups: dict[str, list] = {}
    for name in names:
        graphs, _ = load_graphs(root / name)
        groups.setdefault(family(name), []).extend(graphs)
    result = {}
    for fam, graphs in groups.items():
        result[fam] = evaluate_mlp(model, graphs, device) if model_kind == "mlp" else \
            evaluate_graph_model(model, graphs, device, batch_size, model_kind)
    return result


def final_overall(
    model: nn.Module,
    model_kind: str,
    root: Path,
    names: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    graphs_all = []
    for name in names:
        graphs, _ = load_graphs(root / name)
        graphs_all.extend(graphs)
    if not graphs_all:
        return metrics_from_arrays([], [], [])
    return evaluate_mlp(model, graphs_all, device) if model_kind == "mlp" else \
        evaluate_graph_model(model, graphs_all, device, batch_size, model_kind)


def forgetting_from_matrix(matrix: list[dict[str, Any]], train_names: list[str]) -> dict[str, Any]:
    if not matrix:
        return {"per_scenario": {}, "average_forgetting_f1": float("nan"),
                "average_backward_transfer_f1": float("nan")}
    per = {}
    forget_vals = []
    bwt_vals = []
    for idx, name in enumerate(train_names):
        observed = []
        learned_value = None
        for stage_idx, row in enumerate(matrix):
            m = row.get("seen_scenarios", {}).get(name)
            if m is None:
                continue
            f1 = float(m.get("f1", float("nan")))
            if np.isfinite(f1):
                observed.append(f1)
                if stage_idx == idx:
                    learned_value = f1
        if not observed:
            continue
        final = observed[-1]
        best = max(observed)
        forgetting = best - final
        bwt = final - learned_value if learned_value is not None else float("nan")
        per[name] = {"best_f1": best, "final_f1": final,
                     "forgetting_f1": forgetting, "backward_transfer_f1": bwt}
        forget_vals.append(forgetting)
        if np.isfinite(bwt):
            bwt_vals.append(bwt)
    return {
        "per_scenario": per,
        "average_forgetting_f1": float(np.mean(forget_vals)) if forget_vals else float("nan"),
        "average_backward_transfer_f1": float(np.mean(bwt_vals)) if bwt_vals else float("nan"),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def aggregate(per_seed: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = ["precision", "recall", "f1", "accuracy", "auc_roc"]
    mean, std = {}, {}
    for key in keys:
        arr = np.asarray([r["test_overall"][key] for r in per_seed], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        mean[key] = float(arr.mean()) if arr.size else float("nan")
        std[key] = float(arr.std(ddof=1)) if arr.size > 1 else (0.0 if arr.size == 1 else float("nan"))
    f = np.asarray([r.get("forgetting", {}).get("average_forgetting_f1", np.nan) for r in per_seed], dtype=float)
    f = f[np.isfinite(f)]
    mean["average_forgetting_f1"] = float(f.mean()) if f.size else float("nan")
    std["average_forgetting_f1"] = float(f.std(ddof=1)) if f.size > 1 else (0.0 if f.size == 1 else float("nan"))
    return mean, std


def main() -> None:
    args = parse_args()
    train_names = csv(args.train_scenarios)
    test_names = csv(args.test_scenarios)
    seeds = [int(s) for s in csv(args.seeds)]
    if not train_names:
        raise SystemExit("--train-scenarios is required")
    if not args.smoke and len(seeds) != 5:
        raise SystemExit("paper comparison runs require exactly five seeds; use --smoke to relax")
    if not args.smoke and args.epochs != 200:
        raise SystemExit("paper comparison runs use 200 epochs; use --smoke to relax")

    args.out.mkdir(parents=True, exist_ok=True)
    first_graphs, meta = load_graphs(args.snapshots_root / train_names[0])
    in_channels = int(meta.get("node_feature_dim", first_graphs[0].x.shape[1]))
    if "ctu13" in train_names[0].lower():
        x0 = first_graphs[0].x
        if x0.numel() and torch.allclose(x0, x0[0].expand_as(x0)):
            raise SystemExit(
                "CTU-13 Prometheus received constant node features (usually type_only). "
                "Prometheus intentionally ignores edge_attr, so build/use *_flowagg snapshots first."
            )
    del first_graphs
    total_graphs = graph_count(args.snapshots_root, train_names)
    buffer_capacity = max(1, int(math.ceil(args.buffer_pct * total_graphs)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = vars(args).copy()
    config["snapshots_root"] = str(args.snapshots_root)
    config["out"] = str(args.out)
    config["train_scenarios"] = train_names
    config["test_scenarios"] = test_names
    config["in_channels"] = in_channels
    config["buffer_capacity_graphs"] = buffer_capacity
    config["device"] = str(device)
    config["ctu13_family_map"] = CTU13_FAMILY
    if args.experiment == "generative_replay":
        config["paper_underspecification_note"] = (
            "The paper names a generative replay IDS but does not publish its generator/classifier architecture; "
            "this run uses a conditional VAE + node MLP as a transparent CTU-13 proxy."
        )
    (args.out / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    per_seed = []
    for seed in seeds:
        print(f"\n===== {args.experiment} seed={seed} =====", flush=True)
        set_seed(seed)
        seed_dir = args.out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        history: dict[str, Any] = {}
        continual_matrix: list[dict[str, Any]] = []

        if args.experiment in {"full_rehearsal", "finetune", "static_gnn"}:
            model: nn.Module = Prometheus(in_channels=in_channels).to(device)
            model_kind = "gnn"
        elif args.experiment in {"cnn_bilstm", "cnn_bilstm_rehearsal"}:
            model = CNNBiLSTMAttention(in_channels=in_channels).to(device)
            model_kind = "cnn"
        else:
            model = NodeMLP(in_channels=in_channels).to(device)
            model_kind = "mlp"

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start_train = time.perf_counter()

        if args.experiment in {"static_gnn", "cnn_bilstm"}:
            history["joint"] = train_joint_scenarios(
                model, args.snapshots_root, train_names, optimizer, device,
                args.epochs, args.batch_size, model_kind, seed
            )
            seen = evaluate_named_scenarios(model, model_kind, args.snapshots_root,
                                            train_names, device, args.batch_size)
            continual_matrix.append({"stage": "joint", "seen_scenarios": seen})

        elif args.experiment in {"full_rehearsal", "finetune", "cnn_bilstm_rehearsal"}:
            use_buffer = args.experiment in {"full_rehearsal", "cnn_bilstm_rehearsal"}
            buffer = RehearsalBuffer(max_size=buffer_capacity, seed=seed) if use_buffer else None
            for stage_idx, name in enumerate(train_names):
                graphs, m = load_graphs(args.snapshots_root / name)
                if int(m.get("node_feature_dim", in_channels)) != in_channels:
                    raise ValueError(f"feature width mismatch in {name}")
                history[name] = train_graph_epochs(
                    model, graphs, optimizer, device, args.epochs, args.batch_size,
                    model_kind, buffer, args.lambda_rehearsal, label=name
                )
                if buffer is not None:
                    buffer.add_many(graphs)
                seen_names = train_names[:stage_idx + 1]
                seen = evaluate_named_scenarios(model, model_kind, args.snapshots_root,
                                                seen_names, device, args.batch_size)
                test_now = evaluate_named_scenarios(model, model_kind, args.snapshots_root,
                                                    test_names, device, args.batch_size) if test_names else {}
                continual_matrix.append({"stage": name, "seen_scenarios": seen, "held_out": test_now,
                                         "buffer_size": len(buffer) if buffer is not None else 0})
                del graphs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:  # generative_replay
            vae = ConditionalVAE(in_channels=in_channels).to(device)
            generator_ready = False
            past_positive_rate = 0.5
            for stage_idx, name in enumerate(train_names):
                graphs, _ = load_graphs(args.snapshots_root / name)
                current_x, current_y = sample_current_nodes(
                    graphs, args.generator_max_current_nodes, seed + stage_idx
                )
                train_x, train_y = current_x, current_y
                if generator_ready and args.generator_replay_nodes > 0:
                    n = args.generator_replay_nodes
                    rng = np.random.default_rng(seed + 1000 + stage_idx)
                    labels_np = (rng.random(n) < past_positive_rate).astype(np.int64)
                    replay_y = torch.from_numpy(labels_np).long().to(device)
                    vae.eval()
                    replay_x = vae.sample(replay_y).cpu()
                    replay_y = replay_y.cpu()
                    train_x = torch.cat([current_x, replay_x], dim=0)
                    train_y = torch.cat([current_y, replay_y], dim=0)
                history[name] = train_mlp_tensor(
                    model, train_x, train_y, optimizer, device, args.epochs,
                    args.batch_size, label=name
                )
                # Preserve the generator itself by training it on current + its
                # own previous synthetic replay distribution.
                train_vae_tensor(
                    vae, train_x, train_y, device, args.generator_epochs,
                    args.batch_size, args.lr
                )
                generator_ready = True
                past_positive_rate = float(train_y.float().mean().item())
                seen_names = train_names[:stage_idx + 1]
                seen = evaluate_named_scenarios(model, "mlp", args.snapshots_root,
                                                seen_names, device, args.batch_size)
                test_now = evaluate_named_scenarios(model, "mlp", args.snapshots_root,
                                                    test_names, device, args.batch_size) if test_names else {}
                continual_matrix.append({"stage": name, "seen_scenarios": seen,
                                         "held_out": test_now,
                                         "generated_replay_nodes": int(args.generator_replay_nodes if stage_idx else 0)})
                del graphs, current_x, current_y, train_x, train_y
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        train_seconds = time.perf_counter() - start_train
        test_per_scenario = evaluate_named_scenarios(
            model, model_kind, args.snapshots_root, test_names, device, args.batch_size
        ) if test_names else {}
        test_by_family = aggregate_family_metrics(
            model, model_kind, args.snapshots_root, test_names, device, args.batch_size
        ) if test_names else {}
        test_overall = final_overall(
            model, model_kind, args.snapshots_root, test_names, device, args.batch_size
        ) if test_names else metrics_from_arrays([], [], [])
        forgetting = forgetting_from_matrix(continual_matrix, train_names)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_kind": model_kind,
            "experiment": args.experiment,
            "in_channels": in_channels,
            "seed": seed,
            "train_scenarios": train_names,
            "test_scenarios": test_names,
            "config": config,
        }
        torch.save(checkpoint, seed_dir / "model.pt")
        result = {
            "seed": seed,
            "experiment": args.experiment,
            "model_kind": model_kind,
            "parameter_count": parameter_count(model),
            "train_seconds": train_seconds,
            "test_overall": test_overall,
            "test_per_scenario": test_per_scenario,
            "test_by_family": test_by_family,
            "continual_matrix": continual_matrix,
            "forgetting": forgetting,
        }
        (seed_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (seed_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        per_seed.append(result)
        print(f"[seed {seed}] final test={test_overall} forgetting={forgetting.get('average_forgetting_f1')}", flush=True)

    mean, std = aggregate(per_seed)
    summary = {
        "experiment": args.experiment,
        "dataset": "CTU-13",
        "target": "node",
        "train_scenarios": train_names,
        "test_scenarios": test_names,
        "per_seed": per_seed,
        "mean": mean,
        "std": std,
        "reporting": "mean +/- sample standard deviation over independent seeds",
    }
    if args.experiment == "generative_replay":
        summary["caveat"] = config["paper_underspecification_note"]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if seeds:
        shutil.copy2(args.out / f"seed_{seeds[0]}" / "model.pt", args.out / "best_model.pt")
    print("\n===== aggregate =====")
    for key, val in mean.items():
        print(f"{key}: {val:.6f} +/- {std.get(key, float('nan')):.6f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

