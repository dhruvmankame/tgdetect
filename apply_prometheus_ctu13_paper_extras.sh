#!/usr/bin/env bash
set -euo pipefail

# Apply the CTU-13 paper-comparison experimental suite around the already
# paper-aligned Prometheus implementation. Run from the tgdetect repository root.

if [[ ! -f "modal_train.py" || ! -d "models" || ! -d "scripts" ]]; then
  echo "ERROR: run this script from the tgdetect repository root." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP=".prometheus-paper-extras-backup-${STAMP}"
mkdir -p "$BACKUP/models" "$BACKUP/scripts"
for f in \
  models/prometheus_baselines.py \
  scripts/train_prometheus_experiment.py \
  scripts/benchmark_prometheus.py \
  scripts/explain_prometheus.py \
  scripts/report_prometheus_paper.py \
  scripts/check_prometheus_paper_suite.py \
  PROMETHEUS_CTU13_PAPER_SUITE.md \
  modal_train.py; do
  if [[ -f "$f" ]]; then
    mkdir -p "$BACKUP/$(dirname "$f")"
    cp -a "$f" "$BACKUP/$f"
  fi
done

echo "Backup created: $BACKUP"
mkdir -p models scripts

cat > models/prometheus_baselines.py <<'TGDETECT_MODELS_PROMETHEUS_BASELINES_PY'
"""Baselines used by the CTU-13 Prometheus paper-comparison suite.

These baselines intentionally live outside models/prometheus.py so the
paper-faithful Prometheus architecture remains untouched.

The base paper names several comparison systems but does not publish enough
implementation detail to reproduce their source code exactly.  Therefore:

* Static GNN / fine-tuned GNN reuse the repository's paper-aligned Prometheus
  GAT architecture and only change the training protocol.
* CNN-BiLSTM+attention is a transparent node-sequence baseline with two 1-D
  convolutions, a bidirectional LSTM whose concatenated width is 128, and
  additive attention.
* Generative replay uses a conditional VAE over CTU-13 node features plus a
  node MLP classifier.  It is explicitly a reproducible proxy for the paper's
  under-specified "generative replay IDS", not a claim about the authors'
  hidden implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTMAttention(nn.Module):
    """Two Conv1d layers -> BiLSTM(128 total width) -> additive attention.

    The model predicts one class per node.  For a PyG mini-batch, each graph is
    treated as one variable-length node sequence and processed independently;
    logits are concatenated back in PyG node order.
    """

    def __init__(
        self,
        in_channels: int,
        conv_channels: int = 64,
        bilstm_total_hidden: int = 128,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if bilstm_total_hidden % 2:
            raise ValueError("bilstm_total_hidden must be even for bidirectional LSTM")
        self.in_channels = int(in_channels)
        self.conv_channels = int(conv_channels)
        self.bilstm_total_hidden = int(bilstm_total_hidden)
        self.dropout_p = float(dropout)
        per_direction = bilstm_total_hidden // 2

        self.conv1 = nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=per_direction,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(bilstm_total_hidden, 1)
        self.classifier = nn.Linear(2 * bilstm_total_hidden, num_classes)

    def _one_graph(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((0, self.classifier.out_features))
        # [N,F] -> [1,F,N]
        h = x.transpose(0, 1).unsqueeze(0)
        h = self.dropout(F.relu(self.conv1(h)))
        h = self.dropout(F.relu(self.conv2(h)))
        # [1,C,N] -> [1,N,C]
        h = h.transpose(1, 2)
        seq, _ = self.bilstm(h)
        scores = self.attention(seq).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * seq, dim=1, keepdim=True)
        context = context.expand(-1, seq.size(1), -1)
        logits = self.classifier(torch.cat([seq, context], dim=-1))
        return logits.squeeze(0)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if batch is None:
            return self._one_graph(x)
        if batch.numel() == 0:
            return x.new_zeros((0, self.classifier.out_features))
        pieces = []
        num_graphs = int(batch.max().item()) + 1
        for graph_id in range(num_graphs):
            pieces.append(self._one_graph(x[batch == graph_id]))
        return torch.cat(pieces, dim=0) if pieces else x.new_zeros((0, 2))


class NodeMLP(nn.Module):
    """Simple node classifier used by the generative-replay proxy baseline."""

    def __init__(self, in_channels: int, hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalVAE(nn.Module):
    """Conditional VAE over node features for a reproducible replay proxy."""

    def __init__(self, in_channels: int, latent_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_channels + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels),
        )

    @staticmethod
    def _one_hot(y: torch.Tensor) -> torch.Tensor:
        return F.one_hot(y.long(), num_classes=2).float()

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, self._one_hot(y)], dim=-1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, self._one_hot(y)], dim=-1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        z = torch.randn(labels.numel(), self.latent_dim, device=labels.device)
        return self.decode(z, labels)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + float(beta) * kl

TGDETECT_MODELS_PROMETHEUS_BASELINES_PY

cat > scripts/train_prometheus_experiment.py <<'TGDETECT_SCRIPTS_TRAIN_PROMETHEUS_EXPERIMENT_PY'
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

TGDETECT_SCRIPTS_TRAIN_PROMETHEUS_EXPERIMENT_PY

cat > scripts/benchmark_prometheus.py <<'TGDETECT_SCRIPTS_BENCHMARK_PROMETHEUS_PY'
#!/usr/bin/env python3
"""Measure Prometheus inference latency and memory on the chosen hardware.

The paper reports latency/memory but does not define its exact timing protocol.
This script makes the protocol explicit and saves enough metadata to reproduce
it.  Use Modal A10G for project measurements; do not compare the raw latency
number directly with the paper's V100 result without stating the hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.prometheus import Prometheus, snapshot_to_data


def load_graphs(root: Path, names: list[str], max_graphs: int) -> list:
    graphs = []
    for name in names:
        files = sorted((root / name).glob("snapshot_*.pkl"))
        for fp in files:
            with open(fp, "rb") as fh:
                graphs.append(snapshot_to_data(pickle.load(fh)))
            if max_graphs and len(graphs) >= max_graphs:
                return graphs
    return graphs


def graph_bytes(g) -> int:
    total = g.x.numel() * g.x.element_size()
    total += g.edge_index.numel() * g.edge_index.element_size()
    total += g.y.numel() * g.y.element_size()
    return int(total)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--snapshots-root", type=Path, required=True)
    p.add_argument("--scenarios", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--max-graphs", type=int, default=100)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    in_channels = int(ckpt.get("meta", {}).get("node_feature_dim", ckpt.get("in_channels", 0)))
    if not in_channels:
        raise KeyError("checkpoint missing node_feature_dim/in_channels")
    model = Prometheus(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    names = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    graphs = load_graphs(args.snapshots_root, names, args.max_graphs)
    if not graphs:
        raise SystemExit("no graphs found")
    loader = list(DataLoader(graphs, batch_size=args.batch_size, shuffle=False))

    with torch.no_grad():
        for i in range(args.warmup):
            b = loader[i % len(loader)].to(device)
            _ = model(b.x, b.edge_index)
        sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latencies_ms = []
        graphs_seen = nodes_seen = 0
        for i in range(args.repeats):
            b = loader[i % len(loader)].to(device)
            sync(device)
            t0 = time.perf_counter()
            _ = model(b.x, b.edge_index)
            sync(device)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt)
            graphs_seen += int(b.num_graphs)
            nodes_seen += int(b.num_nodes)

    total_ms = sum(latencies_ms)
    sorted_l = sorted(latencies_ms)
    p95 = sorted_l[min(len(sorted_l) - 1, int(round(0.95 * (len(sorted_l) - 1))))]
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    sample_bytes = [graph_bytes(g) for g in graphs]
    capacity = int(ckpt.get("paper_config", {}).get("rehearsal_buffer_capacity_graphs", 0))
    replay_est = (statistics.median(sample_bytes) * capacity) if capacity and sample_bytes else 0

    result = {
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "device": str(device),
        "protocol": {
            "warmup_batches": args.warmup,
            "timed_batches": args.repeats,
            "batch_size_graphs": args.batch_size,
            "max_loaded_graphs": args.max_graphs,
            "synchronizes_cuda_before_and_after_each_timing": device.type == "cuda",
        },
        "latency_ms_per_batch_mean": float(statistics.mean(latencies_ms)),
        "latency_ms_per_batch_median": float(statistics.median(latencies_ms)),
        "latency_ms_per_batch_p95": float(p95),
        "latency_ms_per_graph": float(total_ms / max(graphs_seen, 1)),
        "latency_ms_per_node": float(total_ms / max(nodes_seen, 1)),
        "nodes_per_second": float(nodes_seen / max(total_ms / 1000.0, 1e-12)),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "parameter_memory_mb": float(param_bytes / (1024 ** 2)),
        "median_serialized_tensor_bytes_per_graph": float(statistics.median(sample_bytes)),
        "replay_capacity_graphs_from_checkpoint": capacity,
        "estimated_replay_tensor_memory_mb": float(replay_est / (1024 ** 2)),
        "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None,
        "cuda_peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)) if device.type == "cuda" else None,
        "caveat": "Paper timing/memory denominator is under-specified; compare only with protocol and hardware disclosed.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

TGDETECT_SCRIPTS_BENCHMARK_PROMETHEUS_PY

cat > scripts/explain_prometheus.py <<'TGDETECT_SCRIPTS_EXPLAIN_PROMETHEUS_PY'
#!/usr/bin/env python3
"""Export attention-backed malicious subgraph evidence for Prometheus.

This is model-derived evidence, not a claim that GAT attention is a complete
causal explanation.  It provides a defensible paper-style malicious-subgraph
artifact: high-risk nodes plus the strongest final-layer attention edges that
support them.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.prometheus import Prometheus


@torch.no_grad()
def explain_one(model: Prometheus, snap: dict, device: torch.device, top_nodes: int, top_edges: int) -> dict:
    x = torch.from_numpy(np.asarray(snap["x"], dtype=np.float32)).to(device)
    edge_index = torch.from_numpy(np.asarray(snap["edge_index"], dtype=np.int64)).to(device)
    h = x
    att_edge_index = edge_index
    att_alpha = None
    for conv in model.convs:
        h, att = conv(h, edge_index, return_attention_weights=True)
        att_edge_index, att_alpha = att
        h = model.dropout(F.relu(h))
    logits = model.classifier(h)
    probs = torch.softmax(logits, dim=-1)[:, 1]
    k = min(top_nodes, probs.numel())
    selected = torch.topk(probs, k=k).indices if k else torch.empty(0, dtype=torch.long, device=device)
    selected_set = set(selected.cpu().tolist())
    node_ids = list(snap.get("node_ids") or [str(i) for i in range(x.size(0))])
    labels = np.asarray(snap.get("node_labels", np.zeros(x.size(0))), dtype=np.int64)

    nodes = [
        {"index": int(i), "node_id": node_ids[int(i)], "malicious_probability": float(probs[int(i)].item()),
         "ground_truth": int(labels[int(i)]) if int(i) < len(labels) else None}
        for i in selected.cpu().tolist()
    ]

    edges = []
    if att_alpha is not None:
        alpha = att_alpha.mean(dim=-1) if att_alpha.ndim > 1 else att_alpha
        src = att_edge_index[0].cpu().tolist()
        dst = att_edge_index[1].cpu().tolist()
        aval = alpha.cpu().tolist()
        for s, d, a in zip(src, dst, aval):
            if s in selected_set or d in selected_set:
                risk = max(float(probs[s].item()), float(probs[d].item()))
                edges.append({
                    "src_index": int(s), "src": node_ids[int(s)],
                    "dst_index": int(d), "dst": node_ids[int(d)],
                    "attention": float(a), "endpoint_risk": risk,
                    "evidence_score": float(a) * risk,
                })
        edges.sort(key=lambda e: e["evidence_score"], reverse=True)
        edges = edges[:top_edges]

    return {
        "ts_start": float(snap.get("ts_start", 0.0)),
        "ts_end": float(snap.get("ts_end", 0.0)),
        "scenario_id": snap.get("scenario_id", ""),
        "top_nodes": nodes,
        "attention_edges": edges,
        "explanation_note": "Final-layer GAT attention is supporting evidence, not guaranteed causal attribution.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--snapshots", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-nodes", type=int, default=25)
    p.add_argument("--top-edges", type=int, default=100)
    p.add_argument("--max-snapshots", type=int, default=10)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    in_channels = int(ckpt.get("meta", {}).get("node_feature_dim", ckpt.get("in_channels", 0)))
    model = Prometheus(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    files = sorted(args.snapshots.glob("snapshot_*.pkl"))
    if args.max_snapshots:
        files = files[:args.max_snapshots]
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for i, fp in enumerate(files):
        with open(fp, "rb") as fh:
            snap = pickle.load(fh)
        result = explain_one(model, snap, device, args.top_nodes, args.top_edges)
        target = args.out / f"explanation_{i:05d}.json"
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        index.append({"snapshot": fp.name, "file": target.name,
                      "max_risk": result["top_nodes"][0]["malicious_probability"] if result["top_nodes"] else None})
        print("saved", target)
    (args.out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

TGDETECT_SCRIPTS_EXPLAIN_PROMETHEUS_PY

cat > scripts/report_prometheus_paper.py <<'TGDETECT_SCRIPTS_REPORT_PROMETHEUS_PAPER_PY'
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

TGDETECT_SCRIPTS_REPORT_PROMETHEUS_PAPER_PY

cat > scripts/check_prometheus_paper_suite.py <<'TGDETECT_SCRIPTS_CHECK_PROMETHEUS_PAPER_SUITE_PY'
#!/usr/bin/env python3
"""Lightweight architecture checks for the CTU-13 paper-comparison suite."""
import torch
from models.prometheus import Prometheus
from models.prometheus_baselines import CNNBiLSTMAttention, ConditionalVAE, NodeMLP

x = torch.randn(12, 39)
edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10,11]])
batch = torch.tensor([0]*6 + [1]*6)

p = Prometheus(39)
assert p(x, edge_index).shape == (12, 2)

c = CNNBiLSTMAttention(39)
assert c(x, batch).shape == (12, 2)

m = NodeMLP(39)
assert m(x).shape == (12, 2)

v = ConditionalVAE(39)
y = torch.randint(0, 2, (12,))
recon, mu, logvar = v(x, y)
assert recon.shape == x.shape and mu.shape == logvar.shape
assert v.sample(y).shape == x.shape
print("PASS: Prometheus CTU-13 paper suite structural checks")

TGDETECT_SCRIPTS_CHECK_PROMETHEUS_PAPER_SUITE_PY

cat > PROMETHEUS_CTU13_PAPER_SUITE.md <<'TGDETECT_PROMETHEUS_CTU13_PAPER_SUITE_MD'
# Prometheus CTU-13 — Base-Paper Experimental Suite

This suite deliberately keeps **CTU-13** as the project dataset. It does not add
Wazuh ingestion and it does not claim that CTU-13 results reproduce the paper's
Wazuh/sandbox dataset results.

## What this patch adds

The core Prometheus architecture/training path was already aligned to every
explicitly reported model parameter in the paper. The missing paper-side work
was primarily experimental evidence. This patch adds:

1. **Static GNN baseline** using the same Prometheus GAT but joint/static training.
2. **Fine-tuning baseline** using sequential tasks with no replay.
3. **CNN-BiLSTM + attention baseline** with two Conv1d layers and a 128-wide
   bidirectional LSTM representation.
4. **Sequential baseline + rehearsal** for the paper's ablation comparison.
5. **Generative replay proxy** using a conditional VAE + node MLP. The paper does
   not disclose the exact generative baseline architecture, so this is explicitly
   a reproducible project proxy rather than an alleged source-identical copy.
6. **After-each-task continual-learning evaluation**, including per-scenario F1,
   average forgetting, and backward transfer.
7. **CTU-13 scenario/family breakdown** instead of inventing MITRE tactic labels.
8. **Latency / memory benchmark** with an explicit timing protocol and GPU name.
9. **Attention-backed malicious-subgraph evidence** for high-risk nodes.
10. **Automatic Markdown comparison report** across the paper-style experiments.

## Critical CTU-13 feature requirement

Paper-style Prometheus ignores edge attributes. The ordinary CTU-13
`type_only` snapshots therefore give every IP node the same input feature and
can make node-wise GAT degenerate. For Prometheus experiments, build and use the
repository's leakage-free **`flow_agg` node features**:

```bash
modal run modal_train.py::snapshots_ctu13_flowagg \
  --names ctu13_c52,ctu13_c46,ctu13_c53,ctu13_c48,ctu13_c45,ctu13_c49,ctu13_c54,ctu13_c47
```

That creates directories such as `ctu13_c52_flowagg` and
`ctu13_c47_flowagg`. Use those names in the experiment commands below.

## Run one experiment on Modal A10G

```bash
modal run --detach modal_train.py::prometheus_experiment \
  --experiment finetune \
  --train-scenarios ctu13_c52_flowagg,ctu13_c46_flowagg,ctu13_c53_flowagg,ctu13_c48_flowagg,ctu13_c45_flowagg,ctu13_c49_flowagg,ctu13_c54_flowagg \
  --test-scenarios ctu13_c47_flowagg \
  --out-name paper_suite/finetune
```

Valid experiment names:

```text
full_rehearsal
finetune
static_gnn
cnn_bilstm
cnn_bilstm_rehearsal
generative_replay
```

If you already completed a correct five-seed `full_rehearsal` run, you do not
need to retrain it merely to add the other baselines.

## Benchmark the trained Prometheus checkpoint

```bash
modal run modal_train.py::benchmark_prometheus_extra \
  --checkpoint-name prometheus_paper \
  --scenarios ctu13_c47_flowagg
```

## Export attention-backed subgraph evidence

```bash
modal run modal_train.py::explain_prometheus_extra \
  --checkpoint-name prometheus_paper \
  --scenario ctu13_c47_flowagg
```

## Generate the comparison report

Store experiment outputs under a common root such as:

```text
/checkpoints/paper_suite/full_rehearsal/
/checkpoints/paper_suite/finetune/
/checkpoints/paper_suite/static_gnn/
/checkpoints/paper_suite/cnn_bilstm/
/checkpoints/paper_suite/cnn_bilstm_rehearsal/
/checkpoints/paper_suite/generative_replay/
```

Then run:

```bash
modal run modal_train.py::report_prometheus_extra --root-name paper_suite
```

The generated report is saved under `/checkpoints/paper_suite/`.

## What remains impossible to make source-identical

The publication still does not disclose the exact attention operator/head
count, lambda value, five seed values, or implementation of the generative
replay baseline. Those choices must remain documented project assumptions.
Using CTU-13 also means tactic-wise Wazuh results cannot be reproduced without
fabricating labels, which this suite intentionally avoids.

TGDETECT_PROMETHEUS_CTU13_PAPER_SUITE_MD

# Replace only our own Modal extras block on reruns, and fix the stale A100
# docstring left by the earlier paper-alignment patch. Existing functions stay intact.
python - <<'PYMODAL'
from pathlib import Path
p = Path("modal_train.py")
text = p.read_text(encoding="utf-8")
start = "# >>> PROMETHEUS_CTU13_PAPER_EXTRAS_BEGIN >>>"
end = "# <<< PROMETHEUS_CTU13_PAPER_EXTRAS_END <<<"
if start in text and end in text:
    a = text.index(start)
    b = text.index(end, a) + len(end)
    text = text[:a].rstrip() + "\n\n" + text[b:].lstrip()
text = text.replace('"""Submit the strict paper configuration to an A100 cloud GPU."""',
                    '"""Submit the strict paper configuration to an A10G cloud GPU."""')
block = r"""
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
"""
text = text.rstrip() + "\n\n" + block.strip() + "\n"
p.write_text(text, encoding="utf-8")
PYMODAL

chmod +x scripts/train_prometheus_experiment.py scripts/benchmark_prometheus.py \
  scripts/explain_prometheus.py scripts/report_prometheus_paper.py \
  scripts/check_prometheus_paper_suite.py

# Syntax-only validation: no heavy training is performed locally.
python -m py_compile \
  models/prometheus_baselines.py \
  scripts/train_prometheus_experiment.py \
  scripts/benchmark_prometheus.py \
  scripts/explain_prometheus.py \
  scripts/report_prometheus_paper.py \
  scripts/check_prometheus_paper_suite.py \
  modal_train.py

echo
echo "PASS: CTU-13 Prometheus paper-extras patch applied and Python syntax validated."
echo "No model training was run locally."
echo
echo "IMPORTANT: Prometheus ignores edge_attr, so use leakage-free *_flowagg snapshots."
echo "Build them on Modal with:"
echo "  modal run modal_train.py::snapshots_ctu13_flowagg --names ctu13_c52,ctu13_c46,ctu13_c53,ctu13_c48,ctu13_c45,ctu13_c49,ctu13_c54,ctu13_c47"
echo
echo "Then run a missing paper experiment, e.g.:"
cat <<'CMDOUT'
  modal run --detach modal_train.py::prometheus_experiment \
    --experiment finetune \
    --train-scenarios ctu13_c52_flowagg,ctu13_c46_flowagg,ctu13_c53_flowagg,ctu13_c48_flowagg,ctu13_c45_flowagg,ctu13_c49_flowagg,ctu13_c54_flowagg \
    --test-scenarios ctu13_c47_flowagg \
    --out-name paper_suite/finetune
CMDOUT
echo
echo "Documentation: PROMETHEUS_CTU13_PAPER_SUITE.md"
echo "Backup: $BACKUP"
