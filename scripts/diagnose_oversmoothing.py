#!/usr/bin/env python3
"""Diagnose over-smoothing in the paper-faithful Prometheus stack.

Why this exists
---------------
The base paper (IEEE Access Vol. 13, Dec 2025, DOI 10.1109/ACCESS.2025.3639270)
specifies 3 GNN layers with attention aggregation and NO residual connection,
NO normalisation, and node-wise softmax (Eq. 4).  On a *dense* graph that
recipe provably collapses: each GAT layer replaces a node's state with an
attention-weighted average over its neighbourhood, so the node's own signal is
retained with weight ~1/deg per layer and ~(1/deg)^L after L layers.  At
deg ~ 24 and L = 3 that is ~7e-5 — every node ends up with the same
representation and AUC-ROC is pinned to 0.5.

This script measures where that transition happens as a function of graph
density, and whether a residual connection (arguably inside the paper's own
Eq. 3, since AGGREGATE(h_v u N(v)) explicitly retains h_v) rescues it.

The synthetic generator is deliberately NOT the CTU-13 pipeline: the point is
to isolate depth and density as the only moving parts, with node features whose
discriminative power is known by construction.  A logistic-regression ceiling
on the raw features is reported so a low GNN score can never be blamed on the
features.

Usage:
    python scripts/diagnose_oversmoothing.py                # full sweep
    python scripts/diagnose_oversmoothing.py --degrees 24   # single density
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data

from models.prometheus import Prometheus

# flow_agg node-feature width on CTU-13 (see graph_builder.temporal.FLOW_AGG_DIM
# plus the 1-dim node-type column), so the sweep runs at the real input width.
FEATURE_DIM = 39
SIGNAL_DIMS = 6      # how many of the 39 carry the botnet signal
COHEN_D = 1.0        # per-dim separation; modest on purpose, not a giveaway


def make_graph(n_nodes: int, avg_degree: int, pos_rate: float,
               rng: np.random.Generator) -> Data:
    """One window: node features with a known signal + edges at a target density."""
    y = (rng.random(n_nodes) < pos_rate).astype(np.int64)

    x = rng.standard_normal((n_nodes, FEATURE_DIM)).astype(np.float32)
    x[y == 1, :SIGNAL_DIMS] += COHEN_D

    n_edges = max(1, int(n_nodes * avg_degree / 2))
    src = rng.integers(0, n_nodes, size=n_edges)
    dst = rng.integers(0, n_nodes, size=n_edges)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    # Undirected: message passing in both directions, so avg_degree is honoured.
    edge_index = np.stack([np.concatenate([src, dst]),
                           np.concatenate([dst, src])]).astype(np.int64)

    return Data(x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_index),
                y=torch.from_numpy(y))


def feature_ceiling(train: list[Data], test: list[Data]) -> tuple[float, float]:
    """Logistic regression on raw node features — the score the GNN must beat."""
    xtr = np.concatenate([d.x.numpy() for d in train])
    ytr = np.concatenate([d.y.numpy() for d in train])
    xte = np.concatenate([d.x.numpy() for d in test])
    yte = np.concatenate([d.y.numpy() for d in test])
    lr = LogisticRegression(max_iter=2000).fit(xtr, ytr)
    p = lr.predict_proba(xte)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def run_gnn(train: list[Data], test: list[Data], layers: int, residual: bool,
            epochs: int, seed: int, device: torch.device) -> tuple[float, float, float]:
    """Train paper-hyperparameter Prometheus; return (auc_roc, auc_pr, distinct_frac)."""
    torch.manual_seed(seed)
    model = Prometheus(
        in_channels=FEATURE_DIM, hidden_channels=128, num_gnn_layers=layers,
        dropout=0.2, edge_dim=0, heads=4, num_classes=2,
        readout="node", use_layer_norm=False, residual=residual,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()

    tr = PyGBatch.from_data_list(train).to(device)
    te = PyGBatch.from_data_list(test).to(device)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(tr.x, tr.edge_index, tr.batch), tr.y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(te.x, te.edge_index, te.batch), dim=-1)[:, 1]
    p = p.cpu().numpy()
    yte = te.y.cpu().numpy()

    # Fraction of distinct predictions: the direct fingerprint of collapse.
    distinct = len(np.unique(np.round(p, 6))) / max(len(p), 1)
    return roc_auc_score(yte, p), average_precision_score(yte, p), distinct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", type=str, default="2,4,8,16,24,48",
                    help="Average node degrees to sweep")
    ap.add_argument("--layers", type=str, default="1,2,3")
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--pos-rate", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    degrees = [int(d) for d in args.degrees.split(",") if d.strip()]
    layer_counts = [int(l) for l in args.layers.split(",") if l.strip()]
    device = torch.device("cpu")

    print("=" * 78)
    print("OVER-SMOOTHING SWEEP — paper hyperparameters (hidden=128, heads=4,")
    print("dropout=0.2, Adam 1e-3/1e-5, no LayerNorm, node-wise Eq. 4 readout)")
    print(f"{args.windows} windows x {args.nodes} nodes, {FEATURE_DIM}-dim features, "
          f"{SIGNAL_DIMS} signal dims at cohen-d {COHEN_D}, "
          f"pos rate {args.pos_rate:.0%}")
    print("=" * 78)

    for deg in degrees:
        rng = np.random.default_rng(args.seed)
        graphs = [make_graph(args.nodes, deg, args.pos_rate, rng)
                  for _ in range(args.windows)]
        split = int(0.7 * len(graphs))
        train, test = graphs[:split], graphs[split:]

        lr_roc, lr_pr = feature_ceiling(train, test)
        print(f"\n--- avg degree {deg} "
              f"(edges/window ~ {int(args.nodes * deg / 2)}) ---")
        print(f"  logistic-regression ceiling on raw features: "
              f"AUC-ROC {lr_roc:.4f}  AUC-PR {lr_pr:.4f}")
        print(f"  {'layers':>6}  {'residual':>8}  {'AUC-ROC':>8}  {'AUC-PR':>8}  "
              f"{'distinct preds':>14}")
        for layers in layer_counts:
            for residual in (False, True):
                roc, pr, distinct = run_gnn(train, test, layers, residual,
                                            args.epochs, args.seed, device)
                tag = "paper" if (layers == 3 and not residual) else ""
                print(f"  {layers:>6}  {str(residual):>8}  {roc:>8.4f}  {pr:>8.4f}  "
                      f"{distinct:>13.1%}  {tag}")

    print("\n" + "=" * 78)
    print("Read: compare each row's AUC-ROC against the logistic-regression ceiling")
    print("for that density.  A row far below the ceiling has collapsed, EVEN IF its")
    print("distinct-prediction share is high: over-smoothing leaves the outputs")
    print("numerically different while stripping out the class information.  Only the")
    print("degenerate-input case (constant node features) pins distinct preds to ~0%.")
    print("=" * 78)


if __name__ == "__main__":
    main()
