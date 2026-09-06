"""Prometheus: 3-layer attention GNN with rehearsal continual learning.

Base paper: "Adaptive Detection of Advanced Persistent Threats (APT) With
Graph Neural Networks and Rehearsal-Based Continual Learning on Wazuh EDR
Telemetry" (IEEE Access, 2025) — Auttapon Pomsathit, KMITL.

Architecture (matches paper):
  3-layer GAT (hidden=128, 4 heads, dropout 0.2, ReLU)
  Global mean pooling → graph-level logit
  L = L_new(D_t) + λ * L_rehearsal(D_buf)   (λ=1.0)
  Rehearsal buffer: 10% reservoir sampling

Task: graph-level classification (snapshot_label) on CTU-13.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Prometheus requires PyTorch and PyTorch Geometric. "
        "Install them with: pip install -r requirements-ml.txt"
    ) from exc


class Prometheus(nn.Module):
    """3-layer GAT for graph-level classification.

    Paper equations:
      h_v^(l+1) = σ(W^(l) * AGGREGATE(h_v^(l) ∪ {h_u^(l): u ∈ N(v)}))
      y = softmax(W_o * POOL(h_v^(L)) + b_o)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_gnn_layers: int = 3,
        dropout: float = 0.2,
        edge_dim: int = 0,
        heads: int = 4,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.heads = heads

        # GAT layers: first (num_gnn_layers-1) layers use multi-head attention;
        # last layer uses single head (concat=False) for classification.
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            if i == num_gnn_layers - 1:
                # Last layer: single head, no concatenation → hidden_channels
                self.convs.append(GATConv(
                    in_ch, hidden_channels, heads=1, concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim if edge_dim > 0 else None,
                ))
            else:
                self.convs.append(GATConv(
                    in_ch, hidden_channels // heads, heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim if edge_dim > 0 else None,
                ))
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

        # Graph-level classifier: single logit (BCE with logits)
        self.classifier = nn.Linear(hidden_channels, 1)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for graph-level classification.

        Args:
            x: [N, in_channels] node features (concatenated across batch)
            edge_index: [2, E] edge connectivity (concatenated across batch)
            batch: [N] batch assignment index per node
            edge_attr: [E, edge_dim] edge features (optional)

        Returns:
            logits: [B, 1] graph-level logits
        """
        h = x
        for conv, ln in zip(self.convs, self.layer_norms):
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = torch.relu(h)
            h = ln(h)
            h = self.dropout_layer(h)

        # Global mean pooling: [N, H] -> [B, H]
        h = global_mean_pool(h, batch)

        # Classifier: [B, H] -> [B, 1]
        return self.classifier(h)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode graphs to embeddings (no classifier, no dropout)."""
        h = x
        for conv, ln in zip(self.convs, self.layer_norms):
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = torch.relu(h)
            h = ln(h)
        return global_mean_pool(h, batch)


def snapshot_to_data(snap: dict, device: torch.device) -> Data:
    """Convert a snapshot dict to a PyG Data object for graph-level training."""
    x = torch.from_numpy(snap["x"]).float().to(device)
    edge_index = torch.from_numpy(snap["edge_index"]).long().to(device)
    edge_attr = (
        torch.from_numpy(snap["edge_attr"]).float().to(device)
        if snap.get("edge_attr") is not None and snap["edge_attr"].size > 0
        else None
    )
    y = torch.tensor([float(snap.get("snapshot_label", 0))], dtype=torch.float32, device=device)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
