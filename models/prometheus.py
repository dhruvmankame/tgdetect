"""Paper-faithful Prometheus GNN.

Reference:
A. Pomsathit, "Adaptive Detection of Advanced Persistent Threats (APT)
With Graph Neural Networks and Rehearsal-Based Continual Learning on
Wazuh EDR Telemetry," IEEE Access, 2025.

The paper explicitly specifies:
  * 3 GNN layers
  * hidden dimension 128
  * ReLU
  * dropout 0.2
  * attention-based neighborhood aggregation
  * node-wise classification with softmax

The paper does NOT specify the exact attention operator or number of heads.
This implementation uses the minimal standard PyG concretization: single-head
GATConv with 128 output features at every layer. No pooling, LayerNorm, edge
conditioning, residual block, or other unreported architectural component is
added.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

PAPER_GNN_LAYERS = 3
PAPER_HIDDEN_DIM = 128
PAPER_DROPOUT = 0.2
PAPER_NUM_CLASSES = 2
PAPER_ATTENTION_HEADS = 1  # paper leaves head count unspecified; minimal choice


class Prometheus(nn.Module):
    """Three-layer attention GNN with node-wise classification.

    Paper equations implemented conceptually as:
      h_v^(l+1) = ReLU(W^(l) * ATTENTION_AGGREGATE(v, N(v)))
      y_v = softmax(W_o * h_v^(L) + b_o)

    ``forward`` returns raw class logits because ``CrossEntropyLoss`` applies
    the softmax/log-softmax operation internally in a numerically stable way.
    Use ``predict_proba`` when explicit probabilities are required.
    """

    def __init__(self, in_channels: int, num_classes: int = PAPER_NUM_CLASSES) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = PAPER_HIDDEN_DIM
        self.num_gnn_layers = PAPER_GNN_LAYERS
        self.dropout_p = PAPER_DROPOUT
        self.num_classes = int(num_classes)

        self.convs = nn.ModuleList(
            [
                GATConv(
                    self.in_channels,
                    PAPER_HIDDEN_DIM,
                    heads=PAPER_ATTENTION_HEADS,
                    concat=False,
                    dropout=0.0,
                ),
                GATConv(
                    PAPER_HIDDEN_DIM,
                    PAPER_HIDDEN_DIM,
                    heads=PAPER_ATTENTION_HEADS,
                    concat=False,
                    dropout=0.0,
                ),
                GATConv(
                    PAPER_HIDDEN_DIM,
                    PAPER_HIDDEN_DIM,
                    heads=PAPER_ATTENTION_HEADS,
                    concat=False,
                    dropout=0.0,
                ),
            ]
        )
        self.dropout = nn.Dropout(PAPER_DROPOUT)
        self.classifier = nn.Linear(PAPER_HIDDEN_DIM, self.num_classes)

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node-wise class logits with shape ``[num_nodes, 2]``."""
        h = self.encode_nodes(x, edge_index)
        return self.classifier(h)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the paper's explicit node-wise softmax probabilities."""
        return torch.softmax(self.forward(x, edge_index), dim=-1)


def snapshot_to_data(snap: dict) -> Data:
    """Convert one temporal snapshot dictionary to a node-classification graph.

    Prometheus intentionally consumes only ``x`` and ``edge_index``. Edge
    attributes are excluded because the base paper defines ``G=(V,E,X)`` with
    a node feature matrix X and does not include edge attributes in Eq. (3).
    """
    if "node_labels" not in snap:
        raise KeyError("snapshot is missing node_labels required by paper-style node classification")

    x_np = np.asarray(snap["x"], dtype=np.float32)
    edge_index_np = np.asarray(snap["edge_index"], dtype=np.int64)
    y_np = np.asarray(snap["node_labels"], dtype=np.int64)

    if x_np.ndim != 2:
        raise ValueError(f"x must be [N,F], got {x_np.shape}")
    if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        raise ValueError(f"edge_index must be [2,E], got {edge_index_np.shape}")
    if y_np.ndim != 1 or y_np.shape[0] != x_np.shape[0]:
        raise ValueError(f"node_labels must be [N] aligned with x; got y={y_np.shape}, x={x_np.shape}")

    return Data(
        x=torch.from_numpy(x_np),
        edge_index=torch.from_numpy(edge_index_np),
        y=torch.from_numpy(y_np),
    )
