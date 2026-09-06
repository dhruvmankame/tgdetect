"""Temporal Graph Neural Network for multi-step attack detection.

This module uses PyTorch Geometric (PyG). If PyG is not installed the import
will fail with a clear error pointing to `requirements-ml.txt`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Framework guard: provide a helpful message instead of a cryptic import error.
try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TGNN models require PyTorch and PyTorch Geometric. "
        "Install them with: pip install -r requirements-ml.txt"
    ) from exc


class TemporalGNN(nn.Module):
    """Baseline TGNN: GNN per snapshot + GRU over node embeddings across time.

    The model processes a sequence of snapshot graphs and predicts a
    maliciousness score for every node (or edge) in the final snapshot.
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_gnn_layers: int = 2,
        num_rnn_layers: int = 1,
        dropout: float = 0.3,
        node_types: Optional[int] = None,
        num_relations: Optional[int] = None,
        gnn_aggregation: str = "sage",  # "sage" | "gat" | "gcn"
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.gnn_aggregation = gnn_aggregation.lower()

        # Optional learned embeddings for heterogeneous types.
        self.node_type_embedding: Optional[nn.Embedding] = None
        self.relation_embedding: Optional[nn.Embedding] = None

        # Node type is already one-hot encoded inside `x` by the snapshot
        # builder, so no extra embedding is concatenated here; the GNN input
        # width must match x exactly.
        gnn_input = in_channels
        if num_relations and edge_dim == 0:
            self.relation_embedding = nn.Embedding(num_relations, 16)
            edge_dim = 16

        # GNN layers operate on each snapshot independently.
        self.convs = nn.ModuleList()
        self.edge_encoders = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            in_ch = gnn_input if i == 0 else hidden_channels
            agg = self.gnn_aggregation
            if agg == "gat":
                # Multi-head GAT: paper-aligned with 4 heads, dropout 0.2.
                self.convs.append(GATConv(in_ch, hidden_channels // 4, heads=4,
                                          dropout=dropout, edge_dim=edge_dim if edge_dim > 0 else None))
                self.edge_encoders.append(None)  # GAT takes edge_attr directly
            elif agg == "gcn":
                self.convs.append(GCNConv(in_ch, hidden_channels))
                self.edge_encoders.append(None)
            else:
                self.convs.append(SAGEConv(in_ch, hidden_channels))
                if edge_dim > 0:
                    self.edge_encoders.append(nn.Linear(edge_dim, hidden_channels))
                else:
                    self.edge_encoders.append(None)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

        # RNN layers aggregate node embeddings across snapshots.
        # NOTE: `sequence` is shaped [T, N, H] = [time, nodes, features]. With
        # batch_first=False the GRU treats dim 0 as the SEQUENCE (time) axis and
        # dim 1 as the batch (nodes) — i.e. recurrence runs over time, per node,
        # which is the whole point of a temporal GNN. (batch_first=True made it
        # recur over nodes with time as the batch — a silent correctness bug.)
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=out_channels,
            num_layers=num_rnn_layers,
            batch_first=False,
            dropout=dropout if num_rnn_layers > 1 else 0.0,
        )

        # Final classifiers.
        self.node_classifier = nn.Linear(out_channels, 1)
        self.snapshot_classifier = nn.Linear(out_channels, 1)
        # Edge (per-flow) head: concat of the two endpoint embeddings plus edge
        # features. Edge attrs (duration, packets, bytes, proto) are directly
        # predictive for per-flow botnet classification and were previously
        # only used via additive aggregation in the GNN, which discards them.
        if edge_dim and edge_dim > 0:
            self.edge_feat_encoder = nn.Sequential(
                nn.Linear(edge_dim, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.edge_classifier = nn.Linear(3 * out_channels, 1)
        else:
            self.edge_feat_encoder = None
            self.edge_classifier = nn.Linear(2 * out_channels, 1)

        self.dropout_layer = nn.Dropout(dropout)

    def _encode_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the GNN stack on a single snapshot."""
        h = x

        num_nodes = x.size(0)
        for i, (conv, edge_enc, ln) in enumerate(
            zip(self.convs, self.edge_encoders, self.layer_norms)
        ):
            if self.gnn_aggregation == "gat":
                h = conv(h, edge_index, edge_attr=edge_attr)  # [N, hidden]
            else:
                h = conv(h, edge_index)  # [N, hidden]
                if (
                    edge_attr is not None
                    and edge_enc is not None
                    and edge_index.numel() > 0
                ):
                    edge_msg = torch.relu(edge_enc(edge_attr))  # [E, hidden]
                    agg = torch.zeros_like(h)
                    counts = torch.zeros(num_nodes, 1, device=h.device)
                    ones = torch.ones(edge_index.size(1), 1, device=h.device)
                    for endpoint in (edge_index[0], edge_index[1]):
                        agg.index_add_(0, endpoint, edge_msg)
                        counts.index_add_(0, endpoint, ones)
                    h = h + agg / counts.clamp(min=1.0)

            h = torch.relu(h)
            h = ln(h)
            h = self.dropout_layer(h)
        return h


    def forward(
        self,
        snapshots: list[dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass over a sequence of snapshots.

        Args:
            snapshots: list of snapshot dicts with keys
                x, edge_index, edge_attr, node_labels, snapshot_label.

        Returns:
            node_logits: [N, 1] logits for nodes in the last snapshot
            snapshot_logit: [1] logit for the last snapshot
            edge_logits: [E, 1] logits for edges (flows) in the last snapshot
        """
        if not snapshots:
            raise ValueError("Empty snapshot sequence")

        # Keep every tensor on the same device as the model weights.
        device = next(self.parameters()).device

        # GNN over each snapshot -> list of [N_t, hidden] tensors.
        snapshot_embeddings = []
        snapshot_node_ids: list[list[str]] = []
        for snap in snapshots:
            x = torch.from_numpy(snap["x"]).float().to(device)
            edge_index = torch.from_numpy(snap["edge_index"]).long().to(device)
            edge_attr = (
                torch.from_numpy(snap["edge_attr"]).float().to(device)
                if snap.get("edge_attr") is not None and snap["edge_attr"].size > 0
                else None
            )
            h = self._encode_snapshot(x, edge_index, edge_attr)
            snapshot_embeddings.append(h)
            snapshot_node_ids.append(snap["node_ids"])

        # Track only last-snapshot nodes through the GRU: each node's
        # embedding at timestep t is its GNN output if it appears in that
        # snapshot, else zeros.  The GRU naturally carries forward the last
        # seen embedding across gaps.  Tracking the full union of all nodes
        # ever seen (commented out below) was tried but hurt: it spends
        # capacity on nodes we never classify and adds noise from brief
        # one-snapshot appearances that don't carry useful temporal signal.
        last_snap = snapshots[-1]
        last_node_ids = last_snap["node_ids"]
        T = len(snapshot_embeddings)
        N = len(last_node_ids)
        node_id_to_idx = {nid: i for i, nid in enumerate(last_node_ids)}
        sequence = torch.zeros(T, N, self.hidden_channels, device=device)

        for t, (snap_nids, emb) in enumerate(zip(snapshot_node_ids, snapshot_embeddings)):
            src = torch.as_tensor(
                [local_idx for local_idx, nid in enumerate(snap_nids) if nid in node_id_to_idx],
                dtype=torch.long, device=device,
            )
            if src.numel() == 0:
                continue
            dst = torch.as_tensor(
                [node_id_to_idx[snap_nids[i.item()]] for i in src],
                dtype=torch.long, device=device,
            )
            sequence[t].index_copy_(0, dst, emb.index_select(0, src))

        # GRU over time for each node: recurrence over the time axis, carrying
        # each node's last-seen embedding forward through gaps.
        rnn_out, _ = self.gru(sequence)  # [T, N, out_channels]
        final_node_emb = rnn_out[-1]  # last timestep per node

        node_logits = self.node_classifier(final_node_emb)  # [N, 1]
        snapshot_emb = final_node_emb.max(dim=0, keepdim=True).values
        snapshot_logit = self.snapshot_classifier(snapshot_emb)  # [1, 1]

        # Edge (per-flow) logits for the last snapshot — edge_index is already
        # aligned with last_node_ids which is our global index, so no remap.
        last_snap = snapshots[-1]
        last_node_ids = last_snap["node_ids"]
        last_edge_index = torch.from_numpy(last_snap["edge_index"]).long().to(device)
        last_edge_attr = (
            torch.from_numpy(last_snap["edge_attr"]).float().to(device)
            if last_snap.get("edge_attr") is not None and last_snap["edge_attr"].size > 0
            else None
        )
        if last_edge_index.numel() > 0:
            src_emb = final_node_emb.index_select(0, last_edge_index[0])
            dst_emb = final_node_emb.index_select(0, last_edge_index[1])
            if self.edge_feat_encoder is not None and last_edge_attr is not None:
                edge_feat = self.edge_feat_encoder(last_edge_attr)
                edge_logits = self.edge_classifier(torch.cat([src_emb, dst_emb, edge_feat], dim=1))
            else:
                edge_logits = self.edge_classifier(torch.cat([src_emb, dst_emb], dim=1))
        else:
            edge_logits = torch.zeros((0, 1), device=device)

        return node_logits, snapshot_logit.squeeze(-1), edge_logits

