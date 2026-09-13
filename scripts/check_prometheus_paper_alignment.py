#!/usr/bin/env python3
"""Tiny architecture-only check. No dataset and no training required."""
import torch
from torch_geometric.nn import GATConv
from models.prometheus import Prometheus

m = Prometheus(in_channels=8)
assert len(m.convs) == 3
assert all(isinstance(c, GATConv) for c in m.convs)
assert all(c.heads == 1 for c in m.convs)
assert m.hidden_channels == 128
assert abs(m.dropout_p - 0.2) < 1e-12
assert m.classifier.in_features == 128
assert m.classifier.out_features == 2
assert not hasattr(m, "layer_norms")

x = torch.randn(7, 8)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=torch.long)
m.eval()
logits = m(x, edge_index)
probs = m.predict_proba(x, edge_index)
assert logits.shape == (7, 2)
assert torch.allclose(probs.sum(dim=1), torch.ones(7), atol=1e-6)
print("PASS: Prometheus structural checks match the paper-explicit model specification.")
