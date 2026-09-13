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

