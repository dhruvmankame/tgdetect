"""Rehearsal buffer with reservoir sampling for Prometheus continual learning.

Paper: 10% reservoir sampling — each incoming sample has probability p =
buffer_size / total_seen of being added; if added, it evicts a uniformly
random existing sample.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from torch_geometric.data import Data


class RehearsalBuffer:
    """Fixed-size reservoir buffer for graph-level Data objects."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.buffer: list[Data] = []
        self.total_seen = 0

    def add(self, data: Data) -> None:
        """Add a single graph to the buffer via reservoir sampling."""
        self.total_seen += 1
        if len(self.buffer) < self.max_size:
            self.buffer.append(data)
        else:
            # Replace a random entry with probability max_size / total_seen
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.max_size:
                self.buffer[idx] = data

    def sample(self, n: int, device: torch.device) -> list[Data]:
        """Sample n items uniformly from the buffer (without replacement)."""
        if not self.buffer:
            return []
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)

    def __len__(self) -> int:
        return len(self.buffer)

    def __bool__(self) -> bool:
        return len(self.buffer) > 0


def buffer_loss(
    model: torch.nn.Module,
    buffer: RehearsalBuffer,
    criterion: torch.nn.Module,
    device: torch.device,
    batch_size: int = 256,
    lambda_rehearsal: float = 1.0,
) -> Optional[torch.Tensor]:
    """Compute rehearsal loss: L_rehearsal(D_buf).

    Returns None if buffer is empty.
    """
    if not buffer:
        return None
    items = buffer.sample(batch_size, device)
    if not items:
        return None

    from torch_geometric.data import Batch as PyGBatch
    batch = PyGBatch.from_data_list(items).to(device)
    logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
    loss = criterion(logits.squeeze(-1), batch.y)
    return lambda_rehearsal * loss
