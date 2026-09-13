"""Reservoir rehearsal buffer for the paper-faithful Prometheus path."""

from __future__ import annotations

import random
from typing import Iterable

import torch
from torch_geometric.data import Batch, Data


class RehearsalBuffer:
    """Fixed-capacity uniform reservoir (Algorithm R) over graph samples.

    Capacity is configured by the trainer to exactly 10% of the unique
    training graph samples, matching the paper's reported rehearsal storage.
    Each graph is inserted only once after its scenario has finished training,
    so 200 epochs do not create 200 duplicate copies of the same sample.
    """

    def __init__(self, max_size: int, seed: int = 0) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = int(max_size)
        self.buffer: list[Data] = []
        self.total_seen = 0
        self._rng = random.Random(seed)

    def add(self, data: Data) -> None:
        # Keep replay storage on CPU even when the model trains on GPU.
        data = data.cpu()
        self.total_seen += 1
        if len(self.buffer) < self.max_size:
            self.buffer.append(data)
            return
        j = self._rng.randrange(self.total_seen)
        if j < self.max_size:
            self.buffer[j] = data

    def add_many(self, items: Iterable[Data]) -> None:
        for item in items:
            self.add(item)

    def sample(self, n: int) -> list[Data]:
        if not self.buffer:
            return []
        n = min(int(n), len(self.buffer))
        return self._rng.sample(self.buffer, n)

    def sample_batch(self, n: int, device: torch.device) -> Batch | None:
        items = self.sample(n)
        if not items:
            return None
        return Batch.from_data_list(items).to(device)

    def __len__(self) -> int:
        return len(self.buffer)

    def __bool__(self) -> bool:
        return bool(self.buffer)
