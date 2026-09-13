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

