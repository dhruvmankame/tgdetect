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

