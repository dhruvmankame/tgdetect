# Prometheus — Base Paper Alignment

This repository path is aligned to the architecture and training choices that the base paper **explicitly reports**.

## Exact-to-reported-spec changes

- Node-wise classification instead of graph-level pooling/classification.
- 3 attention GNN layers, hidden dimension 128.
- ReLU and dropout 0.2.
- No global mean pooling.
- No LayerNorm.
- No edge attributes inside Prometheus message passing.
- Linear 128 -> 2 node classifier with softmax probabilities.
- Cross-entropy classification loss.
- Adam, lr=1e-3, weight decay=1e-5.
- Batch size 256.
- 200 epochs per scenario/task.
- Continual scenario-by-scenario training.
- Rehearsal capacity 10% of unique training graph samples.
- Reservoir sampling.
- Five independent seed runs with mean +/- standard deviation reporting.
- No early stopping, ReduceLROnPlateau, gradient clipping, threshold tuning, or positive-class weighting in the paper-faithful path.

## What the paper does not specify

A literal source-code-identical reproduction is impossible from the publication alone because it does **not** report:

1. the exact attention operator (GAT/GATv2/custom),
2. the number of attention heads,
3. the numerical value of lambda in `L_new + lambda * L_rehearsal`,
4. full node-feature encoding/preprocessing details,
5. the exact five random seed values.

This patch therefore uses the minimum-assumption realization: standard single-head PyG `GATConv`, and records `lambda_rehearsal` in every run (default 1.0).

## Data caveat

The base paper uses Wazuh endpoint telemetry plus sandbox-executed APT traces. Using CTU-13 or another dataset can test the same model architecture, but it is **not an exact reproduction of the paper's dataset/experiment**. Matching the experiment also requires equivalent Wazuh-derived graphs and node features.

## Cloud-only training

Use the Modal entrypoint added by the patch. It runs model training on a cloud GPU; the local machine only submits code and receives artifacts.
