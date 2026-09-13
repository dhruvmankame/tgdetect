# Prometheus CTU-13 — Base-Paper Experimental Suite

This suite deliberately keeps **CTU-13** as the project dataset. It does not add
Wazuh ingestion and it does not claim that CTU-13 results reproduce the paper's
Wazuh/sandbox dataset results.

## What this patch adds

The core Prometheus architecture/training path was already aligned to every
explicitly reported model parameter in the paper. The missing paper-side work
was primarily experimental evidence. This patch adds:

1. **Static GNN baseline** using the same Prometheus GAT but joint/static training.
2. **Fine-tuning baseline** using sequential tasks with no replay.
3. **CNN-BiLSTM + attention baseline** with two Conv1d layers and a 128-wide
   bidirectional LSTM representation.
4. **Sequential baseline + rehearsal** for the paper's ablation comparison.
5. **Generative replay proxy** using a conditional VAE + node MLP. The paper does
   not disclose the exact generative baseline architecture, so this is explicitly
   a reproducible project proxy rather than an alleged source-identical copy.
6. **After-each-task continual-learning evaluation**, including per-scenario F1,
   average forgetting, and backward transfer.
7. **CTU-13 scenario/family breakdown** instead of inventing MITRE tactic labels.
8. **Latency / memory benchmark** with an explicit timing protocol and GPU name.
9. **Attention-backed malicious-subgraph evidence** for high-risk nodes.
10. **Automatic Markdown comparison report** across the paper-style experiments.

## Critical CTU-13 feature requirement

Paper-style Prometheus ignores edge attributes. The ordinary CTU-13
`type_only` snapshots therefore give every IP node the same input feature and
can make node-wise GAT degenerate. For Prometheus experiments, build and use the
repository's leakage-free **`flow_agg` node features**:

```bash
modal run modal_train.py::snapshots_ctu13_flowagg \
  --names ctu13_c52,ctu13_c46,ctu13_c53,ctu13_c48,ctu13_c45,ctu13_c49,ctu13_c54,ctu13_c47
```

That creates directories such as `ctu13_c52_flowagg` and
`ctu13_c47_flowagg`. Use those names in the experiment commands below.

## Run one experiment on Modal A10G

```bash
modal run --detach modal_train.py::prometheus_experiment \
  --experiment finetune \
  --train-scenarios ctu13_c52_flowagg,ctu13_c46_flowagg,ctu13_c53_flowagg,ctu13_c48_flowagg,ctu13_c45_flowagg,ctu13_c49_flowagg,ctu13_c54_flowagg \
  --test-scenarios ctu13_c47_flowagg \
  --out-name paper_suite/finetune
```

Valid experiment names:

```text
full_rehearsal
finetune
static_gnn
cnn_bilstm
cnn_bilstm_rehearsal
generative_replay
```

If you already completed a correct five-seed `full_rehearsal` run, you do not
need to retrain it merely to add the other baselines.

## Benchmark the trained Prometheus checkpoint

```bash
modal run modal_train.py::benchmark_prometheus_extra \
  --checkpoint-name prometheus_paper \
  --scenarios ctu13_c47_flowagg
```

## Export attention-backed subgraph evidence

```bash
modal run modal_train.py::explain_prometheus_extra \
  --checkpoint-name prometheus_paper \
  --scenario ctu13_c47_flowagg
```

## Generate the comparison report

Store experiment outputs under a common root such as:

```text
/checkpoints/paper_suite/full_rehearsal/
/checkpoints/paper_suite/finetune/
/checkpoints/paper_suite/static_gnn/
/checkpoints/paper_suite/cnn_bilstm/
/checkpoints/paper_suite/cnn_bilstm_rehearsal/
/checkpoints/paper_suite/generative_replay/
```

Then run:

```bash
modal run modal_train.py::report_prometheus_extra --root-name paper_suite
```

The generated report is saved under `/checkpoints/paper_suite/`.

## What remains impossible to make source-identical

The publication still does not disclose the exact attention operator/head
count, lambda value, five seed values, or implementation of the generative
replay baseline. Those choices must remain documented project assumptions.
Using CTU-13 also means tactic-wise Wazuh results cannot be reproduced without
fabricating labels, which this suite intentionally avoids.

