# TG-Detect: Anticipated Panel Questions and Answers

**Date:** 2026-09-06
**Context:** Held-out-family botnet detection on CTU-13 using a GraphSAGE+GRU temporal GNN. Winning result: AUC-PR=0.9998, Recall@1%FPR=1.0000, best-F1=0.9944 on unseen donbot (c47).

---

## 1. Threshold Selection

### Q1: "Your best-F1 is 0.9944, but you say the deployable F1 is only 0.636. Which number should we believe?"

The honest answer is **both**, for different purposes:

- **0.9944 (oracle best-F1):** This is the maximum F1 achievable by any threshold, found by sweeping the PR curve on test labels. It tells you the model's *ceiling* — how well the scores separate the two classes. It is useful for comparing against other papers that report oracle-tuned F1.
- **0.636 (1%-FPR budget):** This is the F1 when you set the threshold to allow exactly 1% false positives, computed **without peeking at test labels** — the threshold is set on the benign score distribution only. This is the *deployable* number because in production you have no labels to tune on.
- **0.9095 (saved threshold):** This is the F1 using the threshold (0.015) tuned on the in-distribution validation set. It is the most realistic number *if* the test distribution matches the validation distribution — which it doesn't here, because donbot is an unseen family.

**What to say to the panel:** "The 0.9944 is the model's proven ceiling. The 0.636 is what you get in practice with a fixed false-positive budget. The gap between them is the calibration problem we haven't solved yet."

### Q2: "Why is your saved threshold 0.015 but the oracle-optimal threshold on test is 0.816? That's a huge gap."

This is the **threshold-transfer artifact**. The threshold is tuned on the in-distribution validation set (same families as training). On that set, the model is well-calibrated and separates classes at a low threshold. But on the shifted donbot distribution, the score distribution is different — the model is less confident on average, so a threshold that worked on val produces a flood of false positives on donbot.

**Why it happens:** The validation set comes from the same families as training (Rbot, fast-flux, NSIS.ay, sogou, qvod). Donbot's C2 communication pattern produces different edge-level scores. The threshold tuned on val families doesn't transfer.

**Solutions we haven't implemented yet:** temperature scaling, isotonic regression on a small labeled test sample, or re-tuning the threshold per-deployment.

### Q3: "How do you set the threshold in deployment where you have no labels at all?"

You have three options, all with trade-offs:

1. **Use the saved val-tuned threshold** — works if test distribution matches val (in-distribution). Doesn't work for held-out-family. This is what we currently do; it gives F1=0.9095 on donbot by luck (the threshold happens to still be somewhat usable), but that's not a guarantee for other unseen families.
2. **Use a fixed-FPR budget** — set the threshold on the benign score distribution from the validation set to allow a fixed false-positive rate (e.g., 1%). This is the most principled deployable policy. It gave F1=0.636 on donbot.
3. **Calibrate on a small labeled sample** — collect a few hundred labeled examples from the target deployment and tune the threshold on those. Most realistic but requires labeling effort.

### Q4: "You report Recall@1%FPR = 1.0000. But your F1 at that operating point is only 0.636. How can recall be perfect but F1 be low?"

Because precision is 0.465 at that operating point. At a 1% false-positive budget on ~1M test edges, 1% of ~990K benign edges = ~9,900 false positives. With ~9,256 true positives, precision = 9256 / (9256 + 9900) = 0.465. F1 = 2 * (0.465 * 1.0) / (0.465 + 1.0) = 0.636. This is the fundamental trade-off of extreme imbalance: even a tiny FPR in absolute terms produces many false positives when the negative class is huge.

---

## 2. Features

### Q5: "You use only a single all-ones feature for CTU-13 nodes. Isn't that absurdly weak? How can the model work with no node features?"

It's not as weak as it sounds. The single all-ones column serves as a **learnable bias per node position** in the GNN — every node gets the same input, so the GNN's job is purely structural: what matters is *who connects to whom, how often, and with what edge features*. The temporal GRU then tracks how each node's structural context evolves.

For CTU-13 this makes sense because:
- Every node is an IP address. There's no natural node feature (unlike Wazuh where nodes are processes with command lines, file paths, etc.).
- The *edge* features (flow duration, packet counts, byte counts, protocol, TCP state flags) carry the predictive signal — these are what distinguish botnet C2 traffic from benign browsing.
- Adding label-derived node features (degree, mal_ratio) creates **target leakage** — those features encode the answer directly.

### Q6: "Why not use richer node features like degree, PageRank, or mal_ratio?"

Because they leak the label:
- **Degree** = how many connections an IP has. In a snapshot containing both benign and malicious flows, a botnet node's degree is elevated *because it's malicious*. This is correlation, not causation.
- **mal_ratio** = the fraction of a node's edges that are malicious. This is literally the target encoded as a feature. A model with this feature will report near-perfect metrics that collapse on any real deployment.

We tried `type_degree` mode (which includes mal_ratio) and it produced inflated metrics that didn't survive the held-out-family test. `type_only` (single all-ones column) is the leakage-free choice.

### Q7: "You deliberately excluded raw port numbers. Why? Don't botnet C2 ports carry signal?"

Yes, they do — in a single scenario. A botnet might always use port 6667 for IRC-based C2. But if you include raw ports as features, the model memorizes "port 6667 = botnet" for one scenario and it **never transfers** to a different botnet that uses a different port. This is a classic overfitting trap.

Instead, we use **port service buckets** (10 binary features): sport<1024, sport>=49152, dport<1024, dport>=49152, dns, http, https, smtp, irc, other. These generalize: "uses IRC port" transfers across IRC-based botnets; "destination port <1024" captures server-like behavior regardless of which specific port.

### Q8: "Walk me through the 35 edge features."

| Group | Count | Features |
|---|---|---|
| Numeric (log1p) | 9 | duration, total bytes, src bytes, dst bytes, total packets, bytes/packet, bytes/sec, packets/sec, src_byte_ratio |
| Protocol one-hot | 4 | TCP, UDP, ICMP, other |
| Direction one-hot | 4 | ->, <->, <-, other |
| TCP state flags | 8 | CON, INT, URP, S, F, R, A, P |
| Port service buckets | 10 | sport<1024, sport>=49152, dport<1024, dport>=49152, dns, http, https, smtp, irc, other |

The port bucket features are computed from raw ports at build time but only the bucket indicators are stored — raw ports are discarded. This prevents the model from memorizing specific port numbers.

### Q9: "Why log1p on the numeric features?"

Flow durations and byte counts span many orders of magnitude (microseconds to hours, bytes to gigabytes). Without log transform, the model spends capacity on scale differences rather than discrimination. `log1p(x) = log(1+x)` handles zero-valued features gracefully.

---

## 3. Architecture

### Q10: "Why GraphSAGE and not GAT or GCN? The paper uses attention."

GraphSAGE was the default when we started and it worked well enough that switching wasn't a priority. We tried GAT (4 heads, hidden//4 per head, dropout 0.2 — paper-aligned) as an alternative but it didn't outperform SAGE on this task. GAT's attention mechanism is designed for heterogeneous neighbor importance weighting, but in CTU-13's near-homogeneous graph (all IPs, mostly similar connection patterns), the added complexity doesn't pay off.

**When GAT would matter:** If you had a graph with clearly different node types (processes, files, IPs, users) where some neighbor types are much more informative than others, attention would help. CTU-13 is structurally simpler.

### Q11: "Why GRU and not LSTM or Transformer?"

- **GRU vs LSTM:** GRU has fewer parameters (no separate cell state) and empirically performs similarly on this task. The temporal sequences in CTU-13 are short (typically 5-20 snapshots per sequence), so LSTM's gating advantage on long sequences doesn't materialize.
- **GRU vs Transformer:** A temporal Transformer would require positional encoding and self-attention across timesteps. For short sequences with a clear "most recent matters most" structure, the GRU's inductive bias (recency through hidden state) is a better fit. We tried temporal attention (learnable weighted combination of all GRU outputs) and it hurt — it overfit to training-family temporal patterns.

### Q12: "You only track last-snapshot nodes through the GRU. Why not all nodes ever seen?"

We tried it (v6 experiment). Tracking every node ever seen through the GRU:
- Spends capacity on nodes that appear once and are never classified
- Adds noise from brief one-snapshot appearances that don't carry useful temporal signal
- Result: F1 dropped from 0.839 to 0.761

The last-snapshot-only approach is cleaner: the GRU only tracks nodes we actually classify, and the GRU's hidden state naturally carries forward the last-seen embedding across gaps.

### Q13: "Your edge classifier concatenates src_emb + dst_emb + edge_feat. Why not just use the edge features alone?"

Because the endpoint embeddings carry the **temporal context** of each IP — how that IP has behaved across all previous snapshots. The edge features alone (duration, bytes, protocol) describe a single flow. The endpoint embeddings describe the *history* of the source and destination IPs. Concatenating both gives the classifier: "this flow looks like C2 (edge features) AND these IPs have been behaving like botnet nodes (temporal embeddings)."

### Q14: "Why 2 GNN layers and not 3 like the paper?"

2 layers was sufficient for this graph structure. Each SAGEConv layer aggregates 1-hop neighbor information, so 2 layers gives 2-hop receptive field. In CTU-13's flow graph, most of the discriminative signal is local (the flow itself and its immediate endpoints). A 3rd layer would aggregate information from nodes 3 hops away, which adds noise without adding signal for this task.

---

## 4. Evaluation

### Q15: "Why held-out-family and not random split?"

A random split (or chronological split within the same families) would let the model memorize family-specific artifacts — timing patterns, port usage, connection cadence — that don't generalize. Held-out-family is the only protocol that tests whether the model has learned *botnet behavior* vs *this specific botnet's behavior*.

The base paper uses a random graph split on Wazuh data. That's a much easier test. Our held-out-family protocol is closer to the real deployment scenario: you train on known botnets and need to detect a new one.

### Q16: "Why AUC-PR and not just AUC-ROC?"

AUC-ROC is misleading under extreme imbalance. At 0.87% positive rate, a model that predicts everything negative gets AUC-ROC = 0.5 (random), but a model with high recall and moderate FPR can get AUC-ROC = 0.998 while still producing thousands of false positives. AUC-PR penalizes false positives directly because precision is in the numerator. Our AUC-PR of 0.9998 means the model ranks almost every positive above almost every negative — that's the honest measure.

### Q17: "Why Recall@1%FPR as a metric?"

Because in deployment, the false-positive budget is the binding constraint. A SOC analyst can investigate maybe 100 alerts per day. If your system generates 10,000 alerts, it doesn't matter how good your recall is — the system is unusable. Recall@1%FPR tells you: "given a fixed alert budget, how many real attacks do you catch?" It's the metric that maps to operational reality.

### Q18: "You only tested on one held-out family (donbot). How do you know it generalizes?"

We don't — not fully. Donbot is a single data point. The right thing to do is rotate the held-out family (Neris, Rbot, Murlo, NSIS.ay, sogou) and report a family-averaged number. This is the most important next step. Donbot was chosen because it's a single-scenario family (c51 has no data), making the split clean.

### Q19: "What about the Garcia split? Why haven't you done that?"

The Garcia split uses all 13 CTU-13 scenarios including the large ones (c42, c43, c44, c49, c50 at 2-4.7M flows each). These are OOM-prone on a single GPU. We deferred the full Garcia split until the smaller-scenario pipeline was green and reproducible. The current 7-scenario subset is a defensible intermediate result.

---

## 5. Comparison with Base Paper

### Q20: "You claim to beat the base paper's F1 of 0.981 with your 0.9944. Is that a fair comparison?"

**No, and we don't claim it is.** The comparison is scientifically invalid for several reasons:

| Dimension | Base paper | TG-Detect |
|---|---|---|
| Data | Wazuh EDR host telemetry | CTU-13 Argus NetFlow |
| Task unit | provenance graph | per-flow edge |
| Split | in-distribution (random graph) | held-out family (donbot unseen) |
| Imbalance | not reported | 0.87% positive |
| Continual learning | rehearsal, 10% reservoir | none |
| Model | 3-layer attention GNN, 128-wide | 2-layer GraphSAGE + GRU, 128-wide |

The honest statement is: TG-Detect demonstrates strong unseen-family generalization with a simpler model and no continual learning. The 0.9944 is not a like-for-like comparison with 0.981.

### Q21: "If you can't compare directly, what's the scientific contribution?"

The contribution is **held-out-family generalization on extreme imbalance with a leakage-free feature set**. The base paper doesn't report:
- Whether their split is family-disjoint
- What their event-level or node-level class ratio is
- Whether their features include label-derived information
- Their false-positive rate or alert volume

TG-Detect's result is stronger *because* it's on a harder protocol, not because the model is better.

### Q22: "The base paper has continual learning. You don't. Isn't that a dealbreaker?"

It's a gap, not a dealbreaker. The current result is a **static train** — one training run, no updates. Continual learning (rehearsal replay) is needed for the scenario-stream setting where new botnet families arrive over time. It's Phase 5 of the roadmap. The static result is the baseline that continual learning must improve upon.

---

## 6. What Was Tried and Failed

### Q23: "You tried focal loss and it gave F1=0.0174. What went wrong?"

Focal loss is designed to emphasize hard-to-classify positives by down-weighting easy negatives. At 0.87% positive rate, "easy negatives" is 99.13% of the data. Focal loss down-weights them so aggressively that the model either:
- Collapses to predicting everything positive (recall=1, precision=0.0087, F1≈0.017)
- Gets stuck in a degenerate loss landscape where the gamma=2.0 exponent amplifies noise

Standard BCE with pos_weight (neg/pos ratio, capped at 50) handles the imbalance without the collapse. The cap prevents the loss from being dominated by the extreme ratio.

### Q24: "Temporal attention hurt. Why would that be?"

Temporal attention learns a weighted combination of all timestep embeddings. On the training families, this lets the model focus on the timesteps that are most discriminative for those specific botnets' C2 patterns. But donbot's C2 cadence is different — the attention weights learned on training families attend to the wrong timesteps for donbot. The simpler "use the last GRU output" is more robust because it doesn't make assumptions about which timesteps matter.

### Q25: "Union-node GRU tracking hurt. Why?"

Tracking every node ever seen means the GRU has to maintain state for thousands of nodes that appear once and are never classified. This dilutes the capacity available for the nodes we actually care about. It also adds noise: a node that appears in one snapshot for 5 seconds gets the same GRU treatment as a node that's been active for the full capture.

---

## 7. Training Details

### Q26: "Why early stopping on auc_pr and not F1?"

Because F1 at a tuned threshold can hit 0.99 on val while the model generalizes poorly. AUC-PR is threshold-free — it measures how well the model ranks positives above negatives regardless of threshold. A model with high val-F1 but low val-auc_pr is overfitting to the threshold; a model with high val-auc_pr is genuinely learning to separate classes.

### Q27: "pos_weight=50. How was that chosen?"

`pos_weight = min(neg_count / pos_count, cap)` where cap defaults to 50. For CTU-13 at ~0.87% positive, the raw ratio is ~115:1. Capping at 50 prevents the loss from being so dominated by the positive class that the model collapses (similar to the focal loss problem). The cap is a hyperparameter; 50 worked well.

### Q28: "You stopped at epoch 9. Was that enough?"

Early stopping patience=5 means we stopped after 5 consecutive epochs with no auc_pr improvement. Val AUC-PR peaked at epoch 4 (0.9999) and declined to 0.9855 by epoch 8 — a clear overfitting signal. 9 epochs on an A10 GPU took ~10 minutes. The model converges fast because the task is structurally simple (flow-level features are highly predictive).

### Q29: "What about the learning rate schedule?"

AdamW with lr=0.001, weight_decay=1e-4, ReduceLROnPlateau(patience=3, factor=0.5). The scheduler reduces LR when val loss plateaus, which helps fine-tune in later epochs. Gradient clipping at 1.0 prevents exploding gradients from the high pos_weight.

---

## 8. Deployment

### Q30: "How would you deploy this in a real network?"

1. **Ingest:** Read NetFlow/sFlow/IPFIX from network taps or routers
2. **Build snapshots:** 60-second windows with 30-second stride (same as training)
3. **Run inference:** Forward pass through TemporalGNN, get edge logits
4. **Apply threshold:** Use the 1%-FPR budget threshold (set on benign scores)
5. **Alert:** Flag edges above threshold as suspicious flows
6. **Aggregate:** Group flagged edges by source IP to identify likely botnet nodes

**What's missing for production:**
- Real-time inference pipeline (currently batch-only)
- Threshold calibration per deployment
- Continual learning for new botnet families
- Explainability (which flows triggered the alert and why)

### Q31: "What's the inference latency?"

Not measured yet. The model is small (83K parameters) and processes one sequence at a time. On GPU, a single forward pass is milliseconds. The bottleneck is graph construction (parsing flows, building snapshots), not inference.

### Q32: "What about concept drift? Botnets change their C2 patterns."

This is the biggest unsolved problem. A model trained on today's botnets may not detect tomorrow's. The base paper's rehearsal continual learning is the proposed solution: maintain a buffer of past samples and replay them during updates to prevent forgetting. We haven't implemented this yet.

---

## 9. Limitations and Weaknesses

### Q33: "What are the biggest limitations of this work?"

1. **Single held-out family** — donbot only. Need family-averaged results across all 7 families.
2. **Threshold-transfer unsolved** — the gap between oracle-F1 (0.9944) and deployable-F1 (0.636) is large. Calibration is needed.
3. **No continual learning** — static model, no adaptation to new families over time.
4. **No latency/throughput measurement** — no production inference benchmark.
5. **CTU-13 is old** — captured in 2011. Modern botnets use encrypted C2, domain fronting, and other evasion that may not be present in this dataset.
6. **No explainability** — the model says "this flow is malicious" but not why.
7. **No comparison with non-GNN baselines** — we haven't shown that a GNN is necessary vs a simpler model (logistic regression on edge features, LightGBM, etc.).

### Q34: "Is CTU-13 too easy? The flows are unencrypted and the botnets are old."

It's a fair concern. CTU-13's botnets (Neris, Rbot, donbot) use plaintext IRC and HTTP C2, which is easier to detect than modern encrypted C2. However:
- The held-out-family protocol is still hard: donbot's C2 pattern is genuinely different from the training families
- The extreme imbalance (0.87% positive) makes the task non-trivial even with "easy" features
- CTU-13 is a standard benchmark in the botnet detection literature, enabling comparison with prior work

The right next step is to test on more modern data (e.g., CICIoT2023, N-BaITS) if available.

### Q35: "You have no ablation study. How do we know which components matter?"

We have informal ablations from the experiments:
- Edge features matter enormously (the v4 baseline with edge features concatenated outperformed the version that only used edge features via GNN aggregation)
- Training family diversity matters (4→5 families: AUC-PR 0.707→0.9998)
- Temporal attention and union-node tracking both hurt (removed)
- Focal loss hurt catastrophically (removed)

What's missing is a systematic ablation: fix everything else and vary one component at a time, report the delta. This should be done for the next paper/report.

---

## 10. Broader Questions

### Q36: "Why should we trust these results? It's a single run on a single dataset."

You shouldn't — not yet. The results need:
- **Multiple seeds:** Run with 5+ random seeds, report mean ± std
- **Multiple held-out families:** Rotate through all 7 families
- **Confidence intervals:** Bootstrap on the test set
- **Non-GNN baselines:** Show that the GNN adds value over simpler models

The current result is a **proof of concept**, not a definitive claim.

### Q37: "What's the novelty here? GNNs for botnet detection already exist."

The novelty is in the **evaluation protocol**, not the model:
- Held-out-family split (not random or chronological)
- Leakage-free feature set (no label-derived node features)
- Three operating points (not just one tuned F1)
- Explicit reporting of the threshold-transfer artifact

Most prior work reports a single F1 on a random split with potentially leaky features. We're trying to be more honest.

### Q38: "How does this compare to commercial NTA solutions?"

We haven't compared. Commercial solutions (Darktrace, Vectra, ExtraHop) use ensembles of deep learning, behavioral analysis, and threat intelligence. They likely outperform our single-model prototype on raw detection. But they're black boxes; our model is interpretable (you can trace which flows contributed to the alert).

### Q39: "What would you do differently if you started over?"

1. **Start with baselines:** Logistic regression on edge features, LightGBM, and a simple MLP. Know what the GNN is adding before building it.
2. **Design the split first:** Define held-out-family before running any experiments. Don't let the data dictate the protocol.
3. **Report calibration from day one:** Always report expected calibration error and Brier score alongside F1.
4. **Run multiple seeds from the start:** Never report a single-run number.
5. **Measure latency early:** If the model can't run in real-time, the detection quality doesn't matter.

### Q40: "What's the next concrete experiment?"

**Rotate the held-out family.** Train on all families except Neris, test on Neris. Then except Rbot, test on Rbot. Report the family-averaged AUC-PR and Recall@1%FPR. This gives a single number that's more defensible than the donbot-only result.

---

## Quick Reference: Numbers to Memorize

| Metric | Value | Operating Point |
|---|---|---|
| AUC-PR | 0.9998 | Threshold-free |
| AUC-ROC | 1.0000 | Threshold-free |
| Recall@1%FPR | 1.0000 | 1% FPR budget |
| Best-F1 | 0.9944 (P=0.9979, R=0.9909) | Oracle (PR sweep) |
| F1 @ saved threshold | 0.9095 | Val-tuned threshold (0.015) |
| F1 @ 1%FPR | 0.6360 | 1% FPR budget |
| Test size | 1,068,851 edges, 9,256 positive | — |
| Training families | 5 (Rbot, fast-flux, NSIS.ay, sogou, qvod) | — |
| Held-out family | donbot (c47) | — |
| Model params | 83,651 | — |
| Training time | ~10 min (9 epochs, A10 GPU) | — |
