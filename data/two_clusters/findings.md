# Decoding the Two Clusters: Findings

Five experiments on binary computational states in LLM hidden states, conducted on Qwen2.5-1.5B-Instruct processing 500 MATH-500 problems, with cross-validation on Phi-2, Pythia-1.4B, StableLM-1.6B, and non-math tasks.

---

## 1. What are the two clusters?

**Answer: The two clusters partially separate numeric/mathematical tokens from natural language tokens, but this is a weak effect layered on top of a universal architectural structure.**

K-means (k=2) on the final-layer token hidden states yields a mean silhouette of 0.242 across all 500 problems. The contingency analysis reveals a statistically significant but small association between token type and cluster membership (chi-squared=745, p<10^-161, Cramer's V=0.130):

- **Cluster A** is enriched for numeric tokens (64.5% of numerics go to A)
- **Cluster B** is enriched for language tokens (52.6% of language tokens go to B)

However, this effect is weak. The two-cluster structure is **not math-specific**: Wikipedia text wrapped in the same prompt template shows *higher* silhouette (0.311 vs 0.242). The structure also appears across non-math tasks (GSM8K: 0.192, TriviaQA: 0.173). This indicates the clusters reflect a general property of the model's token representation, not math-specific computation.

Correctness has a marginal effect: correctly-solved problems show slightly *lower* silhouette (0.222 vs 0.242, p=0.028), suggesting that successful computation slightly dissolves the binary partition rather than reinforcing it.

**The per-token per-layer analysis reveals the real story.** At layers 7-21 (mid-network), the two-cluster structure is extraordinarily strong (silhouette 0.98), with zero token switching between layers. By layer 27 (final), silhouette drops to 0.40. The clusters form in the early-to-mid network and partially dissolve by the output layer. At the mid-network, the clusters are a near-perfect binary partition of all tokens.

## 2. What is the 1D flow?

**Answer: The residual stream traces a strongly 1D trajectory, with the dominant flow axis explaining 74.3% of all inter-layer displacement variance. It represents computational progress — correlated with entropy decrease, but non-monotonically.**

PCA on the displacement vectors (consecutive-layer differences) across all 500 problems reveals a single dominant direction capturing 74.3% of variance. The next four components together add only 10.4%. The flow is overwhelmingly one-dimensional.

The flow axis correlates moderately with logit lens entropy (r=-0.39, p<10^-52): as the hidden state moves along the flow axis, the predicted output distribution becomes more peaked. However, entropy does NOT decrease monotonically — 0% of problems show strict monotonic decrease. Instead, entropy fluctuates at intermediate layers before converging.

All difficulty levels start at the same flow-axis position (0.16) but end at different positions. Harder problems undergo slightly larger displacement (-43.5 for Level 1 vs -46.4 for Level 5), suggesting the model does more "work" on difficult problems along this axis.

The mean answer convergence layer is 23.7 — the correct answer token first appears in the logit lens top-5 at layer ~24 of 29.

## 3. What is the bridge?

**Answer: Bridge tokens (|silhouette| < 0.1) are causally important for model output but do NOT explain the Goldilocks H1 effect. They are predominantly natural language tokens. Paradoxically, core cluster tokens can be swapped with near-zero impact.**

Bridge tokens constitute 16-27% of all tokens, with the fraction *decreasing* with difficulty (Spearman r=-0.34, p<10^-14). Easy problems have the highest bridge fraction (26.5%), not medium problems — this contradicts the Goldilocks H1 hypothesis. Bridge tokens are overwhelmingly language tokens (70%) and structural tokens (17%), with very few numerics (3%).

The causal intervention results are striking:

| Intervention | Mean KL Divergence | Answer Changed |
|---|---|---|
| Bridge ablation (replace with mean) | 2.25 | 20.0% |
| Steer bridge to Cluster A | 2.75 | 16.7% |
| Steer bridge to Cluster B | 2.38 | 16.7% |
| **Core swap (A↔B centroids)** | **0.069** | **0.0%** |

**Core swap has near-zero effect** (KL=0.069, 0% answer changes). This means swapping all Cluster A tokens to Cluster B centroids (and vice versa) at layer 14 barely changes the output. The cluster identity of firmly-classified tokens is computationally irrelevant at the mid-network. In contrast, bridge token ablation changes 20% of answers — the tokens *between* clusters carry the signal.

## 4. Is this universal?

**Answer: Yes. k=2 is universally optimal and the 1D flow is present in all four architectures tested.**

| Model | k_opt | Mean Sil | Corr Dim | Flow PC1 Var |
|---|---|---|---|---|
| Qwen2.5-1.5B | 2 | 0.242 | 2.05 | 74.3% |
| Phi-2 | 2 | 0.177 | 2.20 | 54.4% |
| Pythia-1.4B | 2 | 0.255 | 1.41 | 78.2% |
| StableLM-1.6B | 2 | 0.243 | 1.97 | 82.1% |

k=2 is optimal across all models. The 1D flow (PC1 > 50% of displacement variance) is universal. Correlation dimension ranges from 1.4 to 2.2, consistent with a near-1D manifold with some perpendicular structure.

Cross-task silhouettes: MATH-500 (0.242) > GSM8K (0.192) > TriviaQA (0.173). The structure exists for all tasks but is somewhat stronger for mathematical reasoning, suggesting math problems activate the binary partition more strongly.

## 5. Where in the model does it come from?

**Answer: The final MLP layer (layer 27) is the dominant cluster-creating component. The cluster axis is a universal direction (mean pairwise |cosine|=0.79) but is orthogonal to the flow axis (|cosine|=0.10). These are independent geometric structures.**

MLP ablation reveals layer 27 as the critical layer (silhouette drop=0.053 when ablated, 6x larger than any other layer). Attention ablation has smaller and more distributed effects. The top-3 critical MLP layers are [27, 5, 25].

At layer 27, neuron 3051 has 2x the importance of any other neuron. The top-10 neurons at layer 27 collectively drive the bulk of the cluster-separating signal. Neurons at earlier layers (e.g., layer 5) have 50x less importance.

The cluster axis is remarkably consistent across problems: mean pairwise |cosine similarity| = 0.79. This means all 500 problems are partitioned along nearly the same direction in hidden-state space. The cluster separation magnitude is constant across difficulty levels (norm ~117 for all levels).

**The cluster axis is orthogonal to the flow axis** (|cosine|=0.105). The 1D flow (Experiment 2) and the cluster separation (Experiment 5) live in perpendicular subspaces. The residual stream simultaneously moves along the flow axis (computational progress) while tokens are separated along an independent direction (binary partition).

## 6. The unified picture

### Confirmed

1. **Universal binary partition**: All transformer models tested produce a two-cluster structure in token hidden states (k=2 universally optimal). This is not math-specific — it appears for all text types, including Wikipedia and trivia.

2. **Universal 1D flow**: The residual stream traces a strongly one-dimensional trajectory (74-82% variance in one axis) across all architectures.

3. **Mid-network crystallization, final-layer dissolution**: The clusters are maximally sharp at layers 7-21 (silhouette 0.98) with zero token switching, then partially dissolve by the output layer (0.40). This suggests the binary partition is an intermediate computational structure, not a final output feature.

4. **Bridge tokens are causally critical; core tokens are not**: Ablating bridge tokens (those between clusters) changes 20% of model outputs; swapping all core cluster tokens changes 0%. The information content lives at the boundary.

5. **Orthogonal geometry**: The flow axis and cluster axis are perpendicular — the model encodes computational progress and binary token partitioning as independent geometric features in the residual stream.

6. **Single dominant MLP layer**: Layer 27 (final) creates the cluster structure, dominated by a small number of neurons (especially neuron 3051).

### Partially supported

1. **Clusters reflect token type (weakly)**: Numeric tokens prefer Cluster A, language tokens prefer Cluster B, but the effect is small (V=0.13). Most of the cluster structure is driven by something other than surface-level token type — possibly positional information, syntactic role, or computational state.

2. **Flow correlates with entropy decrease**: r=-0.39 is significant but moderate, and entropy is not monotonically decreasing. The flow axis partially represents "computational convergence" but is not a simple progress metric.

### Remains open

1. **What exactly drives the binary partition?** Token type explains only ~2% of cluster variance (V^2). The cluster axis is a consistent universal direction, but we haven't identified what semantic or computational property it encodes. Candidates: input vs prompt tokens, positional encoding effects, or a learned "attention routing" signal.

2. **Why does mid-network crystallization dissolve?** The 0.98→0.40 silhouette drop from layer 21→27 suggests the final layers actively mix the clusters. Is this mixing necessary for correct output, or is it an artifact?

3. **Bridge fraction and Goldilocks**: The bridge fraction peaks at Level 1 (easiest), not Level 3. The Goldilocks H1 effect must be driven by something other than the proportion of between-cluster tokens. The H1 cycle likely reflects higher-dimensional topology (loop structure) that is independent of the binary partition.

4. **Training dynamics**: We could not determine when the two-cluster structure emerges during training (Pythia checkpoint data lacked per-token silhouette). This remains a key question for understanding whether the structure is learned or architectural.
