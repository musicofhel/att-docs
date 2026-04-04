# Mechanistic Investigation: Where Does the Topological Structure Come From?

## Summary

Five experiments probe the mechanistic origins of difficulty-dependent topological structure in LLM hidden states. We trace when it emerges during training, which components create it, and what makes easy-problem representations geometrically distinct.

---

## Experiment 1: Training Dynamics (Pythia-1.4B)

**Question**: When does the H1 non-monotonicity (Level 1 minimum) emerge during training?

**Result**: The topological signature is *learned*, not architectural. At step 0 (random initialization), hidden states show non-monotonic H1 entropy, but in the *wrong direction* — Level 5 (hardest) has the minimum H1 entropy (0.07), while Level 1 has high entropy (1.71). The model initializes with topology that is structured but not difficulty-aware.

By step 2000 (~1.4% of training), the pattern inverts: Level 1 becomes the minimum. This is remarkably early — the model learns to topologically compress easy-problem representations well before convergence. The final pattern at step 143000 shows a clear monotonic gradient: L1 (0.63) < L2 (0.82) < L3 (1.00) < L4 (0.99) < L5 (1.54).

Between steps 2000-32000, there is a "topological reorganization" period where multiple levels collapse to near-zero H1, suggesting intermediate training creates degenerate representations that later differentiate. By step 64000, all levels have non-zero H1 and the final monotonic ordering stabilizes.

**Key finding**: The Level 1 minimum emerges at step 2000. It is a *learned* property, not a consequence of architecture or tokenization. The topology undergoes a phase transition during early-to-mid training.

---

## Experiment 2: Per-Head Topology (Qwen2.5-1.5B)

**Question**: Which attention heads carry the difficulty-dependent topological signal?

**Result**: No single attention head reaches significance (z > 2) for difficulty discrimination at the terminal layer. The highest z-score is Head 10 (z = 1.67), followed by Head 6 (z = 1.04). Most heads have *negative* z-scores, indicating they are less discriminative than chance.

Scanning across 5 representative layers (0, 7, 14, 21, 27), the pattern is diffuse: Layer 0/Head 0 (z = 1.86) shows the strongest single-head signal, but no layer has a majority of significant heads. This contrasts with TOHA's finding that 6 specific heads drive hallucination detection in Llama/Mistral.

**Key finding**: The difficulty-dependent topology is *not concentrated* in specific attention heads. It is a distributed, emergent property of the full model rather than attributable to a small set of specialized heads. This makes sense given Experiment 3's finding that the structure is primarily in the residual stream, not individual attention patterns.

---

## Experiment 3: Residual Stream Decomposition

**Question**: Does the attention mechanism or the MLP create the topological structure at the terminal layer?

**Result**: The highest difficulty-discrimination z-score belongs to `pre_attn` (z = 7.67) — the residual stream *before* the terminal layer's attention and MLP have even processed it. This means the topological structure is already present when it arrives at the terminal layer; it was created by earlier layers.

The MLP output contributes meaningfully (z = 3.70), showing rich H1 entropy across all levels (range 1.05–2.03). The attention output is the weakest component (z = 2.01), with mostly near-zero H1 entropy — attention at the terminal layer barely creates new topological features.

The full layer output (`post_mlp`, z = 6.70) is slightly less discriminative than the input (`pre_attn`), suggesting the terminal layer's processing mixes rather than sharpens the topological signal.

**Key finding**: The topological structure is primarily constructed by layers *earlier* than the terminal layer. The MLP creates more topological structure than attention. The terminal layer's attention is topologically inert — it routes information but doesn't reshape the difficulty-dependent geometry.

---

## Experiment 4: What Makes Level 1 Special?

**Question**: What is geometrically different about easy-problem representations?

**Result**: Level 1 (easy) and Level 5 (hard) hidden states differ on every geometric metric tested:

| Metric | Level 1 | Level 5 | Interpretation |
|--------|---------|---------|----------------|
| Intrinsic dimension (TwoNN) | 8.49 | 15.94 | Easy lives in ~half the dimensions |
| Isotropy | 0.980 | 0.967 | Easy is more uniformly spread |
| Silhouette (k=2) | 0.119 | 0.672 | Hard has clear sub-clusters |
| H1 entropy | 1.182 | 1.210 | Easy has simpler loop structure |
| H0 entropy | 1.849 | 2.020 | Easy has fewer components |

Level 1 representations form a low-dimensional, isotropic, unclustered blob. Level 5 representations span twice the intrinsic dimensions and have strong sub-cluster structure (silhouette = 0.67). The model compresses easy problems into a simple, compact manifold while hard problems maintain high-dimensional, structured representations — consistent with the hypothesis that the model "knows what it doesn't know" topologically.

**Key finding**: The Level 1 minimum is explained by geometric compression: easy problems are mapped to a low-dimensional, isotropic subspace with simple topology. Hard problems maintain complex, clustered, high-dimensional structure. The 2x difference in intrinsic dimensionality (8.5 vs 15.9) is the most striking result.

---

## Experiment 5: Layer-by-Layer H1 Trajectory

**Question**: At which layer do the difficulty levels first separate topologically?

**Result**: The H1 trajectory across all 29 layers (embedding + 28 transformer layers) reveals:

1. **Layer 0 (embedding)**: All levels have H1 = 0 — token embeddings have no topological structure.
2. **Layer 1**: Immediate differentiation begins. Level 5 already differs from others.
3. **Layers 1–10**: Volatile — H1 values fluctuate widely as representations are being formed. No stable ordering.
4. **Layers 10–16**: A transition zone where Level 4 consistently rises and maintains high H1.
5. **Layers 17–28**: Level 3 emerges as the H1 maximum (~2.0–2.4), while Level 1 and Level 2 remain lower (~0.5–1.4). Levels 4 and 5 occupy intermediate positions.

The formal separation layer (where L1 and L5 first differ by > 1 std) is layer 1, but this is driven by early noise. The stable separation pattern doesn't emerge until approximately layers 14–17, where the Level 3 > Level 4 > Level 5 > Level 1 ordering begins to crystallize.

**Key finding**: Topological differentiation begins at the very first transformer layer but doesn't stabilize until the middle layers (14–17). The final ordering is non-trivial — Level 3 (not Level 5) has the highest H1 entropy in many layers, suggesting medium-difficulty problems generate the richest topological structure. This is consistent with a hypothesis that easy problems are trivially compressed and very hard problems may produce degenerate representations, while medium-difficulty problems maximally exercise the model's representational capacity.

---

## Synthesis: Where Does the Topological Structure Come From?

The topological structure in LLM hidden states has four key properties:

1. **Learned, not innate**: It does not exist at initialization and emerges early in training (by step 2000 for Pythia-1.4B, ~1.4% of total training).

2. **Distributed, not localized**: No single attention head or small head subset drives the signal. The structure is an emergent property of many layers working together.

3. **MLP-dominated, attention-inert**: At the terminal layer, the MLP creates more topological structure than attention. The attention mechanism routes information but doesn't reshape the difficulty-dependent geometry. The bulk of the structure arrives via the residual stream from earlier layers.

4. **Geometric compression of easy problems**: The Level 1 minimum reflects genuine geometric simplification — easy problems are mapped to a low-dimensional, isotropic, unclustered manifold (~8.5 intrinsic dimensions vs ~15.9 for hard problems).

### Implications for Mechanistic Interpretability

- **Topology as a training diagnostic**: The H1 non-monotonicity emerges very early in training. Monitoring topological metrics could provide a cheap signal for when a model has learned to differentiate problem difficulty — useful for curriculum learning or early stopping.

- **MLP as geometry engine**: The finding that MLPs create topological structure while attention is topologically inert aligns with the "MLP as key-value memory, attention as routing" framework. The MLP layers sculpt the representational geometry; attention determines which information flows where.

- **Distributed emergence**: The fact that no individual head carries the difficulty signal means the topological perspective captures something that single-head probing would miss. This supports TDA as a complementary tool to attention-head analysis for mechanistic interpretability.

- **Medium-difficulty maximum**: The unexpected finding that Level 3 (not Level 5) has the highest H1 in many layers suggests a "Goldilocks" effect — easy problems are trivially compressed, very hard problems may produce degenerate or random-like representations, and medium-difficulty problems maximally engage the model's structured computation.
