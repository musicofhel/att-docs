# Handoff: Branch A — Cubical Persistence on Layer-Token Activation Grid

**Date**: 2026-04-04
**Branch**: `experiment/tda-cubical` (from `experiment/neuromorphic-snn` @ 60e67f2)
**Commit**: 3861e96
**Script**: `scripts/branches/cubical_persistence.py`
**Results**: `data/cubical/results.json`
**Figures**: `figures/cubical/` (4 PNGs)

---

## What was done

Replaced point-cloud Vietoris-Rips PH with cubical persistence (via `cripser`) on the raw activation landscape. Five experiments ran on 500 problems from MATH-500 (Qwen, aligned dataset):

### Experiment 1: 1D Cubical on Token Activation Norms
- Computed L2 norm of each token's hidden state at final layer → 1D signal per problem
- **H0 count scales with difficulty**: 18.3 (L1) → 34.3 (L5) — harder problems have nearly 2× more local minima in the activation landscape
- Mean persistence also increases: 25.3 → 28.4
- Permutation test (200 perms, Wasserstein on L1 vs L5): p=1.0 — the greedy Wasserstein approximation saturated; the permutation test is unreliable here

### Experiment 2: 2D Cubical on PCA-Reduced Token Trajectories
- PCA-reduced token trajectories to 3 components → (n_tokens, 3) grid per problem
- **H1 entropy monotonically increases with difficulty**: 1.40 (L1) → 1.82 (L5)
- H1 count: 6.0 (L1) → 11.2 (L5)
- This is the cleanest difficulty gradient from cubical persistence

### Experiment 3: Spatial Localization
- Core question: do hard-problem topological features concentrate at specific token positions?
- **Answer: No.** KS test L1 vs L5: p=0.32. Late-region (>75%) birth fractions nearly uniform (~28% across all levels)
- Cubical persistence gives spatial coordinates, but they don't carry difficulty-dependent signal in this data
- Likely because token trajectories are final-layer only — no true layer×token grid exists

### Experiment 4: Correctness Prediction
- 10 cubical features (H0 count/persistence/entropy, birth stats, H1 count/entropy), logistic regression, 5-fold stratified CV
- **Cubical AUROC = 0.615** vs VR baseline 0.787
- Combined cubical + layer-norm features: 0.584 (worse — overfitting on small positive class n=38)
- Cubical features are weaker predictors than VR features for correctness

### Experiment 5: Cubical vs VR Direct Comparison
- 100 problems (20/level), same PCA-reduced data, both cubical and GUDHI Rips
- **H0 entropy correlation: r=0.976** — nearly identical information
- **H1 entropy correlation: r=0.735** — moderate, cubical captures different H1 structure
- VR produces more features (106 vs 62 mean) — simplicial complexes are richer than cubical

---

## Key Takeaways

1. **Cubical persistence captures the difficulty gradient** — H0 count and H1 entropy both scale with difficulty. This confirms the VR findings from a completely different topological framework.

2. **Spatial localization is a dead end** (with current data). The (layer, token) localization promise of cubical PH can't be realized because the data lacks full layer×token activation tensors. Only final-layer token trajectories and per-layer last-token states exist.

3. **VR still wins for classification** (0.787 vs 0.615 AUROC). Cubical persistence on low-dimensional PCA grids produces fewer features with less discriminative power. VR's higher-dimensional simplicial complexes capture richer topological structure.

4. **H0 is redundant** between methods (r=0.98). The novel information is in H1, where cubical and VR diverge (r=0.74).

---

## Technical Notes

- **cripser essential features**: `computePH` returns death=`float64.max` (~1.8e308) for essential H0 features (global minimum). These pass `np.isfinite()`. Must filter with `death < 1e300`.
- **Correctness alignment**: 433 of 500 problems have correctness labels. Mapping: first N per difficulty level (L1:43, L2:90, L3-5:100 each).
- **Dependencies added**: `cripser==0.0.32` (pip installed `--break-system-packages`), `gudhi` already present.

---

## What could come next

- **Full layer×token extraction**: Re-run `extract_hidden_states.py` saving all 29 layers × all tokens (memory-intensive but would unlock true 2D cubical grids and real spatial localization)
- **Cubical on attention maps**: Attention weight matrices are naturally 2D grids — cubical PH on attention would give head/token localization of topological features
- **Hybrid features**: Use VR features for H0, cubical for H1 (since they decorrelate there), could improve D2 AUROC beyond 0.787
- **Permutation test fix**: Replace greedy Wasserstein with proper bottleneck distance or use persistence images for group comparison
