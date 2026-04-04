# Which Simplicial Complex Should You Use for TDA on Transformer Internals?

**Summary**: Use Vietoris-Rips (VR) on hidden-state point clouds as your default. Cubical persistence on attention grids is useful for visualization but statistically inferior. Directed flag complexes add nothing over VR. Avoid Dowker complexes on self-attention.

---

## 1. When to Use Cubical Persistence

**Input**: Attention matrices reshaped as 2D grids (layer × token position).

**Strengths**:
- High interpretability — birth/death positions map to specific layers and token indices
- Fast computation via `cripser`
- Natural for gridded data where spatial adjacency matters (e.g., image patches in ViTs)

**Limitations (empirical)**:
- No spatial localization of difficulty-dependent features (KS p=0.321 between easy/hard birth distributions)
- Correctness prediction AUROC = 0.615 vs VR's 0.787
- Combined cubical+VR features *decrease* performance (AUROC 0.584) — the two feature sets are highly correlated (r=0.975 for H0) so combining them introduces noise without new information

**Recommendation**: Use cubical persistence for exploratory visualization (e.g., "where in the layer × token grid do topological features appear?"). Do not rely on it for quantitative analysis or prediction. If your data is inherently gridded (e.g., vision transformers with spatial patch layouts), cubical persistence may be more appropriate than for language models.

## 2. When to Use Vietoris-Rips (VR)

**Input**: Hidden-state vectors treated as a point cloud in R^d.

**Strengths**:
- Highest permutation z-score for difficulty discrimination (z=8.11 on full dataset, z=2.79 in 50-problem comparison)
- Best correctness AUROC (0.787)
- No threshold tuning required
- Mature library support (Ripser/ripser.py — fast, well-tested, GPU-accelerated variants available)
- Established in the TDA-on-neural-networks literature

**Limitations**:
- Low interpretability — features are defined by abstract simplices in high-dimensional space
- Point cloud construction discards sequential structure
- O(n^3) worst case for large point clouds (mitigated by Ripser's clearing optimization)

**Recommendation**: Use VR as your primary construction for any quantitative analysis (permutation tests, classification, cross-model comparison). This is the construction that should appear in papers and be compared against.

## 3. When to Use Directed Flag Complexes

**Input**: Attention matrix binarized as a directed graph (A[i,j] > threshold → edge i→j).

**Strengths**:
- Preserves the asymmetric structure of attention (A ≠ A^T)
- Theoretically motivated — attention flow has direction
- `pyflagser` library exists (maintained, reasonable performance)

**Limitations (empirical)**:
- Produces nearly identical results to symmetric VR (persistence landscape L1 distance = 0.23)
- Requires threshold calibration — unstable below k=5 neighbors (z=0.12 at k=3 vs z=2.83 at k=8)
- Permutation z-score (2.64) slightly worse than VR (2.79)
- Combined with VR, does not improve prediction

**Why symmetrization loses nothing**: In multi-head self-attention, the asymmetric structure (A[i,j] ≠ A[j,i]) reflects which tokens attend to which. However, for difficulty discrimination, what matters is the *overall connectivity pattern* — whether tokens form tightly coupled or loosely coupled clusters — not the direction of coupling. Symmetrizing via max(A[i,j], A[j,i]) preserves this structure.

**Recommendation**: Only use directed flag complexes if attention asymmetry is your specific research question (e.g., "does information flow direction change with task difficulty?"). For general difficulty discrimination, VR is simpler and equally effective.

## 4. When to Skip Dowker Complexes

**Input**: Attention matrix treated as a bipartite relation between "source" tokens (rows) and "target" tokens (columns).

**Problems (empirical)**:
- Extremely high variance in feature counts (H1: 35–70 ± 47–89 vs VR: 16–35 ± 6–14)
- Permutation z-score = −0.98 (below chance — no difficulty discrimination)
- Correctness AUROC = 0.65 (barely above chance)
- Library (`pyDowker`) is immature with limited documentation

**Why Dowker fails on self-attention**: Dowker complexes are designed for bipartite relations — entity A relates to entity B. In self-attention, both rows and columns are the same set of tokens. The bipartite construction creates a degenerate structure where the two "sides" are topologically identical, producing unstable and uninformative persistence.

**Recommendation**: Avoid Dowker complexes for self-attention matrices. They may be appropriate for cross-attention (encoder-decoder models where source ≠ target tokens), but this has not been tested.

## 5. Practical Guidance

**Starting a new TDA-on-transformers project**:

1. **Explore** with cubical persistence on attention grids — fast, visual, helps build intuition about where topological structure lives
2. **Quantify** with VR on hidden states — this is your main result. Use permutation tests for significance, logistic regression on persistence features for prediction
3. **Check asymmetry** only if VR leaves unexplained variance — use directed flag at k=8 (middle of stable range). If results match VR within noise, report that symmetrization is safe and stop
4. **Skip Dowker** for self-attention. Consider it only for cross-attention or other genuinely bipartite relations

**Feature selection**: H0 persistence entropy and H1 max lifetime are the most predictive features (single-feature AUROCs of 0.824 and 0.803). Start with these before engineering complex feature sets.

**Sample size**: The 50-problem comparison produced z-scores of 2–3 (significant but modest). The full 500-problem analysis produced z=8.11. For publication-quality results, use at least 200 problems.

---

*Based on experiments from ATT (Attractor Topology Toolkit) on Qwen2.5-1.5B-Instruct with MATH-500.*
