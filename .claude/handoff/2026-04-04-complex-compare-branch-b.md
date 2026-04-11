# Branch B: Simplicial Complex Comparison — 2026-04-04

## Branch
`experiment/tda-complex-compare` (commit 0bb5c51), branched from `experiment/neuromorphic-snn` (60e67f2)

## What Was Done

Head-to-head comparison of three simplicial complex constructions on 50 attention matrices (10 per difficulty level) from Qwen2.5-1.5B terminal layer (layer 27) on MATH-500.

### Three Constructions
1. **Symmetric VR (baseline)**: Symmetrize attention → distance → Ripser. Standard approach in all published TDA-on-attention work.
2. **Directed Flag Complex**: Build directed graph (top-k edges per query token), add edges + directed 3-clique triangles, compute PH via GUDHI SimplexTree. Preserves attention asymmetry.
3. **Dowker Complex**: Treat attention as bipartite relation (query→key). Simplex exists if all query tokens attend to a common key token. Manual construction via GUDHI.

### Five Experiments
1. **Feature Count & Entropy** — H0/H1 counts and entropy per difficulty level per construction
2. **Permutation Test** — 200-permutation shuffle test on H1 entropy, z-score per construction
3. **Correctness Prediction** — Logistic regression AUROC (5-fold CV) on persistence features
4. **Threshold Sensitivity** — Directed flag at k={3,5,8,12,20}, stability curve
5. **Diagram Structure** — Pairwise Wasserstein distances between constructions

### Results

| Metric | Symmetric VR | Directed Flag | Dowker |
|--------|-------------|---------------|--------|
| H1 entropy (L5) | 4.40 ± 0.60 | 4.39 ± 0.56 | 3.22 ± 3.28 |
| Permutation z | **2.79** (p=0.015) | 2.64 (p=0.02) | -0.98 (p=0.82) |
| AUROC | 0.700 | 0.700 | 0.650 |
| d(·, VR) | — | 0.23 | 2.58 |

**Verdict: `symmetrization_loses_nothing`**

### Key Findings
- Directed flag ≈ symmetric VR (Wasserstein d=0.23). Head-averaged attention is approximately symmetric, so directed structure adds nothing.
- Dowker is noisy (H1 count std 49–89) and non-discriminative (z=-0.98).
- Combined features (0.625 AUROC) < individual (0.700) — no complementary information.
- Directed flag stable for k≥5 (z=2.0–2.8), collapses at k=3.

## Debugging Notes

### pyflagser Filtration Issues
- `pyflagser.flagser_weighted` with `filtration='max'` produces zero finite H1 features on attention graphs (both directed and undirected).
- `filtration='product'` produces features with birth > death (negative lifetimes) — invalid.
- `filtration='max_plus_one'` produces constant lifetime=1.0 — no discrimination.
- **Solution**: Abandoned pyflagser entirely. Built directed flag complex manually via GUDHI SimplexTree with proper max-edge filtration. This produces meaningful H1 with varied lifetimes.

### Dowker H1 = 0 Bug
- Original code only added triangles when `max_dim >= 2`. But H1 computation needs 2-simplices (triangles) to kill 1-cycles. Without them, all H1 features have infinite death → filtered out.
- **Fix**: Always add triangles regardless of max_dim, capped at 40 attendees per key token to prevent combinatorial explosion.

### NaN/Inf in Features
- `persistence_entropy` could produce NaN when all lifetimes sum to 0 or contain inf.
- **Fix**: Filter to finite positive lifetimes before computing entropy. `np.nan_to_num` on feature matrices before sklearn.

## Files
- `scripts/branches/complex_compare.py` — Full experiment script (~880 lines)
- `data/complex_compare/results.json` — All numeric results
- `data/complex_compare/raw_attention_matrices.npz` — 50 raw attention matrices (not committed, 2MB)
- `figures/complex_compare/` — 4 figures (entropy, AUROC, threshold stability, diagram grid)

## Dependencies Installed
- `pyflagser==0.4.7` (installed but ultimately not used — GUDHI used instead)
- `gudhi`, `ripser`, `cripser` were already present

## Implications for ATT Preprint
This is a strong null result: the standard symmetrized VR approach used throughout the ATT paper is justified. No need to add complexity with directed or Dowker constructions. Can be cited as supplementary validation if reviewers question the symmetrization step.
