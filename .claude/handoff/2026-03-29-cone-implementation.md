# Cone Implementation — Handoff

**Date**: 2026-03-29
**Status**: IMPLEMENTATION COMPLETE — all methods, tests, visualizations, notebook experiments

## What Was Done

### 1. ConeDetector — All 6 Methods Implemented (`att/cone/detector.py`)

- **fit()**: TakensEmbedder for source (1D), JointEmbedder for receivers (multi-channel), tail-align, axis estimation, CCA
- **estimate_projection_axis()**: Conditional-mean PCA on 20 source quantile bins, SVD for first PC, fallback to full-cloud PCA if <3 bins
- **slice_at_depth()**: Equal-count quantile bins on depth projections
- **availability_profile()**: PH per depth bin via PersistenceAnalyzer, Betti counting (persistence > 10% of max), trend slope via polyfit, monotonicity check
- **coupling_influence_subspace()**: CCA between embedded source and receiver cloud, returns projected receiver
- **depth_asymmetry()**: Two BindingDetector instances (shallow vs deep), returns score comparison
- **full_chain_emergence()**: 3 pairwise + 3-way joint via JointEmbedder, PI subtraction against max-of-marginals baseline

### 2. Visualizations — All 6 Functions (`att/cone/visualize.py`)

- plot_availability_profile, plot_coupling_sweep (2-panel heatmap + slope), plot_cross_sections (2-row PCA + persistence), plot_subspace_comparison, plot_cascade_verification (2x4 xcorr grid), plot_directed_vs_symmetric

### 3. Tests — 13 New Tests (`tests/test_cone_synthetic.py`)

TestConeDetector class: fit_returns_self, fit_populates_state, source_receiver_aligned, projection_axis_is_unit, projection_axis_dimension, slice_at_depth_valid_bins, slice_at_depth_invalid, availability_profile_keys, availability_profile_shapes, availability_profile_cca, cca_subspace_shape, depth_asymmetry_keys (slow), full_chain_emergence_keys (slow)

**All 35 tests passed in 19s** (22 synthetic + 13 ConeDetector, including 2 slow).
Initial slow tests OOM-killed at 19.9GB — fixed by adding `subsample` param to `depth_asymmetry()` and `full_chain_emergence()`, tests now pass with `subsample=500` in 8s.

### 4. Notebook — All 6 Experiments Filled In (`notebooks/cone_prototype.ipynb`)

- Exp 0: Already runnable (cascade verification)
- Exp 1: Pairwise bindings + depth_asymmetry + full_chain_emergence + zero-coupling control
- Exp 2: ConeDetector fit + availability profiles (full + CCA), comparison plots
- Exp 3: Coupling sweep (10 values) with availability profiles + asymmetry
- Exp 4: Timescale ratio sweep (7 values)
- Exp 5: Directed vs symmetric comparison

### 5. Ripser++ Decision: DEFERRED

No pre-built wheel for Python 3.12, nvcc not installed. CPU ripser fast enough for 2000-point subsamples (<1s each). Single callsite at `PersistenceAnalyzer._ripser_compute()` makes future migration trivial.

## Key Early Result (Smoke Test)

Using 8000 post-transient steps, 3 depth bins, subsample=500:

| Subspace | Bin 0 β₁ | Bin 1 β₁ | Bin 2 β₁ | Slope |
|----------|----------|----------|----------|-------|
| Full     | 105      | 73       | 95       | -3.09 |
| CCA      | 44       | 49       | 62       | +5.32 |

**The cone appears in the CCA coupling-influence subspace but not the full embedding.** This matches the spec's prediction: "the cone is a low-dimensional feature embedded in a high-dimensional attractor state space."

## Files Modified

| File | Changes |
|------|---------|
| `att/cone/detector.py` | All 6 methods implemented (was pseudocode stubs) |
| `att/cone/visualize.py` | All 6 plot functions implemented (was stubs) |
| `tests/test_cone_synthetic.py` | Added TestConeDetector class (13 tests) |
| `notebooks/cone_prototype.ipynb` | Filled in Exp 1-5 cells, updated imports |

## OOM Fix

`depth_asymmetry()` and `full_chain_emergence()` now accept optional `subsample` and `seed` params, passed through to `BindingDetector.fit()` and `PersistenceAnalyzer.fit_transform()`. Without subsampling, auto embedding estimation + Ripser on full 4000-point clouds consumed 19.9GB and got OOM-killed. With `subsample=500`, both complete in 8s total.

## Doc Updates

All intent-layer documents updated with cone module:
- **ARCHITECTURE.md** — `att.cone` module section + Aizawa/network generators in `att.synthetic`
- **API.md** — Full `att.cone` API docs with ConeDetector, all methods, example code
- **CHANGELOG.md** — v0.2.0-dev entry for cone prototype
- **ROADMAP.md** — Phase 6 (Cone Prototype) with completed/remaining tasks
- **README.md** — API overview table updated

## What's Next

1. **Run the full notebook** — Exp 0 and 2 are quick, Exp 1/3/4/5 are compute-intensive (BindingDetector per coupling/ratio value). Use `subsample=500` for Exp 1/3/4/5 to stay under memory limits.
2. **Analyze results** — fill in the decision table, determine if cone construction works
3. **Ripser++ GPU** — when ready: `sudo apt install nvidia-cuda-toolkit`, `pip install ripserplusplus`, add backend to PersistenceAnalyzer
4. **Scale to grid** — if prototype succeeds, move to 4x4 grids per the spec
