# Cone Prototype Scaffold — Handoff

**Date**: 2026-03-29
**Status**: SCAFFOLD COMPLETE — generators working, detector/notebook pseudocode stubs

## What Was Done

### 1. Aizawa Attractor Generator
Added `aizawa_system()` to `att/synthetic/generators.py`. Same pattern as lorenz/rossler (solve_ivp, seed, noise). Roughly spherical geometry with helical escape tube — chosen for cleaner cross-sections than Lorenz/Rossler.

### 2. Layered Network Integrator
Created `att/synthetic/layered_network.py`:
- `layered_aizawa_network()` — 5-node directed network (C→A3→A5, C→B3→B5), per-layer dt, xy-only diffusive coupling, Euler integration
- `layered_aizawa_network_symmetric()` — all-to-all control for Experiment 5, Frobenius-norm matched
- Lazy import in `att/synthetic/__init__.py` (module-level `__getattr__`)

### 3. Cone Module Scaffold
Created `att/cone/` with pseudocode stubs:
- `detector.py`: `ConeDetector` class with 6 methods — `fit()`, `estimate_projection_axis()` (conditional-mean PCA), `slice_at_depth()`, `availability_profile()` (Betti vs depth), `coupling_influence_subspace()` (CCA), `depth_asymmetry()`, `full_chain_emergence()`
- `visualize.py`: 6 plot functions — availability profile, coupling sweep, cross-sections, subspace comparison, cascade verification, directed vs symmetric

### 4. Tests
`tests/test_cone_synthetic.py` — 22 tests all green:
- TestAizawa: shape, reproducibility, boundedness, noise, custom params
- TestLayeredNetwork: shapes, coupling effects, column symmetry, cascade
- TestLayeredNetworkSymmetric: shapes, boundedness, cross-correlation

Full suite: 217 passed, 0 regressions.

### 5. Notebook Skeleton
`notebooks/cone_prototype.ipynb` — Experiments 0-5 with markdown headers, pseudocode cells showing exact ATT API calls. Exp 0 (cascade verification) is runnable now.

### 6. Spec Saved
`CONE_PROTOTYPE_SPEC-1.md` was corrupted (all null bytes from Windows copy). Replaced with actual content from user paste.

## Files Created

| File | Status |
|------|--------|
| `att/synthetic/layered_network.py` | Working |
| `att/cone/__init__.py` | Working (imports ConeDetector) |
| `att/cone/detector.py` | Pseudocode stubs (NotImplementedError) |
| `att/cone/visualize.py` | Pseudocode stubs (NotImplementedError) |
| `tests/test_cone_synthetic.py` | 22 tests, all green |
| `notebooks/cone_prototype.ipynb` | Skeleton with pseudocode |
| `CONE_PROTOTYPE_SPEC-1.md` | Full spec |

## Files Modified

| File | Changes |
|------|---------|
| `att/synthetic/generators.py` | Added `aizawa_system()` |
| `att/synthetic/__init__.py` | Added aizawa + lazy import for layered_network |

## Remaining Task List

| Priority | Task | Notes |
|----------|------|-------|
| **1** | Install Ripser++ (GPU) | Task #82. Need `nvcc` — `sudo apt install nvidia-cuda-toolkit` then `pip install ripserplusplus`. Have 2060 Super + CUDA 12 runtime but no compiler. Update `PersistenceAnalyzer` to use as backend. |
| **2** | Implement `ConeDetector.fit()` + `estimate_projection_axis()` | ~80 lines. Unlocks Experiment 2. |
| **3** | Implement `availability_profile()` | ~60 lines. Core deliverable. |
| **4** | Implement `depth_asymmetry()` + `full_chain_emergence()` | ~40 lines. Uses existing BindingDetector. |
| **5** | Implement visualize.py functions | ~100 lines. |
| **6** | Run Experiments 0-5 in notebook | Exp 0-1 don't need ConeDetector. Exp 2-5 do. |
| **7** | Coupling sweep (Exp 3) + timescale sweep (Exp 4) | Computationally intensive. |

## Key Design Decisions

- Aizawa added to existing `generators.py` (not separate file) — follows codebase pattern
- Layered network uses Euler integration (not RK45) — per-node dt with coupling forces doesn't fit solve_ivp's single-system interface cleanly
- `layered_aizawa_network_symmetric` scales per-edge coupling by sqrt(4/20) to match Frobenius norm of directed topology
- ConeDetector composes with (not inherits from) BindingDetector — different interface (multi-channel vs pairwise)
- Lazy import for layered_network to avoid import error before file exists

## Environment Notes

- GPU: NVIDIA 2060 Super, driver 591.74, CUDA 13.1 (WSL2)
- `nvcc` NOT installed — needed for Ripser++ compilation
- PyTorch 2.10.0 with CUDA 12 runtime is installed
- Current ripser: CPU-only v0.6.14
- User confirmed they used Ripser++ "last night" but it's not currently in the environment
