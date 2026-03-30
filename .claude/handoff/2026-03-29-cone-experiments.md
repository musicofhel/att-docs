# Cone Experiments — Handoff

**Date**: 2026-03-29
**Status**: ALL 6 EXPERIMENTS COMPLETE — Decision: PROCEED with revisions

## Prior Session

Implementation session completed all code: ConeDetector (6 methods), visualize.py (6 plots), 35 tests passing, notebook cells filled in. See `2026-03-29-cone-implementation.md`.

## This Session

### OOM Bug Fix

Notebook cells 5, 9, 11, 13 called `depth_asymmetry()` and `full_chain_emergence()` without `subsample` parameter — would OOM on 75k-point series. Fixed by adding `subsample=2000, seed=42` to all calls. Also added `subsample=2000, seed=42` to `BindingDetector.fit()` calls in Exp 1.

### Experiment Results

**Parameters**: N_STEPS=80,000, TRANSIENT=5,000 (75k points), coupling_source=coupling_down=0.15, seed=42, n_depth_bins=5, subsample=2000

#### Exp 0: Cascade Verification — PASS

| Pair | Peak Lag | Peak |r| |
|------|----------|---------|
| C→A3 | -115 | 0.265 |
| C→A5 | -399 | 0.041 |
| A3→A5 | -76 | 0.214 |
| A3-B3 | -367 | 0.636 |

Coupling works as designed. Lag increases along chain, correlation decreases with depth.

#### Exp 1: Depth Asymmetry & Emergence

| Measure | Column A | Column B | Control (ε=0) |
|---------|----------|----------|----------------|
| Shallow binding | 48,196 | 22,225 | 19,235 |
| Deep binding | 49,941 | 45,467 | 15,467 |
| **Asymmetry** | **+1,745** | **+23,242** | **-3,768** |
| 3-way emergence | NO (-62,059) | NO (-62,996) | — |

Depth asymmetry positive for coupled, negative for uncoupled. No 3-way emergence — cross-column pairwise bindings dominate.

#### Exp 2: Core Cone Detection — CONE DETECTED

| Subspace | Bin 0 β₁ | Bin 1 β₁ | Bin 2 β₁ | Bin 3 β₁ | Bin 4 β₁ | Slope |
|----------|----------|----------|----------|----------|----------|-------|
| Full (16D) | 431 | 540 | 595 | 787 | 483 | **+42.25** |
| CCA (3D) | 364 | 334 | 306 | 314 | 347 | -7.63 |

**Cone appears in full embedding, NOT CCA.** This reverses the 8k-step smoke test. At full scale, the cone is higher-dimensional than CCA's 3 components can capture.

#### Exp 3: Coupling Sweep (10 values, ε=0.0→0.5)

| Coupling | Slope | Asymmetry |
|----------|-------|-----------|
| 0.000 | +25.6 | -3,768 |
| 0.111 | +16.8 | +6,846 |
| 0.222 | -49.9 | +24,231 |
| 0.333 | +21.5 | +34,032 |
| 0.500 | +20.7 | +40,889 |

Slope is noisy, but **asymmetry increases monotonically**. No inverted-U collapse at ε=0.5.

#### Exp 4: Timescale Ratio Sweep (7 values)

| Ratio | Slope | Asymmetry |
|-------|-------|-----------|
| 1.0 | +40.6 | -30,070 |
| 2.0 | +20.0 | **+12,590** |
| 2.4 | +42.3 | +1,745 |
| 3.0 | +35.1 | +1,373 |
| 5.0 | -96.6 | -28,302 |

**Peak asymmetry at ratio=2.0**, confirming ATT's heterogeneous-timescale finding. Extreme ratios produce negative asymmetry.

#### Exp 5: Directed vs Symmetric

| Network | Slope | Asymmetry |
|---------|-------|-----------|
| Directed | **+42.25** | +1,745 |
| Symmetric | +5.54 | +3,325 |

Directed coupling produces **7.6x steeper** availability slope. Symmetric Betti_1 is U-shaped, not monotonic.

### Decision: PROCEED (with revisions)

The cone prototype successfully detects directed projection geometry. Key revisions for scaling:
1. **Full embedding** as primary (not CCA) — cone is higher-dimensional than 3 CCA components
2. **Depth asymmetry** as robust measure (monotonic with coupling, clear control separation)
3. **Drop 3-way emergence** — pairwise cross-column dominates
4. **Timescale ratios 2.0-3.0** for optimal sensitivity
5. **Extend coupling range** past ε=0.5 to find collapse

### Surprising Findings

1. Full vs CCA flip at scale — smoke test (8k) showed CCA positive, full negative. At 80k, it's reversed. Likely: with more data, the 16D embedding captures cone structure that 3D CCA projection destroys.
2. No inverted-U — expected synchronization collapse didn't appear. Aizawa's helical escape tube may resist full sync.
3. Slope measure is noisy — availability profile slope oscillates wildly across coupling values, while depth asymmetry is smooth and monotonic. The slope's sensitivity to the bin-4 "edge dip" makes it unreliable.

## Files Modified

| File | Changes |
|------|---------|
| `notebooks/cone_prototype.ipynb` | Fixed OOM (subsample params), updated decision table |
| `.claude/handoff/2026-03-29-cone-experiments.md` | This file |

## What's Next

1. **Scale to 4×4 grids** — replace 2-column/3-layer with 4×4 lattice
2. **Robust slope measure** — drop bin-4 edge effects, or use trimmed slope (bins 0-3 only)
3. **Extended coupling sweep** — ε=0.5→2.0 to find synchronization collapse
4. **Ripser++ GPU** — when nvcc available, for faster PH on larger grids
