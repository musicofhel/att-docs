# Grid Scaling Plan — Handoff

**Date**: 2026-03-29
**Status**: PLAN APPROVED, implementation not yet started
**Git**: All cone prototype work committed and pushed (`5e45f63` on master, 17 files, 2225 lines)

## What Was Done This Session

1. Read all handoff files from prior cone sessions (scaffold, implementation, experiments)
2. Explored codebase thoroughly: ConeDetector internals, JointEmbedder, PersistenceAnalyzer, BindingDetector — assessed scaling limits for 48D point clouds
3. Read CONE_PROTOTYPE_SPEC-1.md, ROADMAP.md, layered_network.py, detector.py, test_cone_synthetic.py
4. Entered plan mode, designed grid scaling architecture with Plan agent
5. Wrote and got approval for the plan at `.claude/plans/quirky-painting-rain.md`
6. Committed all uncommitted cone prototype work (was entirely uncommitted) and pushed to remote

## The Approved Plan (Summary)

Full plan at `.claude/plans/quirky-painting-rain.md`. 7 tasks:

### Task 1: Grid Network Generator (NEW FILE)
**File**: `att/synthetic/grid_network.py` (~150 lines)

External source S + N×M grid of receivers (default 4×4 = 17 nodes total):
```
Source S → row 0: [N_0_0] [N_0_1] [N_0_2] [N_0_3]  (dt_fast)
                      ↓        ↓        ↓        ↓
           row 1: [N_1_0] [N_1_1] [N_1_2] [N_1_3]
                      ↓        ↓        ↓        ↓
           row 2: [N_2_0] [N_2_1] [N_2_2] [N_2_3]
                      ↓        ↓        ↓        ↓
           row 3: [N_3_0] [N_3_1] [N_3_2] [N_3_3]  (dt_slow)
```

Three coupling params: `coupling_source` (S→row0), `coupling_down` (row r→r+1), `coupling_lateral` (within-row bidirectional, default 0).

Per-row timescale: `dt_row[r] = dt_top * ratio^(r/(n_rows-1))`, default ratio=2.0.

Functions:
- `grid_aizawa_network(n_rows, n_cols, n_steps, dt_top, timescale_ratio, coupling_source, coupling_down, coupling_lateral, seed)` → `dict[str, ndarray]`
- `grid_aizawa_network_symmetric(...)` — all-to-all, Frobenius-norm matched
- `grid_node_names(n_rows, n_cols)` → `list[str]` including "S"
- `row_nodes(traj, row, n_cols)` → `list[ndarray]` (x-component series)
- `column_nodes(traj, col, n_rows)` → `list[ndarray]`

Import `_aizawa_deriv` from `layered_network.py`. Use adjacency list built once from params (not hardcoded edges). Update `att/synthetic/__init__.py` with lazy import.

### Task 2: Trimmed Slope
**File**: `att/cone/detector.py` — add `trim_bins: int = 0` parameter to `availability_profile()`. Excludes edge bins from slope/monotonicity computation. Default 0 preserves current behavior.

### Task 3: Grid Visualization
**File**: `att/cone/visualize.py` — add `plot_grid_asymmetry_heatmap()`, `plot_lateral_sweep()`, `plot_grid_cascade()`.

### Task 4: Grid Generator Tests (NEW FILE)
**File**: `tests/test_grid_network.py` (~200 lines, 12 tests)
Tests: returns_all_nodes (17), shapes, reproducible, different_seeds, bounded, not_degenerate, zero_coupling_independent, coupling_vs_uncoupled, row_symmetry, depth_gradient, lateral_coupling_effect, parameterized_grid_size.

### Task 5: Cone-on-Grid Integration Tests (NEW FILE)
**File**: `tests/test_cone_grid.py` (~120 lines, 5 tests)
Tests: per_column_detector_fit, per_column_availability, depth_asymmetry_on_grid, asymmetry_increases_with_row, zero_coupling_no_asymmetry.

### Task 6: Grid Experiments G0-G5
Run as standalone Python scripts (not notebook execution — scripts were more reliable last session).

| Exp | Question | Method |
|-----|----------|--------|
| G0 | Grid cascade works? | Lagged xcorr all source-to-node pairs |
| G1 | Depth asymmetry scales to 4×4? | Per-column depth_asymmetry at rows 1,2,3 + zero-coupling control |
| G2 | Lateral coupling helps/hurts? | Sweep coupling_lateral 0.0→0.3, measure per-column asymmetry |
| G3 | Where does grid collapse? | Sweep coupling 0.0→1.0 (extended past 0.5) |
| G4 | Per-column vs per-row profiles | Per-column + per-row availability profiles (~12D each) |
| G5 | Directed vs symmetric at grid scale | Grid vs symmetric, Frobenius-norm matched |

Fixed params: `n_steps=80000, timescale_ratio=2.0, seed=42, subsample=2000`

### Task 7: Analyze & Update Docs
Handoff, ROADMAP, memory updates with grid findings.

## Key Design Decisions

1. **New file, not extend layered_network.py** — different data structures, existing tests must not break
2. **External source S (17 nodes)** — keeps source free-running, required for ConeDetector's conditional-mean axis estimation
3. **Do NOT attempt full 48D joint embedding** — distance concentration in 48D destroys PH signal. Use pairwise depth asymmetry (6D, proven robust) as primary; per-column availability profiles (4 receivers → ~12D, same as proven 2-column scale) as secondary
4. **Depth asymmetry is primary measure, availability profile is secondary** — proven by 2-column experiments
5. **G2 (lateral coupling) is the KEY new experiment** — what the grid uniquely enables that 2-column cannot test
6. **Scripts over notebook execution** — last session showed standalone Python scripts via Bash are more reliable

## The 48D Trap (Critical Context)

16 receivers × ~3D Takens embedding each = ~48D joint receiver cloud. Problems:
- Distance concentration: in high D, max/min distance ratio → 1, PH filtration loses scale separation
- Sampling: 2000 points in 48D is catastrophically sparse for meaningful PH
- Untested: largest tested embedding was 16D (current 4-receiver cone prototype)

Solution: hierarchical analysis at three scales:
1. **Pairwise depth asymmetry** (source + 1 receiver = ~6D) — primary measure, proven robust
2. **Per-column availability** (source + 4 receivers = ~12D) — secondary, same as 2-column scale
3. **Per-row availability** (source + 4 receivers in one row = ~12D) — tests cross-section spread

## Scaling Concerns from Exploration

- `JointEmbedder`: no hard dimension limit, concatenates per-channel Takens embeddings. Handles arbitrary channel count.
- `PersistenceAnalyzer`: Ripser computes distance matrix O(n²) regardless of dimension. No hard dim cap. But topology in 48D with 2000 points = noise.
- `BindingDetector`: embedding quality gate checks condition number. At 48D, condition numbers will be high → degeneracy warnings expected.
- `ConeDetector`: API is receiver-agnostic — `fit(source_ts, receiver_channels)` works with any number of channels. No code changes needed for grid.
- FNN dimension estimation: `max_dim=10` hardcoded per channel. 16 channels × up to 10D = up to 160D theoretical max (but typically 3-5D per channel for Aizawa).

## Reusable Existing Code

| What | Where |
|------|-------|
| `_aizawa_deriv()` | `att/synthetic/layered_network.py:22` — import for grid ODE |
| `get_rng()` | `att/config/seed.py` — seed management |
| `ConeDetector` | `att/cone/detector.py` — used as-is with per-column receiver lists |
| `layered_aizawa_network_symmetric` Frobenius scaling | `layered_network.py:134` — adapt for grid edge count |
| `TestLayeredNetwork` test pattern | `tests/test_cone_synthetic.py:63` — follow for grid tests |

## Task Tracker State

| ID | Task | Status |
|----|------|--------|
| #98 | Create grid network generator | in_progress (not yet started, just marked) |
| #99 | Add trim_bins to availability_profile | pending |
| #100 | Add grid visualization functions | pending |
| #101 | Write grid generator tests | pending |
| #102 | Write cone-on-grid integration tests | pending |
| #103 | Run grid experiments G0-G5 | pending |
| #104 | Analyze results and update docs | pending |

## Execution Order

```
#98 (grid generator)           ← start here
#101 (generator tests)         ← validate generator
  ↓
#99 (trim_bins) + #100 (viz)   ← parallel
#102 (integration tests)       ← after generator validated
  ↓
#103 (experiments G0→G5)       ← sequential within
  ↓
#104 (analysis & docs)         ← after all experiments
```

## 2-Column Experiment Results (Context for Grid Comparison)

| Experiment | Result |
|------------|--------|
| Exp 0: Cascade | PASS — lag C→A3=-115, A3→A5=-76, cross-column r=0.636 |
| Exp 1: Depth asymmetry | Col A: +1,745, Col B: +23,242, Control: -3,768. No 3-way emergence. |
| Exp 2: Core cone | Full slope=+42.25 (β₁: 431→787), CCA slope=-7.63. Cone in full embedding. |
| Exp 3: Coupling sweep | Asymmetry monotonic 0→+40,889. No inverted-U at ε=0.5. |
| Exp 4: Timescale ratio | Peak asymmetry at ratio=2.0 (+12,590). Extreme ratios negative. |
| Exp 5: Dir vs sym | Directed slope=+42.25 vs symmetric=+5.54 (7.6x steeper). |

## Files Modified This Session

| File | Change |
|------|--------|
| `.claude/plans/quirky-painting-rain.md` | Overwritten with grid scaling plan |
| `.claude/handoff/2026-03-29-grid-scaling-plan.md` | This file |

## Git State

- All cone prototype work committed: `5e45f63` "Cone prototype: ConeDetector, 5-node network, all 6 experiments complete"
- Pushed to `origin/master`
- Working tree clean (except this handoff and plan file)
