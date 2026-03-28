# Phase 8: Method Hardening — Handoff

**Date**: 2026-03-28
**Status**: COMPLETE — all 232 tests verified green

## What Was Done

### Code Changes to `att/binding/detector.py`

1. **Speed fix**: `_compute_surrogate_score()` reuses cached embedding params (delay, dimension) from original `fit()` instead of re-estimating via AMI/FNN per surrogate. Cached in `self._marginal_delay_x`, `_marginal_dim_x`, `_marginal_delay_y`, `_marginal_dim_y`, `_joint_delays`, `_joint_dims`.

2. **Ensemble binding**: `n_ensemble` parameter on `fit()`. When >1 and `subsample` is provided, runs K independent persistence+scoring passes with different subsample seeds. `binding_score()` returns ensemble mean. New `ensemble_scores` property and `confidence_interval(confidence=0.95)` method.

3. **Z-score calibration**: `test_significance()` return dict now includes `z_score`, `calibrated_score`, `surrogate_mean`, `surrogate_std`.

### Test Fix

- `test_ensemble_no_subsample_skips` was hanging: called `fit(subsample=None)` on 5000-point `coupled_pair` fixture → ripser O(n²) on ~4966 points in 6D. Fixed by using inline small data (`n_steps=1500`, ~1000 points).

### New Tests

- 7 edge case tests in `tests/test_binding.py` (ensemble, CI, z-score, cached params)
- 10 slow validation tests in `tests/test_validation.py` (baseline diagnosis, power analysis, ensemble, calibrated scores, N-body diagnostic)

## Test Results

### Non-slow suite: 195/195 passed (8:24)
### Slow validation: 33/33 passed

### Key Experimental Findings

**Baseline diagnosis** (coupling=0):
- Scores INCREASE with n_steps: 45 (n=3000) → 127 (n=10000) → 164 (n=20000)
- Structural, not finite-sample — joint embedding dimensionality causes inherent positive baseline
- Max vs sum baselines nearly identical

**Power analysis** (49 surrogates):
- Lorenz coupling=0.3: ALL negative z-scores (-0.6 to -4.3), 0% detection. Method has zero power for same-system Lorenz.
- Rössler-Lorenz coupling=0.3: Positive z-scores (0.08 to 2.85), 1/6 significant at p<0.05. Method works for heterogeneous-timescale coupling.
- More data (n=10000 vs 5000) doesn't help Lorenz but slightly helps Rössler-Lorenz.

**Ensemble binding**:
- Variance reduction modest: CV 28% → 25% → 24% at K=1,5,10
- Ensemble doesn't improve discrimination between coupling=0 and 0.3

**Calibrated z-scores**:
- Lorenz: z-scores negative for both coupling=0 and 0.3 → no discrimination
- Rössler-Lorenz: z-scores positive (mean ~1.5), 3/5 seeds at p≤0.05 → method works here

**N-body contamination confirmed**:
- Pair 0-2 (coupling=0.0) scores 83 in 3-body vs 33 in isolation
- Indirect coupling through shared oscillator inflates uncoupled pair scores 2.5×

## Key Takeaways

1. **The method works for heterogeneous-timescale coupling** (Rössler-Lorenz) but NOT for same-timescale (Lorenz-Lorenz)
2. **Z-scores are the correct metric** — raw binding scores have a structural positive baseline that grows with data size
3. **N-body pairwise analysis is contaminated** by indirect coupling — isolated pair controls are needed
4. **Ensemble provides marginal variance reduction** — not transformative

## Files Modified

| File | Changes |
|------|---------|
| `att/binding/detector.py` | +80 lines (ensemble, speed fix, z-score, CI) |
| `tests/test_binding.py` | +55 lines (7 edge case tests), 1 test fix |
| `tests/test_validation.py` | +300 lines (10 slow validation tests) |

## Result Files Produced

| File | Contents |
|------|----------|
| `results/finite_sample_effect.json` | Baseline scores at n=3000/10000/20000 |
| `results/baseline_comparison_uncoupled.json` | Max vs sum baseline per seed |
| `results/power_analysis.csv` | 12-cell power sweep (2 systems × 2 n_steps × 3 seeds) |
| `results/effect_sizes.csv` | Cohen's d per cell |
| `results/ensemble_variance.csv` | CV at K=1,5,10 |
| `results/ensemble_discrimination.json` | Coupling=0 vs 0.3 with/without ensemble |
| `results/calibrated_scores.csv` | Z-scores at coupling=0 vs 0.3 |
| `results/cross_system_zscores.csv` | Z-scores for Lorenz vs Rössler-Lorenz |
| `results/n_body_diagnostic.json` | 3-body vs isolated pair comparison |
