# ATT Phase 7: Test Expansion + Validation Experiments

**Date**: 2026-03-28
**Repo**: `~/att-docs` (github.com/musicofhel/att-docs, private)
**Commit**: `e12c6ce` on `origin/master`

## What Was Done

### 1. Edge-Case Tests (55 new tests across 11 files)

**test_binding.py** — `TestBindingEdgeCases` (12 tests):
Invalid method/baseline raises, very short series, identical signals, NaN input, empty H1 diagrams, n_surrogates=0 (p=1.0), time_shuffle/twin_surrogate significance, invalid surrogate method, diagram_matching significance raises NotImplementedError, plot_comparison returns Figure.

**test_benchmarks.py** — `TestBenchmarkEdgeCases` (10 tests):
Constant/zero inputs for TE/PAC/CRQA, short signals, `_delay_embed` empty output, `_count_line_points` direct test, `_discretize` identical values, unknown normalization raises.

**test_transitions.py** — `TestTransitionEdgeCases` (7 tests):
Threshold method, custom threshold=0, CUSUM on iid data, invalid method raises, input too short raises, single window, 1D without embedding params raises.

**test_topology.py** — `TestTopologyEdgeCases` (5 tests):
Single-point cloud, two-point cloud, to_image explicit ranges, to_image before fit raises, distance with minimal clouds.

**test_surrogates.py** — `TestSurrogateEdgeCases` (4 tests):
5-sample phase randomize, 3-sample time shuffle, minimal twin surrogate, block_size > signal length.

**test_embedding.py** — `TestEmbeddingEdgeCases` (5 tests):
Constant signal delay/dimension, single-channel JointEmbedder, short series raises, degenerate cloud detection.

**test_viz.py** — `TestVizEdgeCases` (4 tests):
plot_surrogate_distribution, plot_benchmark_sweep (with/without score_normalized, with/without ax).

**test_cli.py** — `TestCLIEdgeCases` (3 tests):
Invalid YAML, unknown system, minimal config.

**test_config.py** — `TestGetSeed` (2 tests):
get_seed after set_seed.

**test_neuro.py** — `TestNeuroEdgeCases` (3 tests):
Preprocessing pipeline, get_events returns None, unsupported format raises.

### 2. Validation Experiments (23 tests in test_validation.py → results/)

All marked `@pytest.mark.slow`. Each writes CSV/JSON to `results/`.

**TestReproducibilityAndVariance** (6 tests):
- Binding score CV: 0.24 (coupling=0.3), 0.35 (coupling=0 and 0.5) — method is noisy (~30% variance)
- False positive rate: 0/20 at coupling=0 — no false positives
- False negative rate: 20/20 at coupling=0.5 with n=3000, 19 surrogates — ZERO statistical power at small sample sizes

**TestSampleSizeSensitivity** (3 tests):
- Works down to n_steps=600 (100 points after transient discard)
- Score doesn't stabilize cleanly — CV at n=2000 is 0.178, at n=8000 is 0.251

**TestEmbeddingParameterSensitivity** (4 tests):
- 41% max deviation from perturbing delay ±50%
- 43% max deviation from perturbing dimension ±1
- Auto vs manual (delay=15, dim=3): scores within same order of magnitude

**TestMethodComparison** (3 tests):
- PI subtraction and diagram matching: Spearman rho=0.2 (barely correlated, measure different things)
- Matching scores (~600-770) are order of magnitude larger than PI scores (~37-107)

**TestNBodyBinding** (3 tests):
- 3-body Rössler: scores DON'T cleanly order by coupling strength
- Binding is perfectly symmetric (ratio=1.0 with fixed seed)

**TestCrossSystemGeneralization** (4 tests):
- 3/3 systems show correct coupling direction (Lorenz, Rössler-Lorenz, Kuramoto)
- Kuramoto effect dramatic: 77x higher binding at zero coupling vs strong (synchronization collapse)

### 3. Bug Fix

`BindingDetector.test_significance()` crashed when `method="diagram_matching"` because `_compute_surrogate_score()` accessed `self._birth_range`, `self._persistence_range`, and `self._images_x` — all `None` for diagram matching. Fixed: early `NotImplementedError` guard in `test_significance()`.

## Test Suite

216 total: 189 non-slow + 23 slow validation + 4 existing slow. All green in ~15min total.

## Key Findings Summary

| Metric | Value | Implication |
|--------|-------|-------------|
| Score CV | 0.24-0.35 | ~30% run-to-run variance — method is noisy |
| FP rate | 0% (n=20) | Good: no false coupling detection |
| FN rate | 100% (n=20, n_surr=19) | Bad: zero power at small n — needs more data or surrogates |
| Embedding sensitivity | ~40% deviation | Scores fragile to parameter choice |
| PI vs matching rho | 0.2 | Methods barely agree — measure different aspects |
| Cross-system direction | 3/3 correct | Method generalizes across system types |
| N-body ordering | Incorrect | Pairwise scores don't rank by coupling strength |
| Min viable length | n_steps=600 | Method runs but quality unknown at tiny sizes |

## Files Created

| File | Description |
|------|-------------|
| `tests/test_validation.py` | 23 validation experiment tests |
| `results/*.csv`, `results/*.json` | 22 experiment result files (in .gitignore) |
| `.claude/handoff/2026-03-28-phase7-test-expansion.md` | This file |

## Files Modified

| File | Changes |
|------|---------|
| `att/binding/detector.py` | NotImplementedError guard in test_significance() for diagram_matching |
| `tests/test_binding.py` | +12 edge-case tests (TestBindingEdgeCases) |
| `tests/test_benchmarks.py` | +10 edge-case tests (TestBenchmarkEdgeCases) |
| `tests/test_transitions.py` | +7 edge-case tests (TestTransitionEdgeCases) |
| `tests/test_topology.py` | +5 edge-case tests (TestTopologyEdgeCases) |
| `tests/test_surrogates.py` | +4 edge-case tests (TestSurrogateEdgeCases) |
| `tests/test_embedding.py` | +5 edge-case tests (TestEmbeddingEdgeCases) |
| `tests/test_viz.py` | +4 edge-case tests (TestVizEdgeCases) |
| `tests/test_cli.py` | +3 edge-case tests (TestCLIEdgeCases) |
| `tests/test_config.py` | +2 tests (TestGetSeed) |
| `tests/test_neuro.py` | +3 edge-case tests (TestNeuroEdgeCases) |
| `.gitignore` | Added `results/` |

## What's Next

Based on the findings, the method's main weaknesses are:

1. **Statistical power**: Significance test has zero power at n=3000 with 19 surrogates. Need to test with n=10000+ and 99+ surrogates to find the power threshold.
2. **Score variance**: 30% CV means individual scores are unreliable. Could ensemble over multiple subsamplings or use a more stable summary statistic.
3. **Embedding sensitivity**: 40% deviation means the method is fragile. Could investigate adaptive parameter selection or score normalization by embedding quality metrics.
4. **N-body failure**: Pairwise scores don't rank by coupling strength for 3-body Rössler. May need different approach for multi-oscillator systems.
5. **PI vs matching divergence**: The two methods measure different things. Could investigate what each captures and when to use which.
6. **Multi-subject EEG**: Only 1 subject downloaded. Need to download more and run batch pipeline.
