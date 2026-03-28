# ATT Phase 6: Extensions — Twin Surrogates, Kuramoto, Diagram Matching, Docs Deployment

**Date**: 2026-03-28
**Repo**: `~/att-docs` (github.com/musicofhel/att-docs, private)

## What Was Done

### 1. Twin Surrogates (Thiel et al. 2006)

Added `twin_surrogate()` to `att/surrogates/core.py`. Algorithm: delay-embed input → recurrence matrix (threshold = 10th percentile of pairwise distances) → walk trajectory via random twin selection → extract first coordinate.

- Generates surrogates from the attractor's recurrence structure itself
- Tests specifically for deterministic coupling (complements AAFT spectrum-preserving and time-shuffle)
- Integrated into `BindingDetector.test_significance(method="twin_surrogate")` with handling for shorter output (embedding padding)
- 5 new tests: shape, reproducibility, coupling destruction, recurrence preservation, not-identical

### 2. Kuramoto Oscillator Generator

Added `kuramoto_oscillators()` to `att/synthetic/generators.py`. Classic Kuramoto model: dθ_i/dt = ω_i + (K/N) Σ sin(θ_j − θ_i) + noise. Euler integration (smooth dynamics).

- Returns (phases, signals) where signals = sin(phases)
- Tests toolkit on oscillatory (non-chaotic) dynamics — closer to neural oscillation coupling
- 5 new tests: shape, uncoupled drift, synchronized convergence (order parameter R), reproducibility, signals bounded
- Benchmark config: `configs/kuramoto_sweep.yaml`

**Key finding**: Binding score *decreases* with Kuramoto coupling (opposite of chaotic systems). Phase synchronization collapses the joint embedding onto a lower-dimensional manifold, reducing excess topology. The benchmark test asserts this correct direction.

### 3. Diagram Matching Binding Method

Added `method="diagram_matching"` to `BindingDetector`. Hungarian algorithm (scipy.optimize.linear_sum_assignment) for optimal matching between joint and concatenated marginal persistence diagrams.

- Augmented cost matrix: L∞ distances for real-to-real matches, persistence/2 diagonal penalties
- Hyperparameter-free (no sigma/resolution dependency unlike PI method)
- Binding score = total assignment cost summed across homology dimensions
- `binding_features()` returns per-dimension details (score, n_joint, n_baseline, n_unmatched)
- `binding_image()` raises RuntimeError (no images for this method)
- 10 new tests: coupled positive, uncoupled valid, sigma independence, features structure, score consistency, reproducibility, image raises, quality available, not fitted raises, H0-only

### 4. GitHub Pages Docs Deployment

Created `.github/workflows/docs.yml`:
- Build job: checkout → Python 3.12 → `pip install -e ".[docs]"` → `sphinx-build -b html docs docs/_build/html` → upload Pages artifact
- Deploy job: `actions/deploy-pages@v4` to `github-pages` environment
- Triggers on push to master and workflow_dispatch

Updated README.md with docs badge and link to https://musicofhel.github.io/att-docs/

**Note**: Needs GitHub Pages enabled in repo settings (Settings > Pages > Source: "GitHub Actions") for first deployment.

### 5. Blog Post Finalization

Polished `blog/post.md`:
- Fixed 6 figure paths (`figures/` → `../figures/`)
- Renumbered figures sequentially (1-6, removed gaps)
- Fixed `test_significance()` return type in code snippets (returns dict, not bool)
- Added hosted docs URL
- Fixed "Rossler" → "Rössler" spelling
- Updated `BLOG.md` to status tracker format

### 6. Integration & Lint Cleanup

- Added `TestKuramotoBenchmark` to `tests/test_benchmarks.py`
- Verified twin_surrogate wiring in `BindingDetector.test_significance()`
- Fixed all 40 ruff lint errors (37 auto-fixed + 3 manual: unused variable in plotting, unused `w` in catch_warnings, import ordering in test_neuro)
- Added `docs/_build/` to `.gitignore`

## Test Suite

134 passed, 4 deselected (slow/witness/changepoint). All green in 419s.

## Files Created

| File | Description |
|------|-------------|
| `.github/workflows/docs.yml` | GitHub Pages deployment workflow |
| `configs/kuramoto_sweep.yaml` | Kuramoto benchmark config |
| `.claude/handoff/2026-03-28-phase6-extensions.md` | This file |

## Files Modified

| File | Changes |
|------|---------|
| `att/surrogates/core.py` | Added `twin_surrogate()` function |
| `att/surrogates/__init__.py` | Export `twin_surrogate` |
| `att/synthetic/generators.py` | Added `kuramoto_oscillators()` function |
| `att/synthetic/__init__.py` | Export `kuramoto_oscillators` |
| `att/binding/detector.py` | Added `diagram_matching` method, `twin_surrogate` in `test_significance()`, `_fitted` flag |
| `att/viz/plotting.py` | Removed unused `residuals` variable |
| `tests/test_surrogates.py` | Added `TestTwinSurrogate` (5 tests) |
| `tests/test_synthetic.py` | Added `TestKuramoto` (5 tests) |
| `tests/test_binding.py` | Added `TestDiagramMatching` (10 tests), lint fix |
| `tests/test_benchmarks.py` | Added `TestKuramotoBenchmark` (1 test) |
| `tests/test_neuro.py` | Fixed import ordering |
| `blog/post.md` | Figure paths, code snippets, docs URL, polish |
| `BLOG.md` | Replaced outline with status tracker |
| `README.md` | Added docs badge and hosted URL |
| `.gitignore` | Added `docs/_build/` |

## What's Next

1. **Enable GitHub Pages**: Settings > Pages > Source: "GitHub Actions" (first push will trigger build)
2. **PyPI upload**: `twine upload dist/* -u __token__ -p pypi-TOKEN` (needs API token)
3. **Multi-subject EEG**: Download remaining subjects, run `scripts/batch_eeg.py`
4. **Publish blog**: dev.to / personal site, update README with link
5. **Phase 2b remaining**: Cross-barcode integration (R-Cross-Barcode on VR complexes)
6. **Preprint update**: Add twin surrogates, Kuramoto, diagram matching results (when ready for arXiv)
