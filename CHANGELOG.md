# Changelog

## v0.2.0-dev (2026-03-29)

Cone prototype: directed cross-layer projection geometry via depth-stratified topology.

### Cone Detection (`att.cone`)

- `ConeDetector` with 6 methods: fit, estimate_projection_axis, slice_at_depth, availability_profile, coupling_influence_subspace, depth_asymmetry, full_chain_emergence
- Conditional-mean PCA for projection axis estimation
- CCA-based coupling-influence subspace analysis
- 6 visualization functions: availability profile, coupling sweep, cross-sections, subspace comparison, cascade verification, directed vs symmetric

### New Synthetic Systems

- Aizawa attractor generator (`aizawa_system()`) — spherical geometry with helical escape tube
- 5-node directed layered network (`layered_aizawa_network()`) — C→A3→A5, C→B3→B5 with per-layer timescale separation
- Symmetric all-to-all control network (`layered_aizawa_network_symmetric()`) — Frobenius-norm matched

### Tests

- 35 cone tests (22 synthetic + 13 ConeDetector), 267 total

### Key Early Finding

- Cone appears in CCA coupling-influence subspace (Betti_1 slope > 0) but NOT in full Takens embedding — supports the theoretical claim that the cone is a low-dimensional feature

## v0.1.1 (2026-03-28)

Phase 11-12: Multi-subject EEG validation + ROADMAP gap closure.

### Multi-Subject EEG (Phase 11-12)

- Transition detection batch: N=80, precision 94.1%±14.9%, recall 40.6%±13.9%
- Cross-region binding batch: N=79, population mean rho=0.10 (p=0.001)
- Parallel processing: `--workers N` flag with ProcessPoolExecutor
- Precision/recall bug fix in `evaluate_alignment()`

### Tutorials & Infrastructure

- Tutorial notebook: Lorenz end-to-end walkthrough (ROADMAP 1.13)
- Tutorial notebook: heterogeneous timescales demo (ROADMAP 1.14)
- Root Makefile: test, lint, docs, demo, notebook-test targets
- Preprint: 22 pages with N=79 binding + N=80 transition results

## v0.1.0 (2026-03-28)

Initial release. 9 development phases, 232 tests, 21-page preprint.

### Core

- Takens delay embedding with AMI delay + FNN dimension estimation
- Joint embedding with per-channel delay/dimension and SVD validation
- Persistent homology via Ripser (H0, H1, H2) with persistence images and landscapes
- Bottleneck and Wasserstein distance metrics

### Binding Detection

- Joint-vs-marginal persistence image subtraction (max and sum baselines)
- Diagram matching via Hungarian algorithm (hyperparameter-free alternative)
- Embedding quality gate (condition number check, prevents degenerate-embedding artifacts)
- Ensemble binding (`n_ensemble` parameter) with confidence intervals
- Z-score calibration against surrogate null distributions
- Significance testing with AAFT, time-shuffle, and twin surrogates

### Benchmarks

- Transfer entropy, phase-amplitude coupling, cross-recurrence quantification
- `CouplingBenchmark` with `sweep()` and `register_method()` plugin interface
- Rank, min-max, and z-score normalization for cross-method comparison

### Transition Detection

- Sliding-window persistent homology with L2 image distances
- CUSUM changepoint detection (threshold = mean + 2*std)
- `TransitionDetector` for tracking topological regime changes over time

### Neural Data

- EEG loader (MNE-Python) with auto/fallback embedding parameters
- Literature-grounded fallback params for standard EEG bands
- Real EEG validation: 100% precision, 80.5% recall on binocular rivalry data (N=1)

### Synthetic Systems

- Lorenz, Rossler, coupled Lorenz, coupled Rossler-Lorenz
- Switching Rossler (bistable dynamics)
- Kuramoto oscillators (oscillatory coupling)

### Infrastructure

- CLI: `att benchmark run --config sweep.yaml`
- PyPI package: `pip install att-toolkit`
- Sphinx docs (furo theme) with GitHub Pages deployment
- GitHub Actions CI (lint + test on Python 3.10/3.11/3.12 + build)
- 232 tests (195 fast + 37 slow validation)
- Streamlit demo (3-page interactive app)
- 21-page preprint with 9 experiments, 9 figures, 2 tables

### Key Findings

- Binding score is selectively sensitive to heterogeneous-timescale coupling
- Zero power for same-timescale coupling (Lorenz-Lorenz)
- Structural positive baseline grows with data size (not finite-sample noise)
- ~30% coefficient of variation; ensemble reduces to ~24% at K=10
- Z-scores against surrogates are the correct calibrated measure
- N-body pairwise scores contaminated by indirect coupling (2.5x inflation)
- Kuramoto: binding *decreases* with oscillatory coupling (77x reduction)
