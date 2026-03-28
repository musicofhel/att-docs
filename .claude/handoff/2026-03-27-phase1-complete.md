# ATT Phase 1 Complete — Handoff

**Date**: 2026-03-27
**Repo**: `~/att-docs`
**Session**: Initial implementation

## What Was Done

Phase 1 of the Attractor Topology Toolkit (ATT) — a Python library for topological analysis of dynamical attractors via persistent homology. Novel contribution: joint-vs-marginal persistent homology on Takens-embedded coupled systems (confirmed novel as of March 2026).

### Modules Implemented

| Module | Files | What It Does |
|--------|-------|-------------|
| `att.config` | `seed.py`, `experiment.py` | `set_seed()`, `load_config()`, `save_config()` — deterministic reproducibility |
| `att.synthetic` | `generators.py` | 6 chaotic system generators: Lorenz, Rössler, coupled Lorenz, coupled Rössler-Lorenz, switching Rössler, coupled oscillators. All seeded. |
| `att.embedding` | `takens.py`, `joint.py`, `delay.py`, `dimension.py`, `validation.py` | `TakensEmbedder`, `JointEmbedder`, AMI delay estimation, FNN dimension estimation, `validate_embedding()` (condition number gating), `svd_embedding()` |
| `att.topology` | `persistence.py` | `PersistenceAnalyzer` — Ripser + GUDHI backends, persistence images (custom Gaussian impl, not persim), landscapes, Betti curves, entropy, bottleneck/Wasserstein distances |
| `att.viz` | `plotting.py` | Persistence diagrams, barcodes, Betti curves, persistence images, 3D attractor (Plotly + matplotlib), surrogate distribution, benchmark sweep, JSON export/load |

### Test Suite

57 tests, all green. `python -m pytest tests/ -v` in ~84s.

- `test_config.py` — seed determinism, config round-trip
- `test_synthetic.py` — shapes, reproducibility, coupling behavior
- `test_embedding.py` — AMI/FNN estimates, auto params, degeneracy detection
- `test_topology.py` — H1 recovery (Lorenz 2 loops), distances, PI comparison, reproducibility
- `test_viz.py` — all plots render, JSON round-trip

### Key Validation Numbers

- Lorenz delay: τ=16 (spec: 15±5) ✓
- Lorenz dimension: d=3 ✓
- Lorenz H1: 2 dominant features (top lifetime 5.738) ✓
- Bottleneck same-system: 0.66, cross-system (Lorenz vs Rössler): 2.87 ✓
- Bitwise reproducibility with `set_seed(42)` ✓

### Package

- `pyproject.toml` configured, `pip install -e .` works
- Dependencies installed: numpy, scipy, scikit-learn, ripser, persim, matplotlib, plotly, pyyaml
- Sample config at `configs/coupled_lorenz_sweep.yaml`

## What's Next — Phase 2 (Binding Detection + Benchmarks + Preprint)

This is the novel contribution. Per ROADMAP.md tasks 2.1–2.22 (67 hours estimated):

### Essential Path

1. **`att.binding.BindingDetector`** — persistence image subtraction (max/sum baselines), embedding quality gate on all 3 clouds, binding score (L1 norm of positive residual), binding image visualization
2. **`att.surrogates`** — phase-randomized (AAFT), time-shuffle, with seed support
3. **`BindingDetector.test_significance()`** — surrogate-tested p-values
4. **`att.benchmarks.CouplingBenchmark`** — transfer entropy (PyInform or Kraskov), PAC (modulation index), CRQA, `register_method()` plugin, `sweep()` with normalization (rank/minmax/zscore/none)
5. **`att benchmark run` CLI** — YAML config → sweep DataFrame → CSV + plot
6. **5 paper figures** — coupling sweep, baseline comparison, benchmark overlay, surrogate null, heterogeneous timescales
7. **Preprint draft** — methods section, experiment descriptions, figure captions

### Completion Criteria (from ROADMAP.md)

- Binding score <0.05 at coupling=0, p>0.05 (no false positive)
- Binding score detects coupling at ≥0.3 (p<0.05)
- Unimodal curve: increases to ~0.5, decreases toward 1.0 (sync collapse)
- Max baseline fewer false positives than sum at coupling=0
- Quality gate fires on degenerate joint, doesn't fire on good one
- Benchmark sweep: binding tracks TE qualitatively
- `register_method()` works for custom 5th method
- Full sweep reproducible with same seed

## Architecture Notes

- Persistence images are computed with custom Gaussian kernel code (not persim's PersistenceImager — it had pixel_size issues producing tiny images). See `att/topology/persistence.py:to_image()`.
- `persim.wasserstein()` only supports order 1 (no `order` kwarg). The `distance()` method treats all `wasserstein_*` metrics the same.
- All generators use `scipy.integrate.solve_ivp` with RK45 except `switching_rossler` and `coupled_oscillators` which use manual RK4/Euler for parameter-switching flexibility.
- The `att.binding`, `att.surrogates`, `att.benchmarks`, `att.transitions`, `att.neuro`, and `att.cli` modules have empty `__init__.py` files — ready for Phase 2+ implementation.

## Files

```
att-docs/
├── att/
│   ├── __init__.py
│   ├── config/{__init__,seed,experiment}.py
│   ├── embedding/{__init__,takens,joint,delay,dimension,validation}.py
│   ├── topology/{__init__,persistence}.py
│   ├── synthetic/{__init__,generators}.py
│   ├── viz/{__init__,plotting}.py
│   ├── binding/__init__.py          # empty — Phase 2
│   ├── surrogates/__init__.py       # empty — Phase 2
│   ├── benchmarks/__init__.py       # empty — Phase 2
│   ├── transitions/__init__.py      # empty — Phase 3
│   ├── neuro/__init__.py            # empty — Phase 3
│   └── cli/__init__.py              # empty — Phase 2
├── tests/
│   ├── test_config.py
│   ├── test_synthetic.py
│   ├── test_embedding.py
│   ├── test_topology.py
│   └── test_viz.py
├── configs/coupled_lorenz_sweep.yaml
├── pyproject.toml
├── README.md                        # original docs README
├── ARCHITECTURE.md, API.md, etc.    # design docs
└── data/, demo/, notebooks/, paper/, blog/, docs/  # empty dirs
```
