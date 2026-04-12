# ATT - Attractor Topology Toolkit

Python library for topological coupling detection in dynamical systems via joint-vs-marginal persistent homology on Takens embeddings.

## API Modules

| Module | Entry Point | Purpose |
|--------|-------------|---------|
| `att.config` | `set_seed()` | Deterministic seeding, YAML experiment configs |
| `att.synthetic` | `lorenz_system()`, `coupled_lorenz()` | Chaotic system generators (Lorenz, Rossler, Aizawa, Kuramoto, coupled, switching) |
| `att.embedding` | `TakensEmbedder` | Delay embedding with auto AMI/FNN estimation |
| `att.topology` | `PersistenceAnalyzer` | Persistent homology via Ripser/GUDHI, persistence images/landscapes |
| `att.binding` | `BindingDetector` | Joint-vs-marginal persistence image subtraction + significance testing |
| `att.transitions` | `TransitionDetector` | Sliding-window PH with CUSUM/PELT changepoint detection |
| `att.surrogates` | `phase_randomize()` | Phase-randomized and time-shuffled surrogates |
| `att.benchmarks` | `CouplingBenchmark` | Transfer entropy, PAC, CRQA comparison framework |
| `att.neuro` | `EEGLoader`, `embed_channel()` | MNE-Python EEG/MEG loader with auto-estimation and literature fallback |
| `att.cone` | `ConeDetector` | Directed cross-layer projection geometry (availability profiles) |
| `att.viz` | `plot_persistence_diagram()` | Publication-quality diagrams, barcodes, Betti curves, binding images |
| `att.cli` | `att benchmark run` | Command-line interface for benchmark sweeps |

## Experimental Results (12 experiments)

| # | Experiment | Domain | Key Metric |
|---|-----------|--------|------------|
| 1 | Coupling sweep | Coupled Lorenz | Binding rises then collapses at sync |
| 2 | Baseline comparison | Coupled Lorenz | Max baseline more conservative than sum |
| 3 | Benchmark comparison | Rossler-Lorenz | Binding > TE, PAC, CRQA |
| 4 | Surrogate validation | Coupled Lorenz | Controlled FPR with AAFT surrogates |
| 5 | Heterogeneous timescales | Rossler-Lorenz | Per-channel delay handles mismatch |
| 6 | Sliding-window transitions | Synthetic bistable EEG | CUSUM detects regime changes |
| 7 | Real EEG validation | Binocular rivalry SSVEP | 94.1% precision, 40.6% recall (N=80) |
| 8 | Cross-region binding | Binocular rivalry SSVEP | Oz-Pz modulates with perception (p=0.001) |
| 9 | Z-score calibration | Lorenz vs Rossler-Lorenz | Selective sensitivity to heterogeneous coupling |
| 10 | Cardiac arrhythmia | MIT-BIH ECG | **AUROC = 0.8014** (mixed TDA + signal) |
| 11 | Künneth confirmation | Reservoir computing | **F1 = 0.985** (inverted binding detection) |
| 12 | Perfect rivalry detection | Binocular rivalry SSVEP | **F1 = 1.000** (adaptive per-subject) |

Experiments 10-12 were discovered by CORAL automated multi-agent search. CORAL is the experiment runner; ATT is the subject under test. Each CORAL agent writes scripts calling ATT's API directly (`BindingDetector`, `TransitionDetector`, `PersistenceAnalyzer`, `TakensEmbedder`). Results validate ATT's Künneth cross-term prediction, extend its domain coverage to cardiac ECG, and close the EEG recall gap.

## Domain of Applicability

The binding score is selectively sensitive to coupling between systems with *different intrinsic timescales* (e.g., Rossler-Lorenz). It has **zero power** for same-timescale coupling (e.g., Lorenz-Lorenz). Always use z-score calibration against surrogates rather than raw binding scores.

## Testing

```bash
python -m pytest tests/ -x -q   # 267 tests
```

## Key Directories

| Path | Contents |
|------|----------|
| `att/` | Source code (12 modules) |
| `tests/` | Test suite (267 tests) |
| `paper/` | LaTeX preprint (`preprint.tex`, `references.bib`) |
| `docs/` | Sphinx site (Furo theme), auto-deployed to GitHub Pages |
| `coral_archive/` | CORAL experiment artifacts (108 attempts, 53 agent notes, 4 summaries) |
| `configs/` | YAML experiment configurations |
| `figures/` | Generated plots |
