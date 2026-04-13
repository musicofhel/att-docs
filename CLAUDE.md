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
| `att.llm` | `HiddenStateLoader`, `LayerwiseAnalyzer`, `TopologicalFeatureExtractor` | LLM hidden-state topology: layer-wise PH, CROCKER, zigzag, intrinsic dimension, attention-hidden binding |
| `att.topology.spectral` | `knn_graph_laplacian()`, `spectral_distance_matrix()` | kNN Laplacian spectral distances for high-dimensional PH |

## Experimental Results (14 experiments)

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
| 13 | LLM hidden-state topology | Qwen2.5-1.5B MATH-500 | **z = 8.12 terminal-layer significance**; AUROC = 0.580 |
| 14 | Topo-confidence validation | Qwen2.5-1.5B MATH-500 | **Holdout AUROC = 0.948** [0.898, 0.986] |

Experiments 10-12 were discovered by CORAL automated multi-agent search. CORAL is the experiment runner; ATT is the subject under test. Each CORAL agent writes scripts calling ATT's API directly (`BindingDetector`, `TransitionDetector`, `PersistenceAnalyzer`, `TakensEmbedder`). Results validate ATT's Künneth cross-term prediction, extend its domain coverage to cardiac ECG, and close the EEG recall gap.

Experiments 13-14 extend ATT to LLM hidden-state analysis. Persistent homology on transformer point clouds reveals terminal-layer topological concentration (z=8.12), universal H1 non-monotonic entropy across 4 models, and attention-hidden coupling that decreases with problem difficulty. CORAL-engineered features push correctness prediction from AUROC 0.580 to holdout AUROC 0.948, orthogonal to output entropy (r=0.062). GO decision issued for Pathways 2-4. See [topo-confidence](https://github.com/musicofhel/topo-confidence).

## Domain of Applicability

The binding score is selectively sensitive to coupling between systems with *different intrinsic timescales* (e.g., Rossler-Lorenz). It has **zero power** for same-timescale coupling (e.g., Lorenz-Lorenz). Always use z-score calibration against surrogates rather than raw binding scores.

## Testing

```bash
python -m pytest tests/ -x -q   # 375 tests
```

## Key Directories

| Path | Contents |
|------|----------|
| `att/` | Source code (~20 modules including `att/llm/`) |
| `att/llm/` | LLM hidden-state topology (8 modules, added in Experiment 13) |
| `tests/` | Test suite (375 tests) |
| `paper/` | LaTeX preprint (`preprint.tex`, `references.bib`) |
| `docs/` | Sphinx site (Furo theme), auto-deployed to GitHub Pages |
| `coral_archive/` | CORAL experiment artifacts (108 attempts, 53 agent notes, 4 summaries) |
| `configs/` | YAML experiment configurations |
| `figures/` | Generated plots |
