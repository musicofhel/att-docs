# Attractor Topology Toolkit (ATT)

**Topological analysis of dynamical attractors in neural and artificial systems.**

ATT is a Python library and interactive visualization tool that extracts, compares, and visualizes the topological structure of chaotic attractors from multivariate time series. It bridges computational neuroscience and AI interpretability by providing clean infrastructure for attractor reconstruction, persistent homology, cross-system binding detection, and benchmarking against established coupling measures.

## Novelty Statement

ATT introduces **joint-vs-marginal persistent homology on Takens-embedded coupled systems** — a construction that, as of March 2026, does not exist in the literature. Adjacent work exists in multivariate delay embeddings for causal inference (CCM/Sugihara), cross-barcodes for graph comparison (R-Cross-Barcode/RTD-Lite), and directed persistent homology on transfer entropy networks (Xi et al.), but nobody has combined joint Takens embeddings with persistence image subtraction to isolate emergent coupling topology. ATT fills that gap with validated tooling, surrogate-tested statistics, and head-to-head benchmarks against transfer entropy, phase-amplitude coupling, and cross-recurrence quantification.

## What It Does

- **Attractor Reconstruction**: Takens delay embedding with per-channel delay estimation (AMI) and dimension estimation (FNN), supporting heterogeneous timescales in multi-system embeddings
- **Embedding Quality Gating**: Automatic degeneracy detection via condition number analysis on delay matrices, preventing downstream topological artifacts from bad embeddings
- **Topological Fingerprinting**: Persistent homology (Betti numbers, persistence diagrams, persistence images/landscapes) via Ripser/GUDHI with witness complex support for large point clouds
- **Cross-Attractor Binding**: Detects excess topological features in joint embeddings absent from marginals, using persistence image subtraction with configurable baselines (max, sum) and permutation-tested significance
- **Transition Detection**: Identifies attractor-switching events via sliding-window topology with CUSUM/PELT changepoint detection
- **Coupling Benchmarks**: Head-to-head comparison of topological binding score against transfer entropy, phase-amplitude coupling, and cross-recurrence quantification on identical coupled systems, with explicit normalization (rank, minmax, z-score) for fair comparison
- **Surrogate Testing**: Phase-randomized and time-shuffled surrogates for statistical validation of all binding and transition results
- **Reproducibility**: Deterministic seeding and YAML experiment configs across all stochastic operations
- **Interactive Visualization**: 3D attractor geometry, persistence diagrams, binding timelines (Streamlit/Plotly)

## Quickstart

```bash
pip install att-toolkit
```

Or from source:
```bash
git clone https://github.com/yourusername/attractor-topology-toolkit.git
cd attractor-topology-toolkit
pip install -e ".[dev]"
```

```python
from att.config import set_seed
from att.embedding import TakensEmbedder
from att.topology import PersistenceAnalyzer
from att.synthetic import lorenz_system

set_seed(42)

# Generate a Lorenz attractor
ts = lorenz_system(n_steps=10000, dt=0.01)

# Reconstruct phase space
embedder = TakensEmbedder(delay="auto", dimension="auto")
cloud = embedder.fit_transform(ts[:, 0])

# Compute persistent homology
analyzer = PersistenceAnalyzer(max_dim=2)
diagrams = analyzer.fit_transform(cloud, subsample=1000)
analyzer.plot()
```

```python
from att.config import set_seed
from att.binding import BindingDetector
from att.synthetic import coupled_lorenz

set_seed(42)

# Detect emergent topology in coupled systems
ts_x, ts_y = coupled_lorenz(coupling=0.5)
detector = BindingDetector(max_dim=1, method="persistence_image")
detector.fit(ts_x[:, 0], ts_y[:, 0])
print(f"Binding score: {detector.binding_score():.4f}")
print(f"Embedding quality: {detector.embedding_quality()}")
print(f"Significant: {detector.test_significance(n_surrogates=100)}")
```

### CLI

```bash
# Run a coupling sweep from a YAML config
att benchmark run --config configs/coupled_lorenz_sweep.yaml --output results/sweep.csv --plot results/sweep.png
```

## Project Structure

```
attractor-topology-toolkit/
├── att/                    # Core library
│   ├── config/             # Seed management, YAML configs
│   ├── embedding/          # Takens, time-delay, SVD, per-channel delays, quality validation
│   ├── topology/           # Persistent homology, persistence images/landscapes
│   ├── binding/            # Cross-attractor joint topology + embedding gate + surrogates
│   ├── transitions/        # Sliding-window attractor switching
│   ├── benchmarks/         # TE, PAC, CRQA comparison + register_method() plugin interface
│   ├── synthetic/          # Lorenz, Rössler, coupled oscillators (seeded)
│   ├── neuro/              # EEG/MEG loaders, preprocessing, fallback parameters
│   ├── surrogates/         # Phase randomization, time shuffle, twin surrogates
│   ├── cli/                # `att benchmark run` CLI entry points
│   └── viz/                # Plotting utilities (matplotlib/plotly)
├── demo/                   # Streamlit app or static HTML interactive explorer
├── notebooks/              # Walkthrough notebooks
├── tests/                  # Pytest suite
├── configs/                # Experiment YAML configs for reproducibility
├── data/                   # Sample datasets + download scripts
├── docs/                   # Sphinx documentation
├── paper/                  # Preprint drafts and figures
├── blog/                   # Blog post drafts
└── pyproject.toml
```

## Requirements

- Python 3.10+
- NumPy, SciPy, scikit-learn
- Ripser (ripser.py) or GUDHI
- Persim (persistence images, diagram distances)
- MNE-Python (for EEG/MEG data)
- PyInform or JIDT (for transfer entropy benchmarks)
- Plotly, Matplotlib
- PyYAML (for experiment configs)
- Streamlit (optional, for interactive demo)

## Documentation

- [ARCHITECTURE.md](./docs/ARCHITECTURE.md) — System design, module contracts, data flow
- [ROADMAP.md](./docs/ROADMAP.md) — Scope, sequence, completion criteria
- [RESEARCH.md](./docs/RESEARCH.md) — Theoretical foundations, novelty analysis, literature map
- [API.md](./docs/API.md) — Library API reference
- [DATA.md](./docs/DATA.md) — Dataset guide, download instructions, preprocessing
- [BLOG.md](./blog/BLOG.md) — Blog post draft structure

## Prior Art & Positioning

ATT sits at the intersection of several existing research threads, none of which fully cover its construction:

| Prior Work | What It Does | What ATT Adds |
|------------|-------------|---------------|
| CCM (Sugihara et al.) | Joint delay embeddings for causal inference | Persistent homology on the joint manifold |
| R-Cross-Barcode / RTD-Lite | Cross-barcodes for graph comparison | Extension to VR complexes on Takens point clouds |
| Xi et al. (TE + directed PH) | PH on transfer entropy networks | PH on state-space embeddings, not derived networks |
| Giusti, Curto et al. | PH on neural correlation matrices | PH on time-evolving attractor dynamics |
| Sliding-window PH (Perea, Harer) | Topological time series analysis | Applied to bistable perception EEG (novel niche) |

## License

MIT
