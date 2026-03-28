# Attractor Topology Toolkit (ATT)

**Topological analysis of dynamical attractors in neural and artificial systems.**

ATT is a Python library that extracts, compares, and visualizes the topological structure of chaotic attractors from multivariate time series. It introduces **joint-vs-marginal persistent homology on Takens-embedded coupled systems** -- a construction that isolates emergent coupling topology by comparing persistence diagrams of joint delay embeddings against their marginal counterparts. ATT provides clean infrastructure for attractor reconstruction, persistent homology, cross-system binding detection, surrogate-tested statistics, and head-to-head benchmarks against transfer entropy, phase-amplitude coupling, and cross-recurrence quantification.

## Installation

```bash
pip install att-toolkit
```

For EEG/MEG data support (requires MNE-Python):

```bash
pip install att-toolkit[eeg]
```

For the GUDHI backend (alpha/witness complexes):

```bash
pip install att-toolkit[gudhi]
```

For all optional dependencies:

```bash
pip install att-toolkit[all]
```

From source:

```bash
git clone https://github.com/musicofhel/attractor-topology-toolkit.git
cd attractor-topology-toolkit
pip install -e ".[dev]"
```

## Quick Start

### Topological fingerprinting of a Lorenz attractor

```python
from att.config import set_seed
from att.synthetic import lorenz_system
from att.embedding import TakensEmbedder
from att.topology import PersistenceAnalyzer

set_seed(42)

# Generate a Lorenz attractor
ts = lorenz_system(n_steps=10000, dt=0.01)

# Reconstruct phase space via Takens embedding
embedder = TakensEmbedder(delay="auto", dimension="auto")
cloud = embedder.fit_transform(ts[:, 0])

# Compute persistent homology
analyzer = PersistenceAnalyzer(max_dim=2)
diagrams = analyzer.fit_transform(cloud, subsample=1000)
analyzer.plot()
```

### Binding detection in coupled systems

```python
from att.config import set_seed
from att.synthetic import coupled_lorenz
from att.binding import BindingDetector

set_seed(42)

# Generate coupled Lorenz systems
ts_x, ts_y = coupled_lorenz(coupling=0.5)

# Detect emergent topology in the joint embedding
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

## API Overview

| Module | Description |
|--------|-------------|
| `att.config` | Deterministic seeding, YAML experiment configs |
| `att.synthetic` | Lorenz, Rossler, coupled oscillators (all seeded) |
| `att.embedding` | Takens delay embedding, AMI delay estimation, FNN dimension estimation, SVD validation |
| `att.topology` | Persistent homology via Ripser/GUDHI, persistence images/landscapes |
| `att.binding` | Joint-vs-marginal persistence image subtraction with significance testing |
| `att.transitions` | Sliding-window topology with CUSUM/PELT changepoint detection |
| `att.surrogates` | Phase-randomized and time-shuffled surrogates |
| `att.benchmarks` | Transfer entropy, PAC, CRQA comparison framework |
| `att.neuro` | EEG/MEG loaders and preprocessing (requires `mne`) |
| `att.viz` | Persistence diagrams, barcodes, Betti curves, 3D attractors, binding images |
| `att.cli` | Command-line interface for benchmark sweeps |

## What Makes ATT Different

ATT sits at the intersection of several existing research threads, none of which fully cover its construction:

| Prior Work | What It Does | What ATT Adds |
|------------|-------------|---------------|
| CCM (Sugihara et al.) | Joint delay embeddings for causal inference | Persistent homology on the joint manifold |
| R-Cross-Barcode / RTD-Lite | Cross-barcodes for graph comparison | Extension to VR complexes on Takens point clouds |
| Xi et al. (TE + directed PH) | PH on transfer entropy networks | PH on state-space embeddings, not derived networks |
| Giusti, Curto et al. | PH on neural correlation matrices | PH on time-evolving attractor dynamics |
| Sliding-window PH (Perea, Harer) | Topological time series analysis | Applied to bistable perception EEG (novel niche) |

## Citation

If you use ATT in your research, please cite:

```bibtex
@software{att2026,
  title = {Attractor Topology Toolkit: Joint-vs-Marginal Persistent Homology on Takens Embeddings},
  author = {{ATT Contributors}},
  year = {2026},
  url = {https://github.com/musicofhel/attractor-topology-toolkit},
}
```

## Documentation

- [ARCHITECTURE.md](./docs/ARCHITECTURE.md) -- System design, module contracts, data flow
- [ROADMAP.md](./docs/ROADMAP.md) -- Scope, sequence, completion criteria
- [RESEARCH.md](./docs/RESEARCH.md) -- Theoretical foundations, novelty analysis, literature map
- [API.md](./docs/API.md) -- Library API reference
- [DATA.md](./docs/DATA.md) -- Dataset guide, download instructions, preprocessing

## License

MIT. See [LICENSE](./LICENSE) for details.
