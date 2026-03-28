# Attractor Topology Toolkit (ATT)

[![PyPI version](https://img.shields.io/pypi/v/att-toolkit)](https://pypi.org/project/att-toolkit/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/att-toolkit)](https://pypi.org/project/att-toolkit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![CI](https://github.com/musicofhel/att-docs/actions/workflows/ci.yml/badge.svg)](https://github.com/musicofhel/att-docs/actions/workflows/ci.yml)

**Joint-vs-marginal persistent homology on Takens embeddings for coupling detection in dynamical systems.**

<!-- TODO: Add animated GIF demo here (e.g. binding detection sweep on coupled Lorenz) -->

```bash
pip install att-toolkit
```

---

ATT extracts, compares, and visualizes the topological structure of chaotic attractors from multivariate time series. Its core construction -- **joint-vs-marginal persistent homology on Takens-embedded coupled systems** -- isolates emergent coupling topology by comparing persistence diagrams of joint delay embeddings against their marginal counterparts. The library provides clean infrastructure for attractor reconstruction, persistent homology, cross-system binding detection, surrogate-tested statistics, and head-to-head benchmarks against transfer entropy, phase-amplitude coupling, and cross-recurrence quantification.

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
att benchmark run \
  --config configs/coupled_lorenz_sweep.yaml \
  --output results/sweep.csv \
  --plot results/sweep.png
```

## Installation

```bash
pip install att-toolkit           # core
pip install att-toolkit[eeg]      # + MNE-Python for EEG/MEG
pip install att-toolkit[gudhi]    # + GUDHI backend (alpha/witness complexes)
pip install att-toolkit[all]      # everything
```

From source:

```bash
git clone https://github.com/musicofhel/att-docs.git
cd att-docs
pip install -e ".[dev]"
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

## Documentation

See [`docs/`](./docs/) for Sphinx API reference and quickstart tutorial.

<!-- TODO: Link to hosted docs once published -->
<!-- TODO: Link to blog post -->

## Citation

If you use ATT in your research, please cite:

```bibtex
@software{att2026,
  title   = {Attractor Topology Toolkit: Joint-vs-Marginal Persistent
             Homology on Takens Embeddings},
  author  = {{ATT Contributors}},
  year    = {2026},
  url     = {https://github.com/musicofhel/att-docs},
}
```

<!-- TODO: Replace with preprint citation once posted to arXiv -->

## License

MIT. See [LICENSE](./LICENSE) for details.
