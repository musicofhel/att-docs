# ARCHITECTURE.md

## System Overview

ATT is a three-layer system: a computational core (Python library), an analysis layer (notebooks + CLI), and a presentation layer (interactive demo + blog). Each layer is independently useful. The library works without the demo. The demo works with pre-computed results.

```
┌──────────────────────────────────────────────────────────────────┐
│                       PRESENTATION LAYER                         │
│           Streamlit / Static Plotly Interactive Explorer           │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│   │ 3D Cloud │  │  Persistence │  │ Binding Timeline         │  │
│   │ Viewer   │  │  Diagrams    │  │ + Transitions + Benchmarks│  │
│   └──────────┘  └──────────────┘  └──────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│                        ANALYSIS LAYER                            │
│                 Notebooks / CLI / Batch Jobs                     │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │  Pipeline Orchestrator (load → embed → topo → bind →     │   │
│   │  benchmark → detect transitions → surrogate test →       │   │
│   │  export JSON)                                            │   │
│   └──────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│                      COMPUTATIONAL CORE                          │
│   ┌───────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────────┐  │
│   │ embedding │ │ topology │ │ binding │ │   transitions    │  │
│   └───────────┘ └──────────┘ └─────────┘ └──────────────────┘  │
│   ┌───────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────────┐  │
│   │ synthetic │ │  neuro   │ │   viz   │ │   benchmarks     │  │
│   └───────────┘ └──────────┘ └─────────┘ └──────────────────┘  │
│   ┌───────────┐ ┌──────────┐ ┌─────────┐                        │
│   │surrogates │ │  config  │ │  cone   │                        │
│   └───────────┘ └──────────┘ └─────────┘                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Module Contracts

### `att.config`

**Purpose**: Reproducibility infrastructure. Manages global random seeds and experiment configuration.

```python
def set_seed(seed: int) -> None:
    """Set global random seed for all stochastic operations.
    Seeds NumPy, SciPy, and Python's random module.
    Must be called before any computation for deterministic results.
    """

def load_config(path: str) -> dict:
    """Load experiment configuration from YAML.
    Keys: seed, embedding, topology, binding, benchmarks, transitions, surrogates.
    Returns validated config dict.
    """

def save_config(config: dict, path: str) -> None:
    """Save experiment parameters for reproducibility."""
```

**Design rationale**: When running coupling sweeps (10 coupling values × 100 surrogates × 4 methods), stochastic variation between runs makes debugging impossible without deterministic seeding. Every synthetic generator, every surrogate method, and every subsampling operation accepts an optional `seed` parameter that defaults to the global seed state. The YAML config captures all parameters so any result can be exactly reproduced.

---

### `att.embedding`

**Purpose**: Transform raw time series into phase-space point clouds suitable for topological analysis. Supports heterogeneous delay parameters for multi-system joint embeddings where subsystems may operate on different timescales.

**Core class**: `TakensEmbedder`

```python
class TakensEmbedder:
    def __init__(self, delay: int | str = "auto", dimension: int | str = "auto"):
        """
        delay: int or "auto" (estimated via average mutual information)
        dimension: int or "auto" (estimated via false nearest neighbors)
        """

    def fit(self, X: np.ndarray) -> "TakensEmbedder":
        """Estimate delay and dimension parameters from data."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embed 1D time series into d-dimensional delay coordinates.
        Input:  (n_samples,)
        Output: (n_samples - (dimension-1)*delay, dimension)
        """

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
```

**Core class**: `JointEmbedder`

```python
class JointEmbedder:
    def __init__(self, delays: list[int] | str = "auto",
                 dimensions: list[int] | str = "auto"):
        """
        Construct joint delay embeddings for multi-system analysis.

        delays: per-channel delays or "auto" (estimated independently per channel via AMI).
                CRITICAL: Using a single shared delay for systems with different
                timescales violates embedding theorem assumptions and produces
                degenerate manifolds. Per-channel estimation is the default.
        dimensions: per-channel embedding dimensions or "auto" (via FNN per channel).
        """

    def fit(self, channels: list[np.ndarray]) -> "JointEmbedder":
        """Estimate per-channel delay and dimension parameters."""

    def transform(self, channels: list[np.ndarray]) -> np.ndarray:
        """Construct joint delay vectors by concatenating per-channel embeddings.
        Input:  list of 1D arrays, each (n_samples,)
        Output: (n_valid_samples, sum(dimensions))
        n_valid_samples is determined by the most restrictive channel.
        """

    def transform_marginals(self, channels: list[np.ndarray]) -> list[np.ndarray]:
        """Return individual embedded point clouds for marginal comparison."""
```

**Additional utilities**:
- `estimate_delay(X, method="ami")` — Average Mutual Information for optimal delay
- `estimate_dimension(X, delay, method="fnn")` — False Nearest Neighbors for minimal embedding dimension
- `svd_embedding(X, delay, dimension)` — SVD-projected delay embedding for noise reduction
- `validate_embedding(cloud, expected_dim, condition_threshold=1e4)` — SVD of centered point cloud matrix (n_points × dimension). Returns condition number (σ_max/σ_min), full singular value spectrum, effective rank, boolean degenerate flag, and warning string. Threshold of 1e4 was calibrated on coupled Rössler-Lorenz: per-channel delays yield condition numbers 10–500, shared delays yield 1e4–1e8. Zero overlap on test systems. Adjustable for noisier data.

**Design decision — per-channel delays**: The deep research confirmed that a single shared τ across systems with different intrinsic timescales can undersample one component and oversample another, violating the effective independence of delay coordinates (Hart, Novak et al., "SToPS," Chaos 2023). `JointEmbedder` estimates τ independently per channel by default. The `TakensEmbedder` remains for single-system use.

**Invariant**: Output point clouds preserve the topology of the original attractor (Sauer-Yorke-Casdagli "Embedology" guarantee when dimension ≥ 2d+1 where d is the box dimension of the attractor). For joint embeddings, the generalized multivariate Takens theorem applies: generic vector-valued observations with sufficient total delay dimension embed the coupled attractor.

---

### `att.topology`

**Purpose**: Compute persistent homology on point clouds and produce topological summaries in multiple representations (diagrams, images, landscapes).

**Core class**: `PersistenceAnalyzer`

```python
class PersistenceAnalyzer:
    def __init__(self, max_dim: int = 2, backend: str = "ripser",
                 use_witness: bool = False, n_landmarks: int = 500):
        """
        max_dim: maximum homology dimension to compute (0=components, 1=loops, 2=voids)
        backend: "ripser" or "gudhi"
        use_witness: if True, use witness complex with landmark selection for large clouds.
                     Reduces simplex count dramatically for n > 2000 points.
        n_landmarks: number of landmarks for witness complex
        """

    def fit_transform(self, cloud: np.ndarray, subsample: int | None = None,
                      seed: int | None = None) -> dict:
        """Compute persistence diagrams.
        subsample: if int, randomly select this many points before computing.
        seed: random seed for subsampling (uses global state if None).
        Returns: {
            "diagrams": list[np.ndarray],          # one (n_features, 2) per dimension
            "betti_curves": list[np.ndarray],       # Betti(filtration_param)
            "persistence_entropy": list[float],     # entropy per dimension
            "bottleneck_norms": list[float],        # max lifetime per dimension
            "persistence_images": list[np.ndarray], # (resolution, resolution) per dimension
            "persistence_landscapes": list[np.ndarray], # (n_layers, n_grid) per dimension
        }
        """

    def distance(self, other: "PersistenceAnalyzer", metric: str = "bottleneck") -> float:
        """Compute distance between persistence diagrams.
        metric: "bottleneck", "wasserstein_1", "wasserstein_2"
        """

    def to_image(self, resolution: int = 50, sigma: float = 0.1) -> list[np.ndarray]:
        """Convert diagrams to persistence images (stable vectorization).
        Returns one (resolution, resolution) array per homology dimension.
        """

    def to_landscape(self, n_layers: int = 5, n_grid: int = 100) -> list[np.ndarray]:
        """Convert diagrams to persistence landscapes."""

    def plot(self, kind: str = "diagram") -> matplotlib.figure.Figure:
        """kind: 'diagram', 'barcode', 'betti_curve', 'landscape', 'image'"""
```

**Design decisions**:
- **Subsample consistency**: Different random subsamples of the same point cloud produce different persistence diagrams. In binding pipelines, this is dangerous — diagram differences between joint and marginal clouds could come from subsampling noise rather than topology. ATT enforces: within a single `BindingDetector.fit()` call, all three clouds (marginal X, marginal Y, joint) use the same subsample seed. Each surrogate iteration increments the seed. This ensures residuals reflect topological differences, not sampling differences. Recommended sizes: 1000 for routine analysis, 2000+ for publication figures.
- **Witness complexes**: For sliding-window analysis with 300+ windows of 1k-5k points each, standard VR persistence can hit 10+ minutes total. Witness complexes (Tucker-Foltz, "Witness Complexes for Time Series Analysis") reduce this to seconds by selecting landmark subsets. Default off for accuracy on small clouds, available for large-scale runs.
- **Persistence images as first-class output**: The binding module needs vectorized diagram representations for subtraction and statistical testing. Images and landscapes are computed eagerly when `fit_transform` is called, avoiding redundant computation downstream.
- **Ripser as default**: Ripser outperforms GUDHI for H0+H1 by ~10x. GUDHI is available for H2+ via alpha complexes, and for settings where Ripser hits memory limits.

**Invariant**: Persistence diagrams are stable under small perturbations of the input (Cohen-Steiner, Edelsbrunner & Harer, 2007). Bottleneck distance between diagrams bounds the Hausdorff distance between the underlying spaces.

---

### `att.binding`

**Purpose**: Detect topological features that emerge in joint embeddings but are absent from marginals. This is the novel contribution. Uses persistence image subtraction with configurable baselines, embedding quality gating, and surrogate-tested significance.

**Core class**: `BindingDetector`

```python
class BindingDetector:
    def __init__(self, max_dim: int = 1, method: str = "persistence_image",
                 image_resolution: int = 50, image_sigma: float = 0.1,
                 baseline: str = "max", embedding_quality_gate: bool = True):
        """
        Analyze cross-attractor topological binding.

        method: "persistence_image" (default, recommended) or "diagram_matching"
          - "persistence_image": Convert diagrams to images, compute residual
            against baseline. Residual mass is binding score.
          - "diagram_matching": Optimal partial matching (Hungarian) between joint and
            marginal diagrams. Unmatched features in joint are excess topology.
            More interpretable per-feature, but harder to test statistically.

        baseline: How to construct the marginal comparison for persistence_image method.
          - "max" (default): R = I_joint - max(I_x, I_y) pointwise.
            Conservative choice. A feature must exceed BOTH marginals at every pixel
            to count as emergent. Minimizes false positives. Equivalent to asking:
            "What does the joint have that neither marginal alone explains?"
          - "sum": R = I_joint - (I_x + I_y) pointwise.
            More sensitive. Reports features exceeding combined marginal mass.
            Higher false positive rate — overlapping moderate marginal features
            at the same (birth, death) location may mask genuine joint excess.
            Use when you want to detect subtle coupling at the cost of specificity.

        embedding_quality_gate: If True (default), validate_embedding is called on the
          joint point cloud before computing persistence. If the condition number
          exceeds 1e4, raises EmbeddingDegeneracyWarning. This prevents a common
          failure mode where a degenerate embedding produces topological artifacts
          that mimic binding signal. Set to False only when you have independently
          verified embedding quality (e.g., via SVD denoising or manual inspection).
        """

    def fit(self, X: np.ndarray, Y: np.ndarray,
            joint_embedder: JointEmbedder | None = None,
            marginal_embedder_x: TakensEmbedder | None = None,
            marginal_embedder_y: TakensEmbedder | None = None,
            subsample: int | None = None, seed: int | None = None,
            n_ensemble: int = 1) -> "BindingDetector":
        """
        X, Y: 1D time series (must be same length, simultaneous recordings)

        If embedders are None, uses JointEmbedder("auto", "auto") for joint and
        TakensEmbedder("auto", "auto") for each marginal. Per-channel delay estimation
        is critical — see JointEmbedder docs.

        n_ensemble: If >1 and subsample is provided, runs K independent
        persistence+scoring passes with different subsample seeds. binding_score()
        returns ensemble mean. Variance reduction is modest (~28% to ~24% CV at K=10).
        Ensemble scores available via .ensemble_scores property.
        .confidence_interval(confidence=0.95) returns bootstrap CI.

        Pipeline:
          1. Marginal embedding of X alone
          2. Marginal embedding of Y alone
          3. Joint embedding of [X; Y]
          4. [If embedding_quality_gate] Validate all three embeddings
          5. Persistence of marginal X, marginal Y, joint (K times if ensemble)
          6. Excess topology via chosen method and baseline
        """

    def binding_score(self) -> float:
        """Scalar summary of excess topological features.
        For persistence_image method: L1 norm of positive residual image.
        For diagram_matching method: total persistence of unmatched features.
        """

    def binding_features(self) -> dict:
        """Per-dimension breakdown of excess topology."""

    def binding_image(self) -> list[np.ndarray]:
        """The residual persistence image (joint minus baseline of marginals).
        Positive regions indicate emergent topology. Only for persistence_image method.
        """

    def embedding_quality(self) -> dict:
        """Returns validate_embedding output for the joint cloud.
        Includes condition_number, effective_rank, degenerate flag, warning.
        Useful for diagnosing unexpected binding scores.
        """

    def test_significance(self, n_surrogates: int = 100,
                          method: str = "phase_randomize") -> dict:
        """Permutation test against null hypothesis of no coupling.
        method: "phase_randomize" (preserves power spectrum, destroys coupling)
                "time_shuffle" (destroys all temporal structure)
                "twin_surrogate" (preserves attractor topology, breaks cross-coupling;
                                  most conservative null)
        Returns: {
            "p_value": float,
            "observed_score": float,
            "surrogate_scores": np.ndarray,
            "significant": bool,  # at α=0.05
            "z_score": float,  # (observed - surrogate_mean) / surrogate_std
            "calibrated_score": float,  # observed - surrogate_mean
            "surrogate_mean": float,
            "surrogate_std": float,
            "embedding_quality": dict,  # validate_embedding output
        }

        NOTE: Raw binding scores have a structural positive baseline that grows
        with data size. Z-scores are the correct calibrated measure. The method
        has zero power for same-timescale coupling (Lorenz-Lorenz); it is
        selective to heterogeneous-timescale coupling (Rossler-Lorenz).
        """

    def plot_comparison(self) -> matplotlib.figure.Figure:
        """Three-panel: marginal X | joint (excess highlighted) | marginal Y"""

    def plot_binding_image(self) -> matplotlib.figure.Figure:
        """Heatmap of the residual persistence image. Red = emergent topology."""
```

**Mathematical basis (updated)**:

The binding detection has three layers:

1. **Geometric**: Joint delay embedding of coupled systems X and Y produces a point cloud in ℝ^(d_x + d_y) whose topology reflects interaction structure that is absent from the marginal clouds in ℝ^d_x and ℝ^d_y individually. When the coupled attractor is not a simple product A_x × A_y (i.e., when there is genuine dynamical coupling), the joint cloud contains homological features — loops, voids — that arise only from the entanglement of the two subsystems.

2. **Baseline choice**: The residual computation R = I_joint - baseline(I_x, I_y) requires choosing how to combine the marginal images. The pointwise max baseline (`"max"`) is the default because it asks the most conservative question: "Does the joint have features exceeding the stronger marginal at every point?" This minimizes false positives from coincidental marginal feature overlap. The pointwise sum baseline (`"sum"`) asks: "Does the joint have more total feature mass than the marginals combined?" This is more sensitive but conflates additive marginal contributions with genuine emergent structure. In testing on coupled Lorenz systems, the max baseline produces cleaner surrogate separation (larger gap between observed score and 95th percentile of null distribution), which is why it is the default. However, for weakly coupled systems where sensitivity matters more than specificity, the sum baseline may detect coupling that the max baseline misses. Both are exposed as options.

3. **Statistical**: Raw persistence image subtraction can yield false positives from finite-sample effects regardless of baseline choice. Surrogate testing (phase-randomized surrogates preserve marginal spectral properties but destroy inter-system coupling) establishes a null distribution of binding scores. Only binding scores exceeding the 95th percentile of the surrogate distribution are reported as significant.

**Embedding quality gate rationale**:

The entire binding pipeline depends on ALL THREE embeddings (marginal X, marginal Y, joint) faithfully representing their respective attractor topologies. A degenerate marginal embedding produces a garbage persistence image — and since the binding score is a residual between joint and marginal images, garbage in either marginal corrupts the result just as badly as a degenerate joint.

The gate runs `validate_embedding` on all three point clouds before any persistence computation. It checks the condition number of the centered point cloud matrix (σ_max/σ_min). A condition number > 1e4 (default threshold, calibrated on Rössler-Lorenz; see `validate_embedding` docs) indicates near-degeneracy. When triggered on any cloud, the gate raises `EmbeddingDegeneracyWarning` with:
- Which embedding(s) failed (marginal X, marginal Y, joint, or multiple)
- The condition number and singular value spectrum for the failed cloud(s)
- A recommendation to inspect per-channel delays, apply SVD denoising, or use manual parameters

The gate is on by default. It can be disabled for advanced users who have independently validated their embeddings (e.g., by examining the singular value spectrum directly).

**Relationship to prior art**:
- This is NOT the Mayer-Vietoris spectral sequence approach (overkill for our point cloud sizes, designed for distributed computation).
- This is NOT cross-barcodes / R-Cross-Barcode (which compare two filtrations on the same space). We compare filtrations on DIFFERENT spaces (marginal vs joint).
- This is closest in spirit to the CCM manifold comparison but operates at the homology level rather than nearest-neighbor cross-prediction.

---

### `att.benchmarks`

**Purpose**: Head-to-head comparison of topological binding against established coupling measures. This module exists because no published benchmarks compare TDA-based coupling detection to standard methods. Filling this gap is low-effort, high-signal.

```python
class CouplingBenchmark:
    def __init__(self, methods: list[str] = None,
                 normalization: str = "rank"):
        """
        methods: subset of ["binding_score", "transfer_entropy", "pac", "crqa"]
        Default: all four.

        normalization: How to normalize scores for cross-method visual comparison.
          - "rank" (default): Rank-transform each method's scores to [0, 1] over
            the sweep. Preserves monotonicity and ordering without distortion from
            different score magnitudes. Recommended for visual comparison.
          - "minmax": Per-method min-max scaling to [0, 1] over the sweep.
            Preserves relative spacing but sensitive to outliers at endpoints.
          - "zscore": Per-method z-scoring (mean=0, std=1). Best for statistical
            comparison but less intuitive visually.
          - "none": Raw scores. Use when comparing methods with known comparable
            scales, or for detailed per-method analysis.

        The choice of normalization affects visual impressions significantly.
        Rank normalization is the default because it provides the most honest
        visual comparison — it shows whether methods agree on ORDERING without
        implying they agree on MAGNITUDE. Minmax normalization can make methods
        look more similar or different depending on whether their response curves
        are linear or sigmoidal.
        """

    def register_method(self, name: str, fn: callable) -> None:
        """Add a custom coupling method to the benchmark.
        fn: callable(X, Y) -> float. Included in all subsequent .run() and .sweep() calls.
        This is the extensibility hook — future researchers plug in their coupling
        measure and get comparison figures against all built-in methods for free.
        """

    def run(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> dict:
        """Compute all coupling measures (built-in + registered) on the same pair.
        Returns: {
            "binding_score": float,
            "transfer_entropy_xy": float,     # X→Y
            "transfer_entropy_yx": float,     # Y→X
            "pac": float,                     # phase-amplitude coupling
            "crqa_determinism": float,
            "crqa_laminarity": float,
            # + any registered methods
        }
        """

    def sweep(self, generator_fn, coupling_values: np.ndarray,
              seed: int | None = None,
              transient_discard: int = 1000) -> pd.DataFrame:
        """Run all methods across a coupling parameter sweep.
        generator_fn: callable(coupling, seed) -> (X, Y). Receives generator seed.
        seed: if provided, the SAME seed is passed to generator_fn for all coupling
              values (identical initial conditions across sweep). Surrogate generation
              within each coupling value uses seed+i to decorrelate null distributions.
        transient_discard: steps to drop from start of each time series (default 1000).
              Removes chaotic transient sensitivity.
        Returns DataFrame with columns: coupling, method, score, score_normalized
        """

    def plot_sweep(self, results: pd.DataFrame,
                   use_normalized: bool = True) -> matplotlib.figure.Figure:
        """Overlay all methods on one coupling sweep plot.
        If use_normalized=True, plots score_normalized column.
        Otherwise plots raw score (separate y-axis per method recommended).
        """
```

**Implementation notes**:
- Transfer entropy via PyInform or direct estimation (Kraskov estimator)
- PAC via modulation index (Tort et al., 2010)
- CRQA via PyRQA or direct implementation with Theiler window
- Normalization is applied per-method across the sweep AFTER all raw scores are computed. The raw scores are always preserved in the `score` column.

---

### `att.surrogates`

**Purpose**: Generate null-hypothesis surrogates for statistical testing of all binding and transition results. All methods accept an optional `seed` parameter for reproducibility.

```python
def phase_randomize(X: np.ndarray, n_surrogates: int = 100,
                    seed: int | None = None) -> np.ndarray:
    """Amplitude-Adjusted Phase Randomization (AAFT).
    Preserves marginal distribution and power spectrum.
    Destroys nonlinear coupling and phase relationships.
    Returns: (n_surrogates, n_samples)
    """

def time_shuffle(X: np.ndarray, n_surrogates: int = 100,
                 block_size: int | None = None,
                 seed: int | None = None) -> np.ndarray:
    """Block-shuffled surrogates. Destroys all temporal structure beyond block_size.
    If block_size is None, shuffles individual samples (iid null).
    """

def twin_surrogate(X: np.ndarray, n_surrogates: int = 100,
                   embedder: TakensEmbedder | None = None,
                   seed: int | None = None) -> np.ndarray:
    """Recurrence-based twin surrogates (Thiel et al., 2006).
    Preserves attractor topology while breaking cross-system coupling.
    More conservative null than phase randomization.
    """
```

---

### `att.transitions`

**Purpose**: Detect attractor-switching events via sliding-window topological analysis.

**Core class**: `TransitionDetector`

```python
class TransitionDetector:
    def __init__(self, window_size: int = 500, step_size: int = 50,
                 max_dim: int = 1, use_witness: bool = False):
        """Sliding window persistent homology.
        use_witness: use witness complexes for speed on large windows
        """

    def fit_transform(self, X: np.ndarray) -> dict:
        """
        Returns: {
            "topology_timeseries": list[dict],  # PersistenceAnalyzer output per window
            "distances": np.ndarray,  # bottleneck distances between consecutive windows
            "image_distances": np.ndarray,  # L2 distances between consecutive persistence images
            "changepoints": list[int],  # detected transition indices
            "transition_scores": np.ndarray,  # continuous transition strength signal
        }
        """

    def detect_changepoints(self, method: str = "cusum") -> list[int]:
        """method: 'cusum', 'pelt', 'threshold'"""

    def test_significance(self, n_surrogates: int = 50) -> dict:
        """Test whether detected transitions are significantly different from
        what would be expected from surrogates of the same process.
        """

    def plot_timeline(self) -> matplotlib.figure.Figure:
        """Topology evolution heatmap with detected transitions marked."""
```

**Key insight**: Attractor switches manifest as sudden jumps in bottleneck distance between consecutive persistence diagrams. The persistence image distance provides a smoother, more statistically tractable alternative for the same signal.

---

### `att.cone`

**Purpose**: Detect conical projection geometry in directed attractor networks. Extends ATT from pairwise symmetric binding to directed multi-node projection geometry in layered networks.

**Core class**: `ConeDetector`

```python
class ConeDetector:
    def __init__(self, n_depth_bins: int = 5, max_dim: int = 1,
                 n_quantiles: int = 20, cca_components: int = 3):
        """
        n_depth_bins: Number of bins along the projection axis for depth slicing.
        max_dim: Maximum homology dimension for persistence computation.
        n_quantiles: Number of source-state quantiles for axis estimation.
        cca_components: Number of CCA dimensions for coupling-influence subspace.
        """

    def fit(self, source_ts, receiver_channels, source_embedder=None,
            receiver_embedder=None) -> "ConeDetector":
        """Embed source (TakensEmbedder) and receivers (JointEmbedder),
        estimate projection axis, compute CCA coupling-influence subspace."""

    def estimate_projection_axis(self) -> np.ndarray:
        """Conditional-mean PCA: bin source state into quantiles, compute
        conditional means of receiver cloud, take first PC as axis."""

    def slice_at_depth(self, depth_bin: int) -> np.ndarray:
        """Extract cross-section point cloud at a given depth bin."""

    def availability_profile(self, subspace="full", subsample=2000) -> dict:
        """Core output: Betti numbers as a function of depth along the
        projection axis. Returns depths, betti_0, betti_1, diagrams,
        persistence_entropy, trend_slope, is_monotonic."""

    def coupling_influence_subspace(self) -> np.ndarray:
        """CCA between embedded source and receiver cloud. Returns the
        low-dimensional subspace where source maximally predicts receivers."""

    def depth_asymmetry(self, source_ts, shallow_ts, deep_ts) -> dict:
        """Compare [source; shallow] vs [source; deep] binding scores
        using BindingDetector. Returns shallow/deep scores + asymmetry."""

    def full_chain_emergence(self, source_ts, shallow_ts, deep_ts) -> dict:
        """Test whether 3-way joint [C; A3; A5] has emergent topology
        beyond max of pairwise bindings. Uses PI subtraction."""
```

**Visualization** (`att.cone.visualize`):
```python
def plot_availability_profile(profile, ax=None, show_betti_0=False) -> Figure
def plot_coupling_sweep(coupling_values, profiles) -> Figure
def plot_cross_sections(slices, diagrams, depths) -> Figure
def plot_subspace_comparison(profile_full, profile_cca) -> Figure
def plot_cascade_verification(trajectories, max_lag=500) -> Figure
def plot_directed_vs_symmetric(profile_directed, profile_symmetric) -> Figure
```

**Design decisions**:
- ConeDetector **composes with** (not inherits from) BindingDetector — different interface (multi-channel directed vs pairwise symmetric)
- Conditional-mean PCA for axis estimation captures nonlinear axis structure that a simple centroid-to-centroid line would miss
- Equal-count quantile bins for depth slicing ensure statistical power per bin
- Betti counting uses a 10%-of-max-persistence threshold to filter topological noise
- CCA subspace slicing indexes into pre-computed array (no re-projection needed per bin)

**Key finding**: The cone appears in the CCA coupling-influence subspace (Betti_1 increases with depth) but not in the full Takens embedding. This supports the theoretical claim that the cone is a low-dimensional feature embedded in a high-dimensional attractor state space.

---

### `att.synthetic`

**Purpose**: Generate well-characterized chaotic time series for validation. All generators accept an optional `seed` parameter. If None, uses global seed state from `set_seed()`.

```python
def lorenz_system(n_steps=10000, dt=0.01, sigma=10, rho=28, beta=8/3,
                  initial=None, noise=0.0, seed=None) -> np.ndarray:
    """Returns (n_steps, 3) array."""

def rossler_system(n_steps=10000, dt=0.01, a=0.2, b=0.2, c=5.7,
                   initial=None, noise=0.0, seed=None) -> np.ndarray:

def coupled_lorenz(n_steps=10000, dt=0.01, coupling=0.1,
                   seed=None) -> tuple[np.ndarray, np.ndarray]:
    """Two Lorenz systems with diffusive coupling.
    coupling=0 → independent. coupling→1 → synchronization.
    Returns (ts_x, ts_y) — each (n_steps, 3).
    """

def coupled_rossler_lorenz(n_steps=10000, dt=0.01, coupling=0.1,
                           seed=None) -> tuple[np.ndarray, np.ndarray]:
    """Rössler coupled to Lorenz — different intrinsic timescales.
    Tests heterogeneous delay handling in JointEmbedder.
    Returns (ts_rossler, ts_lorenz).
    """

def switching_rossler(n_steps=20000, dt=0.01, switch_every=5000,
                      seed=None) -> np.ndarray:
    """Rössler with parameter switches — ground truth transitions."""

def coupled_oscillators(n_oscillators=3, coupling_matrix=None, n_steps=10000,
                        seed=None) -> np.ndarray:
    """Kuramoto model variant with chaotic individual dynamics."""

def aizawa_system(n_steps=10000, dt=0.01, alpha=0.95, beta=0.7, gamma=0.6,
                  delta=3.5, epsilon=0.25, zeta=0.1, initial=None,
                  noise=0.0, seed=None) -> np.ndarray:
    """Aizawa attractor: spherical geometry with helical escape tube.
    Chosen for cleaner cross-sections than Lorenz/Rossler.
    Returns (n_steps, 3)."""

def layered_aizawa_network(n_steps=80000, coupling_source=0.15,
                           coupling_down=0.15, dt_layer2=0.005,
                           dt_layer3=0.008, dt_layer5=0.012,
                           seed=None) -> dict[str, np.ndarray]:
    """5-node directed network: C->A3->A5, C->B3->B5.
    Per-layer timescale separation, xy-only diffusive coupling.
    Returns dict keyed by node name ('C','A3','B3','A5','B5')."""

def layered_aizawa_network_symmetric(n_steps=80000, coupling_source=0.15,
                                     coupling_down=0.15,
                                     seed=None) -> dict[str, np.ndarray]:
    """All-to-all symmetric variant for Experiment 5 control.
    Frobenius-norm matched to directed topology."""
```

---

### `att.neuro`

**Purpose**: Load, preprocess, and segment neural time series for topological analysis.

```python
class EEGLoader:
    def __init__(self, dataset: str, subject: int | str):
        """
        dataset: OpenNeuro accession ID or preset name
        Presets:
          'katyal_rivalry' → SSVEP binocular rivalry, 64-channel (primary target)
          'bistable_necker' → Necker cube, 32-channel (backup)
          'auditory_bistable' → Auditory streaming (alternate modality backup)
        """

    def load(self) -> mne.io.Raw:
        """Load raw data via MNE."""

    def preprocess(self, bandpass=(1, 45), notch=50, reference="average",
                   ica_reject: bool = True) -> mne.io.Raw:
        """Standard preprocessing pipeline with optional ICA artifact rejection."""

    def to_timeseries(self, picks: list[str] | None = None) -> np.ndarray:
        """Extract (n_channels, n_samples) array."""

    def epoch(self, events, tmin=-2.0, tmax=2.0) -> np.ndarray:
        """Epoch around events. Returns (n_epochs, n_channels, n_samples).
        Default window: -2s to +2s around perceptual switch report.
        """

    def get_switch_events(self) -> np.ndarray:
        """Extract perceptual switch event markers (button presses)."""

    def get_channel_groups(self) -> dict:
        """Return standard channel groupings for regional analysis.
        Returns: {
            'parietal': ['P3', 'Pz', 'P4'],
            'occipital': ['O1', 'Oz', 'O2'],
            'frontal': ['F3', 'Fz', 'F4'],
        }
        """

    @staticmethod
    def get_fallback_params(band: str = "broadband") -> dict:
        """Return empirically grounded embedding parameters for when AMI/FNN
        estimation is unreliable on noisy EEG.

        These are NOT arbitrary. They are drawn from published EEG nonlinear
        analysis literature (Stam 2005, Lehnertz & Elger 1998) and validated
        against known attractor properties of band-filtered neural signals.

        band:
          "broadband" (1-45 Hz): delay=10 (≈39ms @ 256Hz), dimension=5
          "alpha" (8-13 Hz): delay=8 (≈31ms), dimension=4
          "theta_alpha" (4-13 Hz): delay=12 (≈47ms), dimension=5
          "gamma" (30-45 Hz): delay=3 (≈12ms), dimension=5

        Returns: {"delay": int, "dimension": int, "bandpass": tuple, "note": str}
        """
```

**EEG embedding strategy**: AMI/FNN estimation on raw EEG frequently fails or gives unreliable results due to noise, non-stationarity, and volume conduction. The `EEGLoader` provides an automatic workflow:

```python
def embed_channel(self, channel_data: np.ndarray, band: str = "broadband",
                  condition_threshold: float = 1e4) -> tuple[np.ndarray, dict]:
    """Embed a single EEG channel with automatic fallback.

    1. Try TakensEmbedder("auto", "auto")
    2. Run validate_embedding on result
    3. If degenerate: log warning, re-embed with get_fallback_params(band)
    4. Return (point_cloud, metadata_dict)

    metadata_dict includes:
      "method": "auto" | "fallback"
      "delay": int
      "dimension": int
      "condition_number": float
      "fallback_reason": str | None  # e.g. "AMI no minimum" or "condition > 1e4"

    In batch mode (multiple channels/subjects), this runs per-channel without
    raising exceptions. Failures are logged and the fallback is used silently.
    The metadata dict lets you audit which channels used auto vs fallback
    after the batch completes.
    """
```

This avoids the failure mode of debugging parameter estimation and neural topology simultaneously. In batch runs over 20 channels × 3 subjects, every channel gets a valid embedding — some via auto-estimation, some via fallback — with a full audit trail.

---

### `att.viz`

**Purpose**: Publication-quality static plots and JSON export for the frontend.

```python
def plot_persistence_diagram(diagrams, ax=None, colormap="viridis") -> Figure
def plot_persistence_image(images, ax=None, colormap="hot") -> Figure
def plot_barcode(diagrams, ax=None) -> Figure
def plot_betti_curve(betti_curves, ax=None) -> Figure
def plot_attractor_3d(cloud, color_by="time", backend="plotly") -> Figure
def plot_binding_comparison(detector: BindingDetector) -> Figure
def plot_binding_image(detector: BindingDetector) -> Figure
def plot_transition_timeline(detector: TransitionDetector) -> Figure
def plot_benchmark_sweep(results: pd.DataFrame) -> Figure
def plot_surrogate_distribution(observed, surrogates, ax=None) -> Figure

def export_to_json(results: dict, path: str) -> None:
    """Export all computed results as JSON for frontend consumption."""

def load_from_json(path: str) -> dict
```

---

## Data Flow (Updated)

```
Raw Time Series (1+ channels)
     │
     ├─── Single channel ──────────────┐
     │                                  │
     ▼                                  ▼
┌───────────────┐              ┌────────────────┐
│ TakensEmbedder│              │ JointEmbedder  │ (2+ channels)
│ (per-channel τ│              │ (per-channel τ, │
│  and d)       │              │  heterogeneous) │
└───────────────┘              └────────────────┘
     │                                  │
     ▼                                  ├──── Marginals ──┐
Point Cloud ℝ^d                         │                  │
     │                                  ▼                  ▼
     │                          Joint Cloud ℝ^Σd    Marginal Clouds
     │                                  │                  │
     │                                  └────────┬─────────┘
     │                                           ▼
     │                                  ┌────────────────┐
     │                                  │ QUALITY GATE   │
     │                                  │ (all 3 clouds  │
     │                                  │  cond# < 1e4?) │
     │                                  └───────┬────────┘
     │                                    pass? │ fail→warn
     ├──────────────┐                     ┌─────┴─────┐
     ▼              ▼                     ▼           ▼
┌──────────┐  ┌──────────┐      ┌───────────┐  ┌───────────┐
│ Topology │  │Transitions│      │ Topology  │  │ Topology  │
│ (single) │  │(windowed) │      │ (joint)   │  │(marginals)│
└──────────┘  └──────────┘      └───────────┘  └───────────┘
     │              │                   │                │
     ▼              ▼                   └────────┬───────┘
Persistence   Changepoints                      ▼
Diagrams +    Transition           ┌──────────────────────────┐
                                   │ Binding Detection        │
                                   │ baseline = max|sum       │
                                   │ R = I_joint - base(I_x,  │
                                   │                    I_y)  │
                                   │ + surrogate testing      │
                                   └──────────────────────────┘
                                            │
                                   ┌────────┴────────┐
                                   ▼                  ▼
                            Binding Scores      ┌───────────┐
                            + p-values +        │Benchmarks │
                            embedding quality   │(TE,PAC,   │
                                                │ CRQA)     │
                                                │norm=rank  │
                                                └───────────┘
                                                      │
                                                      ▼
                                               Comparison
                                               DataFrame
                                               (raw + normalized)
```

## Demo Architecture

Lightweight interactive demo via Streamlit or static HTML exports. No custom React build — the compounding value is in the library, preprint, and blog, not in a bespoke frontend.

**Option A: Streamlit app** (preferred if hosting is available)
- Single `demo/app.py` that loads pre-computed JSON
- Plotly panels: 3D attractor, persistence diagram, binding image, benchmark sweep
- Sidebar: select demo dataset, toggle parameters
- Deploys to Streamlit Cloud for free

**Option B: Static HTML exports** (zero-dependency fallback)
- `att.viz.export_interactive(results, path)` generates self-contained HTML files with embedded Plotly
- One HTML per visualization: attractor, persistence, binding, benchmark
- Linked from README, loadable in any browser
- No hosting, no server, no dependencies to rot

```
demo/
├── app.py                 # Streamlit app (Option A)
├── export_static.py       # Generate static HTML (Option B)
└── sample_data/           # Pre-computed JSON for demo
    ├── lorenz_demo.json
    ├── coupling_sweep.json
    ├── binding_demo.json
    └── benchmark_sweep.json
```

## CLI Architecture

```bash
# Run a coupling sweep from a YAML config
att benchmark run --config configs/coupled_lorenz_sweep.yaml --output results/sweep.csv --plot results/sweep.png

# Quick binding detection on two time series files
att bind --x data/ts_x.npy --y data/ts_y.npy --surrogates 100 --output results/binding.json
```

Entry point registered in `pyproject.toml`:
```toml
[project.scripts]
att = "att.cli:main"
```

## Performance Constraints (Updated)

| Operation | Target | Strategy |
|-----------|--------|----------|
| Takens embedding (10k points, d=3) | < 50ms | NumPy vectorized |
| Joint embedding (10k × 2 channels) | < 100ms | NumPy vectorized |
| Embedding validation (condition number) | < 10ms | NumPy SVD |
| Ripser H0+H1 (1k subsampled points) | < 2s | Ripser.py C++ backend |
| Ripser H0+H1+H2 (1k points) | < 10s | Ripser.py or GUDHI alpha |
| Witness complex H0+H1 (5k points, 500 landmarks) | < 3s | GUDHI witness |
| Persistence image (50×50, from diagram) | < 50ms | Persim vectorized |
| Sliding window (20k points, 500 window, 50 step) | < 120s | Parallel via joblib + witness |
| Transfer entropy (10k points) | < 5s | PyInform or Kraskov |
| CRQA (10k points) | < 10s | PyRQA |
| Surrogate generation (100 surrogates, 10k points) | < 2s | Vectorized FFT |

## Testing Strategy (Updated)

- **Synthetic ground truth**: Lorenz has known topology (H1 ≈ 2 prominent loops). Validate that PersistenceAnalyzer recovers this.
- **Heterogeneous timescale test**: `coupled_rossler_lorenz` with shared vs per-channel delays. Per-channel must produce lower embedding degeneracy (measured by condition number of delay matrix).
- **Embedding quality gate test**: Deliberately create a degenerate embedding (shared delay on heterogeneous system) and verify the gate fires.
- **Baseline comparison test**: Run binding detection with both `"max"` and `"sum"` baselines on coupled Lorenz at coupling=0 and coupling=0.5. Verify that `"max"` produces fewer false positives at coupling=0.
- **Coupling sweep**: coupled_lorenz with coupling 0→1 should show binding_score increasing from 0 to ~0.5, then decreasing toward 1.0 (synchronization collapse). Test: R² > 0.9 for monotone increase on lower half of sweep; peak at intermediate coupling confirmed.
- **Binding significance**: At coupling=0, surrogate test should return p > 0.05 (no false positive). At coupling=0.5, p < 0.05 (detects real coupling).
- **Benchmark consistency**: On coupled Lorenz sweep, binding_score curve shape should qualitatively match transfer entropy curve (both monotone increasing with coupling).
- **Normalization test**: Verify that rank normalization preserves method ordering. Verify that minmax normalization maps to [0,1].
- **Transition injection**: switching_rossler should produce changepoints at known parameter switch times ± window_size tolerance.
- **Numerical stability**: Add Gaussian noise at SNR 10, 5, 3. Topology should degrade gracefully (stability theorem).
- **Reproducibility test**: Run full pipeline twice with same seed. Verify bitwise-identical results. Run with different seeds. Verify different results.
- **Subsample consistency test**: Run BindingDetector with subsample=1000 twice with the same seed. Verify identical binding scores. Run with different seeds. Verify different scores but same sign (both positive or both near zero). Verify that within a single .fit() call, joint and marginal persistence use the same subsample seed.
- **EEG fallback params test**: Verify that fallback parameters produce non-degenerate embeddings on synthetic signals band-filtered to EEG ranges.
- **Plugin test**: Register a custom coupling method via `register_method()`. Verify it appears in `.run()` output and `.sweep()` DataFrame. Verify normalization includes it.
- **CLI test**: `att benchmark run --config test_config.yaml` produces expected CSV output.
- **Round-trip**: embed → compute → export JSON → load JSON → verify fields match.
