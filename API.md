# API.md

## att.config

### `set_seed(seed: int) -> None`
Set global random seed for reproducibility across all stochastic operations (NumPy, SciPy, surrogate generation, subsampling). Call once at the start of any experiment or notebook.

### `load_config(path: str) -> dict`
Load experiment configuration from a YAML file. Supported keys: `seed`, `embedding`, `topology`, `binding`, `benchmarks`, `transitions`, `surrogates`.

### `save_config(config: dict, path: str) -> None`
Save current experiment parameters for reproducibility.

---

## att.embedding

### `TakensEmbedder(delay="auto", dimension="auto")`

Reconstruct a phase-space attractor from a scalar time series using time-delay embedding.

**Parameters**:
- `delay` (int | "auto"): Time steps between coordinates. "auto" estimates via AMI first minimum.
- `dimension` (int | "auto"): Number of delay coordinates. "auto" estimates via FNN.

**Methods**:

#### `.fit(X: np.ndarray) -> TakensEmbedder`
Estimate parameters. Stores `.delay_` and `.dimension_`.

#### `.transform(X: np.ndarray) -> np.ndarray`
- Input: `(n_samples,)`
- Output: `(n_samples - (dimension-1)*delay, dimension)`

#### `.fit_transform(X: np.ndarray) -> np.ndarray`

**Example**:
```python
from att.config import set_seed
from att.embedding import TakensEmbedder
from att.synthetic import lorenz_system

set_seed(42)
ts = lorenz_system(n_steps=10000)
embedder = TakensEmbedder()
cloud = embedder.fit_transform(ts[:, 0])
print(f"Estimated delay: {embedder.delay_}, dimension: {embedder.dimension_}")
```

---

### `JointEmbedder(delays="auto", dimensions="auto")`

Construct joint delay embeddings for multi-system analysis with per-channel delay estimation.

**Parameters**:
- `delays` (list[int] | "auto"): Per-channel delays. "auto" estimates independently per channel via AMI. Using "auto" is strongly recommended for systems with different timescales.
- `dimensions` (list[int] | "auto"): Per-channel embedding dimensions. "auto" estimates per channel via FNN.

**Methods**:

#### `.fit(channels: list[np.ndarray]) -> JointEmbedder`
Estimate per-channel parameters. Stores `.delays_` and `.dimensions_`.

#### `.transform(channels: list[np.ndarray]) -> np.ndarray`
Construct joint delay vectors by concatenating per-channel embeddings.
- Input: list of 1D arrays, each `(n_samples,)`
- Output: `(n_valid_samples, sum(dimensions))`

#### `.transform_marginals(channels: list[np.ndarray]) -> list[np.ndarray]`
Return individually embedded point clouds for marginal comparison.

**Example**:
```python
from att.embedding import JointEmbedder
from att.synthetic import coupled_rossler_lorenz

ts_r, ts_l = coupled_rossler_lorenz(coupling=0.3)
embedder = JointEmbedder()
embedder.fit([ts_r[:, 0], ts_l[:, 0]])
print(f"Per-channel delays: {embedder.delays_}")  # Will differ
joint_cloud = embedder.transform([ts_r[:, 0], ts_l[:, 0]])
marginals = embedder.transform_marginals([ts_r[:, 0], ts_l[:, 0]])
```

---

### Utility Functions

#### `estimate_delay(X, method="ami", max_lag=100) -> int`
#### `estimate_dimension(X, delay, method="fnn", max_dim=10, threshold=0.01) -> int`
#### `svd_embedding(X, delay, dimension, n_components=None) -> np.ndarray`
SVD-projected delay embedding for noise reduction.

#### `validate_embedding(cloud, expected_dim=None) -> dict`
Check embedding quality by computing the SVD of the centered point cloud matrix (n_points × dimension). The condition number is the ratio of the largest to smallest singular value. A high condition number means some embedding dimensions are near-linear combinations of others — the manifold is collapsed along those directions.

**Threshold justification**: The 1e4 threshold was calibrated on coupled Rössler-Lorenz systems. With per-channel delays, condition numbers are typically 10–500. With a shared delay on heterogeneous timescales, condition numbers jump to 1e4–1e8 as the fast component's delay coordinates become redundant. The 1e4 threshold separates these regimes with zero overlap on our test systems. Users working with noisier data may need to adjust via `validate_embedding(cloud, condition_threshold=...)`.

Returns:
```python
{
    "condition_number": float,       # σ_max / σ_min of centered point cloud
    "singular_values": np.ndarray,   # full singular value spectrum for inspection
    "effective_rank": int,           # number of singular values > 1e-3 * σ_max
    "degenerate": bool,              # True if condition_number > threshold (default 1e4)
    "warning": str | None,           # human-readable warning if near-degenerate
}
```

---

## att.topology

### `PersistenceAnalyzer(max_dim=2, backend="ripser", use_witness=False, n_landmarks=500)`

**Parameters**:
- `max_dim`: Max homology dimension (0=components, 1=loops, 2=voids)
- `backend`: `"ripser"` (fast for H0+H1) or `"gudhi"` (H2+, alpha complexes)
- `use_witness`: Use witness complex for large point clouds (n > 2000)
- `n_landmarks`: Number of landmarks for witness complex

**Methods**:

#### `.fit_transform(cloud, subsample=None, seed=None) -> dict`
- `subsample`: If int, randomly select this many points before computing persistence. Uses `seed` (or global seed state) for deterministic subsampling.
- **Subsample consistency in binding pipelines**: When `BindingDetector` subsamples internally, it uses the SAME seed for all three clouds (marginal X, marginal Y, joint) within a single `.fit()` call. Each surrogate iteration increments the seed. This ensures that diagram differences come from topology, not from different random subsets. Recommended subsample size: 1000 for routine analysis, 2000+ for publication figures. If no subsampling is needed (cloud < 1500 points), leave as None.

```python
{
    "diagrams": list[np.ndarray],           # (n_features, 2) per dimension
    "betti_curves": list[np.ndarray],        # Betti(filtration_param)
    "persistence_entropy": list[float],
    "bottleneck_norms": list[float],
    "persistence_images": list[np.ndarray],  # (resolution, resolution) per dim
    "persistence_landscapes": list[np.ndarray],
}
```

#### `.distance(other, metric="bottleneck") -> float`
Metrics: `"bottleneck"`, `"wasserstein_1"`, `"wasserstein_2"`

#### `.to_image(resolution=50, sigma=0.1) -> list[np.ndarray]`
Convert diagrams to persistence images.

#### `.to_landscape(n_layers=5, n_grid=100) -> list[np.ndarray]`

#### `.plot(kind="diagram") -> Figure`
Options: `"diagram"`, `"barcode"`, `"betti_curve"`, `"landscape"`, `"image"`

---

## att.binding

### `BindingDetector(max_dim=1, method="persistence_image", image_resolution=50, image_sigma=0.1, baseline="max", embedding_quality_gate=True)`

Detect emergent topological features in joint embeddings absent from marginals.

**Parameters**:
- `method`: `"persistence_image"` (recommended) or `"diagram_matching"`
- `baseline`: How to combine marginal persistence images for comparison.
  - `"max"` (default): Pointwise max of I_X and I_Y. Conservative — only reports features exceeding BOTH marginals at every pixel. Chosen as default because it minimizes false positives: a feature must be absent from both marginals to count as emergent. Equivalent to asking "what does the joint have that neither marginal alone explains?"
  - `"sum"`: Pointwise sum of I_X and I_Y. Reports features exceeding the combined marginal mass. More sensitive but higher false positive rate — if both marginals have moderate features at the same (birth, death) location, the sum baseline may mask genuine excess in the joint.
- `embedding_quality_gate` (bool): If True (default), runs `validate_embedding` on ALL THREE point clouds (marginal X, marginal Y, joint). Raises `EmbeddingDegeneracyWarning` if any has condition number exceeding the threshold (default 1e4). A degenerate marginal produces a garbage persistence image, making the residual meaningless even if the joint is fine. Set to False to bypass (e.g., when you have manually validated all embeddings or are using SVD denoising).

**Methods**:

#### `.fit(X, Y, joint_embedder=None, marginal_embedder_x=None, marginal_embedder_y=None, subsample=None, seed=None, n_ensemble=1) -> BindingDetector`
- `X`, `Y`: 1D time series of same length
- If embedders are None, uses auto-estimated per-channel parameters
- `n_ensemble` (int): If >1 and `subsample` is provided, runs K independent persistence+scoring passes with different subsample seeds. `binding_score()` returns the ensemble mean. Variance reduction is modest: CV ~28% → ~24% at K=10.
- **Embedding quality check**: If `embedding_quality_gate=True`, validates all three point clouds (marginal X, marginal Y, joint). If any has condition number exceeding the threshold, raises `EmbeddingDegeneracyWarning` identifying which embedding(s) failed and recommending inspection or SVD denoising. The result dict includes `"embedding_quality"` with validation output for all three clouds.

#### `.binding_score() -> float`
For `persistence_image` method: L1 norm of positive residual image (joint minus baseline).
For `diagram_matching` method: total persistence of unmatched features.
If `n_ensemble > 1`, returns the ensemble mean.

#### `.ensemble_scores -> np.ndarray | None`
Array of K individual binding scores from ensemble runs. None if `n_ensemble=1` or `subsample=None`.

#### `.confidence_interval(confidence=0.95) -> tuple[float, float]`
Bootstrap confidence interval from ensemble scores. Raises ValueError if ensemble was not used.

#### `.binding_features() -> dict`
```python
{
    0: {"n_excess": int, "total_persistence": float, "max_persistence": float},
    1: {"n_excess": int, "total_persistence": float, "max_persistence": float},
}
```

#### `.binding_image() -> list[np.ndarray]`
Residual persistence image (joint minus baseline of marginals). Positive regions = emergent topology. Only for `persistence_image` method.

#### `.embedding_quality() -> dict`
Returns `validate_embedding` output for all three clouds. Useful for diagnosing unexpected binding scores.
```python
{
    "marginal_x": dict,  # validate_embedding output
    "marginal_y": dict,
    "joint": dict,
    "any_degenerate": bool,
}
```

#### `.test_significance(n_surrogates=100, method="phase_randomize") -> dict`
```python
{
    "p_value": float,
    "observed_score": float,
    "surrogate_scores": np.ndarray,
    "significant": bool,  # at α=0.05
    "z_score": float,  # (observed - surrogate_mean) / surrogate_std
    "calibrated_score": float,  # observed - surrogate_mean
    "surrogate_mean": float,
    "surrogate_std": float,
    "embedding_quality": dict,  # validate_embedding output for joint cloud
}
```
Methods: `"phase_randomize"`, `"time_shuffle"`, `"twin_surrogate"`

**Important**: Raw binding scores have a structural positive baseline that grows with data size. Z-scores are the correct calibrated measure for coupling evidence. The method has zero power for same-timescale coupling (e.g., Lorenz–Lorenz); it is selective to heterogeneous-timescale coupling (e.g., Rössler–Lorenz). See the preprint, Experiment 9.

#### `.plot_comparison() -> Figure`
Three-panel: marginal X diagram | joint diagram (excess highlighted) | marginal Y diagram

#### `.plot_binding_image() -> Figure`
Heatmap of residual persistence image.

**Example**:
```python
from att.config import set_seed
from att.binding import BindingDetector
from att.synthetic import coupled_lorenz

set_seed(42)
ts_x, ts_y = coupled_lorenz(coupling=0.5)
detector = BindingDetector(max_dim=1, method="persistence_image")
detector.fit(ts_x[:, 0], ts_y[:, 0])
print(f"Binding score: {detector.binding_score():.4f}")
print(f"Embedding quality: {detector.embedding_quality()}")

sig = detector.test_significance(n_surrogates=100)
print(f"p-value: {sig['p_value']:.4f}")

detector.plot_binding_image()
```

---

## att.surrogates

### `phase_randomize(X, n_surrogates=100, seed=None) -> np.ndarray`
Amplitude-Adjusted Phase Randomization. Preserves power spectrum, destroys nonlinear coupling.
Returns `(n_surrogates, n_samples)`. If `seed` is None, uses global seed state.

### `time_shuffle(X, n_surrogates=100, block_size=None, seed=None) -> np.ndarray`
Block-shuffled surrogates. `block_size=None` → iid shuffle.

### `twin_surrogate(X, n_surrogates=100, embedder=None, seed=None) -> np.ndarray`
Recurrence-based twin surrogates (Thiel et al., 2006). Preserves attractor topology, breaks cross-system coupling. Most conservative null.

---

## att.benchmarks

### `CouplingBenchmark(methods=None, normalization="rank")`

**Parameters**:
- `methods`: Subset of `["binding_score", "transfer_entropy", "pac", "crqa"]`. Default: all.
- `normalization`: How to normalize scores for cross-method visual comparison in sweep plots.
  - `"rank"` (default): Rank-transform each method's scores to [0, 1] over the sweep. Preserves monotonicity and ordering without distortion from different score magnitudes. Recommended for visual comparison.
  - `"minmax"`: Per-method min-max scaling to [0, 1] over the sweep. Preserves relative spacing but sensitive to outliers at endpoints.
  - `"zscore"`: Per-method z-scoring (mean=0, std=1). Best for statistical comparison but less intuitive visually.
  - `"none"`: Raw scores. Useful when comparing methods with known comparable scales.

**Methods**:

#### `.register_method(name, fn) -> None`
Add a custom coupling method to the benchmark suite.
- `name`: String identifier for the method (appears in DataFrame output)
- `fn`: callable(X, Y) → float. Takes two 1D time series, returns a scalar coupling score.
The registered method is included in all subsequent `.run()` and `.sweep()` calls.

```python
# Example: add coherence as a 5th method
def mean_coherence(X, Y):
    from scipy.signal import coherence
    _, coh = coherence(X, Y, nperseg=256)
    return coh.mean()

bench = CouplingBenchmark()
bench.register_method("coherence", mean_coherence)
# Now .run() and .sweep() include coherence alongside the 4 built-in methods
```

#### `.run(X, Y, **kwargs) -> dict`
Compute all coupling measures on the same time series pair.
```python
{
    "binding_score": float,
    "transfer_entropy_xy": float,
    "transfer_entropy_yx": float,
    "pac": float,
    "crqa_determinism": float,
    "crqa_laminarity": float,
}
```

#### `.sweep(generator_fn, coupling_values, seed=None, transient_discard=1000) -> pd.DataFrame`
Run all methods across a coupling parameter sweep.
- `generator_fn`: callable(coupling, seed) → (X, Y). Receives the generator seed.
- `seed`: If provided, the SAME seed is used for all coupling values in `generator_fn` so that initial conditions are identical across the sweep. Surrogate generation within each coupling value uses `seed + i` to avoid correlated surrogates across coupling levels.
- `transient_discard`: Number of initial time steps to discard before analysis (default 1000). Removes sensitivity to initial transients on chaotic attractors.
- Returns DataFrame: columns `coupling`, `method`, `score`, `score_normalized`

#### `.plot_sweep(results, use_normalized=True) -> Figure`
All methods overlaid on one coupling sweep plot. If `use_normalized=True`, plots the `score_normalized` column; otherwise plots raw `score`.

**Example**:
```python
from att.config import set_seed
from att.benchmarks import CouplingBenchmark
from att.synthetic import coupled_lorenz
import numpy as np

set_seed(42)
bench = CouplingBenchmark(normalization="rank")
results = bench.sweep(
    generator_fn=lambda c, s: (coupled_lorenz(coupling=c, seed=s)[0][:, 0],
                                coupled_lorenz(coupling=c, seed=s)[1][:, 0]),
    coupling_values=np.linspace(0, 1, 10),
    seed=42,
    transient_discard=1000,
)
bench.plot_sweep(results)
```

### CLI: `att benchmark run`

Run a coupling sweep from the command line using a YAML config:

```bash
att benchmark run --config configs/coupled_lorenz_sweep.yaml --output results/sweep.csv
att benchmark run --config configs/coupled_lorenz_sweep.yaml --output results/sweep.csv --plot results/sweep.png
```

Reads the YAML config (system, coupling values, methods, normalization, seed), runs the full sweep, and writes the DataFrame to CSV. With `--plot`, also generates the sweep figure. Designed for batch jobs and reproducible experiments.

---

## att.transitions

### `TransitionDetector(window_size=500, step_size=50, max_dim=1, use_witness=False)`

**Methods**:

#### `.fit_transform(X) -> dict`
```python
{
    "topology_timeseries": list[dict],
    "distances": np.ndarray,           # bottleneck between consecutive windows
    "image_distances": np.ndarray,     # L2 between consecutive persistence images
    "changepoints": list[int],
    "transition_scores": np.ndarray,
}
```

#### `.detect_changepoints(method="cusum") -> list[int]`
Methods: `"cusum"`, `"pelt"`, `"threshold"`

#### `.test_significance(n_surrogates=50) -> dict`
Permutation test: are transitions more extreme than surrogate baseline?

#### `.plot_timeline() -> Figure`

---

## att.synthetic

All generators accept an optional `seed` parameter. If None, uses global seed state from `set_seed()`.

### `lorenz_system(n_steps=10000, dt=0.01, sigma=10, rho=28, beta=8/3, initial=None, noise=0.0, seed=None) -> np.ndarray`
Returns `(n_steps, 3)`.

### `rossler_system(n_steps=10000, dt=0.01, a=0.2, b=0.2, c=5.7, initial=None, noise=0.0, seed=None) -> np.ndarray`

### `coupled_lorenz(n_steps=10000, dt=0.01, coupling=0.1, seed=None) -> tuple[np.ndarray, np.ndarray]`
Diffusive coupling. Returns `(ts_x, ts_y)`, each `(n_steps, 3)`.

### `coupled_rossler_lorenz(n_steps=10000, dt=0.01, coupling=0.1, seed=None) -> tuple[np.ndarray, np.ndarray]`
Heterogeneous timescales. Tests per-channel delay handling.

### `switching_rossler(n_steps=20000, dt=0.01, switch_every=5000, seed=None) -> np.ndarray`
Ground truth transitions at `switch_every` intervals.

### `coupled_oscillators(n_oscillators=3, coupling_matrix=None, n_steps=10000, seed=None) -> np.ndarray`
Returns `(n_steps, n_oscillators, 3)`.

### `kuramoto_oscillators(n_oscillators=5, coupling=1.0, n_steps=10000, dt=0.01, natural_frequencies=None, noise=0.0, seed=None) -> tuple[np.ndarray, np.ndarray]`
Classic Kuramoto model: $d\theta_i/dt = \omega_i + (K/N) \sum \sin(\theta_j - \theta_i) + \text{noise}$.
Returns `(phases, signals)` where `phases` is `(n_steps, n_oscillators)` and `signals = sin(phases)`.
**Note**: Binding score *decreases* with Kuramoto coupling (77× reduction at strong coupling). Phase synchronization collapses the joint manifold — opposite of chaotic systems. See preprint Discussion.

---

## att.neuro

### `EEGLoader(dataset, subject)`
Presets: `"katyal_rivalry"`, `"bistable_necker"`, `"auditory_bistable"` or any OpenNeuro accession ID.

#### `.load() -> mne.io.Raw`
#### `.preprocess(bandpass=(1, 45), notch=50, reference="average", ica_reject=True) -> mne.io.Raw`
#### `.to_timeseries(picks=None) -> np.ndarray`
#### `.epoch(events, tmin=-2.0, tmax=2.0) -> np.ndarray`
#### `.get_switch_events() -> np.ndarray`
#### `.get_channel_groups() -> dict`

#### `.embed_channel(channel_data, band="broadband", condition_threshold=1e4) -> tuple[np.ndarray, dict]`
Embed a single EEG channel with automatic fallback. Tries auto-estimation first; if the resulting embedding is degenerate (condition number > threshold), re-embeds using `get_fallback_params(band)`. Returns `(point_cloud, metadata)` where metadata includes `"method"` ("auto" or "fallback"), estimated parameters, condition number, and fallback reason if applicable. Never raises on estimation failure — logs and falls back silently for batch compatibility.

### EEG Fallback Parameters

When AMI/FNN estimation is unreliable on noisy EEG (common), use these empirically grounded defaults:

| Band | Bandpass (Hz) | Delay τ (samples @ 256 Hz) | Dimension d | Source |
|------|--------------|---------------------------|-------------|--------|
| Broadband (1–45 Hz) | (1, 45) | 10 (≈39 ms) | 5 | Stam (2005), typical for EEG nonlinear analysis |
| Alpha (8–13 Hz) | (8, 13) | 8 (≈31 ms) | 4 | Narrowband, lower dimension sufficient |
| Theta-Alpha (4–13 Hz) | (4, 13) | 12 (≈47 ms) | 5 | Bistable perception primary band |
| Gamma (30–45 Hz) | (30, 45) | 3 (≈12 ms) | 5 | Fast dynamics, short delay |

These are accessible via `EEGLoader.get_fallback_params(band="broadband")` and can be passed directly to `TakensEmbedder(delay=..., dimension=...)`.

---

## att.viz

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

def export_to_json(results: dict, path: str) -> None
def load_from_json(path: str) -> dict
```
