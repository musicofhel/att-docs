"""Standalone coupling measurement methods for benchmarking."""

import numpy as np
from scipy.signal import hilbert
from scipy.spatial.distance import cdist


def transfer_entropy(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 1,
    bins: int = 8,
    **kwargs,
) -> float:
    """Compute transfer entropy TE(X -> Y) using histogram estimation.

    Measures information transfer from X to Y beyond Y's own history.

    Parameters
    ----------
    X, Y : 1D time series (same length)
    k : history length (lag)
    bins : number of bins for discretization

    Returns
    -------
    float : TE(X -> Y) in nats
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]

    # Discretize using percentile bins
    X_d = _discretize(X, bins)
    Y_d = _discretize(Y, bins)

    # Build conditional variables
    # TE(X->Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    Y_future = Y_d[k:]
    Y_past = Y_d[:n - k]
    X_past = X_d[:n - k]

    # Compute as MI(X_past; Y_future | Y_past)
    # = H(Y_future, Y_past) + H(X_past, Y_past) - H(Y_past) - H(Y_future, Y_past, X_past)
    h_yf_yp = _joint_entropy(Y_future, Y_past)
    h_xp_yp = _joint_entropy(X_past, Y_past)
    h_yp = _entropy(Y_past)
    h_yf_yp_xp = _triple_entropy(Y_future, Y_past, X_past)

    te = h_yf_yp + h_xp_yp - h_yp - h_yf_yp_xp
    return max(0.0, float(te))


def pac(
    X: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 18,
    **kwargs,
) -> float:
    """Compute phase-amplitude coupling modulation index (Tort et al. 2010).

    Measures how much the amplitude of Y is modulated by the phase of X.
    Uses raw Hilbert transform (no bandpass) — appropriate for broadband
    chaotic signals where PAC is expected to return near-zero.

    Parameters
    ----------
    X, Y : 1D time series (same length)
    n_bins : number of phase bins

    Returns
    -------
    float : modulation index in [0, 1]
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]

    # Extract phase of X and amplitude of Y via Hilbert transform
    phase_x = np.angle(hilbert(X))
    amp_y = np.abs(hilbert(Y))

    # Bin phases into equal-width bins in [-pi, pi)
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_indices = np.digitize(phase_x, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Mean amplitude in each phase bin
    mean_amp = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.any():
            mean_amp[b] = amp_y[mask].mean()

    # Normalize to probability distribution
    total = mean_amp.sum()
    if total < 1e-15:
        return 0.0
    p = mean_amp / total

    # Modulation index: KL divergence from uniform, normalized
    uniform = np.ones(n_bins) / n_bins
    # Avoid log(0)
    p_safe = np.where(p > 1e-15, p, 1e-15)
    kl = np.sum(p_safe * np.log(p_safe / uniform))
    mi = kl / np.log(n_bins)

    return float(max(0.0, mi))


def crqa(
    X: np.ndarray,
    Y: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 1,
    radius: float | None = None,
    min_line: int = 2,
    **kwargs,
) -> float:
    """Compute cross-recurrence quantification analysis determinism.

    Parameters
    ----------
    X, Y : 1D time series
    embedding_dim : embedding dimension for delay vectors
    delay : time delay for embedding
    radius : recurrence threshold; if None, auto-select 10th percentile of distances
    min_line : minimum diagonal line length for determinism

    Returns
    -------
    float : determinism (fraction of recurrence points on diagonal lines >= min_line)
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()

    # Delay embedding
    X_emb = _delay_embed(X, embedding_dim, delay)
    Y_emb = _delay_embed(Y, embedding_dim, delay)

    # Truncate to same length
    n = min(len(X_emb), len(Y_emb))
    X_emb = X_emb[:n]
    Y_emb = Y_emb[:n]

    # Subsample for speed if too large
    max_points = 1000
    if n > max_points:
        step = n // max_points
        X_emb = X_emb[::step]
        Y_emb = Y_emb[::step]
        n = min(len(X_emb), len(Y_emb))

    # Cross-distance matrix
    dists = cdist(X_emb, Y_emb, metric="euclidean")

    # Auto-select radius if not provided
    if radius is None:
        radius = np.percentile(dists, 10)
        if radius < 1e-10:
            radius = np.percentile(dists, 20)

    # Cross-recurrence matrix
    crm = (dists <= radius).astype(int)
    total_recurrence = crm.sum()
    if total_recurrence == 0:
        return 0.0

    # Determinism: fraction of recurrence points on diagonal lines >= min_line
    diag_points = 0
    rows, cols = crm.shape
    for offset in range(-rows + 1, cols):
        diag = np.diag(crm, offset)
        diag_points += _count_line_points(diag, min_line)

    det = diag_points / total_recurrence if total_recurrence > 0 else 0.0
    return float(np.clip(det, 0.0, 1.0))


# ---- Private helpers ----

def _discretize(x: np.ndarray, bins: int) -> np.ndarray:
    """Discretize continuous values into bins via percentile edges."""
    edges = np.percentile(x, np.linspace(0, 100, bins + 1))
    edges[-1] += 1e-10  # include max value
    return np.digitize(x, edges[1:])


def _entropy(x: np.ndarray) -> float:
    """Shannon entropy of discrete variable."""
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + 1e-15))


def _joint_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """Joint entropy of two discrete variables."""
    xy = np.stack([x, y], axis=1)
    _, counts = np.unique(xy, axis=0, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + 1e-15))


def _triple_entropy(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Joint entropy of three discrete variables."""
    xyz = np.stack([x, y, z], axis=1)
    _, counts = np.unique(xyz, axis=0, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + 1e-15))


def _delay_embed(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
    """Create delay embedding vectors."""
    n = len(x) - (dim - 1) * delay
    if n <= 0:
        return np.empty((0, dim))
    return np.array([x[i * delay: i * delay + n] for i in range(dim)]).T


def _count_line_points(diag: np.ndarray, min_line: int) -> int:
    """Count points on lines of length >= min_line in a binary diagonal."""
    count = 0
    current_len = 0
    for val in diag:
        if val:
            current_len += 1
        else:
            if current_len >= min_line:
                count += current_len
            current_len = 0
    if current_len >= min_line:
        count += current_len
    return count
