"""Embedding quality validation via condition number analysis."""

import warnings
import numpy as np


class EmbeddingDegeneracyWarning(UserWarning):
    """Raised when an embedding has a dangerously high condition number."""
    pass


def validate_embedding(
    cloud: np.ndarray,
    expected_dim: int | None = None,
    condition_threshold: float = 1e4,
) -> dict:
    """Check embedding quality via SVD of the centered point cloud matrix.

    The condition number is σ_max / σ_min. High values mean some embedding
    dimensions are near-linear combinations of others — the manifold is
    collapsed along those directions.

    Parameters
    ----------
    cloud : (n_points, dimension) array
    expected_dim : expected intrinsic dimension (informational only)
    condition_threshold : condition number above which embedding is flagged
        as degenerate. Default 1e4 calibrated on coupled Rössler-Lorenz.

    Returns
    -------
    dict with keys:
        condition_number: float
        singular_values: np.ndarray
        effective_rank: int (singular values > 1e-3 * σ_max)
        degenerate: bool
        warning: str | None
    """
    cloud = np.asarray(cloud)
    if cloud.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {cloud.shape}")

    # Center the cloud
    centered = cloud - cloud.mean(axis=0)

    # SVD
    singular_values = np.linalg.svd(centered, compute_uv=False)

    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]

    if sigma_min < 1e-15:
        condition_number = float("inf")
    else:
        condition_number = sigma_max / sigma_min

    effective_rank = int(np.sum(singular_values > 1e-3 * sigma_max))
    degenerate = condition_number > condition_threshold

    warning_msg = None
    if degenerate:
        warning_msg = (
            f"Embedding is near-degenerate: condition number = {condition_number:.1e} "
            f"(threshold = {condition_threshold:.1e}). "
            f"Effective rank = {effective_rank}/{cloud.shape[1]}. "
            f"Consider per-channel delay estimation or SVD denoising."
        )

    return {
        "condition_number": condition_number,
        "singular_values": singular_values,
        "effective_rank": effective_rank,
        "degenerate": degenerate,
        "warning": warning_msg,
    }


def svd_embedding(
    X: np.ndarray,
    delay: int,
    dimension: int,
    n_components: int | None = None,
) -> np.ndarray:
    """SVD-projected delay embedding for noise reduction.

    Constructs the delay matrix then projects onto the top n_components
    principal components.

    Parameters
    ----------
    X : (n_samples,) 1D time series
    delay : time delay
    dimension : embedding dimension
    n_components : number of SVD components to keep (default: dimension)

    Returns
    -------
    (n_valid, n_components) projected point cloud
    """
    X = np.asarray(X).ravel()
    n = len(X) - (dimension - 1) * delay

    if n <= 0:
        raise ValueError("Time series too short for given delay and dimension.")

    if n_components is None:
        n_components = dimension

    # Build delay matrix
    cloud = np.zeros((n, dimension))
    for i in range(dimension):
        cloud[:, i] = X[i * delay: i * delay + n]

    # Center and SVD
    centered = cloud - cloud.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Project onto top components
    return U[:, :n_components] * S[:n_components]
