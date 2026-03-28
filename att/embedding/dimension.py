"""Embedding dimension estimation via False Nearest Neighbors."""

import numpy as np
from sklearn.neighbors import KDTree


def estimate_dimension(
    X: np.ndarray,
    delay: int,
    method: str = "fnn",
    max_dim: int = 10,
    threshold: float = 0.01,
    rtol: float = 15.0,
    atol: float = 2.0,
) -> int:
    """Estimate minimal embedding dimension via False Nearest Neighbors.

    Parameters
    ----------
    X : (n_samples,) 1D time series
    delay : time delay for embedding
    method : "fnn" (only option currently)
    max_dim : maximum dimension to test
    threshold : FNN fraction below which to stop (default 0.01 = 1%)
    rtol : relative distance threshold for FNN criterion 1
    atol : absolute distance threshold for FNN criterion 2

    Returns
    -------
    int : estimated embedding dimension
    """
    if method != "fnn":
        raise ValueError(f"Unknown dimension method: {method}")

    n = len(X)

    for d in range(1, max_dim + 1):
        n_embed = n - d * delay
        n_embed_next = n - (d + 1) * delay

        if n_embed_next < 10:
            return d

        # Build embedding at dimension d
        cloud = np.zeros((n_embed, d))
        for i in range(d):
            cloud[:, i] = X[i * delay: i * delay + n_embed]

        # Build embedding at dimension d+1 (for the same points)
        cloud_next = np.zeros((n_embed_next, d + 1))
        for i in range(d + 1):
            cloud_next[:, i] = X[i * delay: i * delay + n_embed_next]

        # Find nearest neighbors in d-dimensional space
        tree = KDTree(cloud[:n_embed_next])
        dists, inds = tree.query(cloud[:n_embed_next], k=2)

        nn_dists = dists[:, 1]
        nn_inds = inds[:, 1]

        # Check for false nearest neighbors
        n_fnn = 0
        n_valid = 0

        for i in range(n_embed_next):
            if nn_dists[i] < 1e-10:
                continue
            n_valid += 1

            j = nn_inds[i]
            # Extra distance in the (d+1)th dimension
            extra_dist = abs(cloud_next[i, d] - cloud_next[j, d])

            # Criterion 1: relative distance increase
            if extra_dist / nn_dists[i] > rtol:
                n_fnn += 1
                continue

            # Criterion 2: absolute distance relative to attractor size
            if extra_dist > atol * np.std(X):
                n_fnn += 1

        fnn_fraction = n_fnn / max(n_valid, 1)

        if fnn_fraction < threshold:
            return d

    return max_dim
