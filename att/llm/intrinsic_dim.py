"""Intrinsic dimension estimators for LLM hidden-state point clouds.

Direction 7: tracks how representation complexity changes across layers
and difficulty levels using TwoNN (Facco et al. 2017) and PHD (Birdal et al.
2021) estimators.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def twonn_dimension(cloud: np.ndarray, fraction: float = 0.9) -> float:
    """Estimate intrinsic dimension via the TwoNN method (Facco et al. 2017).

    Uses the ratio of distances to the second and first nearest neighbours.
    The ID is estimated as d = 1 / mean(log(mu)) where mu = r2/r1.

    Parameters
    ----------
    cloud : (n, d) point cloud.
    fraction : float
        Fraction of points to use after trimming high-mu outliers (0, 1].

    Returns
    -------
    float : estimated intrinsic dimension.
    """
    n = cloud.shape[0]
    if n < 3:
        return 0.0

    # Pairwise distances
    if n <= 5000:
        D = cdist(cloud, cloud)
    else:
        # For large clouds, use sklearn for efficiency
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=3, algorithm="auto")
        nn.fit(cloud)
        dists, _ = nn.kneighbors(cloud)
        r1 = dists[:, 1]  # skip self
        r2 = dists[:, 2]
        valid = r1 > 1e-15
        r1 = r1[valid]
        r2 = r2[valid]
        if len(r1) == 0:
            return 0.0
        mu = r2 / r1
        mu = np.sort(mu)
        n_use = max(1, int(len(mu) * fraction))
        mu = mu[:n_use]
        log_mu = np.log(mu)
        mean_log_mu = np.mean(log_mu)
        if mean_log_mu < 1e-15:
            return 0.0
        return 1.0 / mean_log_mu

    # Full pairwise path
    np.fill_diagonal(D, np.inf)
    sorted_dists = np.sort(D, axis=1)
    r1 = sorted_dists[:, 0]
    r2 = sorted_dists[:, 1]

    valid = r1 > 1e-15
    r1 = r1[valid]
    r2 = r2[valid]

    if len(r1) == 0:
        return 0.0

    mu = r2 / r1
    mu = np.sort(mu)
    n_use = max(1, int(len(mu) * fraction))
    mu = mu[:n_use]

    log_mu = np.log(mu)
    mean_log_mu = np.mean(log_mu)

    if mean_log_mu < 1e-15:
        return 0.0

    return 1.0 / mean_log_mu


def phd_dimension(diagrams: list[np.ndarray], dim: int = 1) -> float:
    """Estimate intrinsic dimension from persistence diagram lifetimes.

    Based on the observation that in dimension d, the expected lifetime of
    H_k features scales as n^(-1/d) (Birdal et al. 2021). We estimate d
    from the distribution of H1 lifetimes using the log-log slope of the
    survival function.

    Parameters
    ----------
    diagrams : list of (n_features, 2) arrays (persistence diagrams).
    dim : int
        Homology dimension to use (default 1 for loops).

    Returns
    -------
    float : estimated intrinsic dimension (0.0 if insufficient features).
    """
    if dim >= len(diagrams):
        return 0.0

    dgm = diagrams[dim]
    if len(dgm) < 5:
        return 0.0

    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[lifetimes > 1e-15]

    if len(lifetimes) < 5:
        return 0.0

    # Sort lifetimes descending
    lifetimes = np.sort(lifetimes)[::-1]
    n = len(lifetimes)

    # Survival function: P(lifetime > t) vs t in log-log space
    # Use empirical CDF complement
    log_t = np.log(lifetimes)
    log_surv = np.log(np.arange(1, n + 1) / n)

    # Fit linear regression to log-log survival
    # Slope = -d (dimension estimate)
    A = np.column_stack([log_t, np.ones(n)])
    result = np.linalg.lstsq(A, log_surv, rcond=None)
    slope = result[0][0]

    # d = -slope (survival decays as t^{-d})
    d_est = max(0.0, -slope)
    return float(d_est)


def id_profile(
    loader,
    levels: list[int] | None = None,
    n_pca_components: int = 50,
    method: str = "twonn",
    fraction: float = 0.9,
) -> dict[int, np.ndarray]:
    """Compute intrinsic dimension profile across layers for each difficulty level.

    Parameters
    ----------
    loader : HiddenStateLoader
        Loaded hidden-state archive.
    levels : list of int or None
        Difficulty levels to analyze. None = all levels.
    n_pca_components : int
        PCA components before ID estimation (avoids curse of ambient dim).
    method : str
        "twonn" (default) or "phd".
    fraction : float
        Fraction parameter for TwoNN trimming.

    Returns
    -------
    dict mapping level -> (n_layers,) array of ID estimates.
    """
    from sklearn.decomposition import PCA
    from att.topology.persistence import PersistenceAnalyzer

    if levels is None:
        levels = sorted(loader.unique_levels.tolist())

    profiles = {}

    for level in levels:
        n_layers = loader.num_layers
        ids = np.zeros(n_layers)

        for layer_idx in range(n_layers):
            cloud = loader.get_level_cloud(level, layer=layer_idx)
            n_pts = cloud.shape[0]
            if n_pts < 5:
                continue

            n_comp = min(n_pca_components, n_pts - 1, cloud.shape[1])
            pca = PCA(n_components=n_comp)
            cloud_pca = pca.fit_transform(cloud)

            if method == "twonn":
                ids[layer_idx] = twonn_dimension(cloud_pca, fraction=fraction)
            elif method == "phd":
                pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
                result = pa.fit_transform(cloud_pca, subsample=min(n_pts, 200))
                ids[layer_idx] = phd_dimension(result["diagrams"], dim=1)
            else:
                raise ValueError(f"Unknown method: {method}")

        profiles[level] = ids

    return profiles
