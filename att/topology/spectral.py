"""Spectral distance matrices for topology-aware persistent homology.

Constructs kNN graph Laplacians and derives effective-resistance distance
matrices that respect intrinsic geometry better than Euclidean distances
in high-dimensional spaces (Direction 3).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


def knn_graph_laplacian(
    cloud: np.ndarray,
    k: int = 15,
    symmetrize: str = "or",
) -> csr_matrix:
    """Build the graph Laplacian from a kNN adjacency graph.

    Parameters
    ----------
    cloud : (n, d) point cloud.
    k : int
        Number of nearest neighbours (excluding self).
    symmetrize : str
        "or" (union — default) or "and" (intersection) for making the
        directed kNN graph undirected.

    Returns
    -------
    L : (n, n) sparse CSR Laplacian (L = D - W), symmetric positive
        semi-definite.
    """
    n = cloud.shape[0]
    k_eff = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto")
    nn.fit(cloud)
    dists, indices = nn.kneighbors(cloud)

    # Build sparse weight matrix (Gaussian kernel with adaptive bandwidth)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_pos in range(1, k_eff + 1):  # skip self at 0
            j = indices[i, j_pos]
            d = dists[i, j_pos]
            # Heat kernel with local bandwidth = distance to k-th neighbour
            sigma_i = dists[i, k_eff]
            w = np.exp(-(d ** 2) / (sigma_i ** 2 + 1e-15))
            rows.append(i)
            cols.append(j)
            vals.append(w)

    W = csr_matrix((vals, (rows, cols)), shape=(n, n))

    # Symmetrize
    if symmetrize == "or":
        W = (W + W.T) / 2.0
    else:
        W = W.minimum(W.T)

    # Laplacian: L = D - W
    deg = np.array(W.sum(axis=1)).ravel()
    D = diags(deg)
    L = D - W

    return L.tocsr()


def spectral_distance_matrix(
    cloud: np.ndarray,
    k: int = 15,
    n_eigenvectors: int | None = None,
) -> np.ndarray:
    """Compute effective-resistance distance matrix from kNN graph Laplacian.

    The effective resistance between nodes i and j is:
        R(i,j) = (e_i - e_j)^T L^+ (e_i - e_j)
    which can be computed efficiently via the spectral decomposition of L.

    Parameters
    ----------
    cloud : (n, d) point cloud.
    k : int
        Number of nearest neighbours for the graph.
    n_eigenvectors : int or None
        Number of Laplacian eigenvectors to use (truncated approximation).
        None = use min(n-1, 100).

    Returns
    -------
    D : (n, n) symmetric distance matrix with zero diagonal.
    """
    n = cloud.shape[0]
    L = knn_graph_laplacian(cloud, k=k)

    if n_eigenvectors is None:
        n_eigenvectors = min(n - 1, 100)
    n_eigenvectors = min(n_eigenvectors, n - 1)

    # Compute smallest eigenvalues/vectors of L (skip the trivial zero eigenvalue)
    n_eig = min(n_eigenvectors + 1, n - 1)
    eigenvalues, eigenvectors = eigsh(L.astype(np.float64), k=n_eig, which="SM")

    # Skip the zero eigenvalue (connected component)
    # Eigenvalues near zero are the trivial ones
    tol = 1e-10
    nonzero_mask = eigenvalues > tol
    eigenvalues = eigenvalues[nonzero_mask]
    eigenvectors = eigenvectors[:, nonzero_mask]

    if len(eigenvalues) == 0:
        # Degenerate case: all points equivalent
        return np.zeros((n, n))

    # Scaled eigenvectors: phi_k / sqrt(lambda_k)
    scaled = eigenvectors / np.sqrt(eigenvalues)[np.newaxis, :]

    # Effective resistance: R(i,j) = ||scaled[i] - scaled[j]||^2
    # Efficiently compute all pairwise squared distances
    sq_norms = np.sum(scaled ** 2, axis=1)
    D = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (scaled @ scaled.T)

    # Clean up numerical noise
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    # Symmetrize
    D = (D + D.T) / 2.0

    # Return sqrt for use as a metric (effective resistance distance)
    return np.sqrt(D)
