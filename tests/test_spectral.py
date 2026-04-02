"""Tests for att.topology.spectral — kNN graph Laplacian and spectral distances."""

import numpy as np
import pytest

from att.topology.spectral import knn_graph_laplacian, spectral_distance_matrix
from att.topology.persistence import PersistenceAnalyzer


class TestKnnGraphLaplacian:
    def test_shape(self):
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((30, 5))
        L = knn_graph_laplacian(cloud, k=5)
        assert L.shape == (30, 30)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((20, 3))
        L = knn_graph_laplacian(cloud, k=5)
        diff = (L - L.T).toarray()
        np.testing.assert_array_almost_equal(diff, 0, decimal=10)

    def test_row_sums_zero(self):
        """Laplacian rows sum to zero: L = D - W."""
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((25, 4))
        L = knn_graph_laplacian(cloud, k=5)
        row_sums = np.array(L.sum(axis=1)).ravel()
        np.testing.assert_array_almost_equal(row_sums, 0, decimal=10)

    def test_positive_semi_definite(self):
        """Eigenvalues of graph Laplacian are non-negative."""
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((20, 3))
        L = knn_graph_laplacian(cloud, k=5)
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        assert np.all(eigenvalues >= -1e-10)

    def test_k_larger_than_n(self):
        """k clamped to n-1 when k >= n."""
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((5, 3))
        L = knn_graph_laplacian(cloud, k=100)
        assert L.shape == (5, 5)


class TestSpectralDistanceMatrix:
    def test_shape(self):
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((20, 5))
        D = spectral_distance_matrix(cloud, k=5)
        assert D.shape == (20, 20)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((20, 5))
        D = spectral_distance_matrix(cloud, k=5)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_zero_diagonal(self):
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((20, 5))
        D = spectral_distance_matrix(cloud, k=5)
        np.testing.assert_array_almost_equal(np.diag(D), 0)

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((20, 5))
        D = spectral_distance_matrix(cloud, k=5)
        assert np.all(D >= -1e-10)

    def test_triangle_inequality(self):
        """Spot-check triangle inequality for a few triples."""
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((15, 3))
        D = spectral_distance_matrix(cloud, k=5)
        for i in range(5):
            for j in range(i + 1, 10):
                for k_idx in range(j + 1, 15):
                    assert D[i, j] <= D[i, k_idx] + D[k_idx, j] + 1e-10

    def test_precomputed_ph_recovers_circle(self):
        """Circle embedded in R^10: spectral PH should recover H1=1."""
        rng = np.random.default_rng(42)
        n_pts = 60
        t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        circle_2d = np.column_stack([np.cos(t), np.sin(t)])
        # Embed in R^10 with noise
        noise = rng.standard_normal((n_pts, 8)) * 0.01
        cloud = np.hstack([circle_2d, noise])

        D = spectral_distance_matrix(cloud, k=10)
        pa = PersistenceAnalyzer(max_dim=1, backend="ripser", metric="precomputed")
        result = pa.fit_transform(D)

        # Should find exactly 1 prominent H1 feature
        h1 = result["diagrams"][1]
        assert len(h1) >= 1
        lifetimes = h1[:, 1] - h1[:, 0]
        # The longest-lived H1 feature should be significantly longer than others
        if len(lifetimes) > 1:
            sorted_lt = np.sort(lifetimes)[::-1]
            assert sorted_lt[0] > 2 * sorted_lt[1]
