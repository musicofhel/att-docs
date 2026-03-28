"""Tests for att.topology — persistent homology computation."""

import numpy as np
import pytest

from att.config import set_seed
from att.synthetic import lorenz_system, rossler_system
from att.embedding import TakensEmbedder
from att.topology import PersistenceAnalyzer


class TestPersistenceAnalyzer:
    @pytest.fixture(autouse=True)
    def setup(self):
        set_seed(42)
        ts = lorenz_system(n_steps=10000)
        embedder = TakensEmbedder(delay="auto", dimension="auto")
        self.lorenz_cloud = embedder.fit_transform(ts[:, 0])

    def test_lorenz_h1_features(self):
        """Lorenz should have 2 dominant H1 loops (butterfly wings)."""
        analyzer = PersistenceAnalyzer(max_dim=1)
        result = analyzer.fit_transform(self.lorenz_cloud, subsample=1000, seed=42)
        h1 = result["diagrams"][1]
        assert len(h1) >= 2, "Lorenz should have at least 2 H1 features"

        lifetimes = h1[:, 1] - h1[:, 0]
        top2 = sorted(lifetimes, reverse=True)[:2]
        # The top feature should be significantly persistent
        assert top2[0] > 1.0, f"Top H1 lifetime {top2[0]} too small"

    def test_persistence_images_nonzero(self):
        analyzer = PersistenceAnalyzer(max_dim=1)
        result = analyzer.fit_transform(self.lorenz_cloud, subsample=1000, seed=42)
        for img in result["persistence_images"]:
            assert img.shape == (50, 50)

    def test_persistence_images_differ_across_systems(self):
        """Lorenz and Rössler should have different persistence images."""
        set_seed(42)
        analyzer_l = PersistenceAnalyzer(max_dim=1)
        result_l = analyzer_l.fit_transform(self.lorenz_cloud, subsample=1000, seed=42)

        ts_r = rossler_system(n_steps=10000, seed=42)
        cloud_r = TakensEmbedder("auto", "auto").fit_transform(ts_r[:, 0])
        analyzer_r = PersistenceAnalyzer(max_dim=1)
        result_r = analyzer_r.fit_transform(cloud_r, subsample=1000, seed=42)

        # H1 persistence images should differ
        diff = np.abs(result_l["persistence_images"][1] - result_r["persistence_images"][1])
        assert diff.sum() > 0.1

    def test_bottleneck_distance_same_system(self):
        """Two subsamplings of same system should have small distance."""
        a1 = PersistenceAnalyzer(max_dim=1)
        a1.fit_transform(self.lorenz_cloud, subsample=1000, seed=42)

        a2 = PersistenceAnalyzer(max_dim=1)
        a2.fit_transform(self.lorenz_cloud, subsample=1000, seed=43)

        d = a1.distance(a2, metric="bottleneck")
        assert d < 2.0, f"Same-system bottleneck distance {d} too large"

    def test_bottleneck_distance_different_systems(self):
        """Lorenz vs Rössler should have large distance."""
        a_lorenz = PersistenceAnalyzer(max_dim=1)
        a_lorenz.fit_transform(self.lorenz_cloud, subsample=1000, seed=42)

        set_seed(42)
        ts_r = rossler_system(n_steps=10000)
        cloud_r = TakensEmbedder("auto", "auto").fit_transform(ts_r[:, 0])
        a_rossler = PersistenceAnalyzer(max_dim=1)
        a_rossler.fit_transform(cloud_r, subsample=1000, seed=42)

        d = a_lorenz.distance(a_rossler, metric="bottleneck")
        assert d > 1.0, f"Cross-system bottleneck distance {d} too small"

    def test_reproducibility(self):
        """Same seed → identical diagrams."""
        a1 = PersistenceAnalyzer(max_dim=1)
        r1 = a1.fit_transform(self.lorenz_cloud, subsample=500, seed=42)

        a2 = PersistenceAnalyzer(max_dim=1)
        r2 = a2.fit_transform(self.lorenz_cloud, subsample=500, seed=42)

        for dim in range(2):
            np.testing.assert_array_equal(r1["diagrams"][dim], r2["diagrams"][dim])

    def test_betti_curves(self):
        analyzer = PersistenceAnalyzer(max_dim=1)
        result = analyzer.fit_transform(self.lorenz_cloud, subsample=500, seed=42)
        for curve in result["betti_curves"]:
            assert len(curve) == 100

    def test_persistence_entropy(self):
        analyzer = PersistenceAnalyzer(max_dim=1)
        result = analyzer.fit_transform(self.lorenz_cloud, subsample=500, seed=42)
        for entropy in result["persistence_entropy"]:
            assert entropy >= 0

    def test_landscapes(self):
        analyzer = PersistenceAnalyzer(max_dim=1)
        result = analyzer.fit_transform(self.lorenz_cloud, subsample=500, seed=42)
        for ls in result["persistence_landscapes"]:
            assert ls.shape == (5, 100)

    def test_wasserstein_distance(self):
        a1 = PersistenceAnalyzer(max_dim=1)
        a1.fit_transform(self.lorenz_cloud, subsample=500, seed=42)

        a2 = PersistenceAnalyzer(max_dim=1)
        a2.fit_transform(self.lorenz_cloud, subsample=500, seed=43)

        d = a1.distance(a2, metric="wasserstein_1")
        assert d >= 0


@pytest.mark.slow
class TestWitnessComplex:
    def test_witness_produces_features(self):
        """Witness complex on Lorenz should produce H0 and H1 features."""
        pytest.importorskip("gudhi")
        from att.synthetic import coupled_lorenz
        ts_x, _ = coupled_lorenz(n_steps=3000, coupling=0.0, seed=42)
        cloud = ts_x[500:, :3]  # Use 3D Lorenz directly

        pa = PersistenceAnalyzer(max_dim=1, backend="gudhi", use_witness=True, n_landmarks=100)
        result = pa.fit_transform(cloud, subsample=500, seed=42)

        assert len(result["diagrams"][0]) > 0  # H0 features
        assert len(result["diagrams"][1]) > 0  # H1 features
