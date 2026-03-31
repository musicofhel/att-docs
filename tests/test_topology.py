"""Tests for att.topology — persistent homology computation."""

import warnings

import numpy as np
import pytest

from att.config import set_seed
from att.synthetic import lorenz_system, rossler_system
from att.embedding import TakensEmbedder
from att.topology import PersistenceAnalyzer, TopologyDimensionalityWarning


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


class TestTopologyEdgeCases:
    """Edge-case tests for PersistenceAnalyzer."""

    def test_single_point_cloud(self):
        """Single-point cloud should produce minimal diagrams."""
        pa = PersistenceAnalyzer(max_dim=1)
        cloud = np.array([[1.0, 2.0, 3.0]])
        result = pa.fit_transform(cloud)
        # Only H0 can have features (trivially: one component)
        # H1 should be empty
        assert len(result["diagrams"]) >= 1
        if len(result["diagrams"]) > 1:
            assert len(result["diagrams"][1]) == 0

    def test_two_point_cloud(self):
        """Two-point cloud should produce H0 merge feature."""
        pa = PersistenceAnalyzer(max_dim=1)
        cloud = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = pa.fit_transform(cloud)
        # H0: two components merge at distance 1
        assert len(result["diagrams"][0]) >= 1
        # H1 should be empty (no loops with 2 points)
        if len(result["diagrams"]) > 1:
            assert len(result["diagrams"][1]) == 0

    def test_to_image_explicit_ranges(self):
        """to_image with custom ranges should produce correct shape."""
        pa = PersistenceAnalyzer(max_dim=1)
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((200, 3))
        pa.fit_transform(cloud)
        images = pa.to_image(
            resolution=25, sigma=0.05,
            birth_range=(0, 5), persistence_range=(0, 2)
        )
        for img in images:
            assert img.shape == (25, 25)

    def test_to_image_before_fit_raises(self):
        """to_image before fit_transform should raise."""
        pa = PersistenceAnalyzer(max_dim=1)
        with pytest.raises((RuntimeError, AttributeError)):
            pa.to_image()

    def test_distance_with_empty_diagrams(self):
        """Distance between two minimal clouds should be defined."""
        pa1 = PersistenceAnalyzer(max_dim=1)
        pa1.fit_transform(np.array([[0.0, 0.0], [1.0, 0.0]]))
        pa2 = PersistenceAnalyzer(max_dim=1)
        pa2.fit_transform(np.array([[0.0, 0.0], [2.0, 0.0]]))
        dist = pa1.distance(pa2, metric="bottleneck")
        assert isinstance(dist, float)
        assert np.isfinite(dist)
        assert dist >= 0


class TestDimensionalityWarning:
    """Tests for effective dimensionality warning in fit_transform."""

    def test_dimensionality_warning_fires(self):
        """50d cloud living on 3d manifold should trigger warning."""
        rng = np.random.default_rng(42)
        # Create a 50d cloud with only 3 effective dimensions
        base = rng.standard_normal((200, 3))
        cloud = base @ rng.standard_normal((3, 50))
        pa = PersistenceAnalyzer(max_dim=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pa.fit_transform(cloud, subsample=100, seed=42)
            dim_warnings = [x for x in w if issubclass(x.category, TopologyDimensionalityWarning)]
            assert len(dim_warnings) > 0, "Should warn about low effective dimensionality"

    def test_dimensionality_warning_silent(self):
        """Well-spread 10d cloud should not trigger warning."""
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((200, 10))
        pa = PersistenceAnalyzer(max_dim=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pa.fit_transform(cloud, subsample=100, seed=42)
            dim_warnings = [x for x in w if issubclass(x.category, TopologyDimensionalityWarning)]
            assert len(dim_warnings) == 0

    def test_effective_rank_in_result(self):
        """effective_rank should be present and an int."""
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((200, 5))
        pa = PersistenceAnalyzer(max_dim=1)
        result = pa.fit_transform(cloud, subsample=100, seed=42)
        assert "effective_rank" in result
        assert isinstance(result["effective_rank"], int)
