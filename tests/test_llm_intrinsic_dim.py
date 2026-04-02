"""Tests for att.llm.intrinsic_dim — TwoNN and PHD dimension estimators."""

import os
import tempfile

import numpy as np
import pytest

from att.llm.intrinsic_dim import twonn_dimension, phd_dimension, id_profile
from att.llm.loader import HiddenStateLoader


class TestTwoNN:
    def test_line_in_3d(self):
        """Points on a line in R^3 should have low ID (near 1)."""
        rng = np.random.default_rng(42)
        n = 500
        t = rng.uniform(0, 10, n)
        cloud = np.column_stack([t, np.zeros(n), np.zeros(n)])
        cloud += rng.standard_normal(cloud.shape) * 0.001
        d = twonn_dimension(cloud)
        assert 0.5 < d < 3.0, f"Expected ~1.0, got {d}"

    def test_plane_in_10d(self):
        """Points on a 2D plane embedded in R^10 should have ID ≈ 2."""
        rng = np.random.default_rng(42)
        n = 300
        coords = rng.standard_normal((n, 2))
        embedding = np.zeros((n, 10))
        embedding[:, 0] = coords[:, 0]
        embedding[:, 1] = coords[:, 1]
        embedding += rng.standard_normal((n, 10)) * 0.01
        d = twonn_dimension(embedding)
        assert 1.3 < d < 3.5, f"Expected ~2.0, got {d}"

    def test_3d_ball_in_10d(self):
        """Points in a 3D ball embedded in R^10 should have ID ≈ 3."""
        rng = np.random.default_rng(42)
        n = 500
        coords = rng.standard_normal((n, 3))
        embedding = np.zeros((n, 10))
        embedding[:, :3] = coords
        embedding += rng.standard_normal((n, 10)) * 0.01
        d = twonn_dimension(embedding)
        assert 2.0 < d < 4.5, f"Expected ~3.0, got {d}"

    def test_too_few_points(self):
        cloud = np.array([[0, 0], [1, 1]])
        d = twonn_dimension(cloud)
        assert d == 0.0

    def test_fraction_parameter(self):
        """Different fractions should give different (but reasonable) estimates."""
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((200, 5))
        d1 = twonn_dimension(cloud, fraction=0.5)
        d2 = twonn_dimension(cloud, fraction=0.9)
        assert d1 > 0
        assert d2 > 0

    def test_large_cloud_uses_knn(self):
        """Clouds > 5000 points trigger the NearestNeighbors path."""
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((5001, 3))
        d = twonn_dimension(cloud)
        assert 2.0 < d < 4.5


class TestPHD:
    def test_basic_output(self):
        """PHD on synthetic diagrams returns a positive value."""
        # Fake persistence diagram with known lifetimes
        dgm = np.array([
            [0.0, 0.5],
            [0.1, 0.6],
            [0.2, 0.8],
            [0.3, 0.4],
            [0.05, 0.9],
            [0.1, 0.3],
        ])
        d = phd_dimension([np.empty((0, 2)), dgm], dim=1)
        assert d > 0

    def test_empty_diagram(self):
        d = phd_dimension([np.empty((0, 2))], dim=0)
        assert d == 0.0

    def test_too_few_features(self):
        dgm = np.array([[0.0, 0.5], [0.1, 0.6]])
        d = phd_dimension([dgm], dim=0)
        assert d == 0.0  # < 5 features

    def test_dim_out_of_range(self):
        dgm = np.array([[0.0, 0.5]] * 10)
        d = phd_dimension([dgm], dim=5)
        assert d == 0.0


@pytest.fixture
def small_loader():
    """Small synthetic loader: 3 layers, 2 levels."""
    n_per_level = 20
    hidden_dim = 16
    num_layers = 3
    n_problems = n_per_level * 2
    levels = np.array([1] * n_per_level + [5] * n_per_level)
    rng = np.random.default_rng(42)

    last_hidden = rng.standard_normal((n_problems, hidden_dim)).astype(np.float32)
    layer_hidden = rng.standard_normal(
        (n_problems, num_layers, hidden_dim)
    ).astype(np.float32)

    token_trajs = np.empty(n_problems, dtype=object)
    seq_lengths = np.full(n_problems, 10, dtype=int)
    for i in range(n_problems):
        token_trajs[i] = rng.standard_normal((10, hidden_dim)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez_compressed(
            f.name,
            last_hidden_states=last_hidden,
            difficulty_levels=levels,
            layer_hidden_states=layer_hidden,
            token_trajectories=token_trajs,
            seq_lengths=seq_lengths,
            model_name=np.array("test-model"),
            hidden_dim=np.array(hidden_dim),
            num_layers=np.array(num_layers),
            skipped_indices=np.array([]),
        )
        path = f.name

    loader = HiddenStateLoader(path)
    yield loader
    os.unlink(path)


class TestIdProfile:
    def test_twonn_profile_shape(self, small_loader):
        profiles = id_profile(small_loader, levels=[1, 5], method="twonn")
        assert set(profiles.keys()) == {1, 5}
        for level, ids in profiles.items():
            assert ids.shape == (3,)  # 3 layers
            assert np.all(ids >= 0)

    def test_phd_profile_shape(self, small_loader):
        profiles = id_profile(small_loader, levels=[1, 5], method="phd")
        assert set(profiles.keys()) == {1, 5}
        for level, ids in profiles.items():
            assert ids.shape == (3,)

    def test_default_all_levels(self, small_loader):
        profiles = id_profile(small_loader)
        assert set(profiles.keys()) == {1, 5}

    def test_invalid_method(self, small_loader):
        with pytest.raises(ValueError, match="Unknown method"):
            id_profile(small_loader, method="invalid")
