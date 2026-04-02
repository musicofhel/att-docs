"""Tests for att.llm.features — TopologicalFeatureExtractor."""

import os
import tempfile

import numpy as np
import pytest

from att.llm.features import TopologicalFeatureExtractor
from att.llm.loader import HiddenStateLoader


@pytest.fixture
def small_loader():
    """Small synthetic loader: 3 layers, 3 levels, 15 problems each."""
    n_per_level = 15
    hidden_dim = 16
    num_layers = 3
    n_problems = n_per_level * 3
    levels = np.array([1] * n_per_level + [3] * n_per_level + [5] * n_per_level)
    rng = np.random.default_rng(42)

    last_hidden = rng.standard_normal((n_problems, hidden_dim)).astype(np.float32)
    layer_hidden = rng.standard_normal(
        (n_problems, num_layers, hidden_dim)
    ).astype(np.float32)

    token_trajs = np.empty(n_problems, dtype=object)
    seq_lengths = np.full(n_problems, 12, dtype=int)
    for i in range(n_problems):
        token_trajs[i] = rng.standard_normal((12, hidden_dim)).astype(np.float32)

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


class TestTopologicalFeatureExtractor:
    def test_summary_feature_names(self):
        tfe = TopologicalFeatureExtractor(max_dim=1, feature_set="summary")
        names = tfe.feature_names
        assert len(names) == tfe.n_features
        # 8 summary features × 2 dims = 16
        assert len(names) == 16
        assert "H0_persistence_entropy" in names
        assert "H1_total_persistence" in names

    def test_image_feature_names(self):
        tfe = TopologicalFeatureExtractor(
            max_dim=1, feature_set="image", pi_resolution=10
        )
        names = tfe.feature_names
        # 16 summary + 2 × 10 × 10 = 216
        assert len(names) == 16 + 200
        assert len(names) == tfe.n_features

    def test_extract_single_summary(self):
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((30, 8))
        tfe = TopologicalFeatureExtractor(
            max_dim=1, feature_set="summary", subsample=None, seed=42
        )
        features = tfe.extract_single(cloud)
        assert features.shape == (tfe.n_features,)
        # At least some features should be non-zero
        assert np.any(features != 0)

    def test_extract_single_image(self):
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((30, 8))
        tfe = TopologicalFeatureExtractor(
            max_dim=1, feature_set="image", pi_resolution=10,
            subsample=None, seed=42,
        )
        features = tfe.extract_single(cloud)
        assert features.shape == (tfe.n_features,)

    def test_extract_single_too_few_points(self):
        cloud = np.array([[0, 0], [1, 1]])
        tfe = TopologicalFeatureExtractor(max_dim=1, feature_set="summary")
        features = tfe.extract_single(cloud)
        assert features.shape == (tfe.n_features,)
        np.testing.assert_array_equal(features, 0)

    def test_extract_batch_shape(self, small_loader):
        tfe = TopologicalFeatureExtractor(
            max_dim=1, feature_set="summary", subsample=None, seed=42,
        )
        X, names = tfe.extract_batch(small_loader, layer=-1)
        # 3 levels → 3 rows
        assert X.shape == (3, tfe.n_features)
        assert len(names) == tfe.n_features

    def test_extract_per_problem_shape(self, small_loader):
        tfe = TopologicalFeatureExtractor(
            max_dim=1, feature_set="summary", subsample=None, seed=42,
        )
        X, levels = tfe.extract_per_problem(small_loader)
        assert X.shape == (45, tfe.n_features)
        assert levels.shape == (45,)

    def test_max_dim_2(self):
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((40, 10))
        tfe = TopologicalFeatureExtractor(
            max_dim=2, feature_set="summary", subsample=None, seed=42,
        )
        features = tfe.extract_single(cloud)
        # 8 features × 3 dims = 24
        assert features.shape == (24,)
        assert tfe.n_features == 24
