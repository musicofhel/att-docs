"""Tests for att.llm.loader — HiddenStateLoader."""

import os
import tempfile

import numpy as np
import pytest

from att.llm.loader import HiddenStateLoader


@pytest.fixture
def synthetic_npz():
    """Create a synthetic .npz mimicking extract_hidden_states.py output."""
    n_problems = 50
    hidden_dim = 32
    num_layers = 5  # includes embedding layer
    levels = np.array([1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10)
    rng = np.random.default_rng(42)

    last_hidden = rng.standard_normal((n_problems, hidden_dim)).astype(np.float32)
    layer_hidden = rng.standard_normal(
        (n_problems, num_layers, hidden_dim)
    ).astype(np.float32)

    # Variable-length token trajectories
    token_trajs = np.empty(n_problems, dtype=object)
    seq_lengths = np.zeros(n_problems, dtype=int)
    for i in range(n_problems):
        t = rng.integers(10, 50)
        token_trajs[i] = rng.standard_normal((t, hidden_dim)).astype(np.float32)
        seq_lengths[i] = t

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez_compressed(
            f.name,
            last_hidden_states=last_hidden,
            difficulty_levels=levels,
            layer_hidden_states=layer_hidden,
            token_trajectories=token_trajs,
            seq_lengths=seq_lengths,
            model_name=np.array("test-model/1.5B"),
            hidden_dim=np.array(hidden_dim),
            num_layers=np.array(num_layers),
            skipped_indices=np.array([]),
        )
        path = f.name

    yield path, n_problems, hidden_dim, num_layers, levels

    os.unlink(path)


class TestHiddenStateLoader:
    def test_basic_properties(self, synthetic_npz):
        path, n_problems, hidden_dim, num_layers, _ = synthetic_npz
        loader = HiddenStateLoader(path)

        assert loader.n_problems == n_problems
        assert loader.hidden_dim == hidden_dim
        assert loader.num_layers == num_layers
        assert loader.model_name == "test-model/1.5B"

    def test_shapes(self, synthetic_npz):
        path, n_problems, hidden_dim, num_layers, _ = synthetic_npz
        loader = HiddenStateLoader(path)

        assert loader.last_hidden.shape == (n_problems, hidden_dim)
        assert loader.levels.shape == (n_problems,)
        assert loader.layer_hidden.shape == (n_problems, num_layers, hidden_dim)
        assert loader.seq_lengths.shape == (n_problems,)

    def test_unique_levels(self, synthetic_npz):
        path, *_ = synthetic_npz
        loader = HiddenStateLoader(path)
        np.testing.assert_array_equal(loader.unique_levels, [1, 2, 3, 4, 5])

    def test_level_counts(self, synthetic_npz):
        path, *_ = synthetic_npz
        loader = HiddenStateLoader(path)
        counts = loader.level_counts()
        for level in range(1, 6):
            assert counts[level] == 10

    def test_get_level_mask(self, synthetic_npz):
        path, *_ = synthetic_npz
        loader = HiddenStateLoader(path)
        mask = loader.get_level_mask(3)
        assert mask.sum() == 10
        assert mask.dtype == bool

    def test_get_level_cloud_no_layer(self, synthetic_npz):
        path, _, hidden_dim, *_ = synthetic_npz
        loader = HiddenStateLoader(path)
        cloud = loader.get_level_cloud(1)
        assert cloud.shape == (10, hidden_dim)

    def test_get_level_cloud_with_layer(self, synthetic_npz):
        path, _, hidden_dim, *_ = synthetic_npz
        loader = HiddenStateLoader(path)
        cloud = loader.get_level_cloud(2, layer=0)
        assert cloud.shape == (10, hidden_dim)

        cloud_last = loader.get_level_cloud(2, layer=-1)
        assert cloud_last.shape == (10, hidden_dim)

    def test_get_layer_cloud_all(self, synthetic_npz):
        path, n_problems, hidden_dim, *_ = synthetic_npz
        loader = HiddenStateLoader(path)
        cloud = loader.get_layer_cloud(0)
        assert cloud.shape == (n_problems, hidden_dim)

    def test_get_layer_cloud_filtered(self, synthetic_npz):
        path, _, hidden_dim, *_ = synthetic_npz
        loader = HiddenStateLoader(path)
        cloud = loader.get_layer_cloud(0, levels=[1, 5])
        assert cloud.shape == (20, hidden_dim)

    def test_token_trajectories(self, synthetic_npz):
        path, n_problems, hidden_dim, *_ = synthetic_npz
        loader = HiddenStateLoader(path)
        trajs = loader.token_trajectories
        assert len(trajs) == n_problems
        for traj in trajs:
            assert traj.shape[1] == hidden_dim
            assert 10 <= traj.shape[0] <= 50

    def test_repr(self, synthetic_npz):
        path, *_ = synthetic_npz
        loader = HiddenStateLoader(path)
        r = repr(loader)
        assert "test-model/1.5B" in r
        assert "n=50" in r
        assert "d=32" in r
