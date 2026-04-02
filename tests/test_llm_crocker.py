"""Tests for att.llm.crocker — CROCKERMatrix."""

import os
import tempfile

import numpy as np
import pytest

from att.llm.loader import HiddenStateLoader
from att.llm.crocker import CROCKERMatrix


@pytest.fixture
def small_loader():
    """Small synthetic loader: 4 layers, 3 levels."""
    n_per_level = 15
    hidden_dim = 16
    num_layers = 4
    n_problems = n_per_level * 3
    levels = np.array([1] * n_per_level + [3] * n_per_level + [5] * n_per_level)
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


class TestCROCKERMatrix:
    def test_fit_by_difficulty_shapes(self, small_loader):
        cm = CROCKERMatrix(
            n_filtration_steps=50, max_dim=1,
            n_pca_components=10, subsample=None, seed=42,
        )
        cm.fit_by_difficulty(small_loader, layer=-1, levels=[1, 3, 5])

        matrices = cm.betti_matrices
        assert 0 in matrices
        assert 1 in matrices
        # (n_filtration_steps, n_levels)
        assert matrices[0].shape == (50, 3)
        assert matrices[1].shape == (50, 3)

    def test_fit_by_layer_shapes(self, small_loader):
        cm = CROCKERMatrix(
            n_filtration_steps=50, max_dim=1,
            n_pca_components=10, subsample=None, seed=42,
        )
        cm.fit_by_layer(small_loader, level=1)

        matrices = cm.betti_matrices
        # (n_filtration_steps, n_layers=4)
        assert matrices[1].shape == (50, 4)

    def test_parameter_labels(self, small_loader):
        cm = CROCKERMatrix(n_filtration_steps=50, max_dim=1, seed=42)
        cm.fit_by_difficulty(small_loader, levels=[1, 3, 5])
        assert cm.parameter_labels == ["L1", "L3", "L5"]

        cm2 = CROCKERMatrix(n_filtration_steps=50, max_dim=1, seed=42)
        cm2.fit_by_layer(small_loader, level=1)
        assert cm2.parameter_labels == ["Ly0", "Ly1", "Ly2", "Ly3"]

    def test_filtration_range(self, small_loader):
        cm = CROCKERMatrix(n_filtration_steps=50, max_dim=1, seed=42)
        cm.fit_by_difficulty(small_loader, levels=[1, 5])
        lo, hi = cm.filtration_range
        assert lo < hi
        assert lo >= 0

    def test_betti_non_negative(self, small_loader):
        cm = CROCKERMatrix(n_filtration_steps=50, max_dim=1, seed=42)
        cm.fit_by_difficulty(small_loader, levels=[1, 5])
        for dim, mat in cm.betti_matrices.items():
            assert np.all(mat >= 0), f"Negative Betti numbers in H{dim}"

    def test_l1_distances_symmetric(self, small_loader):
        cm = CROCKERMatrix(n_filtration_steps=50, max_dim=1, seed=42)
        cm.fit_by_difficulty(small_loader, levels=[1, 3, 5])
        l1 = cm.pairwise_l1_distances(dim=1)

        assert l1.shape == (3, 3)
        # Symmetric
        np.testing.assert_array_almost_equal(l1, l1.T)
        # Non-negative
        assert np.all(l1 >= 0)
        # Zero diagonal
        np.testing.assert_array_almost_equal(np.diag(l1), 0)

    def test_raises_before_fit(self):
        cm = CROCKERMatrix()
        with pytest.raises(RuntimeError, match="Call fit"):
            _ = cm.betti_matrices
        with pytest.raises(RuntimeError, match="Call fit"):
            _ = cm.parameter_labels
        with pytest.raises(RuntimeError, match="Call fit"):
            _ = cm.filtration_range
