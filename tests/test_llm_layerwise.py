"""Tests for att.llm.layerwise — LayerwiseAnalyzer."""

import os
import tempfile

import numpy as np
import pytest

from att.llm.loader import HiddenStateLoader
from att.llm.layerwise import LayerwiseAnalyzer


@pytest.fixture
def small_loader():
    """Small synthetic loader: 3 layers, 2 levels, 20 problems each."""
    n_per_level = 20
    hidden_dim = 16
    num_layers = 3
    n_problems = n_per_level * 2
    levels = np.array([1] * n_per_level + [5] * n_per_level)
    rng = np.random.default_rng(42)

    # Make levels topologically distinguishable: different means
    last_hidden = rng.standard_normal((n_problems, hidden_dim)).astype(np.float32)
    last_hidden[n_per_level:] += 2.0  # shift level 5

    layer_hidden = rng.standard_normal(
        (n_problems, num_layers, hidden_dim)
    ).astype(np.float32)
    layer_hidden[n_per_level:] += 2.0

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


class TestLayerwiseAnalyzer:
    def test_fit_basic(self, small_loader):
        analyzer = LayerwiseAnalyzer(
            n_pca_components=10, max_dim=1, subsample=None,
            n_permutations=5, seed=42,
        )
        analyzer.fit(small_loader, levels=[1, 5])

        results = analyzer.results_per_layer
        # Should have entries for 2 levels × 3 layers = 6
        assert len(results) == 6
        for (level, layer_idx), result in results.items():
            assert level in [1, 5]
            assert 0 <= layer_idx < 3
            assert "diagrams" in result
            assert "persistence_entropy" in result

    def test_entropy_profile_shape(self, small_loader):
        analyzer = LayerwiseAnalyzer(
            n_pca_components=10, max_dim=1, subsample=None,
            n_permutations=5, seed=42,
        )
        analyzer.fit(small_loader, levels=[1, 5])

        entropy = analyzer.entropy_profile()
        assert set(entropy.keys()) == {1, 5}
        for level, ent in entropy.items():
            assert ent.shape == (3, 2)  # 3 layers, max_dim+1=2
            assert np.all(ent >= 0)

    def test_bottleneck_profile_shape(self, small_loader):
        analyzer = LayerwiseAnalyzer(
            n_pca_components=10, max_dim=1, subsample=None,
            n_permutations=5, seed=42,
        )
        analyzer.fit(small_loader, levels=[1, 5])

        bottleneck = analyzer.bottleneck_profile()
        assert set(bottleneck.keys()) == {1, 5}
        for level, dists in bottleneck.items():
            assert dists.shape == (2,)  # 3 layers - 1 = 2 transitions
            assert np.all(dists >= 0)

    def test_zscore_profile_shape(self, small_loader):
        analyzer = LayerwiseAnalyzer(
            n_pca_components=10, max_dim=1, subsample=None,
            n_permutations=5, seed=42,
        )
        analyzer.fit(small_loader, levels=[1, 5])

        result = analyzer.zscore_profile(small_loader, metric="wasserstein_1")

        assert result["z_scores"].shape == (3,)
        assert result["p_values"].shape == (3,)
        assert result["observed"].shape == (3,)
        assert result["null_mean"].shape == (3,)
        assert result["null_std"].shape == (3,)
        assert 0 in result["per_dim"]
        assert 1 in result["per_dim"]

    def test_zscore_p_values_valid(self, small_loader):
        analyzer = LayerwiseAnalyzer(
            n_pca_components=10, max_dim=1, subsample=None,
            n_permutations=10, seed=42,
        )
        analyzer.fit(small_loader, levels=[1, 5])
        result = analyzer.zscore_profile(small_loader)

        assert np.all(result["p_values"] >= 0)
        assert np.all(result["p_values"] <= 1)

    def test_raises_before_fit(self):
        analyzer = LayerwiseAnalyzer()
        with pytest.raises(RuntimeError, match="Call fit"):
            _ = analyzer.results_per_layer
        with pytest.raises(RuntimeError, match="Call fit"):
            analyzer.entropy_profile()
        with pytest.raises(RuntimeError, match="Call fit"):
            analyzer.bottleneck_profile()
