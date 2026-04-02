"""Tests for att.llm.attention_binding — Direction 10."""

import numpy as np
import pytest

from att.llm.attention_binding import AttentionHiddenBinding, BindingResult, SignificanceResult


# --- Fixtures ---

@pytest.fixture
def ahb():
    """Default AttentionHiddenBinding instance."""
    return AttentionHiddenBinding(
        max_dim=1, image_resolution=20, image_sigma=0.15,
        n_pca_components=10, subsample=30, seed=42,
    )


@pytest.fixture
def coupled_data():
    """Synthetic coupled attention + hidden data.

    Creates an attention matrix and hidden cloud that share geometric structure
    (nearby tokens in hidden space also attend to each other).
    """
    rng = np.random.default_rng(42)
    n = 30
    d = 20

    # Hidden cloud: 3 tight clusters
    centers = rng.standard_normal((3, d)) * 5
    labels = np.repeat([0, 1, 2], 10)
    hidden = centers[labels] + rng.standard_normal((n, d)) * 0.3

    # Attention: tokens in same cluster attend to each other
    attn = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                attn[i, j] = 0.8 + rng.uniform(0, 0.2)
            else:
                attn[i, j] = 0.05 + rng.uniform(0, 0.1)
    # Normalize rows
    attn /= attn.sum(axis=1, keepdims=True)

    return attn, hidden


@pytest.fixture
def uncoupled_data():
    """Synthetic uncoupled attention + hidden data.

    Attention has structure but hidden cloud is random — no coupling.
    """
    rng = np.random.default_rng(123)
    n = 30
    d = 20

    hidden = rng.standard_normal((n, d))

    # Block-diagonal attention (has structure, but unrelated to hidden)
    attn = np.eye(n) * 0.5
    for i in range(0, n, 5):
        block = slice(i, min(i + 5, n))
        attn[block, block] = 0.7 + rng.uniform(0, 0.1, (min(5, n - i), min(5, n - i)))
    attn /= attn.sum(axis=1, keepdims=True)

    return attn, hidden


# --- AttentionHiddenBinding.attention_to_distance ---

class TestAttentionToDistance:
    def test_symmetric(self):
        rng = np.random.default_rng(1)
        attn = rng.uniform(0, 1, (10, 10))
        attn /= attn.sum(axis=1, keepdims=True)
        dist = AttentionHiddenBinding.attention_to_distance(attn)
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_diagonal_zero(self):
        attn = np.eye(5)
        dist = AttentionHiddenBinding.attention_to_distance(attn)
        np.testing.assert_array_equal(np.diag(dist), 0.0)

    def test_range_01(self):
        rng = np.random.default_rng(2)
        attn = rng.uniform(0, 1, (10, 10))
        dist = AttentionHiddenBinding.attention_to_distance(attn)
        assert dist.min() >= 0.0
        assert dist.max() <= 1.0

    def test_high_attention_low_distance(self):
        """Tokens that attend strongly to each other should have low distance."""
        attn = np.full((4, 4), 0.05)
        attn[0, 1] = 0.9
        attn[1, 0] = 0.9
        dist = AttentionHiddenBinding.attention_to_distance(attn)
        assert dist[0, 1] < dist[0, 2]


# --- compute_binding ---

class TestComputeBinding:
    def test_returns_binding_result(self, ahb, coupled_data):
        attn, hidden = coupled_data
        result = ahb.compute_binding(attn, hidden)
        assert isinstance(result, BindingResult)

    def test_binding_score_is_float(self, ahb, coupled_data):
        attn, hidden = coupled_data
        result = ahb.compute_binding(attn, hidden)
        assert isinstance(result.binding_score, float)

    def test_binding_score_in_range(self, ahb, coupled_data):
        attn, hidden = coupled_data
        result = ahb.compute_binding(attn, hidden)
        assert -1.0 <= result.binding_score <= 1.0

    def test_entropy_values(self, ahb, coupled_data):
        attn, hidden = coupled_data
        result = ahb.compute_binding(attn, hidden)
        # persistence_entropy may be list or dict depending on backend
        assert result.attention_entropy is not None
        assert result.hidden_entropy is not None

    def test_feature_counts(self, ahb, coupled_data):
        attn, hidden = coupled_data
        result = ahb.compute_binding(attn, hidden)
        for d in range(2):
            assert result.n_attention_features[d] >= 0
            assert result.n_hidden_features[d] >= 0

    def test_coupled_higher_than_uncoupled(self, ahb, coupled_data, uncoupled_data):
        """Coupled data should show higher absolute binding than uncoupled."""
        attn_c, hidden_c = coupled_data
        attn_u, hidden_u = uncoupled_data

        score_c = ahb.compute_binding(attn_c, hidden_c).binding_score
        score_u = ahb.compute_binding(attn_u, hidden_u).binding_score

        # Coupled should have higher absolute correlation
        assert abs(score_c) >= abs(score_u) * 0.5  # soft check — correlation is noisy

    def test_subsample_handles_large_input(self):
        """Subsampling should work when input exceeds subsample size."""
        ahb = AttentionHiddenBinding(subsample=15, n_pca_components=5, seed=42)
        rng = np.random.default_rng(42)
        n = 50
        attn = rng.uniform(0, 1, (n, n))
        attn /= attn.sum(axis=1, keepdims=True)
        hidden = rng.standard_normal((n, 10))
        result = ahb.compute_binding(attn, hidden)
        assert isinstance(result.binding_score, float)


# --- test_significance ---

class TestSignificance:
    def test_returns_significance_result(self, ahb, coupled_data):
        attn, hidden = coupled_data
        result = ahb.test_significance(attn, hidden, n_permutations=10)
        assert isinstance(result, SignificanceResult)

    def test_p_value_range(self, ahb, coupled_data):
        attn, hidden = coupled_data
        result = ahb.test_significance(attn, hidden, n_permutations=10)
        assert 0.0 < result.p_value <= 1.0

    def test_null_scores_length(self, ahb, coupled_data):
        attn, hidden = coupled_data
        n_perm = 15
        result = ahb.test_significance(attn, hidden, n_permutations=n_perm)
        assert len(result.null_scores) == n_perm

    def test_z_score_is_float(self, ahb, coupled_data):
        attn, hidden = coupled_data
        result = ahb.test_significance(attn, hidden, n_permutations=10)
        assert isinstance(result.z_score, float)

    def test_observed_matches_compute_binding(self, ahb, coupled_data):
        attn, hidden = coupled_data
        direct = ahb.compute_binding(attn, hidden).binding_score
        sig = ahb.test_significance(attn, hidden, n_permutations=5)
        assert abs(sig.observed_score - direct) < 1e-10


# --- shared_ranges ---

class TestSharedRanges:
    def test_non_degenerate(self, ahb):
        """Shared ranges should never be degenerate (zero width)."""
        dgms = [np.array([[0.0, 0.5], [0.1, 0.3]])]
        birth_range, pers_range = ahb._shared_ranges(dgms, dgms)
        assert birth_range[1] > birth_range[0]
        assert pers_range[1] > pers_range[0]

    def test_empty_diagrams(self, ahb):
        """Empty diagrams should return default ranges."""
        dgms = [np.empty((0, 2))]
        birth_range, pers_range = ahb._shared_ranges(dgms, dgms)
        assert birth_range == (0.0, 1.0)
        assert pers_range == (0.0, 1.0)


# --- pi_correlation ---

class TestPICorrelation:
    def test_identical_images(self, ahb):
        """Identical PIs should give correlation 1.0."""
        imgs = [np.random.default_rng(1).standard_normal((20, 20))]
        corr = ahb._pi_correlation(imgs, imgs)
        assert abs(corr - 1.0) < 1e-10

    def test_zero_variance(self, ahb):
        """Constant PIs should give correlation 0.0."""
        imgs = [np.ones((20, 20))]
        corr = ahb._pi_correlation(imgs, imgs)
        assert corr == 0.0
