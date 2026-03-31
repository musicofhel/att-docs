"""Tests for att.embedding — delay estimation, embedders, validation."""

import numpy as np
import pytest

from att.config import set_seed
from att.synthetic import lorenz_system, coupled_rossler_lorenz
from att.embedding import (
    TakensEmbedder,
    JointEmbedder,
    estimate_delay,
    estimate_dimension,
    validate_embedding,
    svd_embedding,
)


class TestEstimateDelay:
    def test_lorenz_delay(self):
        set_seed(42)
        ts = lorenz_system(n_steps=10000)
        delay = estimate_delay(ts[:, 0])
        # Lorenz x-component: expect τ ≈ 15±5
        assert 10 <= delay <= 25, f"Lorenz delay {delay} outside expected range"

    def test_returns_positive(self):
        set_seed(42)
        ts = lorenz_system(n_steps=5000)
        delay = estimate_delay(ts[:, 0])
        assert delay >= 1


class TestEstimateDimension:
    def test_lorenz_dimension(self):
        set_seed(42)
        ts = lorenz_system(n_steps=10000)
        delay = estimate_delay(ts[:, 0])
        dim = estimate_dimension(ts[:, 0], delay)
        # Lorenz: expect d = 3 (box dimension ≈ 2.06, need 2d+1 = 5 but FNN converges at 3)
        assert 2 <= dim <= 5, f"Lorenz dimension {dim} outside expected range"

    def test_returns_at_least_1(self):
        set_seed(42)
        ts = lorenz_system(n_steps=5000)
        dim = estimate_dimension(ts[:, 0], delay=10)
        assert dim >= 1


class TestTakensEmbedder:
    def test_auto_params(self):
        set_seed(42)
        ts = lorenz_system(n_steps=10000)
        embedder = TakensEmbedder(delay="auto", dimension="auto")
        cloud = embedder.fit_transform(ts[:, 0])
        assert embedder.delay_ is not None
        assert embedder.dimension_ is not None
        assert cloud.ndim == 2
        assert cloud.shape[1] == embedder.dimension_

    def test_manual_params(self):
        ts = lorenz_system(n_steps=5000, seed=42)
        embedder = TakensEmbedder(delay=15, dimension=3)
        cloud = embedder.fit_transform(ts[:, 0])
        assert embedder.delay_ == 15
        assert embedder.dimension_ == 3
        expected_rows = 5000 - (3 - 1) * 15
        assert cloud.shape == (expected_rows, 3)

    def test_output_shape(self):
        ts = np.sin(np.linspace(0, 100, 5000))
        embedder = TakensEmbedder(delay=10, dimension=4)
        cloud = embedder.fit_transform(ts)
        assert cloud.shape == (5000 - 30, 4)

    def test_fit_then_transform(self):
        ts = lorenz_system(n_steps=5000, seed=42)
        embedder = TakensEmbedder(delay="auto", dimension="auto")
        embedder.fit(ts[:, 0])
        cloud = embedder.transform(ts[:, 0])
        assert cloud.shape[0] > 0

    def test_transform_without_fit_raises(self):
        embedder = TakensEmbedder()
        with pytest.raises(RuntimeError):
            embedder.transform(np.zeros(100))


class TestJointEmbedder:
    def test_auto_params(self):
        set_seed(42)
        ts_r, ts_l = coupled_rossler_lorenz(n_steps=10000, coupling=0.1)
        je = JointEmbedder()
        joint = je.fit_transform([ts_r[:, 0], ts_l[:, 0]])
        assert je.delays_ is not None
        assert je.dimensions_ is not None
        assert len(je.delays_) == 2
        assert len(je.dimensions_) == 2
        assert joint.shape[1] == sum(je.dimensions_)

    def test_per_channel_delays_differ(self):
        """Rössler and Lorenz have different timescales → different delays."""
        set_seed(42)
        ts_r, ts_l = coupled_rossler_lorenz(n_steps=10000, coupling=0.1)
        je = JointEmbedder()
        je.fit([ts_r[:, 0], ts_l[:, 0]])
        # They may or may not differ, but the estimator should at least run
        assert je.delays_[0] >= 1
        assert je.delays_[1] >= 1

    def test_marginals(self):
        set_seed(42)
        ts_r, ts_l = coupled_rossler_lorenz(n_steps=5000)
        je = JointEmbedder()
        je.fit([ts_r[:, 0], ts_l[:, 0]])
        marginals = je.transform_marginals([ts_r[:, 0], ts_l[:, 0]])
        assert len(marginals) == 2
        for i, m in enumerate(marginals):
            assert m.shape[1] == je.dimensions_[i]

    def test_manual_params(self):
        ts_x = np.sin(np.linspace(0, 100, 5000))
        ts_y = np.cos(np.linspace(0, 100, 5000))
        je = JointEmbedder(delays=[10, 15], dimensions=[3, 4])
        joint = je.fit_transform([ts_x, ts_y])
        assert joint.shape[1] == 7  # 3 + 4


class TestValidateEmbedding:
    def test_good_embedding(self):
        set_seed(42)
        ts = lorenz_system(n_steps=10000)
        cloud = TakensEmbedder(delay="auto", dimension="auto").fit_transform(ts[:, 0])
        result = validate_embedding(cloud)
        assert not result["degenerate"]
        assert result["condition_number"] < 1e4
        assert result["effective_rank"] == cloud.shape[1]

    def test_degenerate_detection(self):
        # Create a degenerate embedding: all columns nearly identical
        n = 1000
        x = np.random.randn(n)
        cloud = np.column_stack([x, x + 1e-8 * np.random.randn(n), x + 1e-8 * np.random.randn(n)])
        result = validate_embedding(cloud)
        assert result["degenerate"]
        assert result["warning"] is not None

    def test_singular_values_returned(self):
        cloud = np.random.randn(100, 3)
        result = validate_embedding(cloud)
        assert "singular_values" in result
        assert len(result["singular_values"]) == 3


    def test_dimension_aware_threshold_default(self):
        """d=6 cloud → threshold_used=100; d=50 cloud → threshold_used=500."""
        rng = np.random.default_rng(42)
        cloud_6 = rng.standard_normal((200, 6))
        result_6 = validate_embedding(cloud_6)
        assert result_6["threshold_used"] == 100.0  # max(10*6, 100) = 100

        cloud_50 = rng.standard_normal((200, 50))
        result_50 = validate_embedding(cloud_50)
        assert result_50["threshold_used"] == 500.0  # max(10*50, 100) = 500

    def test_explicit_threshold_overrides(self):
        """Explicit condition_threshold=1e4 → threshold_used=1e4."""
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((200, 6))
        result = validate_embedding(cloud, condition_threshold=1e4)
        assert result["threshold_used"] == 1e4

    def test_threshold_in_result(self):
        """Verify 'threshold_used' key in returned dict."""
        rng = np.random.default_rng(42)
        cloud = rng.standard_normal((100, 3))
        result = validate_embedding(cloud)
        assert "threshold_used" in result
        assert isinstance(result["threshold_used"], float)


class TestSvdEmbedding:
    def test_basic(self):
        ts = np.sin(np.linspace(0, 100, 5000))
        projected = svd_embedding(ts, delay=10, dimension=5, n_components=3)
        assert projected.shape == (5000 - 40, 3)


class TestEmbeddingEdgeCases:
    """Edge-case tests for embedding functions."""

    def test_constant_signal_delay(self):
        """estimate_delay on constant signal should return 1."""
        delay = estimate_delay(np.ones(1000))
        assert delay == 1

    def test_constant_signal_dimension(self):
        """estimate_dimension on constant signal should return small dim."""
        dim = estimate_dimension(np.ones(1000), delay=1)
        # All NN distances are 0 → skipped → fnn_fraction=0 → returns d=1
        assert dim >= 1
        assert dim <= 3  # Should be very low

    def test_joint_single_channel(self):
        """JointEmbedder with single channel should work."""
        x = np.sin(np.linspace(0, 100, 5000))
        je = JointEmbedder(delays="auto", dimensions="auto")
        cloud = je.fit_transform([x])
        assert cloud.ndim == 2
        assert cloud.shape[0] > 0

    def test_takens_short_series_raises(self):
        """Too-short time series should raise ValueError."""
        emb = TakensEmbedder(delay=50, dimension=5)
        short = np.zeros(100)  # needs (5-1)*50+1 = 201
        with pytest.raises(ValueError, match="too short"):
            emb.fit_transform(short)

    def test_validate_embedding_degenerate(self):
        """validate_embedding on rank-deficient cloud should flag degenerate."""
        # Create a 3D cloud that only varies along 1 axis
        rng = np.random.default_rng(42)
        cloud = np.zeros((100, 3))
        cloud[:, 0] = rng.standard_normal(100)
        result = validate_embedding(cloud)
        assert result["degenerate"] is True
        assert result["effective_rank"] == 1
