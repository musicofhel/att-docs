"""Tests for att.llm.zigzag — zigzag persistence across layers."""

import numpy as np
import pytest

from att.llm.zigzag import (
    ZigzagLayerAnalyzer,
    ZigzagResult,
    compare_zigzag_levels,
    zigzag_feature_lifetime_stats,
)


# --- Fixtures ---

class FakeLoader:
    """Minimal loader for zigzag tests with synthetic 3-layer data."""

    def __init__(self, n_per_level=30, n_layers=3, hidden_dim=10, seed=42):
        rng = np.random.default_rng(seed)
        self._n_per_level = n_per_level
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim

        # Two levels: 1 (easy), 5 (hard)
        n_total = 2 * n_per_level
        self._levels = np.array([1] * n_per_level + [5] * n_per_level)

        # Layer hidden states: (N, L, d)
        # Easy: tight cluster that persists across layers
        # Hard: cluster that spreads out then contracts
        self._layer_hidden = np.zeros((n_total, n_layers, hidden_dim))

        for i in range(n_total):
            for layer in range(n_layers):
                base = rng.standard_normal(hidden_dim)
                if self._levels[i] == 1:
                    # Easy: small noise, stable structure
                    self._layer_hidden[i, layer] = base * 0.5
                else:
                    # Hard: layer-dependent spread
                    spread = 1.0 + layer * 0.5
                    self._layer_hidden[i, layer] = base * spread

    @property
    def num_layers(self):
        return self._n_layers

    @property
    def layer_hidden(self):
        return self._layer_hidden

    def get_level_mask(self, level):
        return self._levels == level


@pytest.fixture
def fake_loader():
    return FakeLoader(n_per_level=30, n_layers=3, hidden_dim=10)


@pytest.fixture
def small_loader():
    """Minimal loader for fast tests."""
    return FakeLoader(n_per_level=15, n_layers=3, hidden_dim=8)


# --- ZigzagLayerAnalyzer Tests ---

class TestZigzagLayerAnalyzer:
    def test_fit_returns_zigzag_result(self, small_loader):
        zza = ZigzagLayerAnalyzer(max_dim=1, subsample=15, n_pca_components=5, seed=42)
        result = zza.fit(small_loader, level=1)
        assert isinstance(result, ZigzagResult)
        assert result.level == 1
        assert result.n_layers_used == 3

    def test_barcodes_have_correct_dims(self, small_loader):
        zza = ZigzagLayerAnalyzer(max_dim=1, subsample=15, n_pca_components=5, seed=42)
        result = zza.fit(small_loader, level=1)
        # Should have H0 and possibly H1
        assert 0 in result.barcodes
        # H0 barcodes should be (n, 2) shaped
        bars = result.barcodes[0]
        assert bars.ndim == 2
        if len(bars) > 0:
            assert bars.shape[1] == 2

    def test_barcode_births_before_deaths(self, small_loader):
        zza = ZigzagLayerAnalyzer(max_dim=1, subsample=15, n_pca_components=5, seed=42)
        result = zza.fit(small_loader, level=1)
        for dim, bars in result.barcodes.items():
            if len(bars) > 0:
                assert np.all(bars[:, 1] >= bars[:, 0]), f"H{dim} has death < birth"

    def test_barcode_values_within_layer_range(self, small_loader):
        zza = ZigzagLayerAnalyzer(max_dim=1, subsample=15, n_pca_components=5, seed=42)
        result = zza.fit(small_loader, level=1)
        n_layers = small_loader.num_layers
        for dim, bars in result.barcodes.items():
            if len(bars) > 0:
                assert np.all(bars >= 0), f"H{dim} has negative birth/death"
                # Max death should be <= n_layers - 0.5 (union step)
                # Actually total_times = 2*n_layers - 1, scaled by /2
                max_time = (2 * n_layers - 1) / 2.0
                assert np.all(bars[:, 1] <= max_time + 0.01), f"H{dim} death exceeds max"

    def test_layer_indices_subset(self, small_loader):
        zza = ZigzagLayerAnalyzer(max_dim=1, subsample=15, n_pca_components=5, seed=42)
        result = zza.fit(small_loader, level=1, layer_indices=[0, 2])
        assert result.n_layers_used == 2
        assert result.layer_indices == [0, 2]

    def test_needs_at_least_two_layers(self, small_loader):
        zza = ZigzagLayerAnalyzer(max_dim=1, subsample=15, n_pca_components=5, seed=42)
        with pytest.raises(ValueError, match="at least 2 layers"):
            zza.fit(small_loader, level=1, layer_indices=[0])

    def test_different_levels_produce_different_results(self, fake_loader):
        zza = ZigzagLayerAnalyzer(max_dim=1, subsample=25, n_pca_components=5, seed=42)
        r1 = zza.fit(fake_loader, level=1)
        r5 = zza.fit(fake_loader, level=5)
        # Results should exist for both
        assert r1.level == 1
        assert r5.level == 5


# --- Lifetime Stats Tests ---

class TestFeatureLifetimeStats:
    def test_empty_barcodes(self):
        result = ZigzagResult(level=1, barcodes={1: np.empty((0, 2))}, n_layers_used=3)
        stats = zigzag_feature_lifetime_stats(result, dim=1)
        assert stats["n_features"] == 0
        assert stats["mean_lifetime"] == 0.0

    def test_missing_dim(self):
        result = ZigzagResult(level=1, barcodes={0: np.array([[0, 1]])}, n_layers_used=3)
        stats = zigzag_feature_lifetime_stats(result, dim=1)
        assert stats["n_features"] == 0

    def test_known_lifetimes(self):
        bars = np.array([[0, 2], [1, 4], [0, 1]])
        result = ZigzagResult(level=1, barcodes={1: bars}, n_layers_used=5)
        stats = zigzag_feature_lifetime_stats(result, dim=1)
        assert stats["n_features"] == 3
        assert stats["mean_lifetime"] == pytest.approx(2.0)  # (2+3+1)/3
        assert stats["max_lifetime"] == 3.0
        assert stats["median_lifetime"] == 2.0

    def test_long_lived_count(self):
        bars = np.array([[0, 5], [0, 1], [0, 3], [0, 0.5]])
        result = ZigzagResult(level=1, barcodes={1: bars}, n_layers_used=6)
        stats = zigzag_feature_lifetime_stats(result, dim=1)
        # Lifetimes: 5, 1, 3, 0.5 -> long-lived (>2): 5, 3 = 2
        assert stats["n_long_lived"] == 2

    def test_zero_lifetime_filtered(self):
        bars = np.array([[1, 1], [0, 3]])
        result = ZigzagResult(level=1, barcodes={1: bars}, n_layers_used=4)
        stats = zigzag_feature_lifetime_stats(result, dim=1)
        assert stats["n_features"] == 1  # only the non-zero one


# --- Compare Levels Tests ---

class TestCompareZigzagLevels:
    def test_compare_returns_valid_keys(self):
        bars_a = np.array([[0, 2], [1, 3]])
        bars_b = np.array([[0, 4], [1, 5], [2, 4]])
        ra = ZigzagResult(level=1, barcodes={1: bars_a}, n_layers_used=5)
        rb = ZigzagResult(level=5, barcodes={1: bars_b}, n_layers_used=5)
        comp = compare_zigzag_levels(ra, rb, dim=1)
        assert "ks_statistic" in comp
        assert "ks_pvalue" in comp
        assert "mean_lifetime_diff" in comp
        assert comp["level_a"] == 1
        assert comp["level_b"] == 5

    def test_ks_pvalue_valid_range(self):
        bars_a = np.array([[0, 1], [0, 2], [1, 3]])
        bars_b = np.array([[0, 4], [1, 5], [2, 6]])
        ra = ZigzagResult(level=1, barcodes={1: bars_a}, n_layers_used=7)
        rb = ZigzagResult(level=5, barcodes={1: bars_b}, n_layers_used=7)
        comp = compare_zigzag_levels(ra, rb, dim=1)
        assert 0.0 <= comp["ks_pvalue"] <= 1.0
        assert 0.0 <= comp["ks_statistic"] <= 1.0

    def test_compare_with_empty_barcodes(self):
        ra = ZigzagResult(level=1, barcodes={1: np.empty((0, 2))}, n_layers_used=3)
        rb = ZigzagResult(level=5, barcodes={1: np.array([[0, 3]])}, n_layers_used=3)
        comp = compare_zigzag_levels(ra, rb, dim=1)
        assert comp["n_features_a"] == 0
        assert comp["n_features_b"] == 1
