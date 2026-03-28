"""Tests for att.benchmarks — coupling measurement methods and sweep framework."""

import numpy as np
import pandas as pd
import pytest

from att.config import set_seed
from att.synthetic import coupled_lorenz, kuramoto_oscillators
from att.binding import BindingDetector
from att.benchmarks import transfer_entropy, pac, crqa, CouplingBenchmark


@pytest.fixture
def coupled_data():
    set_seed(42)
    ts_x, ts_y = coupled_lorenz(n_steps=5000, coupling=0.5, seed=42)
    return ts_x[1000:, 0], ts_y[1000:, 0]  # discard transient, use x-component


@pytest.fixture
def uncoupled_data():
    set_seed(42)
    ts_x, ts_y = coupled_lorenz(n_steps=5000, coupling=0.0, seed=42)
    return ts_x[1000:, 0], ts_y[1000:, 0]


class TestTransferEntropy:
    def test_returns_float(self, coupled_data):
        X, Y = coupled_data
        te = transfer_entropy(X, Y)
        assert isinstance(te, float)

    def test_nonnegative(self, coupled_data):
        X, Y = coupled_data
        te = transfer_entropy(X, Y)
        assert te >= 0.0

    def test_driven_signal_positive_te(self):
        """Directly driven signal should have clearly positive TE."""
        set_seed(42)
        rng = np.random.default_rng(42)
        n = 5000
        X = rng.standard_normal(n)
        # Y is a noisy copy of lagged X → TE(X->Y) should be positive
        Y = np.zeros(n)
        for t in range(1, n):
            Y[t] = 0.8 * X[t - 1] + 0.2 * rng.standard_normal()
        te = transfer_entropy(X, Y)
        assert te > 0.01, f"TE for driven signal {te} too low"


class TestPAC:
    def test_returns_float(self, coupled_data):
        X, Y = coupled_data
        mi = pac(X, Y)
        assert isinstance(mi, float)

    def test_nonnegative(self, coupled_data):
        X, Y = coupled_data
        mi = pac(X, Y)
        assert mi >= 0.0

    def test_bounded(self, coupled_data):
        X, Y = coupled_data
        mi = pac(X, Y)
        assert mi <= 1.0


class TestCRQA:
    def test_returns_float(self, coupled_data):
        X, Y = coupled_data
        det = crqa(X, Y)
        assert isinstance(det, float)

    def test_in_range(self, coupled_data):
        X, Y = coupled_data
        det = crqa(X, Y)
        assert 0.0 <= det <= 1.0

    def test_identical_signals_high_determinism(self):
        """Identical signals should have high cross-recurrence determinism."""
        set_seed(42)
        x = np.sin(np.linspace(0, 10 * np.pi, 1000))
        det = crqa(x, x)
        assert det > 0.5, f"Identical signal determinism {det} too low"


class TestCouplingBenchmark:
    def test_run_returns_dict(self, coupled_data):
        bench = CouplingBenchmark(methods=["transfer_entropy", "pac", "crqa"])
        result = bench.run(*coupled_data)
        assert "transfer_entropy" in result
        assert "pac" in result
        assert "crqa" in result
        for v in result.values():
            assert isinstance(v, float)

    def test_register_method(self, coupled_data):
        bench = CouplingBenchmark(methods=["transfer_entropy"])
        bench.register_method("custom", lambda X, Y: float(np.corrcoef(X, Y)[0, 1]))
        result = bench.run(*coupled_data)
        assert "custom" in result

    def test_sweep_returns_dataframe(self):
        bench = CouplingBenchmark(
            methods=["transfer_entropy", "crqa"],
            normalization="rank",
        )
        df = bench.sweep(
            generator_fn=lambda c, s: coupled_lorenz(n_steps=3000, coupling=c, seed=s),
            coupling_values=[0.0, 0.3, 0.6],
            seed=42,
            transient_discard=500,
        )
        assert isinstance(df, pd.DataFrame)
        assert "coupling" in df.columns
        assert "method" in df.columns
        assert "score" in df.columns
        assert "score_normalized" in df.columns
        # 3 coupling values * 2 methods = 6 rows
        assert len(df) == 6

    def test_rank_normalization_bounds(self):
        bench = CouplingBenchmark(methods=["transfer_entropy"], normalization="rank")
        df = bench.sweep(
            generator_fn=lambda c, s: coupled_lorenz(n_steps=3000, coupling=c, seed=s),
            coupling_values=[0.0, 0.5, 1.0],
            seed=42,
            transient_discard=500,
        )
        assert df["score_normalized"].min() >= 0.0
        assert df["score_normalized"].max() <= 1.0

    def test_minmax_normalization(self):
        bench = CouplingBenchmark(methods=["transfer_entropy"], normalization="minmax")
        df = bench.sweep(
            generator_fn=lambda c, s: coupled_lorenz(n_steps=3000, coupling=c, seed=s),
            coupling_values=[0.0, 0.5, 1.0],
            seed=42,
            transient_discard=500,
        )
        assert df["score_normalized"].min() >= -1e-10
        assert df["score_normalized"].max() <= 1.0 + 1e-10

    def test_reproducible_sweep(self):
        bench = CouplingBenchmark(methods=["transfer_entropy"], normalization="none")
        df1 = bench.sweep(
            generator_fn=lambda c, s: coupled_lorenz(n_steps=3000, coupling=c, seed=s),
            coupling_values=[0.0, 0.5],
            seed=42,
            transient_discard=500,
        )
        df2 = bench.sweep(
            generator_fn=lambda c, s: coupled_lorenz(n_steps=3000, coupling=c, seed=s),
            coupling_values=[0.0, 0.5],
            seed=42,
            transient_discard=500,
        )
        np.testing.assert_array_equal(df1["score"].values, df2["score"].values)


class TestKuramotoBenchmark:
    def test_kuramoto_binding_decreases_with_synchronization(self):
        """Binding score should decrease as Kuramoto coupling synchronizes oscillators.

        In the Kuramoto model, coupling drives oscillators toward phase
        synchronization.  Synchronized signals collapse onto a lower-dimensional
        manifold in the joint embedding, *reducing* excess topology compared to
        uncoupled oscillators with independent frequencies.
        """
        scores = []
        for coupling in [0.0, 1.0, 5.0]:
            phases, signals = kuramoto_oscillators(
                n_oscillators=2, n_steps=5000, dt=0.01,
                coupling=coupling, omega_spread=0.5, seed=42,
            )
            det = BindingDetector(max_dim=1, baseline="max")
            det.fit(signals[:, 0], signals[:, 1], subsample=300, seed=42)
            scores.append(det.binding_score())

        # Uncoupled oscillators with different frequencies produce richer
        # joint topology than synchronized (coupled) ones.
        assert scores[0] > scores[1], (
            f"Scores {scores}: uncoupled should exceed moderately coupled"
        )


class TestBenchmarkEdgeCases:
    """Edge-case tests for benchmark methods."""

    def test_te_constant_input(self):
        """TE of constant signals should be 0."""
        score = transfer_entropy(np.ones(500), np.ones(500))
        assert score == 0.0

    def test_te_zero_input(self):
        """TE of zero signals should be 0."""
        score = transfer_entropy(np.zeros(500), np.zeros(500))
        assert score == 0.0

    def test_pac_constant_input(self):
        """PAC of constant signals should not crash and return a valid float."""
        # Hilbert transform on constant input produces numerical artifacts,
        # so the score may not be exactly 0.
        score = pac(np.ones(500), np.ones(500))
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0

    def test_crqa_constant_input(self):
        """CRQA of constant signals should not crash."""
        score = crqa(np.ones(500), np.ones(500))
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_te_short_signal(self):
        """TE should work on very short signals."""
        rng = np.random.default_rng(42)
        score = transfer_entropy(rng.standard_normal(10), rng.standard_normal(10))
        assert isinstance(score, float)
        assert score >= 0

    def test_crqa_short_signal(self):
        """CRQA should work on short signals."""
        rng = np.random.default_rng(42)
        score = crqa(rng.standard_normal(30), rng.standard_normal(30))
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_delay_embed_empty(self):
        """_delay_embed with too-short input should return empty array."""
        from att.benchmarks.methods import _delay_embed
        result = _delay_embed(np.array([1.0, 2.0, 3.0]), dim=3, delay=2)
        assert result.shape == (0, 3)

    def test_count_line_points_direct(self):
        """_count_line_points should count points on lines of min_line length."""
        from att.benchmarks.methods import _count_line_points
        # [1,1,1,0,1,1,0,1] -> lines of length 3 and 2 -> 5 points total (min_line=2)
        diagonal = np.array([1, 1, 1, 0, 1, 1, 0, 1])
        assert _count_line_points(diagonal, min_line=2) == 5
        # min_line=3 -> only the length-3 line counts -> 3 points
        assert _count_line_points(diagonal, min_line=3) == 3

    def test_discretize_identical_values(self):
        """_discretize should handle all-identical values without crash."""
        from att.benchmarks.methods import _discretize
        result = _discretize(np.ones(100), bins=8)
        assert len(result) == 100
        # All should map to the same bin
        assert len(np.unique(result)) == 1

    def test_unknown_normalization_raises(self):
        """Unknown normalization should raise ValueError."""
        with pytest.raises(ValueError):
            CouplingBenchmark(normalization="bad")
