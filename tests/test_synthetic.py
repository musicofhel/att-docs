"""Tests for att.synthetic — chaotic system generators."""

import numpy as np

from att.synthetic import (
    lorenz_system,
    rossler_system,
    coupled_lorenz,
    coupled_rossler_lorenz,
    switching_rossler,
    coupled_oscillators,
    kuramoto_oscillators,
)


class TestLorenz:
    def test_shape(self):
        ts = lorenz_system(n_steps=1000, seed=42)
        assert ts.shape == (1000, 3)

    def test_reproducible(self):
        a = lorenz_system(n_steps=500, seed=42)
        b = lorenz_system(n_steps=500, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = lorenz_system(n_steps=500, seed=42)
        b = lorenz_system(n_steps=500, seed=99)
        assert not np.array_equal(a, b)

    def test_no_nan(self):
        ts = lorenz_system(n_steps=5000, seed=42)
        assert not np.any(np.isnan(ts))

    def test_noise(self):
        clean = lorenz_system(n_steps=500, seed=42, noise=0.0)
        noisy = lorenz_system(n_steps=500, seed=42, noise=1.0)
        assert not np.array_equal(clean, noisy)


class TestRossler:
    def test_shape(self):
        ts = rossler_system(n_steps=1000, seed=42)
        assert ts.shape == (1000, 3)

    def test_reproducible(self):
        a = rossler_system(n_steps=500, seed=42)
        b = rossler_system(n_steps=500, seed=42)
        np.testing.assert_array_equal(a, b)


class TestCoupledLorenz:
    def test_shape(self):
        ts_x, ts_y = coupled_lorenz(n_steps=1000, seed=42)
        assert ts_x.shape == (1000, 3)
        assert ts_y.shape == (1000, 3)

    def test_uncoupled_differ(self):
        ts_x, ts_y = coupled_lorenz(coupling=0.0, n_steps=1000, seed=42)
        assert not np.allclose(ts_x, ts_y, atol=0.1)

    def test_strong_coupling_sync(self):
        ts_x, ts_y = coupled_lorenz(coupling=5.0, n_steps=5000, seed=42)
        # At very strong coupling, systems should synchronize
        diff = np.mean(np.abs(ts_x[-1000:] - ts_y[-1000:]))
        assert diff < 1.0  # Should be close to synchronized

    def test_reproducible(self):
        a = coupled_lorenz(n_steps=500, seed=42)
        b = coupled_lorenz(n_steps=500, seed=42)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class TestCoupledRosslerLorenz:
    def test_shape(self):
        ts_r, ts_l = coupled_rossler_lorenz(n_steps=1000, seed=42)
        assert ts_r.shape == (1000, 3)
        assert ts_l.shape == (1000, 3)


class TestSwitchingRossler:
    def test_shape(self):
        ts = switching_rossler(n_steps=10000, seed=42)
        assert ts.shape == (10000, 3)

    def test_no_nan(self):
        ts = switching_rossler(n_steps=10000, seed=42)
        assert not np.any(np.isnan(ts))


class TestCoupledOscillators:
    def test_shape(self):
        ts = coupled_oscillators(n_oscillators=3, n_steps=1000, seed=42)
        assert ts.shape == (1000, 3, 3)

    def test_custom_coupling(self):
        coupling = np.array([[0, 0.2, 0], [0.2, 0, 0.1], [0, 0.1, 0]])
        ts = coupled_oscillators(n_oscillators=3, coupling_matrix=coupling, n_steps=500, seed=42)
        assert ts.shape == (500, 3, 3)


class TestKuramoto:
    def test_shape(self):
        phases, signals = kuramoto_oscillators(
            n_oscillators=4, n_steps=1000, seed=42
        )
        assert phases.shape == (1000, 4)
        assert signals.shape == (1000, 4)

    def test_uncoupled_drift(self):
        """At coupling=0, phases should drift apart over time."""
        phases, _ = kuramoto_oscillators(
            n_oscillators=5, n_steps=5000, coupling=0.0, omega_spread=1.0, seed=42
        )
        # Phase differences at start vs end should show increasing variance
        diffs_start = np.var(phases[10])  # skip step 0 (all random but close)
        diffs_end = np.var(phases[-1])
        assert diffs_end > diffs_start

    def test_synchronized(self):
        """At high coupling, the Kuramoto order parameter R should approach 1."""
        phases, _ = kuramoto_oscillators(
            n_oscillators=10, n_steps=10000, coupling=5.0, omega_spread=0.5, seed=42
        )
        # Order parameter R = |1/N * sum(exp(i*theta))|
        last_quarter = phases[-2500:]
        order_param = np.abs(np.mean(np.exp(1j * last_quarter), axis=1))
        mean_R = np.mean(order_param)
        assert mean_R > 0.8, f"Expected R > 0.8 for strong coupling, got {mean_R}"

    def test_reproducible(self):
        a_phases, a_signals = kuramoto_oscillators(n_steps=500, seed=42)
        b_phases, b_signals = kuramoto_oscillators(n_steps=500, seed=42)
        np.testing.assert_array_equal(a_phases, b_phases)
        np.testing.assert_array_equal(a_signals, b_signals)

    def test_signals_bounded(self):
        """sin(phase) should always be in [-1, 1]."""
        _, signals = kuramoto_oscillators(
            n_oscillators=3, n_steps=2000, noise=0.5, seed=42
        )
        assert np.all(signals >= -1.0)
        assert np.all(signals <= 1.0)
