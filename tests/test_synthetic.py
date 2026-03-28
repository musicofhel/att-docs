"""Tests for att.synthetic — chaotic system generators."""

import numpy as np
import pytest

from att.config import set_seed
from att.synthetic import (
    lorenz_system,
    rossler_system,
    coupled_lorenz,
    coupled_rossler_lorenz,
    switching_rossler,
    coupled_oscillators,
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
