"""Tests for att.surrogates — surrogate generation."""

import numpy as np
import pytest

from att.config import set_seed
from att.synthetic import lorenz_system
from att.surrogates import phase_randomize, time_shuffle


class TestPhaseRandomize:
    @pytest.fixture(autouse=True)
    def setup(self):
        set_seed(42)
        ts = lorenz_system(n_steps=2000)
        self.x = ts[:, 0]

    def test_output_shape(self):
        surr = phase_randomize(self.x, n_surrogates=10, seed=42)
        assert surr.shape == (10, len(self.x))

    def test_preserves_spectrum(self):
        """Power spectrum of surrogate should approximate original."""
        surr = phase_randomize(self.x, n_surrogates=1, seed=42)
        psd_orig = np.abs(np.fft.rfft(self.x)) ** 2
        psd_surr = np.abs(np.fft.rfft(surr[0])) ** 2
        # Correlation of power spectra should be high
        corr = np.corrcoef(psd_orig, psd_surr)[0, 1]
        assert corr > 0.8, f"Spectrum correlation {corr} too low"

    def test_preserves_distribution(self):
        """AAFT preserves the exact marginal distribution."""
        surr = phase_randomize(self.x, n_surrogates=1, seed=42)
        np.testing.assert_array_almost_equal(
            np.sort(surr[0]), np.sort(self.x), decimal=10,
        )

    def test_reproducible(self):
        s1 = phase_randomize(self.x, n_surrogates=5, seed=42)
        s2 = phase_randomize(self.x, n_surrogates=5, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds(self):
        s1 = phase_randomize(self.x, n_surrogates=1, seed=42)
        s2 = phase_randomize(self.x, n_surrogates=1, seed=99)
        assert not np.allclose(s1, s2)


class TestTimeShuffle:
    @pytest.fixture(autouse=True)
    def setup(self):
        set_seed(42)
        ts = lorenz_system(n_steps=2000)
        self.x = ts[:, 0]

    def test_output_shape(self):
        surr = time_shuffle(self.x, n_surrogates=10, seed=42)
        assert surr.shape == (10, len(self.x))

    def test_iid_shuffle_destroys_autocorrelation(self):
        surr = time_shuffle(self.x, n_surrogates=1, seed=42)
        # Autocorrelation at lag 1 should drop significantly
        orig_ac = np.corrcoef(self.x[:-1], self.x[1:])[0, 1]
        surr_ac = np.corrcoef(surr[0, :-1], surr[0, 1:])[0, 1]
        assert abs(surr_ac) < abs(orig_ac) * 0.5

    def test_block_shuffle_shape(self):
        surr = time_shuffle(self.x, n_surrogates=5, block_size=50, seed=42)
        assert surr.shape == (5, len(self.x))

    def test_reproducible(self):
        s1 = time_shuffle(self.x, n_surrogates=5, seed=42)
        s2 = time_shuffle(self.x, n_surrogates=5, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_preserves_values(self):
        """Shuffled data should have the same set of values."""
        surr = time_shuffle(self.x, n_surrogates=1, seed=42)
        np.testing.assert_array_almost_equal(
            np.sort(surr[0]), np.sort(self.x), decimal=10,
        )
