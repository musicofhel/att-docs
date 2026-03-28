"""Tests for att.surrogates — surrogate generation."""

import numpy as np
import pytest

from att.config import set_seed
from att.synthetic import lorenz_system, coupled_lorenz
from att.surrogates import phase_randomize, time_shuffle, twin_surrogate


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


class TestTwinSurrogate:
    @pytest.fixture(autouse=True)
    def setup(self):
        set_seed(42)
        ts = lorenz_system(n_steps=2000)
        self.x = ts[:, 0]

    def test_output_shape(self):
        embedding_dim = 3
        embedding_delay = 1
        expected_len = len(self.x) - (embedding_dim - 1) * embedding_delay
        surr = twin_surrogate(
            self.x, n_surrogates=10,
            embedding_dim=embedding_dim, embedding_delay=embedding_delay,
            seed=42,
        )
        assert surr.shape == (10, expected_len)

    def test_reproducible(self):
        s1 = twin_surrogate(self.x, n_surrogates=5, seed=42)
        s2 = twin_surrogate(self.x, n_surrogates=5, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_different_from_original(self):
        """Surrogates should not be identical to the original embedded series."""
        embedding_dim = 3
        embedding_delay = 1
        pad = (embedding_dim - 1) * embedding_delay
        original_first_coord = self.x[:len(self.x) - pad]

        surr = twin_surrogate(
            self.x, n_surrogates=5,
            embedding_dim=embedding_dim, embedding_delay=embedding_delay,
            seed=42,
        )
        for i in range(surr.shape[0]):
            assert not np.allclose(surr[i], original_first_coord), (
                f"Surrogate {i} is identical to original"
            )

    def test_preserves_recurrence_structure(self):
        """Recurrence density of surrogates should be similar to the original."""
        from scipy.spatial.distance import cdist

        embedding_dim = 3
        embedding_delay = 1
        pad = (embedding_dim - 1) * embedding_delay
        n_embedded = len(self.x) - pad

        # Embed the original
        embedded = np.empty((n_embedded, embedding_dim))
        for d in range(embedding_dim):
            embedded[:, d] = self.x[d * embedding_delay: d * embedding_delay + n_embedded]

        dist_orig = cdist(embedded, embedded)
        upper_tri = dist_orig[np.triu_indices(n_embedded, k=1)]
        threshold = np.percentile(upper_tri, 10)
        orig_density = np.mean(dist_orig < threshold)

        # Generate surrogates and re-embed them
        surr = twin_surrogate(
            self.x, n_surrogates=3,
            embedding_dim=embedding_dim, embedding_delay=embedding_delay,
            seed=42,
        )
        # Re-embedding the surrogate (length n_embedded) reduces it further
        n_surr_embedded = n_embedded - pad
        for s in range(surr.shape[0]):
            surr_embedded = np.empty((n_surr_embedded, embedding_dim))
            for d in range(embedding_dim):
                surr_embedded[:, d] = surr[s, d * embedding_delay: d * embedding_delay + n_surr_embedded]

            dist_surr = cdist(surr_embedded, surr_embedded)
            surr_density = np.mean(dist_surr < threshold)

            # Surrogate recurrence density within 50% of original
            assert surr_density > orig_density * 0.5, (
                f"Surrogate {s} recurrence density {surr_density:.4f} "
                f"is less than 50% of original {orig_density:.4f}"
            )
            assert surr_density < orig_density * 1.5, (
                f"Surrogate {s} recurrence density {surr_density:.4f} "
                f"exceeds 150% of original {orig_density:.4f}"
            )

    def test_destroys_coupling(self):
        """Twin surrogates of Y should reduce cross-correlation with X."""
        set_seed(42)
        ts_x, ts_y = coupled_lorenz(n_steps=2000, coupling=0.1, seed=42)
        x_series = ts_x[:, 0]
        y_series = ts_y[:, 0]

        embedding_dim = 3
        embedding_delay = 1
        pad = (embedding_dim - 1) * embedding_delay
        n_embedded = len(y_series) - pad

        # Original cross-correlation (truncated to match surrogate length)
        orig_corr = abs(np.corrcoef(x_series[:n_embedded], y_series[:n_embedded])[0, 1])

        surr = twin_surrogate(
            y_series, n_surrogates=10,
            embedding_dim=embedding_dim, embedding_delay=embedding_delay,
            seed=42,
        )

        surr_corrs = []
        for i in range(surr.shape[0]):
            c = abs(np.corrcoef(x_series[:n_embedded], surr[i])[0, 1])
            surr_corrs.append(c)

        mean_surr_corr = np.mean(surr_corrs)
        assert mean_surr_corr < orig_corr, (
            f"Mean surrogate cross-correlation {mean_surr_corr:.4f} "
            f"should be less than original {orig_corr:.4f}"
        )


class TestSurrogateEdgeCases:
    """Edge-case tests for surrogate generation."""

    def test_phase_randomize_very_short(self):
        """Phase randomization should work on 5-sample signal."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        surr = phase_randomize(x, n_surrogates=2, seed=42)
        assert surr.shape == (2, 5)

    def test_time_shuffle_very_short(self):
        """Time shuffle should work on 3-sample signal."""
        x = np.array([1.0, 2.0, 3.0])
        surr = time_shuffle(x, n_surrogates=2, seed=42)
        assert surr.shape == (2, 3)

    def test_twin_surrogate_minimal(self):
        """Twin surrogate should work on minimal viable input."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        surr = twin_surrogate(x, n_surrogates=2, embedding_dim=2, embedding_delay=1, seed=42)
        assert surr.shape[0] == 2
        assert surr.shape[1] > 0  # Shorter than input due to embedding

    def test_time_shuffle_block_larger_than_signal(self):
        """Block size larger than signal should still work."""
        x = np.arange(10, dtype=float)
        surr = time_shuffle(x, n_surrogates=1, block_size=20, seed=42)
        assert surr.shape == (1, 10)
        # With one block containing everything, output should be same as input
        np.testing.assert_array_equal(surr[0], x)
