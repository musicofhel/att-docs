import numpy as np
import pytest
from att.neuro.eeg_params import get_fallback_params


class TestFallbackParams:
    def test_all_bands_valid(self):
        for band in ["broadband", "alpha", "theta_alpha", "gamma"]:
            params = get_fallback_params(band)
            assert "delay" in params
            assert "dimension" in params
            assert "bandpass" in params
            assert isinstance(params["bandpass"], tuple)
            assert len(params["bandpass"]) == 2

    def test_delay_scales_at_512hz(self):
        p256 = get_fallback_params("broadband", sfreq=256.0)
        p512 = get_fallback_params("broadband", sfreq=512.0)
        assert p512["delay"] == p256["delay"] * 2  # 10 -> 20

    def test_unknown_band_raises(self):
        with pytest.raises(ValueError, match="Unknown band"):
            get_fallback_params("delta")



class TestEmbedChannel:
    def test_clean_signal_uses_auto(self):
        """Switching Rossler should embed cleanly with auto."""
        from att.neuro.embedding import embed_channel
        from att.synthetic import switching_rossler
        ts = switching_rossler(n_steps=5000, seed=42)
        x = ts[:, 0]  # first variable
        cloud, meta = embed_channel(x)
        assert meta["method"] == "auto"
        assert meta["fallback_reason"] is None
        assert cloud.ndim == 2

    def test_constant_signal_uses_fallback(self):
        """Constant signal is degenerate and should trigger fallback.

        White noise is well-conditioned (fills all dimensions uniformly),
        so we use a constant signal whose embedding has condition = inf.
        """
        from att.neuro.embedding import embed_channel
        constant = np.ones(3000)
        cloud, meta = embed_channel(constant)
        assert meta["method"] == "fallback"
        assert meta["fallback_reason"] is not None
        assert cloud.ndim == 2

    def test_metadata_structure(self):
        """Metadata dict has all required keys."""
        from att.neuro.embedding import embed_channel
        from att.synthetic import switching_rossler
        ts = switching_rossler(n_steps=5000, seed=42)
        _, meta = embed_channel(ts[:, 0])
        for key in ["method", "delay", "dimension", "condition_number", "fallback_reason"]:
            assert key in meta


class TestEEGLoader:
    @pytest.fixture(autouse=True)
    def _skip_if_no_mne(self):
        pytest.importorskip("mne")

    def test_load_synthetic_fif(self, tmp_path):
        """Load a synthetic FIF file."""
        import mne
        from att.neuro.loader import EEGLoader

        # Create synthetic EEG
        sfreq = 256.0
        n_channels = 4
        n_samples = 2560  # 10 seconds
        ch_names = ["O1", "O2", "Pz", "Fz"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        data = np.random.default_rng(42).standard_normal((n_channels, n_samples))
        raw = mne.io.RawArray(data, info, verbose=False)
        fif_path = tmp_path / "test_raw.fif"
        raw.save(str(fif_path), overwrite=True, verbose=False)

        loader = EEGLoader(fif_path, subject="test")
        loaded = loader.load()
        assert loaded.info["sfreq"] == sfreq
        assert len(loaded.ch_names) == n_channels

    def test_to_timeseries(self, tmp_path):
        """to_timeseries returns array and channel names."""
        import mne
        from att.neuro.loader import EEGLoader

        sfreq = 256.0
        ch_names = ["O1", "O2"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        data = np.random.default_rng(42).standard_normal((2, 1280))
        raw = mne.io.RawArray(data, info, verbose=False)
        fif_path = tmp_path / "test_raw.fif"
        raw.save(str(fif_path), overwrite=True, verbose=False)

        loader = EEGLoader(fif_path)
        loader.load()
        ts_data, names = loader.to_timeseries()
        assert ts_data.shape[0] == 2
        assert names == ch_names
