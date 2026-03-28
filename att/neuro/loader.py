"""EEG data loading and preprocessing via MNE-Python."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mne


class EEGLoader:
    """Load and preprocess EEG data from common formats.

    Supports BDF, EDF, SET (EEGLAB), FIF, and .mat files via MNE-Python.

    Parameters
    ----------
    data_path : str or Path
        Path to the EEG data file.
    subject : int or str
        Subject identifier (informational, stored as metadata).
    """

    def __init__(self, data_path: str | Path, subject: int | str = 1):
        self.data_path = Path(data_path)
        self.subject = subject
        self._raw: mne.io.Raw | None = None

    def load(self) -> "mne.io.Raw":
        """Load raw EEG data based on file extension."""
        import mne

        suffix = self.data_path.suffix.lower()

        if suffix == ".fif":
            raw = mne.io.read_raw_fif(str(self.data_path), preload=True, verbose=False)
        elif suffix == ".edf":
            raw = mne.io.read_raw_edf(str(self.data_path), preload=True, verbose=False)
        elif suffix == ".bdf":
            raw = mne.io.read_raw_bdf(str(self.data_path), preload=True, verbose=False)
        elif suffix == ".set":
            raw = mne.io.read_raw_eeglab(str(self.data_path), preload=True, verbose=False)
        elif suffix == ".mat":
            raw = self._load_mat()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        self._raw = raw
        return raw

    def _load_mat(self) -> "mne.io.Raw":
        """Load .mat file using heuristic: largest 2D array = channels x samples."""
        import mne
        from scipy.io import loadmat

        mat = loadmat(str(self.data_path))
        # Find the largest 2D numeric array
        best_key = None
        best_size = 0
        for key, val in mat.items():
            if key.startswith("_"):
                continue
            if isinstance(val, np.ndarray) and val.ndim == 2:
                if val.size > best_size:
                    best_key = key
                    best_size = val.size

        if best_key is None:
            raise ValueError("No suitable 2D array found in .mat file")

        data = mat[best_key].astype(np.float64)
        # Ensure shape is (n_channels, n_samples) — wider dimension is samples
        if data.shape[0] > data.shape[1]:
            data = data.T

        n_channels = data.shape[0]
        # Create channel names and info
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
        ch_types = ["eeg"] * n_channels
        sfreq = 256.0  # Default; user should set via raw.info["sfreq"] if known

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw

    def preprocess(
        self,
        bandpass: tuple[float, float] = (1, 45),
        notch: float | None = 50.0,
        reference: str = "average",
        ica_reject: bool = False,
    ) -> "mne.io.Raw":
        """Apply standard preprocessing pipeline.

        Parameters
        ----------
        bandpass : (low, high) Hz
        notch : line noise frequency (None to skip)
        reference : re-referencing scheme ("average" or channel name)
        ica_reject : whether to run ICA artifact rejection (slow)
        """
        if self._raw is None:
            raise RuntimeError("Call load() first.")

        raw = self._raw

        # Bandpass filter
        raw.filter(bandpass[0], bandpass[1], verbose=False)

        # Notch filter
        if notch is not None:
            raw.notch_filter(notch, verbose=False)

        # Re-reference
        if reference == "average":
            raw.set_eeg_reference("average", projection=False, verbose=False)
        elif reference is not None:
            raw.set_eeg_reference([reference], verbose=False)

        # ICA
        if ica_reject:
            import mne
            ica = mne.preprocessing.ICA(n_components=0.95, random_state=42, verbose=False)
            ica.fit(raw, verbose=False)
            # Auto-detect EOG artifacts if EOG channels present
            eog_ch = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
            if eog_ch:
                eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_ch[0], verbose=False)
                ica.exclude = eog_indices
            ica.apply(raw, verbose=False)

        self._raw = raw
        return raw

    def to_timeseries(
        self, picks: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract channel data as numpy array.

        Parameters
        ----------
        picks : channel names to extract (None = all EEG)

        Returns
        -------
        (n_channels, n_samples) array, list of channel names
        """
        if self._raw is None:
            raise RuntimeError("Call load() first.")

        if picks is None:
            picks = "eeg"

        data = self._raw.get_data(picks=picks)
        if isinstance(picks, str):
            ch_names = self._raw.copy().pick(picks).ch_names
        else:
            ch_names = list(picks)

        return data, ch_names

    def get_events(self) -> np.ndarray | None:
        """Extract events from annotations or STIM channels.

        Returns
        -------
        (n_events, 3) array [sample, 0, event_id] or None if no events found.
        """
        import mne

        if self._raw is None:
            raise RuntimeError("Call load() first.")

        # Try annotations first
        if self._raw.annotations and len(self._raw.annotations) > 0:
            try:
                events, event_id = mne.events_from_annotations(
                    self._raw, verbose=False
                )
                return events
            except Exception:
                pass

        # Try STIM channels
        stim_ch = [ch for ch in self._raw.ch_names if "STI" in ch.upper()]
        if stim_ch:
            try:
                events = mne.find_events(self._raw, stim_channel=stim_ch[0], verbose=False)
                return events
            except Exception:
                pass

        return None

    def get_sfreq(self) -> float:
        """Return sampling frequency."""
        if self._raw is None:
            raise RuntimeError("Call load() first.")
        return float(self._raw.info["sfreq"])

    @staticmethod
    def get_channel_groups() -> dict[str, list[str]]:
        """Return standard 10-20 channel groups for region-of-interest analysis."""
        return {
            "frontal": ["F3", "Fz", "F4", "Fp1", "Fp2"],
            "central": ["C3", "Cz", "C4"],
            "parietal": ["P3", "Pz", "P4"],
            "occipital": ["O1", "Oz", "O2"],
            "temporal": ["T7", "T8", "P7", "P8"],
        }

    @staticmethod
    def get_fallback_params(band: str = "broadband", sfreq: float = 256.0) -> dict:
        """Convenience method delegating to eeg_params.get_fallback_params."""
        from att.neuro.eeg_params import get_fallback_params
        return get_fallback_params(band, sfreq)
