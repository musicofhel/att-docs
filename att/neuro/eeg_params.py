"""Fallback Takens-embedding parameters for common EEG frequency bands.

Sources:
    Stam, C. J. (2005). Nonlinear dynamical analysis of EEG and MEG.
    Lehnertz, K. & Elger, C. E. (1998). Can epileptic seizures be predicted?
"""

from __future__ import annotations

FALLBACK_PARAMS: dict[str, dict] = {
    "broadband": {"delay": 10, "dimension": 5, "bandpass": (1, 45)},
    "alpha": {"delay": 8, "dimension": 4, "bandpass": (8, 13)},
    "theta_alpha": {"delay": 12, "dimension": 5, "bandpass": (4, 13)},
    "gamma": {"delay": 3, "dimension": 5, "bandpass": (30, 45)},
}

_BASE_SFREQ = 256.0


def get_fallback_params(band: str = "broadband", sfreq: float = 256.0) -> dict:
    """Return default Takens-embedding parameters for an EEG frequency band.

    Parameters
    ----------
    band : str
        One of ``"broadband"``, ``"alpha"``, ``"theta_alpha"``, ``"gamma"``.
    sfreq : float
        Sampling frequency in Hz.  Delay values are scaled relative to the
        base rate of 256 Hz so that the *physical* lag stays approximately
        constant across sampling rates.

    Returns
    -------
    dict
        Keys: ``delay``, ``dimension``, ``bandpass``, ``note``.

    Raises
    ------
    ValueError
        If *band* is not one of the known bands.
    """
    if band not in FALLBACK_PARAMS:
        valid = ", ".join(sorted(FALLBACK_PARAMS))
        raise ValueError(
            f"Unknown band {band!r}. Valid bands: {valid}"
        )

    entry = FALLBACK_PARAMS[band]
    base_delay = entry["delay"]
    scaled_delay = max(1, round(base_delay * sfreq / _BASE_SFREQ))

    return {
        "delay": scaled_delay,
        "dimension": entry["dimension"],
        "bandpass": entry["bandpass"],
        "note": "Stam (2005), Lehnertz & Elger (1998)",
    }
