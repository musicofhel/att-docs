"""Auto-with-fallback embedding for EEG channel data."""

import numpy as np

from att.embedding.takens import TakensEmbedder
from att.embedding.validation import validate_embedding
from att.neuro.eeg_params import get_fallback_params


def embed_channel(
    channel_data: np.ndarray,
    band: str = "broadband",
    sfreq: float = 256.0,
    condition_threshold: float = 1e4,
) -> tuple[np.ndarray, dict]:
    """Embed a single EEG channel with auto-estimation and fallback.

    Strategy:
    1. Try TakensEmbedder("auto", "auto")
    2. validate_embedding() — check condition number
    3. If degenerate or estimation fails: re-embed with fallback_params

    Parameters
    ----------
    channel_data : 1D array of EEG samples
    band : frequency band for fallback params
    sfreq : sampling frequency in Hz
    condition_threshold : condition number threshold for degeneracy

    Returns
    -------
    (point_cloud, metadata) where metadata is a dict with:
        method: "auto" or "fallback"
        delay: int
        dimension: int
        condition_number: float
        fallback_reason: str or None
    """
    channel_data = np.asarray(channel_data).ravel()

    # Try auto-estimation first
    try:
        embedder = TakensEmbedder("auto", "auto")
        embedder.fit(channel_data)
        cloud = embedder.transform(channel_data)

        val = validate_embedding(cloud, condition_threshold=condition_threshold)

        if not val["degenerate"]:
            return cloud, {
                "method": "auto",
                "delay": embedder.delay_,
                "dimension": embedder.dimension_,
                "condition_number": val["condition_number"],
                "fallback_reason": None,
            }

        fallback_reason = f"degenerate (condition={val['condition_number']:.1e})"
    except Exception as e:
        fallback_reason = f"auto failed: {e}"

    # Fallback to literature-grounded parameters
    params = get_fallback_params(band, sfreq)
    embedder = TakensEmbedder(delay=params["delay"], dimension=params["dimension"])
    embedder.fit(channel_data)
    cloud = embedder.transform(channel_data)

    val = validate_embedding(cloud, condition_threshold=condition_threshold)

    return cloud, {
        "method": "fallback",
        "delay": params["delay"],
        "dimension": params["dimension"],
        "condition_number": val["condition_number"],
        "fallback_reason": fallback_reason,
    }
