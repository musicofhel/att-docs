from att.neuro.eeg_params import get_fallback_params, FALLBACK_PARAMS
from att.neuro.embedding import embed_channel

try:
    from att.neuro.loader import EEGLoader
except ImportError:
    pass  # MNE not installed

__all__ = ["get_fallback_params", "FALLBACK_PARAMS", "embed_channel", "EEGLoader"]
