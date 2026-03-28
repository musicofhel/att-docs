"""Global seed management for deterministic reproducibility."""

import random
import numpy as np

_GLOBAL_SEED: int | None = None
_GLOBAL_RNG: np.random.Generator | None = None


def set_seed(seed: int) -> None:
    """Set global random seed for all stochastic operations.

    Seeds NumPy, SciPy (via NumPy), and Python's random module.
    Call once at the start of any experiment or notebook.
    """
    global _GLOBAL_SEED, _GLOBAL_RNG
    _GLOBAL_SEED = seed
    _GLOBAL_RNG = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Get a numpy Generator, using provided seed or global state.

    If seed is given, creates a new Generator from that seed.
    If seed is None, returns the global Generator (or creates one from seed=0).
    """
    if seed is not None:
        return np.random.default_rng(seed)
    global _GLOBAL_RNG
    if _GLOBAL_RNG is None:
        _GLOBAL_RNG = np.random.default_rng(0)
    return _GLOBAL_RNG


def get_seed() -> int | None:
    """Return the current global seed, or None if not set."""
    return _GLOBAL_SEED
