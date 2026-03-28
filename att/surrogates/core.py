"""Surrogate generation for significance testing."""

import numpy as np
from att.config.seed import get_rng


def phase_randomize(
    X: np.ndarray,
    n_surrogates: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """Generate amplitude-adjusted phase-randomized (AAFT) surrogates.

    Preserves the power spectrum and marginal distribution of X while
    destroying nonlinear coupling and phase relationships.

    Parameters
    ----------
    X : 1D time series of length n_samples
    n_surrogates : number of surrogates to generate
    seed : random seed for reproducibility

    Returns
    -------
    (n_surrogates, n_samples) array of surrogate time series
    """
    X = np.asarray(X).ravel()
    n = len(X)
    rng = get_rng(seed)

    # Precompute FFT of original
    fft_x = np.fft.rfft(X)
    amplitudes = np.abs(fft_x)

    # Sorted original values for amplitude adjustment
    sorted_x = np.sort(X)

    surrogates = np.empty((n_surrogates, n))
    for i in range(n_surrogates):
        # Random phases for non-DC, non-Nyquist frequencies
        n_freq = len(fft_x)
        random_phases = rng.uniform(0, 2 * np.pi, n_freq)
        # DC component: keep phase 0
        random_phases[0] = 0.0
        # Nyquist: phase must be 0 or pi for real-valued output
        if n % 2 == 0:
            random_phases[-1] = rng.choice([0.0, np.pi])

        # Apply random phases to original amplitudes
        randomized_fft = amplitudes * np.exp(1j * random_phases)
        surrogate = np.fft.irfft(randomized_fft, n=n)

        # Amplitude adjustment: rank-reorder to match original distribution
        rank_order = np.argsort(np.argsort(surrogate))
        surrogates[i] = sorted_x[rank_order]

    return surrogates


def time_shuffle(
    X: np.ndarray,
    n_surrogates: int = 100,
    block_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate time-shuffled surrogates.

    Destroys temporal structure while preserving univariate statistics.
    With block_size > 1, preserves short-range autocorrelation within blocks.

    Parameters
    ----------
    X : 1D time series of length n_samples
    n_surrogates : number of surrogates to generate
    block_size : if None, iid permutation; if int, shuffle blocks of this size
    seed : random seed for reproducibility

    Returns
    -------
    (n_surrogates, n_samples) array of surrogate time series
    """
    X = np.asarray(X).ravel()
    n = len(X)
    rng = get_rng(seed)

    surrogates = np.empty((n_surrogates, n))

    if block_size is None or block_size <= 1:
        for i in range(n_surrogates):
            surrogates[i] = rng.permutation(X)
    else:
        n_blocks = int(np.ceil(n / block_size))
        for i in range(n_surrogates):
            # Split into blocks
            blocks = [X[j * block_size: (j + 1) * block_size] for j in range(n_blocks)]
            # Shuffle block order
            perm = rng.permutation(n_blocks)
            shuffled = np.concatenate([blocks[p] for p in perm])
            surrogates[i] = shuffled[:n]

    return surrogates
