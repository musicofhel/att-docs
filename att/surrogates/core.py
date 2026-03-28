"""Surrogate generation for significance testing."""

import numpy as np
from scipy.spatial.distance import cdist

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


def twin_surrogate(
    X: np.ndarray,
    n_surrogates: int = 100,
    embedding_dim: int = 3,
    embedding_delay: int = 1,
    recurrence_threshold: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate twin surrogates from recurrence structure (Thiel et al. 2006).

    Preserves the recurrence properties of the attractor while destroying
    the specific temporal ordering. Tests specifically for deterministic
    coupling structure in the dynamics.

    Parameters
    ----------
    X : 1D time series of length n_samples
    n_surrogates : number of surrogates to generate
    embedding_dim : dimension for Takens delay embedding
    embedding_delay : delay for Takens embedding
    recurrence_threshold : distance threshold for recurrence matrix;
        if None, uses 10th percentile of pairwise distances
    seed : random seed for reproducibility

    Returns
    -------
    (n_surrogates, n_output) array where n_output = n_samples - (embedding_dim - 1) * embedding_delay
    """
    X = np.asarray(X).ravel()
    n = len(X)
    rng = get_rng(seed)

    # 1. Delay-embed the time series
    pad = (embedding_dim - 1) * embedding_delay
    n_embedded = n - pad
    embedded = np.empty((n_embedded, embedding_dim))
    for d in range(embedding_dim):
        embedded[:, d] = X[d * embedding_delay: d * embedding_delay + n_embedded]

    # 2. Pairwise distance matrix
    dist_matrix = cdist(embedded, embedded, metric="euclidean")

    # 3. Build recurrence matrix
    if recurrence_threshold is None:
        # Use 10th percentile of all pairwise distances (excluding diagonal zeros)
        upper_tri = dist_matrix[np.triu_indices(n_embedded, k=1)]
        recurrence_threshold = np.percentile(upper_tri, 10)

    recurrence = dist_matrix < recurrence_threshold

    # 4. Precompute twin lists: twins[i] = indices j where recurrence[i,j] is True
    # (simplified definition: twins are recurrent neighbors)
    twins = [np.where(recurrence[i])[0] for i in range(n_embedded)]

    # 5. Generate surrogates
    surrogates = np.empty((n_surrogates, n_embedded))
    for s in range(n_surrogates):
        trajectory = np.empty(n_embedded, dtype=int)
        # Pick random starting index
        trajectory[0] = rng.integers(0, n_embedded)

        for t in range(1, n_embedded):
            current = trajectory[t - 1]
            tw = twins[current]

            if len(tw) > 0:
                # Jump to a random twin's next time step
                chosen_twin = tw[rng.integers(0, len(tw))]
                # Advance to next step, wrapping around if at the end
                trajectory[t] = (chosen_twin + 1) % n_embedded
            else:
                # No twins — advance sequentially
                trajectory[t] = (current + 1) % n_embedded

        # Extract the first coordinate of the embedded point at each visited index
        surrogates[s] = embedded[trajectory, 0]

    return surrogates
