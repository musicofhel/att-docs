"""Time-delay estimation via Average Mutual Information."""

import numpy as np


def estimate_delay(
    X: np.ndarray,
    method: str = "ami",
    max_lag: int = 100,
) -> int:
    """Estimate optimal time delay for Takens embedding.

    Uses first minimum of Average Mutual Information (Fraser & Swinney, 1986).

    Parameters
    ----------
    X : (n_samples,) array
    method : "ami" (only option currently)
    max_lag : maximum lag to consider

    Returns
    -------
    int : optimal delay (first AMI minimum, or max_lag if no minimum found)
    """
    if method != "ami":
        raise ValueError(f"Unknown delay method: {method}")

    n = len(X)
    max_lag = min(max_lag, n // 4)

    ami_values = np.zeros(max_lag)
    n_bins = max(int(np.sqrt(n / 5)), 16)

    # Histogram-based AMI estimation
    x_min, x_max = X.min(), X.max()
    x_range = x_max - x_min
    if x_range == 0:
        return 1

    # Bin edges
    edges = np.linspace(x_min - 1e-10, x_max + 1e-10, n_bins + 1)

    for lag in range(1, max_lag):
        x_now = X[: n - lag]
        x_lag = X[lag: n]

        # Joint and marginal histograms
        h_joint, _, _ = np.histogram2d(x_now, x_lag, bins=edges)
        h_x = np.histogram(x_now, bins=edges)[0]
        h_y = np.histogram(x_lag, bins=edges)[0]

        # Normalize
        n_pairs = len(x_now)
        p_joint = h_joint / n_pairs
        p_x = h_x / n_pairs
        p_y = h_y / n_pairs

        # Mutual information
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if p_joint[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_x[i] * p_y[j]))

        ami_values[lag] = mi

    # Find first minimum
    for lag in range(2, max_lag - 1):
        if ami_values[lag] < ami_values[lag - 1] and ami_values[lag] < ami_values[lag + 1]:
            return lag

    # No minimum found — return lag at 1/e of AMI(1)
    if ami_values[1] > 0:
        threshold = ami_values[1] / np.e
        for lag in range(2, max_lag):
            if ami_values[lag] < threshold:
                return lag

    return max_lag // 2
