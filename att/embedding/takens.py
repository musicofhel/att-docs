"""Takens delay embedding for attractor reconstruction."""

import numpy as np

from att.embedding.delay import estimate_delay
from att.embedding.dimension import estimate_dimension


class TakensEmbedder:
    """Reconstruct a phase-space attractor from a scalar time series.

    Parameters
    ----------
    delay : int or "auto"
        Time steps between coordinates. "auto" estimates via AMI first minimum.
    dimension : int or "auto"
        Number of delay coordinates. "auto" estimates via FNN.
    """

    def __init__(self, delay: int | str = "auto", dimension: int | str = "auto"):
        self.delay = delay
        self.dimension = dimension
        self.delay_: int | None = None
        self.dimension_: int | None = None

    def fit(self, X: np.ndarray) -> "TakensEmbedder":
        """Estimate parameters from data. Stores .delay_ and .dimension_."""
        X = np.asarray(X).ravel()

        if self.delay == "auto":
            self.delay_ = estimate_delay(X)
        else:
            self.delay_ = int(self.delay)

        if self.dimension == "auto":
            self.dimension_ = estimate_dimension(X, self.delay_)
        else:
            self.dimension_ = int(self.dimension)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embed 1D time series into delay coordinates.

        Input: (n_samples,)
        Output: (n_samples - (dimension-1)*delay, dimension)
        """
        X = np.asarray(X).ravel()

        if self.delay_ is None or self.dimension_ is None:
            raise RuntimeError("Call .fit() before .transform()")

        d = self.dimension_
        tau = self.delay_
        n = len(X) - (d - 1) * tau

        if n <= 0:
            raise ValueError(
                f"Time series too short ({len(X)}) for delay={tau}, dim={d}. "
                f"Need at least {(d - 1) * tau + 1} samples."
            )

        cloud = np.zeros((n, d))
        for i in range(d):
            cloud[:, i] = X[i * tau: i * tau + n]

        return cloud

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(X).transform(X)
