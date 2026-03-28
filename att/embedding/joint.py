"""Joint delay embedding for multi-system analysis."""

import numpy as np

from att.embedding.delay import estimate_delay
from att.embedding.dimension import estimate_dimension


class JointEmbedder:
    """Construct joint delay embeddings with per-channel delay estimation.

    Using "auto" is strongly recommended for systems with different timescales.

    Parameters
    ----------
    delays : list[int] or "auto"
        Per-channel delays. "auto" estimates independently per channel via AMI.
    dimensions : list[int] or "auto"
        Per-channel embedding dimensions. "auto" estimates per channel via FNN.
    """

    def __init__(
        self,
        delays: list[int] | str = "auto",
        dimensions: list[int] | str = "auto",
    ):
        self.delays = delays
        self.dimensions = dimensions
        self.delays_: list[int] | None = None
        self.dimensions_: list[int] | None = None

    def fit(self, channels: list[np.ndarray]) -> "JointEmbedder":
        """Estimate per-channel parameters. Stores .delays_ and .dimensions_."""
        n_channels = len(channels)

        if self.delays == "auto":
            self.delays_ = [estimate_delay(np.asarray(ch).ravel()) for ch in channels]
        else:
            self.delays_ = [int(d) for d in self.delays]
            if len(self.delays_) != n_channels:
                raise ValueError(f"Expected {n_channels} delays, got {len(self.delays_)}")

        if self.dimensions == "auto":
            self.dimensions_ = [
                estimate_dimension(np.asarray(ch).ravel(), self.delays_[i])
                for i, ch in enumerate(channels)
            ]
        else:
            self.dimensions_ = [int(d) for d in self.dimensions]
            if len(self.dimensions_) != n_channels:
                raise ValueError(f"Expected {n_channels} dimensions, got {len(self.dimensions_)}")

        return self

    def transform(self, channels: list[np.ndarray]) -> np.ndarray:
        """Construct joint delay vectors by concatenating per-channel embeddings.

        Input: list of 1D arrays, each (n_samples,)
        Output: (n_valid_samples, sum(dimensions))
        """
        if self.delays_ is None or self.dimensions_ is None:
            raise RuntimeError("Call .fit() before .transform()")

        embeddings = []
        min_length = float("inf")

        for i, ch in enumerate(channels):
            ch = np.asarray(ch).ravel()
            d = self.dimensions_[i]
            tau = self.delays_[i]
            n = len(ch) - (d - 1) * tau

            if n <= 0:
                raise ValueError(
                    f"Channel {i} too short ({len(ch)}) for delay={tau}, dim={d}."
                )

            cloud = np.zeros((n, d))
            for j in range(d):
                cloud[:, j] = ch[j * tau: j * tau + n]

            embeddings.append(cloud)
            min_length = min(min_length, n)

        # Truncate all to same length (determined by most restrictive channel)
        truncated = [emb[:min_length] for emb in embeddings]
        return np.hstack(truncated)

    def transform_marginals(self, channels: list[np.ndarray]) -> list[np.ndarray]:
        """Return individually embedded point clouds for marginal comparison."""
        if self.delays_ is None or self.dimensions_ is None:
            raise RuntimeError("Call .fit() before .transform_marginals()")

        marginals = []
        for i, ch in enumerate(channels):
            ch = np.asarray(ch).ravel()
            d = self.dimensions_[i]
            tau = self.delays_[i]
            n = len(ch) - (d - 1) * tau

            cloud = np.zeros((n, d))
            for j in range(d):
                cloud[:, j] = ch[j * tau: j * tau + n]

            marginals.append(cloud)

        return marginals

    def fit_transform(self, channels: list[np.ndarray]) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(channels).transform(channels)
