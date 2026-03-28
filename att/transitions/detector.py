"""Sliding-window persistent homology for topology transition detection."""

import numpy as np
from att.topology.persistence import PersistenceAnalyzer


class TransitionDetector:
    """Detect topological transitions via sliding-window persistent homology.

    Parameters
    ----------
    window_size : int
        Number of points per window.
    step_size : int
        Step between consecutive windows.
    max_dim : int
        Maximum homology dimension.
    backend : str
        PersistenceAnalyzer backend ("ripser" or "gudhi").
    subsample : int or None
        Points to subsample per window (None = use all).
    """

    def __init__(
        self,
        window_size: int = 500,
        step_size: int = 50,
        max_dim: int = 1,
        backend: str = "ripser",
        subsample: int | None = None,
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.max_dim = max_dim
        self.backend = backend
        self.subsample = subsample
        self._result = None

    def fit_transform(
        self,
        X: np.ndarray,
        seed: int | None = None,
        embedding_dim: int | None = None,
        embedding_delay: int | None = None,
    ) -> dict:
        """Run sliding-window PH on input data.

        Parameters
        ----------
        X : array
            If 2D (n_points, dim): pre-embedded point cloud. Windows the cloud directly.
            If 1D (n_samples,): time series. Embeds each window separately
            (requires embedding_dim and embedding_delay).
        seed : random seed for subsampling
        embedding_dim : embedding dimension (required for 1D input)
        embedding_delay : embedding delay (required for 1D input)

        Returns
        -------
        dict with keys:
            topology_timeseries: list of fit_transform results per window
            distances: list of bottleneck distances between consecutive windows
            image_distances: list of L2 distances between consecutive persistence images
            window_centers: array of center sample indices
            transition_scores: array (same as image_distances, the default score)
        """
        X = np.asarray(X)
        is_1d = X.ndim == 1

        if is_1d:
            if embedding_dim is None or embedding_delay is None:
                raise ValueError("1D input requires embedding_dim and embedding_delay")
            from att.embedding.takens import TakensEmbedder

        # Generate windows
        n_samples = len(X)

        window_starts = list(
            range(0, n_samples - self.window_size + 1, self.step_size)
        )
        if not window_starts:
            raise ValueError(
                f"Input ({n_samples} points) too short for window_size={self.window_size}"
            )

        # Phase 1: Compute PH per window
        analyzers = []
        topology_timeseries = []
        window_centers = []

        for start in window_starts:
            end = start + self.window_size
            if is_1d:
                embedder = TakensEmbedder(delay=embedding_delay, dimension=embedding_dim)
                embedder.fit(X[start:end])
                cloud = embedder.transform(X[start:end])
            else:
                cloud = X[start:end]

            pa = PersistenceAnalyzer(max_dim=self.max_dim, backend=self.backend)
            result = pa.fit_transform(cloud, subsample=self.subsample, seed=seed)
            analyzers.append(pa)
            topology_timeseries.append(result)
            window_centers.append(start + self.window_size // 2)

        window_centers = np.array(window_centers)

        # Phase 2: Compute shared birth/persistence ranges across ALL windows
        all_births = []
        all_persistences = []
        for res in topology_timeseries:
            for dgm in res["diagrams"]:
                if len(dgm) > 0:
                    all_births.extend(dgm[:, 0].tolist())
                    pers = dgm[:, 1] - dgm[:, 0]
                    all_persistences.extend(pers[pers > 1e-10].tolist())

        if all_births and all_persistences:
            birth_range = (min(all_births), max(all_births))
            persistence_range = (0.0, max(all_persistences))
        else:
            birth_range = (0.0, 1.0)
            persistence_range = (0.0, 1.0)

        # Phase 3: Re-compute images on shared grid
        shared_images = []
        for pa in analyzers:
            imgs = pa.to_image(birth_range=birth_range, persistence_range=persistence_range)
            shared_images.append(imgs)

        # Phase 4: Compute distances between consecutive windows
        distances = []
        image_distances = []

        for i in range(len(analyzers) - 1):
            # Bottleneck distance
            d = analyzers[i].distance(analyzers[i + 1], metric="bottleneck")
            distances.append(d)

            # L2 image distance (sum across dimensions)
            img_dist = 0.0
            for dim in range(self.max_dim + 1):
                diff = shared_images[i][dim] - shared_images[i + 1][dim]
                img_dist += float(np.sqrt(np.sum(diff**2)))
            image_distances.append(img_dist)

        distances = np.array(distances)
        image_distances = np.array(image_distances)

        self._result = {
            "topology_timeseries": topology_timeseries,
            "distances": distances,
            "image_distances": image_distances,
            "window_centers": window_centers,
            "transition_scores": image_distances,
            "_analyzers": analyzers,
            "_shared_images": shared_images,
        }
        return self._result

    def detect_changepoints(
        self,
        method: str = "cusum",
        threshold: float | None = None,
    ) -> list[int]:
        """Detect changepoints in the transition score series.

        Parameters
        ----------
        method : "cusum" or "threshold"
        threshold : detection threshold. Default: mean + 2*std for cusum,
            mean + 2*std for threshold.

        Returns
        -------
        List of indices into window_centers[:-1] where transitions detected.
        """
        if self._result is None:
            raise RuntimeError("Call fit_transform first.")

        scores = self._result["transition_scores"]
        if len(scores) == 0:
            return []

        if method == "cusum":
            return self._cusum_changepoints(scores, threshold)
        elif method == "threshold":
            if threshold is None:
                threshold = float(np.mean(scores) + 2 * np.std(scores))
            return [int(i) for i in np.where(scores > threshold)[0]]
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _cusum_changepoints(
        scores: np.ndarray,
        threshold: float | None = None,
    ) -> list[int]:
        """Forward CUSUM changepoint detection.

        Accumulates positive deviations from the mean. A changepoint is
        detected where the cumulative sum exceeds the threshold, then
        the accumulator resets.
        """
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if threshold is None:
            threshold = mean + 2 * std

        cusum = 0.0
        changepoints = []
        for i, s in enumerate(scores):
            cusum = max(0.0, cusum + (s - mean))
            if cusum > threshold:
                changepoints.append(i)
                cusum = 0.0  # Reset after detection

        return changepoints

    def plot_timeline(self, ground_truth: list[int] | None = None):
        """Plot transition timeline. Delegates to viz module."""
        from att.viz.plotting import plot_transition_timeline

        return plot_transition_timeline(self, ground_truth=ground_truth)
