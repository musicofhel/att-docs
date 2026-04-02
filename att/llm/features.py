"""Topological feature extraction for ML prediction (Direction 2).

Vectorizes persistence diagrams into fixed-length feature vectors suitable
for logistic regression, random forests, etc. Supports summary statistics
and persistence image features.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from att.topology.persistence import PersistenceAnalyzer


class TopologicalFeatureExtractor:
    """Extract fixed-length topological feature vectors from point clouds.

    Parameters
    ----------
    max_dim : int
        Maximum homology dimension.
    n_pca_components : int
        PCA dimensions before PH computation.
    subsample : int or None
        Max points per cloud for PH.
    feature_set : str
        "summary" (8 features per dim) or "image" (summary + flattened PI).
    pi_resolution : int
        Persistence image resolution (only used when feature_set="image").
    pi_sigma : float
        Persistence image Gaussian bandwidth.
    seed : int
        Random seed.
    """

    # Summary feature names per homology dimension
    _SUMMARY_NAMES = [
        "persistence_entropy",
        "total_persistence",
        "n_features",
        "max_lifetime",
        "mean_lifetime",
        "std_lifetime",
        "max_birth",
        "mean_birth",
    ]

    def __init__(
        self,
        max_dim: int = 1,
        n_pca_components: int = 50,
        subsample: int | None = 200,
        feature_set: str = "summary",
        pi_resolution: int = 20,
        pi_sigma: float = 0.1,
        seed: int = 42,
    ):
        self.max_dim = max_dim
        self.n_pca_components = n_pca_components
        self.subsample = subsample
        self.feature_set = feature_set
        self.pi_resolution = pi_resolution
        self.pi_sigma = pi_sigma
        self.seed = seed

    @property
    def feature_names(self) -> list[str]:
        """List of feature names matching the output columns."""
        names = []
        for dim in range(self.max_dim + 1):
            for name in self._SUMMARY_NAMES:
                names.append(f"H{dim}_{name}")
        if self.feature_set == "image":
            for dim in range(self.max_dim + 1):
                for i in range(self.pi_resolution):
                    for j in range(self.pi_resolution):
                        names.append(f"H{dim}_pi_{i}_{j}")
        return names

    @property
    def n_features(self) -> int:
        """Total number of features."""
        n_summary = len(self._SUMMARY_NAMES) * (self.max_dim + 1)
        if self.feature_set == "image":
            n_summary += (self.pi_resolution ** 2) * (self.max_dim + 1)
        return n_summary

    def _summarize_diagram(self, diagrams: list[np.ndarray]) -> np.ndarray:
        """Extract summary statistics from persistence diagrams.

        Returns (n_summary_features,) array.
        """
        features = []
        for dim in range(self.max_dim + 1):
            dgm = diagrams[dim] if dim < len(diagrams) else np.empty((0, 2))

            if len(dgm) == 0:
                features.extend([0.0] * len(self._SUMMARY_NAMES))
                continue

            lifetimes = dgm[:, 1] - dgm[:, 0]
            lifetimes = lifetimes[lifetimes > 0]
            births = dgm[:, 0]

            if len(lifetimes) == 0:
                features.extend([0.0] * len(self._SUMMARY_NAMES))
                continue

            # Persistence entropy
            total = lifetimes.sum()
            probs = lifetimes / (total + 1e-15)
            entropy = -np.sum(probs * np.log(probs + 1e-15))

            features.extend([
                float(entropy),                  # persistence_entropy
                float(total),                    # total_persistence
                float(len(lifetimes)),           # n_features
                float(np.max(lifetimes)),        # max_lifetime
                float(np.mean(lifetimes)),       # mean_lifetime
                float(np.std(lifetimes)),        # std_lifetime
                float(np.max(births)),           # max_birth
                float(np.mean(births)),          # mean_birth
            ])

        return np.array(features)

    def extract_single(self, cloud: np.ndarray) -> np.ndarray:
        """Extract topological features from a single point cloud.

        Parameters
        ----------
        cloud : (n_points, d) point cloud.

        Returns
        -------
        (n_features,) feature vector.
        """
        n_pts = cloud.shape[0]
        if n_pts < 3:
            return np.zeros(self.n_features)

        n_comp = min(self.n_pca_components, n_pts - 1, cloud.shape[1])
        pca = PCA(n_components=n_comp)
        cloud_pca = pca.fit_transform(cloud)

        pa = PersistenceAnalyzer(max_dim=self.max_dim, backend="ripser")
        sub = min(n_pts, self.subsample) if self.subsample else None
        result = pa.fit_transform(cloud_pca, subsample=sub, seed=self.seed)
        diagrams = result["diagrams"]

        summary = self._summarize_diagram(diagrams)

        if self.feature_set == "summary":
            return summary

        # Image features
        images = pa.to_image(
            resolution=self.pi_resolution,
            sigma=self.pi_sigma,
        )
        image_features = []
        for dim in range(self.max_dim + 1):
            img = images[dim] if dim < len(images) else np.zeros(
                (self.pi_resolution, self.pi_resolution)
            )
            image_features.append(img.ravel())

        return np.concatenate([summary, *image_features])

    def extract_batch(
        self,
        loader,
        layer: int = -1,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract features for all problems in a loader, per difficulty level.

        Computes PH on the level-cloud at the given layer for each difficulty
        level, producing one feature vector per level.

        Parameters
        ----------
        loader : HiddenStateLoader
        layer : int
            Layer index (-1 = final layer).

        Returns
        -------
        X : (n_levels, n_features) feature matrix.
        feature_names : list of feature name strings.
        """
        levels = sorted(loader.unique_levels.tolist())
        X = np.zeros((len(levels), self.n_features))

        for i, level in enumerate(levels):
            cloud = loader.get_level_cloud(level, layer=layer)
            X[i] = self.extract_single(cloud)

        return X, self.feature_names

    def extract_per_problem(
        self,
        loader,
        layer: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features per problem using token trajectories.

        Each problem's token trajectory (T_i, d) is treated as a point cloud.

        Parameters
        ----------
        loader : HiddenStateLoader
        layer : int
            Not used for token trajectories (included for API consistency).

        Returns
        -------
        X : (n_problems, n_features) feature matrix.
        levels : (n_problems,) difficulty levels.
        """
        n = loader.n_problems
        X = np.zeros((n, self.n_features))

        for i in range(n):
            traj = loader.token_trajectories[i]
            if traj is not None and len(traj) >= 3:
                X[i] = self.extract_single(traj)

        return X, loader.levels.copy()
