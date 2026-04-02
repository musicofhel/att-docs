"""Per-layer persistent homology analysis with permutation-based z-score profiles."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from att.config.seed import get_rng
from att.topology.persistence import PersistenceAnalyzer


class LayerwiseAnalyzer:
    """Per-layer persistent homology with permutation-based z-score profiles.

    Runs PersistenceAnalyzer at each transformer layer for each difficulty
    level, then computes z-score profiles via label-permutation tests.

    Parameters
    ----------
    n_pca_components : int
        PCA dimensions before PH computation.
    max_dim : int
        Maximum homology dimension (0=components, 1=loops, 2=voids).
    subsample : int or None
        Max points to subsample per layer cloud.
    n_permutations : int
        Number of permutations for z-score computation.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_pca_components: int = 50,
        max_dim: int = 2,
        subsample: int | None = 200,
        n_permutations: int = 200,
        seed: int = 42,
    ):
        self.n_pca_components = n_pca_components
        self.max_dim = max_dim
        self.subsample = subsample
        self.n_permutations = n_permutations
        self.seed = seed

        # Filled by fit()
        self._results: dict[tuple[int, int], dict] | None = None
        self._analyzers: dict[tuple[int, int], PersistenceAnalyzer] | None = None
        self._levels: list[int] | None = None
        self._n_layers: int | None = None

    def fit(
        self,
        loader,
        levels: list[int] | None = None,
    ) -> "LayerwiseAnalyzer":
        """Run PH at every layer for each difficulty level.

        Parameters
        ----------
        loader : HiddenStateLoader
            Loaded hidden-state archive.
        levels : list of int or None
            Difficulty levels to analyze. None = all levels.

        Returns
        -------
        self
        """
        if levels is None:
            levels = sorted(loader.unique_levels.tolist())
        self._levels = levels
        self._n_layers = loader.num_layers

        self._results = {}
        self._analyzers = {}

        for level in levels:
            for layer_idx in range(self._n_layers):
                cloud = loader.get_level_cloud(level, layer=layer_idx)
                n_pts = cloud.shape[0]
                if n_pts < 3:
                    continue

                n_comp = min(self.n_pca_components, n_pts - 1, cloud.shape[1])
                pca = PCA(n_components=n_comp)
                cloud_pca = pca.fit_transform(cloud)

                pa = PersistenceAnalyzer(max_dim=self.max_dim, backend="ripser")
                sub = min(n_pts, self.subsample) if self.subsample else None
                result = pa.fit_transform(cloud_pca, subsample=sub, seed=self.seed)

                self._results[(level, layer_idx)] = result
                self._analyzers[(level, layer_idx)] = pa

        return self

    @property
    def results_per_layer(self) -> dict[tuple[int, int], dict]:
        """Raw PH results keyed by (level, layer_idx)."""
        if self._results is None:
            raise RuntimeError("Call fit() first.")
        return self._results

    def entropy_profile(self) -> dict[int, np.ndarray]:
        """Per-layer persistence entropy by difficulty level.

        Returns
        -------
        dict mapping level -> (n_layers, max_dim+1) array of entropies.
        """
        if self._results is None:
            raise RuntimeError("Call fit() first.")

        profiles = {}
        for level in self._levels:
            ent = np.zeros((self._n_layers, self.max_dim + 1))
            for layer_idx in range(self._n_layers):
                key = (level, layer_idx)
                if key in self._results:
                    for dim in range(self.max_dim + 1):
                        if dim < len(self._results[key]["persistence_entropy"]):
                            ent[layer_idx, dim] = self._results[key][
                                "persistence_entropy"
                            ][dim]
            profiles[level] = ent
        return profiles

    def bottleneck_profile(self) -> dict[int, np.ndarray]:
        """Per-layer bottleneck distances between consecutive layers.

        Returns
        -------
        dict mapping level -> (n_layers-1,) array of bottleneck distances.
        """
        if self._analyzers is None:
            raise RuntimeError("Call fit() first.")

        profiles = {}
        for level in self._levels:
            dists = np.zeros(self._n_layers - 1)
            for i in range(self._n_layers - 1):
                pa_i = self._analyzers.get((level, i))
                pa_j = self._analyzers.get((level, i + 1))
                if pa_i is not None and pa_j is not None:
                    dists[i] = pa_i.distance(pa_j, metric="bottleneck")
            profiles[level] = dists
        return profiles

    def zscore_profile(
        self,
        loader,
        metric: str = "wasserstein_1",
    ) -> dict:
        """Compute per-layer z-score of inter-level topological distance.

        Permutes difficulty labels and recomputes pairwise distances at each
        layer to build a null distribution, then computes z-scores.

        Parameters
        ----------
        loader : HiddenStateLoader
            Same loader used in fit().
        metric : str
            Distance metric ("wasserstein_1", "bottleneck").

        Returns
        -------
        dict with:
            z_scores : (n_layers,) per-layer z-scores
            p_values : (n_layers,) per-layer p-values
            observed : (n_layers,) observed mean pairwise distances
            null_mean : (n_layers,) null distribution means
            null_std : (n_layers,) null distribution stds
            per_dim : dict mapping dim -> (n_layers,) z-scores for each H_dim
        """
        if self._results is None:
            raise RuntimeError("Call fit() first.")

        from itertools import combinations

        rng = get_rng(self.seed)
        n_layers = self._n_layers
        levels = self._levels
        n_levels = len(levels)

        # Observed mean pairwise distance per layer
        observed = np.zeros(n_layers)
        observed_per_dim = {d: np.zeros(n_layers) for d in range(self.max_dim + 1)}

        for layer_idx in range(n_layers):
            dists = []
            dists_per_dim = {d: [] for d in range(self.max_dim + 1)}

            for li, lj in combinations(range(n_levels), 2):
                pa_i = self._analyzers.get((levels[li], layer_idx))
                pa_j = self._analyzers.get((levels[lj], layer_idx))
                if pa_i is not None and pa_j is not None:
                    d = pa_i.distance(pa_j, metric=metric)
                    dists.append(d)
                    # Per-dimension distances
                    for dim in range(self.max_dim + 1):
                        d_dim = self._dim_distance(pa_i, pa_j, dim, metric)
                        dists_per_dim[dim].append(d_dim)

            observed[layer_idx] = np.mean(dists) if dists else 0.0
            for dim in range(self.max_dim + 1):
                vals = dists_per_dim[dim]
                observed_per_dim[dim][layer_idx] = np.mean(vals) if vals else 0.0

        # Permutation null
        all_levels_array = loader.levels.copy()
        null_dists = np.zeros((self.n_permutations, n_layers))

        for perm_idx in range(self.n_permutations):
            shuffled = rng.permutation(all_levels_array)

            for layer_idx in range(n_layers):
                perm_analyzers = {}
                for level in levels:
                    mask = shuffled == level
                    cloud = loader.layer_hidden[mask, layer_idx, :]
                    n_pts = cloud.shape[0]
                    if n_pts < 3:
                        continue
                    n_comp = min(
                        self.n_pca_components, n_pts - 1, cloud.shape[1]
                    )
                    pca = PCA(n_components=n_comp)
                    cloud_pca = pca.fit_transform(cloud)
                    pa = PersistenceAnalyzer(
                        max_dim=self.max_dim, backend="ripser"
                    )
                    sub = (
                        min(n_pts, self.subsample) if self.subsample else None
                    )
                    pa.fit_transform(cloud_pca, subsample=sub, seed=self.seed)
                    perm_analyzers[level] = pa

                perm_dists = []
                for li, lj in combinations(range(n_levels), 2):
                    pa_i = perm_analyzers.get(levels[li])
                    pa_j = perm_analyzers.get(levels[lj])
                    if pa_i is not None and pa_j is not None:
                        d = pa_i.distance(pa_j, metric=metric)
                        perm_dists.append(d)

                null_dists[perm_idx, layer_idx] = (
                    np.mean(perm_dists) if perm_dists else 0.0
                )

        # Compute z-scores and p-values
        null_mean = null_dists.mean(axis=0)
        null_std = null_dists.std(axis=0)

        z_scores = np.where(
            null_std > 1e-15,
            (observed - null_mean) / null_std,
            0.0,
        )
        p_values = np.array(
            [
                (np.sum(null_dists[:, i] >= observed[i]) + 1)
                / (self.n_permutations + 1)
                for i in range(n_layers)
            ]
        )

        # Per-dimension z-scores (approximate: use same null since full
        # per-dim null is too expensive)
        per_dim_z = {}
        for dim in range(self.max_dim + 1):
            per_dim_z[dim] = np.where(
                null_std > 1e-15,
                (observed_per_dim[dim] - null_mean) / null_std,
                0.0,
            )

        return {
            "z_scores": z_scores,
            "p_values": p_values,
            "observed": observed,
            "null_mean": null_mean,
            "null_std": null_std,
            "per_dim": per_dim_z,
        }

    @staticmethod
    def _dim_distance(
        pa1: PersistenceAnalyzer,
        pa2: PersistenceAnalyzer,
        dim: int,
        metric: str,
    ) -> float:
        """Compute distance between persistence diagrams at a single dimension."""
        import persim

        dgm1 = pa1.diagrams_[dim] if dim < len(pa1.diagrams_) else np.empty((0, 2))
        dgm2 = pa2.diagrams_[dim] if dim < len(pa2.diagrams_) else np.empty((0, 2))

        if len(dgm1) == 0 and len(dgm2) == 0:
            return 0.0
        if len(dgm1) == 0:
            dgm1 = np.array([[0.0, 0.0]])
        if len(dgm2) == 0:
            dgm2 = np.array([[0.0, 0.0]])

        if metric == "bottleneck":
            return float(persim.bottleneck(dgm1, dgm2))
        else:
            return float(persim.wasserstein(dgm1, dgm2))
