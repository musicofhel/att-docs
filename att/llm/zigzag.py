"""Direction 4: Zigzag persistence across transformer layers.

Tracks topological features as they are born and die across successive layers
using zigzag persistent homology (Carlsson & de Silva 2010). Each layer's
point cloud defines a VR complex; the zigzag filtration connects consecutive
layers via their union complexes.

Requires dionysus>=2.0 (optional dependency).
Install: pip install dionysus  OR  pip install att-toolkit[zigzag]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from att.llm.loader import HiddenStateLoader

try:
    import dionysus
except ImportError:
    dionysus = None  # type: ignore[assignment]


def _require_dionysus():
    if dionysus is None:
        raise ImportError(
            "dionysus>=2.0 is required for zigzag persistence. "
            "Install with: pip install dionysus  OR  pip install att-toolkit[zigzag]"
        )


@dataclass
class ZigzagResult:
    """Container for zigzag persistence results at one difficulty level."""

    level: int
    barcodes: dict[int, np.ndarray] = field(default_factory=dict)
    """dim -> (n_features, 2) array of (birth_layer, death_layer) bars."""
    n_layers_used: int = 0
    layer_indices: list[int] = field(default_factory=list)


class ZigzagLayerAnalyzer:
    """Zigzag persistent homology across transformer layers.

    Constructs a zigzag filtration:
        VR(X_0) <-> VR(X_0 ∪ X_1) <-> VR(X_1) <-> ... <-> VR(X_{L-1})

    where X_i is the point cloud at layer i. The union complexes use the
    minimum pairwise distance across both layers' embeddings of each point.

    Parameters
    ----------
    max_dim : int
        Maximum homology dimension (default 1 -> H0, H1).
    n_pca_components : int
        PCA dimension reduction before computing distances.
    subsample : int or None
        Subsample points per layer to manage runtime.
    threshold : float or None
        VR complex distance threshold. If None, uses adaptive threshold
        based on data scale.
    seed : int
        Random seed for subsampling.
    """

    def __init__(
        self,
        max_dim: int = 1,
        n_pca_components: int = 50,
        subsample: int | None = 100,
        threshold: float | None = None,
        seed: int = 42,
    ):
        _require_dionysus()
        self.max_dim = max_dim
        self.n_pca_components = n_pca_components
        self.subsample = subsample
        self.threshold = threshold
        self.seed = seed

    def fit(
        self,
        loader: HiddenStateLoader,
        level: int,
        layer_indices: list[int] | None = None,
    ) -> ZigzagResult:
        """Compute zigzag persistence across layers for a difficulty level.

        Parameters
        ----------
        loader : HiddenStateLoader
            Hidden state data.
        level : int
            Difficulty level (1-5).
        layer_indices : list of int or None
            Which layers to include. If None, uses all layers.

        Returns
        -------
        ZigzagResult with barcodes per dimension.
        """
        if layer_indices is None:
            layer_indices = list(range(loader.num_layers))

        n_layers = len(layer_indices)
        if n_layers < 2:
            raise ValueError("Need at least 2 layers for zigzag persistence")

        # Get point clouds per layer, subsample consistently
        clouds = self._get_clouds(loader, level, layer_indices)
        n_pts = clouds[0].shape[0]

        # PCA reduce each cloud
        clouds_pca = self._pca_reduce(clouds)

        # Compute distance matrices per layer
        dist_matrices = [squareform(pdist(c)) for c in clouds_pca]

        # Determine threshold
        thresh = self.threshold
        if thresh is None:
            # Adaptive: median of all pairwise distances across layers
            all_dists = np.concatenate([dm[np.triu_indices(n_pts, k=1)] for dm in dist_matrices])
            thresh = float(np.percentile(all_dists, 30))

        # Build zigzag filtration
        barcodes = self._build_and_compute_zigzag(
            dist_matrices, n_pts, n_layers, thresh
        )

        result = ZigzagResult(
            level=level,
            barcodes=barcodes,
            n_layers_used=n_layers,
            layer_indices=layer_indices,
        )
        return result

    def _get_clouds(
        self, loader: HiddenStateLoader, level: int, layer_indices: list[int]
    ) -> list[np.ndarray]:
        """Extract and subsample point clouds per layer."""
        rng = np.random.default_rng(self.seed)
        mask = loader.get_level_mask(level)
        n_total = mask.sum()

        # Consistent subsampling across layers
        if self.subsample and self.subsample < n_total:
            problem_indices = np.where(mask)[0]
            sub_idx = rng.choice(len(problem_indices), size=self.subsample, replace=False)
            sub_idx.sort()
            selected = problem_indices[sub_idx]
        else:
            selected = np.where(mask)[0]

        clouds = []
        for layer in layer_indices:
            cloud = loader.layer_hidden[selected, layer, :]
            clouds.append(cloud)
        return clouds

    def _pca_reduce(self, clouds: list[np.ndarray]) -> list[np.ndarray]:
        """PCA reduce each cloud independently."""
        result = []
        for cloud in clouds:
            n_comp = min(self.n_pca_components, cloud.shape[0] - 1, cloud.shape[1])
            if n_comp < cloud.shape[1]:
                pca = PCA(n_components=n_comp)
                result.append(pca.fit_transform(cloud))
            else:
                result.append(cloud)
        return result

    def _build_and_compute_zigzag(
        self,
        dist_matrices: list[np.ndarray],
        n_pts: int,
        n_layers: int,
        threshold: float,
    ) -> dict[int, np.ndarray]:
        """Build the zigzag filtration and compute persistence.

        The zigzag has 2*n_layers - 1 time steps:
            t=0: VR(X_0)
            t=1: VR(X_0 ∪ X_1) — union complex
            t=2: VR(X_1)
            t=3: VR(X_1 ∪ X_2) — union complex
            ...
            t=2(L-1): VR(X_{L-1})

        Vertices are always present (all times). Edges appear/disappear based on
        whether they are within threshold at the corresponding layer(s).
        """
        total_times = 2 * n_layers - 1

        # Collect all simplices with their [appear, disappear] intervals
        # Vertices: always present
        simplex_list = []
        times_list = []

        # Add vertices — present throughout
        for i in range(n_pts):
            simplex_list.append(dionysus.Simplex([i]))
            times_list.append([0, total_times])

        # For each pair of points, determine which time intervals they form an edge
        if self.max_dim >= 1:
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    intervals = self._edge_intervals(
                        i, j, dist_matrices, threshold, total_times
                    )
                    if intervals:
                        simplex_list.append(dionysus.Simplex([i, j]))
                        times_list.append(intervals)

        # For triangles (H1 requires 2-simplices)
        if self.max_dim >= 1:
            # Precompute edge presence at each time step for triangle check
            edge_at_time = self._edge_presence_matrix(
                n_pts, dist_matrices, threshold, total_times
            )
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    for k in range(j + 1, n_pts):
                        intervals = self._triangle_intervals(
                            i, j, k, edge_at_time, total_times
                        )
                        if intervals:
                            simplex_list.append(dionysus.Simplex([i, j, k]))
                            times_list.append(intervals)

        f = dionysus.Filtration(simplex_list)
        zz, dgms, cells = dionysus.zigzag_homology_persistence(f, times_list)

        # Convert diagrams to arrays, mapping times back to layer indices
        barcodes = {}
        for dim in range(len(dgms)):
            bars = []
            for pt in dgms[dim]:
                b, d = pt.birth, pt.death
                if d == float("inf"):
                    d = total_times
                # Map time to layer: t=0 -> layer 0, t=2 -> layer 1, etc.
                birth_layer = b / 2.0
                death_layer = d / 2.0
                bars.append([birth_layer, death_layer])
            barcodes[dim] = np.array(bars) if bars else np.empty((0, 2))

        return barcodes

    def _edge_intervals(
        self,
        i: int,
        j: int,
        dist_matrices: list[np.ndarray],
        threshold: float,
        total_times: int,
    ) -> list[float]:
        """Compute appearance/disappearance times for edge (i,j).

        An edge is present at:
        - t=2k if dist_matrices[k][i,j] <= threshold  (layer k)
        - t=2k+1 if min(dist_matrices[k][i,j], dist_matrices[k+1][i,j]) <= threshold  (union)
        """
        n_layers = len(dist_matrices)
        # Determine presence at each time step
        present = np.zeros(total_times, dtype=bool)

        for k in range(n_layers):
            # Layer time
            t_layer = 2 * k
            if dist_matrices[k][i, j] <= threshold:
                present[t_layer] = True

            # Union time (between layer k and k+1)
            if k < n_layers - 1:
                t_union = 2 * k + 1
                min_dist = min(dist_matrices[k][i, j], dist_matrices[k + 1][i, j])
                if min_dist <= threshold:
                    present[t_union] = True

        # Convert boolean presence to [appear, disappear] pairs
        return self._presence_to_intervals(present)

    def _edge_presence_matrix(
        self,
        n_pts: int,
        dist_matrices: list[np.ndarray],
        threshold: float,
        total_times: int,
    ) -> np.ndarray:
        """(n_pts, n_pts, total_times) boolean: whether edge is present at each time."""
        n_layers = len(dist_matrices)
        present = np.zeros((n_pts, n_pts, total_times), dtype=bool)

        for k in range(n_layers):
            t_layer = 2 * k
            within = dist_matrices[k] <= threshold
            present[:, :, t_layer] = within

            if k < n_layers - 1:
                t_union = 2 * k + 1
                min_dist = np.minimum(dist_matrices[k], dist_matrices[k + 1])
                present[:, :, t_union] = min_dist <= threshold

        return present

    def _triangle_intervals(
        self,
        i: int,
        j: int,
        k: int,
        edge_at_time: np.ndarray,
        total_times: int,
    ) -> list[float]:
        """Compute intervals for triangle (i,j,k): present when all 3 edges are."""
        present = (
            edge_at_time[i, j, :total_times]
            & edge_at_time[i, k, :total_times]
            & edge_at_time[j, k, :total_times]
        )
        return self._presence_to_intervals(present)

    @staticmethod
    def _presence_to_intervals(present: np.ndarray) -> list[float]:
        """Convert boolean presence array to [appear, disappear, appear, ...] list."""
        intervals = []
        in_interval = False
        for t, p in enumerate(present):
            if p and not in_interval:
                intervals.append(float(t))
                in_interval = True
            elif not p and in_interval:
                intervals.append(float(t))
                in_interval = False
        if in_interval:
            intervals.append(float(len(present)))
        return intervals


def zigzag_feature_lifetime_stats(result: ZigzagResult, dim: int = 1) -> dict:
    """Compute summary statistics on zigzag barcode lifetimes.

    Parameters
    ----------
    result : ZigzagResult
        Output from ZigzagLayerAnalyzer.fit().
    dim : int
        Homology dimension to summarize.

    Returns
    -------
    dict with keys: mean_lifetime, median_lifetime, max_lifetime, std_lifetime,
    n_features, n_long_lived (> 2 layers).
    """
    if dim not in result.barcodes or len(result.barcodes[dim]) == 0:
        return {
            "mean_lifetime": 0.0,
            "median_lifetime": 0.0,
            "max_lifetime": 0.0,
            "std_lifetime": 0.0,
            "n_features": 0,
            "n_long_lived": 0,
        }

    bars = result.barcodes[dim]
    lifetimes = bars[:, 1] - bars[:, 0]
    # Filter out zero-lifetime features
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return {
            "mean_lifetime": 0.0,
            "median_lifetime": 0.0,
            "max_lifetime": 0.0,
            "std_lifetime": 0.0,
            "n_features": 0,
            "n_long_lived": 0,
        }

    return {
        "mean_lifetime": float(np.mean(lifetimes)),
        "median_lifetime": float(np.median(lifetimes)),
        "max_lifetime": float(np.max(lifetimes)),
        "std_lifetime": float(np.std(lifetimes)),
        "n_features": int(len(lifetimes)),
        "n_long_lived": int(np.sum(lifetimes > 2.0)),
    }


def compare_zigzag_levels(
    result_a: ZigzagResult,
    result_b: ZigzagResult,
    dim: int = 1,
) -> dict:
    """Compare zigzag barcodes between two difficulty levels.

    Parameters
    ----------
    result_a, result_b : ZigzagResult
        Zigzag results for two different levels.
    dim : int
        Homology dimension to compare.

    Returns
    -------
    dict with keys: ks_statistic, ks_pvalue, mean_lifetime_diff, n_features_diff.
    """
    def _lifetimes(r):
        if dim not in r.barcodes or len(r.barcodes[dim]) == 0:
            return np.array([0.0])
        bars = r.barcodes[dim]
        lt = bars[:, 1] - bars[:, 0]
        lt = lt[lt > 0]
        return lt if len(lt) > 0 else np.array([0.0])

    lt_a = _lifetimes(result_a)
    lt_b = _lifetimes(result_b)

    ks_stat, ks_p = ks_2samp(lt_a, lt_b)

    stats_a = zigzag_feature_lifetime_stats(result_a, dim)
    stats_b = zigzag_feature_lifetime_stats(result_b, dim)

    return {
        "level_a": result_a.level,
        "level_b": result_b.level,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "mean_lifetime_a": stats_a["mean_lifetime"],
        "mean_lifetime_b": stats_b["mean_lifetime"],
        "mean_lifetime_diff": stats_b["mean_lifetime"] - stats_a["mean_lifetime"],
        "n_features_a": stats_a["n_features"],
        "n_features_b": stats_b["n_features"],
        "n_features_diff": stats_b["n_features"] - stats_a["n_features"],
    }
