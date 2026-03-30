"""ConeDetector: directed projection geometry via depth-stratified topology.

Core idea: if a source attractor C projects through receivers at increasing
depth (C -> A3 -> A5), the joint state-space cross-section topology should
change systematically with depth along the projection axis. This progressive
topological enrichment IS the cone.

Key methods:
  - estimate_projection_axis(): conditional-mean PCA of receiver cloud
    conditioned on source state quantiles
  - slice_at_depth(): extract cross-section point clouds at each depth bin
  - availability_profile(): Betti numbers as function of depth (core output)
  - coupling_influence_subspace(): CCA-based low-dimensional projection
  - depth_asymmetry(): binding score difference [C;A5] vs [C;A3]
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA

from att.embedding.joint import JointEmbedder
from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer


class ConeDetector:
    """Detect conical projection geometry in directed attractor networks.

    Parameters
    ----------
    n_depth_bins : int
        Number of bins along the projection axis for depth slicing.
        Start with 5 for statistical power (~3200+ pts/bin at 80k steps).
    max_dim : int
        Maximum homology dimension for persistence computation.
    n_quantiles : int
        Number of source-state quantiles for axis estimation.
    cca_components : int
        Number of CCA dimensions for coupling-influence subspace.
    """

    def __init__(
        self,
        n_depth_bins: int = 5,
        max_dim: int = 1,
        n_quantiles: int = 20,
        cca_components: int = 3,
    ) -> None:
        self.n_depth_bins = n_depth_bins
        self.max_dim = max_dim
        self.n_quantiles = n_quantiles
        self.cca_components = cca_components

        # Fitted state (populated by fit())
        self._source_embedded: np.ndarray | None = None
        self._receiver_cloud: np.ndarray | None = None
        self._projection_axis: np.ndarray | None = None
        self._depth_projections: np.ndarray | None = None
        self._cca_subspace: np.ndarray | None = None
        self._availability: dict | None = None

    def fit(
        self,
        source_ts: np.ndarray,
        receiver_channels: list[np.ndarray],
        source_embedder: JointEmbedder | None = None,
        receiver_embedder: JointEmbedder | None = None,
    ) -> "ConeDetector":
        """Fit the cone detector on source + receiver time series.

        Parameters
        ----------
        source_ts : 1D array
            Source node time series (e.g., C's x-component).
        receiver_channels : list of 1D arrays
            Receiver node time series (e.g., [A3_x, B3_x, A5_x, B5_x]).
        source_embedder : optional pre-configured embedder for source
        receiver_embedder : optional pre-configured embedder for receivers

        Returns self for fluent API.
        """
        # Embed source (single channel)
        src_emb = source_embedder or TakensEmbedder(delay="auto", dimension="auto")
        self._source_embedded = src_emb.fit_transform(source_ts)

        # Joint-embed all receiver channels
        rcv_emb = receiver_embedder or JointEmbedder(delays="auto", dimensions="auto")
        self._receiver_cloud = rcv_emb.fit_transform(receiver_channels)

        # Tail-align (both embedders trim from the start)
        n = min(len(self._source_embedded), len(self._receiver_cloud))
        self._source_embedded = self._source_embedded[-n:]
        self._receiver_cloud = self._receiver_cloud[-n:]

        # Estimate projection axis and project
        self._projection_axis = self.estimate_projection_axis()
        self._depth_projections = self._receiver_cloud @ self._projection_axis

        # Compute coupling-influence subspace
        self._cca_subspace = self.coupling_influence_subspace()

        return self

    def estimate_projection_axis(self) -> np.ndarray:
        """Estimate the cone's projection axis via conditional-mean PCA.

        For each quantile of the source's state, compute the conditional
        mean of the receiver joint cloud. The first PC of those conditional
        means is the projection axis — the direction along which the source's
        variation maximally structures the receiver cloud.

        Returns: unit vector in receiver embedding space.
        """
        source_vals = self._source_embedded[:, 0]
        edges = np.quantile(source_vals, np.linspace(0, 1, self.n_quantiles + 1))

        # Conditional means of receiver cloud per source quantile bin
        conditional_means = []
        for i in range(self.n_quantiles):
            if i < self.n_quantiles - 1:
                mask = (source_vals >= edges[i]) & (source_vals < edges[i + 1])
            else:
                mask = (source_vals >= edges[i]) & (source_vals <= edges[i + 1])
            if mask.sum() > 0:
                conditional_means.append(self._receiver_cloud[mask].mean(axis=0))

        # Fallback: PCA of full cloud if too few conditional means
        if len(conditional_means) < 3:
            centered = self._receiver_cloud - self._receiver_cloud.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            return Vt[0] / np.linalg.norm(Vt[0])

        cond_matrix = np.array(conditional_means)
        cond_centered = cond_matrix - cond_matrix.mean(axis=0)
        _, _, Vt = np.linalg.svd(cond_centered, full_matrices=False)
        axis = Vt[0]
        return axis / np.linalg.norm(axis)

    def slice_at_depth(self, depth_bin: int) -> np.ndarray:
        """Extract the cross-section point cloud at a given depth bin.

        Parameters
        ----------
        depth_bin : int in [0, n_depth_bins)

        Returns: (n_points_in_bin, embedding_dim) array
        """
        if depth_bin < 0 or depth_bin >= self.n_depth_bins:
            raise ValueError(
                f"depth_bin must be in [0, {self.n_depth_bins}), got {depth_bin}"
            )
        edges = np.quantile(
            self._depth_projections,
            np.linspace(0, 1, self.n_depth_bins + 1),
        )
        lo, hi = edges[depth_bin], edges[depth_bin + 1]
        if depth_bin < self.n_depth_bins - 1:
            mask = (self._depth_projections >= lo) & (self._depth_projections < hi)
        else:
            mask = (self._depth_projections >= lo) & (self._depth_projections <= hi)
        return self._receiver_cloud[mask]

    def availability_profile(
        self,
        subspace: str = "full",
        subsample: int | None = 2000,
    ) -> dict:
        """Compute the availability profile: Betti numbers vs depth.

        This is the core output — the shape of the cone expressed as
        topology-vs-depth.

        Parameters
        ----------
        subspace : "full" for full Takens embedding, "cca" for
            coupling-influence subspace
        subsample : max points per depth bin for persistence computation

        Returns
        -------
        dict with:
          'depths': array of bin centers
          'betti_0': array of Betti_0 per bin
          'betti_1': array of Betti_1 per bin
          'persistence_entropy': list of entropies per bin
          'diagrams': list of persistence diagrams per bin
          'is_monotonic': bool — does Betti_1 increase with depth?
          'trend_slope': float — linear regression slope of Betti_1 vs depth
        """
        edges = np.quantile(
            self._depth_projections,
            np.linspace(0, 1, self.n_depth_bins + 1),
        )
        centers = (edges[:-1] + edges[1:]) / 2

        betti_0 = np.zeros(self.n_depth_bins, dtype=int)
        betti_1 = np.zeros(self.n_depth_bins, dtype=int)
        all_entropy = []
        all_diagrams = []
        rng = np.random.default_rng(42)

        for i in range(self.n_depth_bins):
            # Select points in this depth bin
            lo, hi = edges[i], edges[i + 1]
            if i < self.n_depth_bins - 1:
                mask = (self._depth_projections >= lo) & (
                    self._depth_projections < hi
                )
            else:
                mask = (self._depth_projections >= lo) & (
                    self._depth_projections <= hi
                )

            if subspace == "cca" and self._cca_subspace is not None:
                cloud = self._cca_subspace[mask]
            else:
                cloud = self._receiver_cloud[mask]

            # Subsample if needed
            if subsample is not None and len(cloud) > subsample:
                idx = rng.choice(len(cloud), size=subsample, replace=False)
                cloud = cloud[idx]

            # Compute persistent homology
            pa = PersistenceAnalyzer(max_dim=self.max_dim, backend="ripser")
            result = pa.fit_transform(cloud)
            diagrams = result["diagrams"]
            all_diagrams.append(diagrams)
            all_entropy.append(result["persistence_entropy"])

            # Count Betti numbers (features with persistence > 10% of max)
            for dim in range(min(2, len(diagrams))):
                dgm = diagrams[dim]
                if len(dgm) == 0:
                    continue
                persistence = dgm[:, 1] - dgm[:, 0]
                threshold = 0.1 * persistence.max() if persistence.max() > 0 else 0
                count = int((persistence > threshold).sum())
                if dim == 0:
                    betti_0[i] = count
                elif dim == 1:
                    betti_1[i] = count

        # Trend analysis
        if self.n_depth_bins >= 2:
            trend_slope = float(np.polyfit(centers, betti_1, 1)[0])
        else:
            trend_slope = 0.0
        is_monotonic = all(
            betti_1[j + 1] >= betti_1[j] for j in range(len(betti_1) - 1)
        )

        self._availability = {
            "depths": centers,
            "betti_0": betti_0,
            "betti_1": betti_1,
            "persistence_entropy": all_entropy,
            "diagrams": all_diagrams,
            "is_monotonic": bool(is_monotonic),
            "trend_slope": trend_slope,
        }
        return self._availability

    def coupling_influence_subspace(self) -> np.ndarray:
        """Estimate the low-dim subspace where source maximally predicts receivers.

        Uses Canonical Correlation Analysis (CCA) between the embedded
        source state and the receiver joint cloud.

        Returns: (n_points, cca_components) projected receiver cloud
        """
        n_components = min(
            self.cca_components,
            self._source_embedded.shape[1],
            self._receiver_cloud.shape[1],
        )
        cca = CCA(n_components=n_components)
        cca.fit(self._source_embedded, self._receiver_cloud)
        _, receiver_cca = cca.transform(
            self._source_embedded, self._receiver_cloud
        )
        return receiver_cca

    def depth_asymmetry(
        self,
        source_ts: np.ndarray,
        shallow_ts: np.ndarray,
        deep_ts: np.ndarray,
        subsample: int | None = None,
        seed: int | None = None,
    ) -> dict:
        """Compute binding asymmetry between shallow and deep pairings.

        Compares [source; shallow] binding vs [source; deep] binding.
        A positive asymmetry (deep > shallow) indicates the cone opens
        with depth.

        This is a simpler measure than the full availability profile,
        using ATT's existing BindingDetector directly.

        Parameters
        ----------
        source_ts : 1D source time series (C's x)
        shallow_ts : 1D shallow receiver time series (A3's x)
        deep_ts : 1D deep receiver time series (A5's x)
        subsample : max points for persistence computation (saves memory)
        seed : random seed for subsampling

        Returns
        -------
        dict with:
          'shallow_binding': float
          'deep_binding': float
          'asymmetry': float (deep - shallow)
          'ratio': float (deep / shallow, if shallow > 0)
        """
        from att.binding import BindingDetector

        bd_shallow = BindingDetector(max_dim=self.max_dim)
        bd_shallow.fit(source_ts, shallow_ts, subsample=subsample, seed=seed)
        score_shallow = bd_shallow.binding_score()

        bd_deep = BindingDetector(max_dim=self.max_dim)
        bd_deep.fit(source_ts, deep_ts, subsample=subsample, seed=seed)
        score_deep = bd_deep.binding_score()

        return {
            "shallow_binding": score_shallow,
            "deep_binding": score_deep,
            "asymmetry": score_deep - score_shallow,
            "ratio": score_deep / max(score_shallow, 1e-10),
        }

    def full_chain_emergence(
        self,
        source_ts: np.ndarray,
        shallow_ts: np.ndarray,
        deep_ts: np.ndarray,
        subsample: int | None = None,
        seed: int | None = None,
    ) -> dict:
        """Test whether the full chain [C; A3; A5] has emergent topology.

        Compares the 3-way joint binding against the max of all pairwise
        bindings. Emergent topology = 3-way > max(pairwise).

        Parameters
        ----------
        subsample : max points for persistence computation (saves memory)
        seed : random seed for subsampling

        Returns
        -------
        dict with:
          'pairwise_bindings': dict of pair -> score
          'full_chain_binding': float
          'max_pairwise': float
          'emergence': float (full_chain - max_pairwise)
          'has_emergence': bool
        """
        from att.binding import BindingDetector

        # Pairwise bindings
        pairs = {
            "source_shallow": (source_ts, shallow_ts),
            "source_deep": (source_ts, deep_ts),
            "shallow_deep": (shallow_ts, deep_ts),
        }
        pairwise_bindings = {}
        for name, (a, b) in pairs.items():
            bd = BindingDetector(max_dim=self.max_dim)
            bd.fit(a, b, subsample=subsample, seed=seed)
            pairwise_bindings[name] = bd.binding_score()

        max_pairwise = max(pairwise_bindings.values())

        # 3-way joint: embed all three, compute PI subtraction vs marginals
        embedder = JointEmbedder(delays="auto", dimensions="auto")
        joint_cloud = embedder.fit_transform([source_ts, shallow_ts, deep_ts])
        marginal_clouds = embedder.transform_marginals(
            [source_ts, shallow_ts, deep_ts]
        )

        # Compute PH for joint and each marginal
        joint_pa = PersistenceAnalyzer(max_dim=self.max_dim)
        joint_pa.fit_transform(joint_cloud, subsample=subsample, seed=seed)

        marginal_pas = []
        for mc in marginal_clouds:
            pa = PersistenceAnalyzer(max_dim=self.max_dim)
            pa.fit_transform(mc, subsample=subsample, seed=seed)
            marginal_pas.append(pa)

        # Shared ranges for persistence images
        all_diagrams = [joint_pa.diagrams_] + [mp.diagrams_ for mp in marginal_pas]
        birth_min, birth_max = float("inf"), float("-inf")
        pers_max = 0.0
        for diags in all_diagrams:
            for dgm in diags:
                if len(dgm) > 0:
                    birth_min = min(birth_min, dgm[:, 0].min())
                    birth_max = max(birth_max, dgm[:, 0].max())
                    pers_max = max(pers_max, (dgm[:, 1] - dgm[:, 0]).max())
        if birth_min == float("inf"):
            birth_min, birth_max, pers_max = 0.0, 1.0, 1.0

        img_kwargs = {
            "birth_range": (birth_min, birth_max),
            "persistence_range": (0.0, pers_max),
        }
        joint_images = joint_pa.to_image(**img_kwargs)
        marginal_image_sets = [mp.to_image(**img_kwargs) for mp in marginal_pas]

        # Baseline: element-wise max across all marginal images
        full_chain_score = 0.0
        for dim in range(len(joint_images)):
            baseline = np.maximum.reduce(
                [mis[dim] for mis in marginal_image_sets]
            )
            residual = joint_images[dim] - baseline
            full_chain_score += float(np.sum(np.maximum(residual, 0)))

        emergence = full_chain_score - max_pairwise
        return {
            "pairwise_bindings": pairwise_bindings,
            "full_chain_binding": full_chain_score,
            "max_pairwise": max_pairwise,
            "emergence": emergence,
            "has_emergence": emergence > 0,
        }
