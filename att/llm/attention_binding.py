"""Attention-hidden topological coupling analysis (Direction 10).

Measures coupling between attention topology and hidden-state topology by
computing persistence images on both and comparing via PI subtraction.
Uses row-permutation surrogates for significance testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA

from att.topology.persistence import PersistenceAnalyzer


@dataclass
class BindingResult:
    """Result of attention-hidden binding analysis for one problem/layer."""

    binding_score: float
    attention_entropy: dict[int, float]
    hidden_entropy: dict[int, float]
    n_attention_features: dict[int, int]
    n_hidden_features: dict[int, int]


@dataclass
class SignificanceResult:
    """Result of permutation significance test."""

    observed_score: float
    null_scores: np.ndarray
    p_value: float
    z_score: float
    n_permutations: int


class AttentionHiddenBinding:
    """Measure topological coupling between attention and hidden-state geometry.

    Treats 1 - attention_weight as a precomputed distance matrix and computes
    PH on it. Compares against PH on hidden-state point clouds using persistence
    image subtraction (same principle as BindingDetector).

    Parameters
    ----------
    max_dim : int
        Maximum homology dimension.
    image_resolution : int
        Resolution of persistence images for comparison.
    image_sigma : float
        Gaussian kernel bandwidth for persistence images.
    n_pca_components : int
        PCA components for hidden-state clouds.
    subsample : int
        Max points for hidden-state PH.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        max_dim: int = 1,
        image_resolution: int = 50,
        image_sigma: float = 0.1,
        n_pca_components: int = 50,
        subsample: int = 100,
        seed: int = 42,
    ):
        self.max_dim = max_dim
        self.image_resolution = image_resolution
        self.image_sigma = image_sigma
        self.n_pca_components = n_pca_components
        self.subsample = subsample
        self.seed = seed

    @staticmethod
    def attention_to_distance(attn: np.ndarray) -> np.ndarray:
        """Convert attention matrix to symmetric distance matrix.

        D = 1 - (A + A^T) / 2, clipped to [0, 1], diagonal zeroed.
        """
        sym = (attn + attn.T) / 2.0
        dist = 1.0 - sym
        np.clip(dist, 0.0, 1.0, out=dist)
        np.fill_diagonal(dist, 0.0)
        return dist

    def compute_binding(
        self,
        attention_matrix: np.ndarray,
        hidden_cloud: np.ndarray,
    ) -> BindingResult:
        """Compute binding score between attention topology and hidden-state topology.

        Parameters
        ----------
        attention_matrix : (n, n) attention weight matrix (head-averaged).
        hidden_cloud : (n, d) hidden-state vectors for the same tokens.

        Returns
        -------
        BindingResult with binding score and per-dim feature counts / entropy.
        """
        n = attention_matrix.shape[0]

        # Subsample both consistently
        if n > self.subsample:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n, size=self.subsample, replace=False)
            idx = np.sort(idx)
            attention_matrix = attention_matrix[np.ix_(idx, idx)]
            hidden_cloud = hidden_cloud[idx]

        # Attention PH via precomputed distance
        attn_dist = self.attention_to_distance(attention_matrix)
        pa_attn = PersistenceAnalyzer(
            max_dim=self.max_dim, backend="ripser", metric="precomputed"
        )
        result_attn = pa_attn.fit_transform(attn_dist)

        # Hidden-state PH via Euclidean distance (with PCA)
        n_pts = hidden_cloud.shape[0]
        n_comp = min(self.n_pca_components, n_pts - 1, hidden_cloud.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp)
            cloud_pca = pca.fit_transform(hidden_cloud)
        else:
            cloud_pca = hidden_cloud

        pa_hidden = PersistenceAnalyzer(max_dim=self.max_dim, backend="ripser")
        result_hidden = pa_hidden.fit_transform(cloud_pca)

        # Compute binding via PI subtraction
        birth_range, pers_range = self._shared_ranges(
            pa_attn.diagrams_, pa_hidden.diagrams_
        )

        imgs_attn = pa_attn.to_image(
            self.image_resolution, self.image_sigma, birth_range, pers_range
        )
        imgs_hidden = pa_hidden.to_image(
            self.image_resolution, self.image_sigma, birth_range, pers_range
        )

        # Binding = L1 of |PI_attn - PI_hidden| (symmetric coupling measure)
        score = 0.0
        for d in range(self.max_dim + 1):
            diff = np.abs(imgs_attn[d] - imgs_hidden[d])
            # Coupling = overlap (complement of difference)
            # High similarity → high coupling → low diff
            # So binding = 1 - normalized_diff, but simpler: use correlation
            pass

        # Alternative: correlation-based coupling
        # Correlation of PI vectors: high correlation = topologies match = tight coupling
        score = self._pi_correlation(imgs_attn, imgs_hidden)

        attn_entropy = result_attn.get("persistence_entropy", {})
        hidden_entropy = result_hidden.get("persistence_entropy", {})

        attn_features = {}
        hidden_features = {}
        for d in range(self.max_dim + 1):
            attn_features[d] = len(result_attn["diagrams"][d]) if d < len(result_attn["diagrams"]) else 0
            hidden_features[d] = len(result_hidden["diagrams"][d]) if d < len(result_hidden["diagrams"]) else 0

        return BindingResult(
            binding_score=score,
            attention_entropy=attn_entropy,
            hidden_entropy=hidden_entropy,
            n_attention_features=attn_features,
            n_hidden_features=hidden_features,
        )

    def _pi_correlation(
        self, imgs_a: list[np.ndarray], imgs_b: list[np.ndarray]
    ) -> float:
        """Correlation-based coupling score between two sets of persistence images.

        Concatenates flattened PIs across dimensions and computes Pearson correlation.
        Returns value in [-1, 1]; higher = tighter topological coupling.
        """
        vec_a = np.concatenate([img.ravel() for img in imgs_a])
        vec_b = np.concatenate([img.ravel() for img in imgs_b])

        if np.std(vec_a) < 1e-15 or np.std(vec_b) < 1e-15:
            return 0.0

        return float(np.corrcoef(vec_a, vec_b)[0, 1])

    def _shared_ranges(
        self,
        diagrams_a: list[np.ndarray],
        diagrams_b: list[np.ndarray],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Compute shared birth and persistence ranges for PI computation."""
        all_births = []
        all_pers = []
        for dgms in [diagrams_a, diagrams_b]:
            for dgm in dgms:
                if len(dgm) > 0:
                    births = dgm[:, 0]
                    deaths = dgm[:, 1]
                    all_births.extend(births.tolist())
                    all_pers.extend((deaths - births).tolist())

        if not all_births:
            return (0.0, 1.0), (0.0, 1.0)

        birth_range = (min(all_births), max(all_births))
        pers_range = (0.0, max(all_pers) if all_pers else 1.0)

        # Ensure non-degenerate ranges
        if birth_range[1] - birth_range[0] < 1e-15:
            birth_range = (birth_range[0], birth_range[0] + 1.0)
        if pers_range[1] - pers_range[0] < 1e-15:
            pers_range = (0.0, 1.0)

        return birth_range, pers_range

    def test_significance(
        self,
        attention_matrix: np.ndarray,
        hidden_cloud: np.ndarray,
        n_permutations: int = 100,
    ) -> SignificanceResult:
        """Test binding significance via row-permutation surrogates.

        Permutes rows (and corresponding columns) of the attention matrix to
        destroy the attention-hidden correspondence while preserving attention
        structure. The observed binding score is compared against the null
        distribution of surrogate scores.

        Parameters
        ----------
        attention_matrix : (n, n) attention weight matrix.
        hidden_cloud : (n, d) hidden-state vectors.
        n_permutations : int
            Number of surrogate permutations.

        Returns
        -------
        SignificanceResult with observed score, null distribution, p-value, z-score.
        """
        observed = self.compute_binding(attention_matrix, hidden_cloud)
        observed_score = observed.binding_score

        rng = np.random.default_rng(self.seed)
        null_scores = np.zeros(n_permutations)

        for i in range(n_permutations):
            perm = rng.permutation(attention_matrix.shape[0])
            # Permute rows and columns of attention to break correspondence with hidden
            attn_perm = attention_matrix[np.ix_(perm, perm)]
            result = self.compute_binding(attn_perm, hidden_cloud)
            null_scores[i] = result.binding_score

        null_mean = np.mean(null_scores)
        null_std = np.std(null_scores)

        if null_std > 1e-15:
            z_score = (observed_score - null_mean) / null_std
        else:
            z_score = 0.0

        p_value = (np.sum(np.abs(null_scores) >= np.abs(observed_score)) + 1) / (n_permutations + 1)

        return SignificanceResult(
            observed_score=observed_score,
            null_scores=null_scores,
            p_value=float(p_value),
            z_score=float(z_score),
            n_permutations=n_permutations,
        )

    @staticmethod
    def _normalize_diagrams(diagrams: list[np.ndarray]) -> list[np.ndarray]:
        """Normalize persistence diagrams to [0, 1] birth and persistence ranges.

        This is essential when comparing PH from different metric spaces
        (e.g., attention distances in [0, 1] vs Euclidean distances in [0, 200+]).
        """
        normalized = []
        for dgm in diagrams:
            if len(dgm) == 0:
                normalized.append(dgm)
                continue
            dgm = dgm.copy()
            births = dgm[:, 0]
            deaths = dgm[:, 1]
            b_max = births.max()
            d_max = deaths.max()
            scale = max(b_max, d_max)
            if scale > 1e-15:
                dgm[:, 0] = births / scale
                dgm[:, 1] = deaths / scale
            normalized.append(dgm)
        return normalized

    def compute_binding_from_diagrams(
        self,
        attn_diagrams: list[np.ndarray],
        hidden_cloud: np.ndarray,
    ) -> BindingResult:
        """Compute binding from pre-extracted attention PH diagrams and hidden cloud.

        Parameters
        ----------
        attn_diagrams : list of (n, 2) arrays, one per homology dimension.
        hidden_cloud : (n, d) hidden-state vectors.

        Returns
        -------
        BindingResult with binding score and feature stats.
        """
        # Hidden-state PH
        n_pts = hidden_cloud.shape[0]
        sub = min(self.subsample, n_pts)
        if sub < n_pts:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n_pts, size=sub, replace=False)
            hidden_cloud = hidden_cloud[np.sort(idx)]

        n_comp = min(self.n_pca_components, hidden_cloud.shape[0] - 1, hidden_cloud.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp)
            cloud_pca = pca.fit_transform(hidden_cloud)
        else:
            cloud_pca = hidden_cloud

        pa_hidden = PersistenceAnalyzer(max_dim=self.max_dim, backend="ripser")
        result_hidden = pa_hidden.fit_transform(cloud_pca)

        # Convert attn_diagrams to numpy if needed
        attn_dgms = [np.array(d) if not isinstance(d, np.ndarray) else d for d in attn_diagrams]
        while len(attn_dgms) <= self.max_dim:
            attn_dgms.append(np.empty((0, 2)))

        # Normalize both to [0, 1] to handle scale mismatch
        # (attention distances ∈ [0,1] vs Euclidean distances ∈ [0, 200+])
        attn_norm = self._normalize_diagrams(attn_dgms)
        hidden_norm = self._normalize_diagrams(pa_hidden.diagrams_)

        # Compute PIs on normalized diagrams with fixed [0,1] range
        birth_range = (0.0, 1.0)
        pers_range = (0.0, 1.0)

        imgs_attn = self._diagrams_to_images(attn_norm, birth_range, pers_range)
        imgs_hidden = self._diagrams_to_images(hidden_norm, birth_range, pers_range)

        score = self._pi_correlation(imgs_attn, imgs_hidden)

        # Feature counts / entropy
        attn_entropy = {}
        attn_features = {}
        hidden_entropy = result_hidden.get("persistence_entropy", {})
        hidden_features = {}
        for d in range(self.max_dim + 1):
            attn_features[d] = len(attn_dgms[d]) if d < len(attn_dgms) else 0
            hidden_features[d] = len(result_hidden["diagrams"][d]) if d < len(result_hidden["diagrams"]) else 0

        return BindingResult(
            binding_score=score,
            attention_entropy=attn_entropy,
            hidden_entropy=hidden_entropy,
            n_attention_features=attn_features,
            n_hidden_features=hidden_features,
        )

    def _diagrams_to_images(
        self,
        diagrams: list[np.ndarray],
        birth_range: tuple[float, float],
        pers_range: tuple[float, float],
    ) -> list[np.ndarray]:
        """Convert persistence diagrams to images using Gaussian kernel."""
        resolution = self.image_resolution
        sigma = self.image_sigma
        images = []

        for dgm in diagrams:
            img = np.zeros((resolution, resolution))
            if len(dgm) == 0:
                images.append(img)
                continue

            births = dgm[:, 0]
            deaths = dgm[:, 1]
            pers = deaths - births

            # Grid
            b_min, b_max = birth_range
            p_min, p_max = pers_range
            b_centers = np.linspace(b_min, b_max, resolution)
            p_centers = np.linspace(p_min, p_max, resolution)

            for b, p in zip(births, pers):
                if p <= 0:
                    continue
                weight = p  # weight by persistence
                b_diffs = (b_centers - b) ** 2
                p_diffs = (p_centers - p) ** 2
                kernel = weight * np.exp(-np.add.outer(p_diffs, b_diffs) / (2 * sigma ** 2))
                img += kernel

            images.append(img)

        return images

    def binding_profile(
        self,
        loader,
        attention_ph_data: dict | None = None,
        levels: list[int] | None = None,
        layer_indices: list[int] | None = None,
    ) -> dict:
        """Compute binding scores across difficulty levels and layers.

        If attention_ph_data is not available, returns binding scores based on
        hidden-state self-coupling (within-layer topology consistency). This
        serves as a template for when attention data becomes available.

        Parameters
        ----------
        loader : HiddenStateLoader
        attention_ph_data : optional pre-computed attention PH (from extract_attention_weights.py).
        levels : difficulty levels to analyze.
        layer_indices : which layers to analyze.

        Returns
        -------
        dict with:
            scores : dict mapping (level, layer) -> binding_score
            levels : list of levels
            layers : list of layer indices
        """
        if levels is None:
            levels = sorted(loader.unique_levels.tolist())
        if layer_indices is None:
            n_layers = loader.num_layers
            layer_indices = list(range(max(0, n_layers - 5), n_layers))

        scores = {}
        for level in levels:
            mask = loader.get_level_mask(level)
            indices = np.where(mask)[0]

            for layer_idx in layer_indices:
                level_scores = []
                for problem_idx in indices:
                    token_traj = loader.token_trajectories[problem_idx]
                    n_tokens = token_traj.shape[0]

                    if n_tokens < 10:
                        continue

                    if attention_ph_data is not None:
                        # Use pre-extracted attention PH
                        attn_entry = attention_ph_data.get(problem_idx, {}).get(layer_idx)
                        if attn_entry is None:
                            continue
                        # Reconstruct a synthetic attention-like distance from PH
                        # (actual use requires raw attention matrices)
                        continue

                    # Self-coupling: split token cloud into two halves and
                    # measure topological similarity (proxy when no attention data)
                    mid = n_tokens // 2
                    if mid < 5:
                        continue

                    cloud_a = token_traj[:mid]
                    cloud_b = token_traj[mid:]

                    # Create synthetic "attention-like" distance from cloud_a
                    from scipy.spatial.distance import cdist
                    dists_a = cdist(cloud_a[:min(self.subsample, len(cloud_a))],
                                    cloud_a[:min(self.subsample, len(cloud_a))])
                    if dists_a.max() > 0:
                        dists_a /= dists_a.max()

                    result = self.compute_binding(dists_a, cloud_b[:min(self.subsample, len(cloud_b))])
                    level_scores.append(result.binding_score)

                scores[(level, layer_idx)] = float(np.mean(level_scores)) if level_scores else 0.0

        return {
            "scores": scores,
            "levels": levels,
            "layers": layer_indices,
        }
