"""Binding detection via persistence image subtraction."""

import warnings
import numpy as np

from att.config.seed import get_rng
from att.embedding.takens import TakensEmbedder
from att.embedding.joint import JointEmbedder
from att.embedding.validation import validate_embedding, EmbeddingDegeneracyWarning
from att.topology.persistence import PersistenceAnalyzer


class BindingDetector:
    """Detect topological binding between coupled dynamical systems.

    Computes persistence images for joint and marginal embeddings, then
    measures excess topology in the joint that is absent from both marginals.

    Parameters
    ----------
    max_dim : int
        Maximum homology dimension (0=components, 1=loops).
    method : str
        "persistence_image" (primary) or "diagram_matching" (not yet implemented).
    image_resolution : int
        Resolution of persistence images.
    image_sigma : float
        Gaussian kernel bandwidth for persistence images.
    baseline : str
        "max" (conservative, pointwise max of marginals) or
        "sum" (sensitive, pointwise sum of marginals).
    embedding_quality_gate : bool
        If True, validate all three embeddings and warn if any is degenerate.
    """

    def __init__(
        self,
        max_dim: int = 1,
        method: str = "persistence_image",
        image_resolution: int = 50,
        image_sigma: float = 0.1,
        baseline: str = "max",
        embedding_quality_gate: bool = True,
    ):
        if method not in ("persistence_image",):
            raise ValueError(f"Unknown method: {method}. Use 'persistence_image'.")
        if baseline not in ("max", "sum"):
            raise ValueError(f"Unknown baseline: {baseline}. Use 'max' or 'sum'.")

        self.max_dim = max_dim
        self.method = method
        self.image_resolution = image_resolution
        self.image_sigma = image_sigma
        self.baseline = baseline
        self.embedding_quality_gate = embedding_quality_gate

        # Fitted state
        self._cloud_x: np.ndarray | None = None
        self._cloud_y: np.ndarray | None = None
        self._cloud_joint: np.ndarray | None = None
        self._result_x: dict | None = None
        self._result_y: dict | None = None
        self._result_joint: dict | None = None
        self._images_x: list[np.ndarray] | None = None
        self._images_y: list[np.ndarray] | None = None
        self._images_joint: list[np.ndarray] | None = None
        self._residual_images: list[np.ndarray] | None = None
        self._embedding_quality: dict | None = None
        self._X_raw: np.ndarray | None = None
        self._Y_raw: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        joint_embedder: JointEmbedder | None = None,
        marginal_embedder_x: TakensEmbedder | None = None,
        marginal_embedder_y: TakensEmbedder | None = None,
        subsample: int | None = None,
        seed: int | None = None,
    ) -> "BindingDetector":
        """Fit the detector on two coupled time series.

        Parameters
        ----------
        X, Y : 1D time series arrays
        joint_embedder : pre-configured JointEmbedder, or None for auto
        marginal_embedder_x, marginal_embedder_y : pre-configured TakensEmbedders
        subsample : subsample point clouds before persistence computation
        seed : seed for subsampling (same seed used for all three clouds)

        Returns
        -------
        self
        """
        X = np.asarray(X).ravel()
        Y = np.asarray(Y).ravel()
        self._X_raw = X
        self._Y_raw = Y

        # 1. Marginal embeddings
        if marginal_embedder_x is None:
            marginal_embedder_x = TakensEmbedder(delay="auto", dimension="auto")
        self._cloud_x = marginal_embedder_x.fit_transform(X)

        if marginal_embedder_y is None:
            marginal_embedder_y = TakensEmbedder(delay="auto", dimension="auto")
        self._cloud_y = marginal_embedder_y.fit_transform(Y)

        # 2. Joint embedding
        if joint_embedder is None:
            joint_embedder = JointEmbedder(delays="auto", dimensions="auto")
        self._cloud_joint = joint_embedder.fit_transform([X, Y])

        # 3. Embedding quality gate
        eq_x = validate_embedding(self._cloud_x)
        eq_y = validate_embedding(self._cloud_y)
        eq_joint = validate_embedding(self._cloud_joint)
        any_degenerate = bool(eq_x["degenerate"] or eq_y["degenerate"] or eq_joint["degenerate"])

        self._embedding_quality = {
            "marginal_x": eq_x,
            "marginal_y": eq_y,
            "joint": eq_joint,
            "any_degenerate": any_degenerate,
        }

        if self.embedding_quality_gate and any_degenerate:
            degen_parts = []
            if eq_x["degenerate"]:
                degen_parts.append(f"marginal_x (cond={eq_x['condition_number']:.1f})")
            if eq_y["degenerate"]:
                degen_parts.append(f"marginal_y (cond={eq_y['condition_number']:.1f})")
            if eq_joint["degenerate"]:
                degen_parts.append(f"joint (cond={eq_joint['condition_number']:.1f})")
            warnings.warn(
                f"Degenerate embedding(s): {', '.join(degen_parts)}. "
                "Binding scores may be unreliable.",
                EmbeddingDegeneracyWarning,
                stacklevel=2,
            )

        # 4. Persistence computation on all three clouds (same subsample seed)
        pa_seed = seed if seed is not None else 42

        pa_x = PersistenceAnalyzer(max_dim=self.max_dim)
        self._result_x = pa_x.fit_transform(self._cloud_x, subsample=subsample, seed=pa_seed)

        pa_y = PersistenceAnalyzer(max_dim=self.max_dim)
        self._result_y = pa_y.fit_transform(self._cloud_y, subsample=subsample, seed=pa_seed)

        pa_joint = PersistenceAnalyzer(max_dim=self.max_dim)
        self._result_joint = pa_joint.fit_transform(
            self._cloud_joint, subsample=subsample, seed=pa_seed
        )

        # 5. Compute PIs on shared grid
        birth_range, persistence_range = self._compute_shared_ranges(
            pa_x.diagrams_, pa_y.diagrams_, pa_joint.diagrams_
        )

        self._images_x = pa_x.to_image(
            self.image_resolution, self.image_sigma, birth_range, persistence_range
        )
        self._images_y = pa_y.to_image(
            self.image_resolution, self.image_sigma, birth_range, persistence_range
        )
        self._images_joint = pa_joint.to_image(
            self.image_resolution, self.image_sigma, birth_range, persistence_range
        )

        # Store analyzers for reuse in significance testing
        self._pa_x = pa_x
        self._pa_y = pa_y
        self._pa_joint = pa_joint
        self._birth_range = birth_range
        self._persistence_range = persistence_range

        # 6. Compute residuals
        self._residual_images = []
        for d in range(self.max_dim + 1):
            img_joint = self._images_joint[d]
            img_x = self._images_x[d]
            img_y = self._images_y[d]

            if self.baseline == "max":
                baseline_img = np.maximum(img_x, img_y)
            else:  # "sum"
                baseline_img = img_x + img_y

            self._residual_images.append(img_joint - baseline_img)

        return self

    def binding_score(self) -> float:
        """L1 norm of positive residual (joint excess over baseline).

        Returns
        -------
        float : binding score (higher = more emergent topology)
        """
        self._check_fitted()
        score = 0.0
        for residual in self._residual_images:
            positive = np.maximum(residual, 0)
            score += float(np.sum(positive))
        return score

    def binding_features(self) -> dict:
        """Per-dimension breakdown of excess topology.

        Returns
        -------
        dict : {dim: {n_excess, total_persistence, max_persistence}}
        """
        self._check_fitted()
        features = {}
        for d, residual in enumerate(self._residual_images):
            positive = np.maximum(residual, 0)
            features[d] = {
                "n_excess": int(np.sum(positive > 1e-10)),
                "total_persistence": float(np.sum(positive)),
                "max_persistence": float(np.max(positive)) if positive.max() > 0 else 0.0,
            }
        return features

    def binding_image(self) -> list[np.ndarray]:
        """Residual persistence images (joint - baseline).

        Returns
        -------
        list of (resolution, resolution) arrays, one per homology dimension
        """
        self._check_fitted()
        return self._residual_images

    def embedding_quality(self) -> dict:
        """Embedding quality metrics for all three clouds.

        Returns
        -------
        dict with keys: marginal_x, marginal_y, joint, any_degenerate
        """
        self._check_fitted()
        return self._embedding_quality

    def test_significance(
        self,
        n_surrogates: int = 100,
        method: str = "phase_randomize",
        seed: int | None = None,
        subsample: int | None = None,
    ) -> dict:
        """Test significance of binding score against surrogate null distribution.

        Generates surrogates of Y, recomputes binding score for each,
        and computes a p-value. Reuses the cached marginal X persistence
        result across all surrogates for efficiency.

        Parameters
        ----------
        n_surrogates : number of surrogate iterations
        method : "phase_randomize" or "time_shuffle"
        seed : seed for surrogate generation
        subsample : subsample for persistence (passed through)

        Returns
        -------
        dict with p_value, observed_score, surrogate_scores, significant,
        embedding_quality
        """
        self._check_fitted()

        from att.surrogates import phase_randomize, time_shuffle

        if method == "phase_randomize":
            surr_fn = phase_randomize
        elif method == "time_shuffle":
            surr_fn = time_shuffle
        else:
            raise ValueError(f"Unknown method: {method}. Use 'phase_randomize' or 'time_shuffle'.")

        surr_Y = surr_fn(self._Y_raw, n_surrogates=n_surrogates, seed=seed)

        observed = self.binding_score()
        surrogate_scores = np.empty(n_surrogates)

        # Cache: marginal X images are the same for every surrogate
        cached_images_x = self._images_x

        for i in range(n_surrogates):
            surr_seed = (seed + i + 1) if seed is not None else None
            score = self._compute_surrogate_score(
                self._X_raw, surr_Y[i], cached_images_x,
                subsample=subsample, seed=surr_seed,
            )
            surrogate_scores[i] = score

        # p-value: proportion of surrogates >= observed (with continuity correction)
        p_value = (np.sum(surrogate_scores >= observed) + 1) / (n_surrogates + 1)

        return {
            "p_value": float(p_value),
            "observed_score": observed,
            "surrogate_scores": surrogate_scores,
            "significant": p_value < 0.05,
            "embedding_quality": self._embedding_quality,
        }

    def _compute_surrogate_score(
        self,
        X_raw: np.ndarray,
        Y_surr: np.ndarray,
        cached_images_x: list[np.ndarray],
        subsample: int | None = None,
        seed: int | None = None,
    ) -> float:
        """Compute binding score for a single surrogate Y, reusing marginal X."""
        pa_seed = seed if seed is not None else 42

        # Marginal Y embedding (auto)
        emb_y = TakensEmbedder(delay="auto", dimension="auto")
        cloud_y = emb_y.fit_transform(Y_surr)

        # Joint embedding (auto)
        je = JointEmbedder(delays="auto", dimensions="auto")
        cloud_joint = je.fit_transform([X_raw, Y_surr])

        # Persistence for Y and joint
        pa_y = PersistenceAnalyzer(max_dim=self.max_dim)
        pa_y.fit_transform(cloud_y, subsample=subsample, seed=pa_seed)

        pa_joint = PersistenceAnalyzer(max_dim=self.max_dim)
        pa_joint.fit_transform(cloud_joint, subsample=subsample, seed=pa_seed)

        # Compute images on the same shared grid as the original fit
        images_y = pa_y.to_image(
            self.image_resolution, self.image_sigma,
            self._birth_range, self._persistence_range,
        )
        images_joint = pa_joint.to_image(
            self.image_resolution, self.image_sigma,
            self._birth_range, self._persistence_range,
        )

        # Compute residual and score
        score = 0.0
        for d in range(self.max_dim + 1):
            if self.baseline == "max":
                baseline_img = np.maximum(cached_images_x[d], images_y[d])
            else:
                baseline_img = cached_images_x[d] + images_y[d]
            residual = images_joint[d] - baseline_img
            score += float(np.sum(np.maximum(residual, 0)))

        return score

    def plot_comparison(self):
        """3-panel comparison: marginal X | joint (excess) | marginal Y."""
        self._check_fitted()
        from att.viz.plotting import plot_binding_comparison
        return plot_binding_comparison(self)

    def plot_binding_image(self):
        """Heatmap of residual persistence images."""
        self._check_fitted()
        from att.viz.plotting import plot_binding_image
        return plot_binding_image(self._residual_images)

    def _check_fitted(self):
        if self._residual_images is None:
            raise RuntimeError("Call .fit() first.")

    @staticmethod
    def _compute_shared_ranges(
        diagrams_x: list[np.ndarray],
        diagrams_y: list[np.ndarray],
        diagrams_joint: list[np.ndarray],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Compute shared birth and persistence ranges across all diagrams."""
        all_births = []
        all_persistences = []

        for diagrams in (diagrams_x, diagrams_y, diagrams_joint):
            for dgm in diagrams:
                if len(dgm) == 0:
                    continue
                births = dgm[:, 0]
                pers = dgm[:, 1] - dgm[:, 0]
                mask = pers > 1e-10
                if mask.any():
                    all_births.extend(births[mask])
                    all_persistences.extend(pers[mask])

        if not all_births:
            return (0.0, 1.0), (0.0, 1.0)

        birth_range = (float(min(all_births)), float(max(all_births)))
        persistence_range = (0.0, float(max(all_persistences)))
        return birth_range, persistence_range
