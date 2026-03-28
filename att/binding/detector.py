"""Binding detection via persistence image subtraction and diagram matching."""

import warnings
import numpy as np

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
        "persistence_image" (PI subtraction) or "diagram_matching"
        (optimal matching via Hungarian algorithm).
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
        if method not in ("persistence_image", "diagram_matching"):
            raise ValueError(
                f"Unknown method: {method}. "
                "Use 'persistence_image' or 'diagram_matching'."
            )
        if baseline not in ("max", "sum"):
            raise ValueError(f"Unknown baseline: {baseline}. Use 'max' or 'sum'.")

        self.max_dim = max_dim
        self.method = method
        self.image_resolution = image_resolution
        self.image_sigma = image_sigma
        self.baseline = baseline
        self.embedding_quality_gate = embedding_quality_gate

        # Fitted state
        self._fitted = False
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
        self._matching_score: float | None = None
        self._matching_details: dict | None = None
        self._embedding_quality: dict | None = None
        self._X_raw: np.ndarray | None = None
        self._Y_raw: np.ndarray | None = None
        # Cached embedding params for surrogate speed optimization
        self._marginal_delay_x: int | None = None
        self._marginal_dim_x: int | None = None
        self._marginal_delay_y: int | None = None
        self._marginal_dim_y: int | None = None
        self._joint_delays: list[int] | None = None
        self._joint_dims: list[int] | None = None
        # Ensemble state
        self._ensemble_scores: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        joint_embedder: JointEmbedder | None = None,
        marginal_embedder_x: TakensEmbedder | None = None,
        marginal_embedder_y: TakensEmbedder | None = None,
        subsample: int | None = None,
        seed: int | None = None,
        n_ensemble: int = 1,
    ) -> "BindingDetector":
        """Fit the detector on two coupled time series.

        Parameters
        ----------
        X, Y : 1D time series arrays
        joint_embedder : pre-configured JointEmbedder, or None for auto
        marginal_embedder_x, marginal_embedder_y : pre-configured TakensEmbedders
        subsample : subsample point clouds before persistence computation
        seed : seed for subsampling (same seed used for all three clouds)
        n_ensemble : int
            Number of independent fits with different subsample seeds.
            If > 1, binding_score() returns the mean, and ensemble_scores
            stores individual scores. Default 1 (no ensembling).

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

        # Cache fitted embedding params for surrogate reuse
        self._marginal_delay_x = marginal_embedder_x.delay_
        self._marginal_dim_x = marginal_embedder_x.dimension_
        self._marginal_delay_y = marginal_embedder_y.delay_
        self._marginal_dim_y = marginal_embedder_y.dimension_
        self._joint_delays = list(joint_embedder.delays_)
        self._joint_dims = list(joint_embedder.dimensions_)

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

        # Store analyzers for reuse in significance testing
        self._pa_x = pa_x
        self._pa_y = pa_y
        self._pa_joint = pa_joint

        if self.method == "diagram_matching":
            # 5b. Compute binding score from raw persistence diagrams
            self._matching_score, self._matching_details = (
                self._diagram_matching_score()
            )
        else:
            # 5a. Compute PIs on shared grid (persistence_image method)
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

            self._birth_range = birth_range
            self._persistence_range = persistence_range

            # 6a. Compute residuals
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

        self._fitted = True

        # Ensemble: re-run persistence + scoring with different subsample seeds
        if n_ensemble > 1 and subsample is not None:
            base_seed = seed if seed is not None else 42
            ensemble_scores = []
            for k in range(n_ensemble):
                ens_seed = base_seed + k
                pa_ek_x = PersistenceAnalyzer(max_dim=self.max_dim)
                pa_ek_x.fit_transform(self._cloud_x, subsample=subsample, seed=ens_seed)
                pa_ek_y = PersistenceAnalyzer(max_dim=self.max_dim)
                pa_ek_y.fit_transform(self._cloud_y, subsample=subsample, seed=ens_seed)
                pa_ek_joint = PersistenceAnalyzer(max_dim=self.max_dim)
                pa_ek_joint.fit_transform(self._cloud_joint, subsample=subsample, seed=ens_seed)

                if self.method == "persistence_image":
                    br, pr = self._compute_shared_ranges(
                        pa_ek_x.diagrams_, pa_ek_y.diagrams_, pa_ek_joint.diagrams_
                    )
                    imgs_x = pa_ek_x.to_image(self.image_resolution, self.image_sigma, br, pr)
                    imgs_y = pa_ek_y.to_image(self.image_resolution, self.image_sigma, br, pr)
                    imgs_j = pa_ek_joint.to_image(self.image_resolution, self.image_sigma, br, pr)
                    score_k = 0.0
                    for d in range(self.max_dim + 1):
                        if self.baseline == "max":
                            bl = np.maximum(imgs_x[d], imgs_y[d])
                        else:
                            bl = imgs_x[d] + imgs_y[d]
                        score_k += float(np.sum(np.maximum(imgs_j[d] - bl, 0)))
                    ensemble_scores.append(score_k)
            self._ensemble_scores = np.array(ensemble_scores)

        return self

    def binding_score(self) -> float:
        """Binding score (higher = more emergent topology).

        For persistence_image method: L1 norm of positive residual.
        For diagram_matching method: optimal matching cost between joint
        and concatenated marginal persistence diagrams.

        If fit() was called with n_ensemble > 1, returns the ensemble mean.

        Returns
        -------
        float : binding score
        """
        self._check_fitted()
        if self._ensemble_scores is not None:
            return float(np.mean(self._ensemble_scores))
        if self.method == "diagram_matching":
            return self._matching_score
        score = 0.0
        for residual in self._residual_images:
            positive = np.maximum(residual, 0)
            score += float(np.sum(positive))
        return score

    @property
    def ensemble_scores(self) -> np.ndarray | None:
        """Individual scores from ensemble fitting, or None if n_ensemble=1."""
        return self._ensemble_scores

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float] | None:
        """Bootstrap percentile confidence interval from ensemble scores.

        Parameters
        ----------
        confidence : float
            Confidence level (default 0.95 for 95% CI).

        Returns
        -------
        (lower, upper) tuple, or None if no ensemble was run.
        """
        if self._ensemble_scores is None or len(self._ensemble_scores) < 2:
            return None
        alpha = 1 - confidence
        lo = float(np.percentile(self._ensemble_scores, 100 * alpha / 2))
        hi = float(np.percentile(self._ensemble_scores, 100 * (1 - alpha / 2)))
        return (lo, hi)

    def binding_features(self) -> dict:
        """Per-dimension breakdown of excess topology.

        For persistence_image: {dim: {n_excess, total_persistence, max_persistence}}
        For diagram_matching: {dim: {score, n_joint, n_baseline, n_unmatched}}

        Returns
        -------
        dict : per-dimension feature dictionary
        """
        self._check_fitted()
        if self.method == "diagram_matching":
            return self._matching_details
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

        Only available for the persistence_image method.

        Returns
        -------
        list of (resolution, resolution) arrays, one per homology dimension

        Raises
        ------
        RuntimeError
            If called with the diagram_matching method.
        """
        self._check_fitted()
        if self.method == "diagram_matching":
            raise RuntimeError(
                "binding_image() is not available for the 'diagram_matching' method. "
                "Use binding_features() for per-dimension matching details."
            )
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
        method : "phase_randomize", "time_shuffle", or "twin_surrogate"
        seed : seed for surrogate generation
        subsample : subsample for persistence (passed through)

        Returns
        -------
        dict with p_value, observed_score, surrogate_scores, significant,
        embedding_quality
        """
        self._check_fitted()

        if self.method == "diagram_matching":
            raise NotImplementedError(
                "Significance testing is not yet supported for the "
                "'diagram_matching' method. Use method='persistence_image'."
            )

        from att.surrogates import phase_randomize, time_shuffle, twin_surrogate

        if method == "phase_randomize":
            surr_fn = phase_randomize
        elif method == "time_shuffle":
            surr_fn = time_shuffle
        elif method == "twin_surrogate":
            surr_fn = None  # handled separately below
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Use 'phase_randomize', 'time_shuffle', or 'twin_surrogate'."
            )

        if method == "twin_surrogate":
            surr_Y = twin_surrogate(self._Y_raw, n_surrogates=n_surrogates, seed=seed)
            # Twin surrogates are shorter due to embedding padding; truncate X_raw
            n_surr_len = surr_Y.shape[1]
            X_raw_trimmed = self._X_raw[:n_surr_len]
        else:
            surr_Y = surr_fn(self._Y_raw, n_surrogates=n_surrogates, seed=seed)
            X_raw_trimmed = self._X_raw

        observed = self.binding_score()
        surrogate_scores = np.empty(n_surrogates)

        # Cache: marginal X images are the same for every surrogate
        cached_images_x = self._images_x

        for i in range(n_surrogates):
            surr_seed = (seed + i + 1) if seed is not None else None
            score = self._compute_surrogate_score(
                X_raw_trimmed, surr_Y[i], cached_images_x,
                subsample=subsample, seed=surr_seed,
            )
            surrogate_scores[i] = score

        # p-value: proportion of surrogates >= observed (with continuity correction)
        p_value = (np.sum(surrogate_scores >= observed) + 1) / (n_surrogates + 1)

        # Z-score: calibrated effect size against surrogate null
        surr_mean = float(np.mean(surrogate_scores))
        surr_std = float(np.std(surrogate_scores, ddof=1)) if n_surrogates > 1 else 1.0
        z_score = (observed - surr_mean) / surr_std if surr_std > 1e-10 else 0.0
        calibrated_score = observed - surr_mean

        return {
            "p_value": float(p_value),
            "observed_score": observed,
            "surrogate_scores": surrogate_scores,
            "surrogate_mean": surr_mean,
            "surrogate_std": surr_std,
            "z_score": z_score,
            "calibrated_score": calibrated_score,
            "significant": p_value < 0.05,
            "embedding_quality": self._embedding_quality,
        }

    def _diagram_matching_score(self) -> tuple[float, dict]:
        """Compute binding score via optimal diagram matching.

        Uses the Hungarian algorithm to find the minimum-cost assignment
        between persistence diagram features of the joint embedding and
        the concatenated marginals. Unmatched joint features (those assigned
        to the diagonal) and poor matches contribute to the binding score.

        Returns
        -------
        total_score : float
            Sum of assignment costs across all homology dimensions.
        details : dict
            Per-dimension breakdown: {dim: {score, n_joint, n_baseline, n_unmatched}}.
        """
        from scipy.optimize import linear_sum_assignment

        total = 0.0
        details = {}

        for d in range(self.max_dim + 1):
            joint_dgm = self._pa_joint.diagrams_[d]
            baseline_dgm = np.concatenate([
                self._pa_x.diagrams_[d],
                self._pa_y.diagrams_[d],
            ])

            # Filter zero-persistence features
            if len(joint_dgm) > 0:
                joint_pers = joint_dgm[:, 1] - joint_dgm[:, 0]
                joint_dgm = joint_dgm[joint_pers > 1e-10]
                joint_pers = joint_dgm[:, 1] - joint_dgm[:, 0] if len(joint_dgm) > 0 else np.array([])
            else:
                joint_pers = np.array([])

            if len(baseline_dgm) > 0:
                baseline_pers = baseline_dgm[:, 1] - baseline_dgm[:, 0]
                baseline_dgm = baseline_dgm[baseline_pers > 1e-10]
                baseline_pers = baseline_dgm[:, 1] - baseline_dgm[:, 0] if len(baseline_dgm) > 0 else np.array([])
            else:
                baseline_pers = np.array([])

            n_j = len(joint_dgm)
            n_b = len(baseline_dgm)

            if n_j == 0:
                # No joint features: baseline features match to diagonal
                # but that cost doesn't reflect binding, so score is 0
                details[d] = {
                    "score": 0.0,
                    "n_joint": 0,
                    "n_baseline": n_b,
                    "n_unmatched": 0,
                }
                continue

            if n_b == 0:
                # All joint features are unmatched (sent to diagonal)
                score_d = float(np.sum(joint_pers) / 2)
                details[d] = {
                    "score": score_d,
                    "n_joint": n_j,
                    "n_baseline": 0,
                    "n_unmatched": n_j,
                }
                total += score_d
                continue

            # Build augmented cost matrix (n_j + n_b) x (n_j + n_b)
            #
            # Layout:
            #   Columns 0..n_b-1      : real baseline features
            #   Columns n_b..n_b+n_j-1: diagonal slots for joint features
            #
            #   Rows 0..n_j-1         : real joint features
            #   Rows n_j..n_j+n_b-1   : diagonal slots for baseline features
            #
            # Top-left  (n_j x n_b): L∞ distance between joint[i] and baseline[j]
            # Top-right (n_j x n_j): joint[i] matched to diagonal, cost = pers_i/2
            #                         only slot (i, n_b+i) is finite
            # Bot-left  (n_b x n_b): baseline[j] matched to diagonal, cost = pers_j/2
            #                         only slot (j, j) is finite  (row=n_j+j, col=j)
            # Bot-right (n_b x n_j): diagonal-to-diagonal padding, zero cost

            N = n_j + n_b
            cost = np.full((N, N), np.inf)

            # Top-left: real-to-real matching (vectorized)
            birth_diff = np.abs(joint_dgm[:, 0:1] - baseline_dgm[:, 0].reshape(1, -1))
            death_diff = np.abs(joint_dgm[:, 1:2] - baseline_dgm[:, 1].reshape(1, -1))
            cost[:n_j, :n_b] = np.maximum(birth_diff, death_diff)

            # Top-right: joint[i] to diagonal — only diagonal entries
            for i in range(n_j):
                cost[i, n_b + i] = joint_pers[i] / 2

            # Bottom-left: baseline[j] to diagonal — only diagonal entries
            for j in range(n_b):
                cost[n_j + j, j] = baseline_pers[j] / 2

            # Bottom-right: diagonal-to-diagonal (zero cost padding)
            cost[n_j:, n_b:] = 0.0

            row_ind, col_ind = linear_sum_assignment(cost)
            total_cost = float(cost[row_ind, col_ind].sum())

            # Count joint features matched to diagonal (col >= n_b)
            n_unmatched = sum(
                1 for i, j in zip(row_ind, col_ind) if i < n_j and j >= n_b
            )

            details[d] = {
                "score": total_cost,
                "n_joint": n_j,
                "n_baseline": n_b,
                "n_unmatched": n_unmatched,
            }
            total += total_cost

        return total, details

    def _compute_surrogate_score(
        self,
        X_raw: np.ndarray,
        Y_surr: np.ndarray,
        cached_images_x: list[np.ndarray],
        subsample: int | None = None,
        seed: int | None = None,
    ) -> float:
        """Compute binding score for a single surrogate Y, reusing marginal X.

        Reuses the embedding parameters (delay, dimension) estimated during
        fit() rather than re-estimating for each surrogate. This eliminates
        redundant AMI/FNN computation and ensures consistent embedding geometry.
        """
        pa_seed = seed if seed is not None else 42

        # Marginal Y embedding — reuse fitted params (skip AMI/FNN)
        emb_y = TakensEmbedder(
            delay=self._marginal_delay_y,
            dimension=self._marginal_dim_y,
        )
        cloud_y = emb_y.fit_transform(Y_surr)

        # Joint embedding — reuse fitted params (skip AMI/FNN)
        je = JointEmbedder(
            delays=self._joint_delays,
            dimensions=self._joint_dims,
        )
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
        if not self._fitted:
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
