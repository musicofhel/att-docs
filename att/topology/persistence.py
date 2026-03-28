"""Persistent homology computation via Ripser/GUDHI."""

import numpy as np
from att.config.seed import get_rng


class PersistenceAnalyzer:
    """Compute persistent homology on point clouds.

    Parameters
    ----------
    max_dim : int
        Max homology dimension (0=components, 1=loops, 2=voids).
    backend : str
        "ripser" (fast for H0+H1) or "gudhi" (H2+, alpha complexes).
    use_witness : bool
        Use witness complex for large point clouds (n > 2000).
    n_landmarks : int
        Number of landmarks for witness complex.
    """

    def __init__(
        self,
        max_dim: int = 2,
        backend: str = "ripser",
        use_witness: bool = False,
        n_landmarks: int = 500,
    ):
        self.max_dim = max_dim
        self.backend = backend
        self.use_witness = use_witness
        self.n_landmarks = n_landmarks
        self.diagrams_: list[np.ndarray] | None = None
        self._cloud: np.ndarray | None = None

    def fit_transform(
        self,
        cloud: np.ndarray,
        subsample: int | None = None,
        seed: int | None = None,
    ) -> dict:
        """Compute persistence diagrams and derived representations.

        Parameters
        ----------
        cloud : (n_points, dimension) point cloud
        subsample : if int, randomly select this many points first
        seed : random seed for subsampling

        Returns
        -------
        dict with diagrams, betti_curves, persistence_entropy,
        bottleneck_norms, persistence_images, persistence_landscapes
        """
        cloud = np.asarray(cloud)
        rng = get_rng(seed)

        if subsample is not None and subsample < len(cloud):
            indices = rng.choice(len(cloud), size=subsample, replace=False)
            cloud = cloud[indices]

        self._cloud = cloud

        if self.backend == "ripser":
            diagrams = self._ripser_compute(cloud)
        elif self.backend == "gudhi":
            diagrams = self._gudhi_compute(cloud)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        self.diagrams_ = diagrams

        # Compute derived representations
        betti_curves = self._compute_betti_curves(diagrams)
        persistence_entropy = self._compute_entropy(diagrams)
        bottleneck_norms = [
            float(np.max(dgm[:, 1] - dgm[:, 0])) if len(dgm) > 0 else 0.0
            for dgm in diagrams
        ]
        persistence_images = self.to_image()
        persistence_landscapes = self.to_landscape()

        return {
            "diagrams": diagrams,
            "betti_curves": betti_curves,
            "persistence_entropy": persistence_entropy,
            "bottleneck_norms": bottleneck_norms,
            "persistence_images": persistence_images,
            "persistence_landscapes": persistence_landscapes,
        }

    def _ripser_compute(self, cloud: np.ndarray) -> list[np.ndarray]:
        """Compute persistence via Ripser."""
        import ripser

        result = ripser.ripser(cloud, maxdim=self.max_dim)
        diagrams = []
        for dim in range(self.max_dim + 1):
            dgm = result["dgms"][dim]
            # Remove infinite death features for H0
            if dim == 0:
                finite_mask = np.isfinite(dgm[:, 1])
                dgm = dgm[finite_mask]
            diagrams.append(dgm)
        return diagrams

    @staticmethod
    def _maxmin_landmarks(cloud: np.ndarray, n_landmarks: int) -> np.ndarray:
        """Greedy farthest-point (maxmin) landmark sampling.

        Parameters
        ----------
        cloud : (n_points, dimension) point cloud
        n_landmarks : number of landmarks to select

        Returns
        -------
        (n_landmarks,) array of indices into cloud
        """
        n = len(cloud)
        n_landmarks = min(n_landmarks, n)
        indices = np.zeros(n_landmarks, dtype=int)
        # Start from point 0
        indices[0] = 0
        # Distance from each point to nearest landmark so far
        dist_to_nearest = np.full(n, np.inf)
        for i in range(1, n_landmarks):
            # Update distances with the most recently added landmark
            last = indices[i - 1]
            d = np.linalg.norm(cloud - cloud[last], axis=1)
            dist_to_nearest = np.minimum(dist_to_nearest, d)
            # Pick the point farthest from all current landmarks
            indices[i] = np.argmax(dist_to_nearest)
        return indices

    def _gudhi_witness_compute(self, cloud: np.ndarray) -> list[np.ndarray]:
        """Compute persistence via GUDHI Euclidean strong witness complex.

        Parameters
        ----------
        cloud : (n_points, dimension) point cloud (witnesses)

        Returns
        -------
        list of persistence diagrams, one per homology dimension
        """
        import gudhi

        landmark_indices = self._maxmin_landmarks(cloud, self.n_landmarks)
        landmarks = cloud[landmark_indices]

        witness_complex = gudhi.EuclideanStrongWitnessComplex(
            witnesses=cloud, landmarks=landmarks
        )
        stree = witness_complex.create_simplex_tree(
            max_alpha_square=float("inf"),
            limit_dimension=self.max_dim + 1,
        )
        stree.compute_persistence()

        diagrams = []
        for dim in range(self.max_dim + 1):
            pairs = stree.persistence_intervals_in_dimension(dim)
            if len(pairs) > 0:
                dgm = np.array(pairs)
                finite_mask = np.isfinite(dgm[:, 1])
                dgm = dgm[finite_mask]
            else:
                dgm = np.empty((0, 2))
            diagrams.append(dgm)
        return diagrams

    def _gudhi_compute(self, cloud: np.ndarray) -> list[np.ndarray]:
        """Compute persistence via GUDHI alpha or witness complex."""
        if self.use_witness:
            return self._gudhi_witness_compute(cloud)

        import gudhi

        alpha = gudhi.AlphaComplex(points=cloud)
        st = alpha.create_simplex_tree()
        st.compute_persistence()

        diagrams = []
        for dim in range(self.max_dim + 1):
            pairs = st.persistence_intervals_in_dimension(dim)
            if len(pairs) > 0:
                dgm = np.array(pairs)
                finite_mask = np.isfinite(dgm[:, 1])
                dgm = dgm[finite_mask]
            else:
                dgm = np.empty((0, 2))
            diagrams.append(dgm)
        return diagrams

    def _compute_betti_curves(
        self, diagrams: list[np.ndarray], n_grid: int = 100
    ) -> list[np.ndarray]:
        """Compute Betti curves (Betti number as function of filtration parameter)."""
        all_births = []
        all_deaths = []
        for dgm in diagrams:
            if len(dgm) > 0:
                all_births.extend(dgm[:, 0])
                all_deaths.extend(dgm[:, 1])

        if not all_births:
            return [np.zeros(n_grid) for _ in diagrams]

        eps_min = min(all_births)
        eps_max = max(all_deaths) if all_deaths else eps_min + 1
        grid = np.linspace(eps_min, eps_max, n_grid)

        curves = []
        for dgm in diagrams:
            betti = np.zeros(n_grid)
            for birth, death in dgm:
                betti += (grid >= birth) & (grid < death)
            curves.append(betti)

        return curves

    def _compute_entropy(self, diagrams: list[np.ndarray]) -> list[float]:
        """Compute persistence entropy per dimension."""
        entropies = []
        for dgm in diagrams:
            if len(dgm) == 0:
                entropies.append(0.0)
                continue
            lifetimes = dgm[:, 1] - dgm[:, 0]
            lifetimes = lifetimes[lifetimes > 0]
            if len(lifetimes) == 0:
                entropies.append(0.0)
                continue
            total = lifetimes.sum()
            probs = lifetimes / total
            entropy = -np.sum(probs * np.log(probs + 1e-15))
            entropies.append(float(entropy))
        return entropies

    def distance(self, other: "PersistenceAnalyzer", metric: str = "bottleneck") -> float:
        """Compute distance between persistence diagrams.

        Parameters
        ----------
        other : another PersistenceAnalyzer with computed diagrams
        metric : "bottleneck", "wasserstein_1", "wasserstein_2"

        Returns
        -------
        float : maximum distance across all dimensions
        """
        if self.diagrams_ is None or other.diagrams_ is None:
            raise RuntimeError("Both analyzers must have computed diagrams.")

        import persim

        max_dist = 0.0
        n_dims = min(len(self.diagrams_), len(other.diagrams_))

        for dim in range(n_dims):
            dgm1 = self.diagrams_[dim]
            dgm2 = other.diagrams_[dim]

            if len(dgm1) == 0 and len(dgm2) == 0:
                continue

            # Ensure non-empty diagrams for persim
            if len(dgm1) == 0:
                dgm1 = np.array([[0, 0]])
            if len(dgm2) == 0:
                dgm2 = np.array([[0, 0]])

            if metric == "bottleneck":
                d = persim.bottleneck(dgm1, dgm2)
            elif metric.startswith("wasserstein"):
                # persim.wasserstein only supports order 1
                d = persim.wasserstein(dgm1, dgm2)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            max_dist = max(max_dist, d)

        return max_dist

    def to_image(
        self,
        resolution: int = 50,
        sigma: float = 0.1,
        birth_range: tuple[float, float] | None = None,
        persistence_range: tuple[float, float] | None = None,
    ) -> list[np.ndarray]:
        """Convert diagrams to persistence images.

        Parameters
        ----------
        resolution : int
            Grid size for the persistence image (resolution x resolution).
        sigma : float
            Gaussian kernel bandwidth.
        birth_range : (min, max) or None
            Explicit birth-axis range. If None, computed from data per diagram.
        persistence_range : (min, max) or None
            Explicit persistence-axis range. If None, computed from data per diagram.

        Returns
        -------
        list of (resolution, resolution) arrays, one per homology dimension.
        """
        if self.diagrams_ is None:
            raise RuntimeError("Call fit_transform first.")

        images = []
        for dgm in self.diagrams_:
            if len(dgm) == 0:
                images.append(np.zeros((resolution, resolution)))
                continue

            # Transform to birth-persistence coordinates
            births = dgm[:, 0]
            persistence = dgm[:, 1] - dgm[:, 0]
            mask = persistence > 1e-10
            if mask.sum() == 0:
                images.append(np.zeros((resolution, resolution)))
                continue

            births = births[mask]
            persistence = persistence[mask]

            # Weight by persistence (linear ramp)
            weights = persistence / persistence.max()

            # Grid ranges — use explicit if provided, else compute from data
            if birth_range is not None:
                b_min, b_max = birth_range
            else:
                b_min, b_max = births.min(), births.max()
            if persistence_range is not None:
                p_min, p_max = persistence_range
            else:
                p_min, p_max = 0.0, persistence.max()

            b_pad = (b_max - b_min) * 0.1 + 1e-6
            p_pad = (p_max - p_min) * 0.1 + 1e-6

            b_grid = np.linspace(b_min - b_pad, b_max + b_pad, resolution)
            p_grid = np.linspace(max(0, p_min - p_pad), p_max + p_pad, resolution)

            # Compute persistence image as sum of weighted Gaussians
            img = np.zeros((resolution, resolution))
            for b, p, w in zip(births, persistence, weights):
                gb = np.exp(-((b_grid - b) ** 2) / (2 * sigma ** 2))
                gp = np.exp(-((p_grid - p) ** 2) / (2 * sigma ** 2))
                img += w * np.outer(gp, gb)

            images.append(img)

        return images

    def to_landscape(self, n_layers: int = 5, n_grid: int = 100) -> list[np.ndarray]:
        """Convert diagrams to persistence landscapes.

        Returns one (n_layers, n_grid) array per homology dimension.
        """
        if self.diagrams_ is None:
            raise RuntimeError("Call fit_transform first.")

        landscapes = []
        for dgm in self.diagrams_:
            if len(dgm) == 0:
                landscapes.append(np.zeros((n_layers, n_grid)))
                continue

            births = dgm[:, 0]
            deaths = dgm[:, 1]
            t_min = births.min()
            t_max = deaths.max()
            grid = np.linspace(t_min, t_max, n_grid)

            # Each feature defines a tent function
            tents = np.zeros((len(dgm), n_grid))
            for i, (b, d) in enumerate(dgm):
                mid = (b + d) / 2.0
                half_life = (d - b) / 2.0
                if half_life < 1e-10:
                    continue
                tents[i] = np.maximum(0, half_life - np.abs(grid - mid))

            # k-th landscape is the k-th largest tent value at each grid point
            landscape = np.zeros((n_layers, n_grid))
            for j in range(n_grid):
                vals = np.sort(tents[:, j])[::-1]
                for k in range(min(n_layers, len(vals))):
                    landscape[k, j] = vals[k]

            landscapes.append(landscape)

        return landscapes

    def plot(self, kind: str = "diagram"):
        """Plot persistence results.

        kind: 'diagram', 'barcode', 'betti_curve', 'landscape', 'image'
        """
        # Defer to viz module
        from att.viz import plotting

        if kind == "diagram":
            return plotting.plot_persistence_diagram(self.diagrams_)
        elif kind == "barcode":
            return plotting.plot_barcode(self.diagrams_)
        elif kind == "betti_curve":
            curves = self._compute_betti_curves(self.diagrams_)
            return plotting.plot_betti_curve(curves)
        elif kind == "image":
            images = self.to_image()
            return plotting.plot_persistence_image(images)
        elif kind == "landscape":
            landscapes = self.to_landscape()
            # Basic plot
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, len(landscapes), figsize=(5 * len(landscapes), 4))
            if len(landscapes) == 1:
                axes = [axes]
            for i, (ls, ax) in enumerate(zip(landscapes, axes)):
                for k in range(ls.shape[0]):
                    ax.plot(ls[k], label=f"λ_{k}")
                ax.set_title(f"H{i} Landscape")
                ax.legend()
            return fig
        else:
            raise ValueError(f"Unknown plot kind: {kind}")
