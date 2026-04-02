"""CROCKER (Contour Realization Of Computed K-dimensional hole Evolution in Rips complex).

Computes Betti number heatmaps with filtration scale on one axis and a
varying parameter (difficulty level or layer index) on the other.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from att.topology.persistence import PersistenceAnalyzer


class CROCKERMatrix:
    """Compute CROCKER matrices for LLM hidden-state topology.

    Produces 2D heatmaps of Betti numbers β_k(ε, p) where ε is the
    filtration radius and p is a varying parameter (difficulty level
    or transformer layer index).

    Parameters
    ----------
    n_filtration_steps : int
        Grid resolution along the filtration axis.
    max_dim : int
        Maximum homology dimension.
    n_pca_components : int
        PCA dimensions before PH.
    subsample : int or None
        Max points per cloud.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_filtration_steps: int = 100,
        max_dim: int = 1,
        n_pca_components: int = 50,
        subsample: int | None = 200,
        seed: int = 42,
    ):
        self.n_filtration_steps = n_filtration_steps
        self.max_dim = max_dim
        self.n_pca_components = n_pca_components
        self.subsample = subsample
        self.seed = seed

        # Filled by fit()
        self._matrices: dict[int, np.ndarray] | None = None
        self._parameter_labels: list[str] | None = None
        self._filtration_range: tuple[float, float] | None = None
        self._mode: str | None = None

    def fit_by_difficulty(
        self,
        loader,
        layer: int = -1,
        levels: list[int] | None = None,
    ) -> "CROCKERMatrix":
        """Compute CROCKER matrices with difficulty level as parameter axis.

        Parameters
        ----------
        loader : HiddenStateLoader
        layer : int
            Layer index to analyze (-1 = final layer).
        levels : list of int or None
            Levels to include (None = all).
        """
        if levels is None:
            levels = sorted(loader.unique_levels.tolist())

        diagrams_list = []
        labels = []

        for level in levels:
            cloud = loader.get_level_cloud(level, layer=layer)
            diagrams = self._compute_diagrams(cloud)
            diagrams_list.append(diagrams)
            labels.append(f"L{level}")

        self._build_matrices(diagrams_list)
        self._parameter_labels = labels
        self._mode = "difficulty"
        return self

    def fit_by_layer(
        self,
        loader,
        level: int = 1,
        layers: list[int] | None = None,
    ) -> "CROCKERMatrix":
        """Compute CROCKER matrices with layer index as parameter axis.

        Parameters
        ----------
        loader : HiddenStateLoader
        level : int
            Difficulty level to analyze.
        layers : list of int or None
            Layer indices to include (None = all).
        """
        if layers is None:
            layers = list(range(loader.num_layers))

        diagrams_list = []
        labels = []

        for layer_idx in layers:
            cloud = loader.get_level_cloud(level, layer=layer_idx)
            diagrams = self._compute_diagrams(cloud)
            diagrams_list.append(diagrams)
            labels.append(f"Ly{layer_idx}")

        self._build_matrices(diagrams_list)
        self._parameter_labels = labels
        self._mode = "layer"
        return self

    def _compute_diagrams(self, cloud: np.ndarray) -> list[np.ndarray]:
        """PCA + PH on a single point cloud."""
        n_pts = cloud.shape[0]
        if n_pts < 3:
            return [np.empty((0, 2)) for _ in range(self.max_dim + 1)]

        n_comp = min(self.n_pca_components, n_pts - 1, cloud.shape[1])
        pca = PCA(n_components=n_comp)
        cloud_pca = pca.fit_transform(cloud)

        pa = PersistenceAnalyzer(max_dim=self.max_dim, backend="ripser")
        sub = min(n_pts, self.subsample) if self.subsample else None
        result = pa.fit_transform(cloud_pca, subsample=sub, seed=self.seed)
        return result["diagrams"]

    def _build_matrices(self, diagrams_list: list[list[np.ndarray]]) -> None:
        """Build Betti matrices from a list of persistence diagrams."""
        # Find global filtration range across all diagrams
        all_births = []
        all_deaths = []
        for diagrams in diagrams_list:
            for dgm in diagrams:
                if len(dgm) > 0:
                    all_births.extend(dgm[:, 0].tolist())
                    all_deaths.extend(dgm[:, 1].tolist())

        if not all_births:
            n_params = len(diagrams_list)
            self._matrices = {
                d: np.zeros((self.n_filtration_steps, n_params))
                for d in range(self.max_dim + 1)
            }
            self._filtration_range = (0.0, 1.0)
            return

        eps_min = min(all_births)
        eps_max = max(all_deaths)
        self._filtration_range = (eps_min, eps_max)
        grid = np.linspace(eps_min, eps_max, self.n_filtration_steps)

        n_params = len(diagrams_list)
        matrices = {
            d: np.zeros((self.n_filtration_steps, n_params))
            for d in range(self.max_dim + 1)
        }

        for p_idx, diagrams in enumerate(diagrams_list):
            for dim in range(self.max_dim + 1):
                if dim < len(diagrams):
                    dgm = diagrams[dim]
                    for birth, death in dgm:
                        matrices[dim][:, p_idx] += (grid >= birth) & (
                            grid < death
                        )

        self._matrices = matrices

    @property
    def betti_matrices(self) -> dict[int, np.ndarray]:
        """Betti matrices keyed by homology dimension.

        Each matrix has shape (n_filtration_steps, n_parameters).
        """
        if self._matrices is None:
            raise RuntimeError("Call fit_by_difficulty() or fit_by_layer() first.")
        return self._matrices

    @property
    def parameter_labels(self) -> list[str]:
        """Labels for the parameter axis."""
        if self._parameter_labels is None:
            raise RuntimeError("Call fit_by_difficulty() or fit_by_layer() first.")
        return self._parameter_labels

    @property
    def filtration_range(self) -> tuple[float, float]:
        """(min, max) of the filtration grid."""
        if self._filtration_range is None:
            raise RuntimeError("Call fit_by_difficulty() or fit_by_layer() first.")
        return self._filtration_range

    def pairwise_l1_distances(self, dim: int = 1) -> np.ndarray:
        """L1 distances between CROCKER slices (columns) at a given homology dim.

        Returns
        -------
        (n_params, n_params) symmetric distance matrix.
        """
        mat = self.betti_matrices[dim]
        n = mat.shape[1]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sum(np.abs(mat[:, i] - mat[:, j]))
                dists[i, j] = d
                dists[j, i] = d
        return dists
