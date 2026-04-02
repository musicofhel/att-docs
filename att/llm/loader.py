"""Standardized loader for LLM hidden-state archives."""

from __future__ import annotations

import numpy as np


class HiddenStateLoader:
    """Load and query LLM hidden-state archives produced by extract_hidden_states.py.

    Parameters
    ----------
    path : str
        Path to .npz archive containing hidden states.
    """

    def __init__(self, path: str):
        self._path = path
        data = np.load(path, allow_pickle=True)

        self._last_hidden: np.ndarray = data["last_hidden_states"]  # (N, d)
        self._levels: np.ndarray = data["difficulty_levels"]  # (N,)
        self._layer_hidden: np.ndarray = data["layer_hidden_states"]  # (N, L+1, d)
        self._token_trajectories: np.ndarray = data["token_trajectories"]  # (N,) object
        self._seq_lengths: np.ndarray = data["seq_lengths"]  # (N,)
        self._model_name: str = str(data["model_name"])
        self._hidden_dim: int = int(data["hidden_dim"])
        self._num_layers: int = int(data["num_layers"])  # L+1 (includes embedding)

    @property
    def last_hidden(self) -> np.ndarray:
        """(N, d) last-token hidden states at the final transformer layer."""
        return self._last_hidden

    @property
    def levels(self) -> np.ndarray:
        """(N,) difficulty levels (1-5)."""
        return self._levels

    @property
    def layer_hidden(self) -> np.ndarray:
        """(N, L+1, d) hidden states at the final token across all layers.

        Index 0 is the embedding layer output; index -1 is the final transformer layer.
        """
        return self._layer_hidden

    @property
    def token_trajectories(self) -> np.ndarray:
        """(N,) object array where each element is (T_i, d) token-position hidden states."""
        return self._token_trajectories

    @property
    def seq_lengths(self) -> np.ndarray:
        """(N,) sequence lengths per problem."""
        return self._seq_lengths

    @property
    def model_name(self) -> str:
        """HuggingFace model ID used for extraction."""
        return self._model_name

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of hidden-state vectors."""
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        """Number of layers including embedding layer (L+1)."""
        return self._num_layers

    @property
    def n_problems(self) -> int:
        """Total number of problems."""
        return len(self._levels)

    @property
    def unique_levels(self) -> np.ndarray:
        """Sorted array of unique difficulty levels."""
        return np.unique(self._levels)

    def get_level_mask(self, level: int) -> np.ndarray:
        """Boolean mask for problems at a given difficulty level."""
        return self._levels == level

    def get_level_cloud(
        self,
        level: int,
        layer: int | None = None,
    ) -> np.ndarray:
        """Point cloud of hidden states for a difficulty level.

        Parameters
        ----------
        level : int
            Difficulty level (1-5).
        layer : int or None
            If None, uses last_hidden (final layer, last token).
            If int, extracts from layer_hidden at that layer index.

        Returns
        -------
        (n_level, d) array of hidden-state vectors.
        """
        mask = self.get_level_mask(level)
        if layer is None:
            return self._last_hidden[mask]
        return self._layer_hidden[mask, layer, :]

    def get_layer_cloud(
        self,
        layer: int,
        levels: list[int] | None = None,
    ) -> np.ndarray:
        """Point cloud of hidden states at a specific layer across problems.

        Parameters
        ----------
        layer : int
            Layer index (0=embedding, -1=final transformer layer).
        levels : list of int or None
            If None, includes all problems. Otherwise filters to specified levels.

        Returns
        -------
        (n_problems, d) array of hidden-state vectors.
        """
        if levels is None:
            return self._layer_hidden[:, layer, :]
        mask = np.isin(self._levels, levels)
        return self._layer_hidden[mask, layer, :]

    def level_counts(self) -> dict[int, int]:
        """Number of problems per difficulty level."""
        return {int(lv): int((self._levels == lv).sum()) for lv in self.unique_levels}

    def __repr__(self) -> str:
        counts = self.level_counts()
        levels_str = ", ".join(f"L{k}={v}" for k, v in sorted(counts.items()))
        return (
            f"HiddenStateLoader(model={self._model_name!r}, "
            f"n={self.n_problems}, d={self._hidden_dim}, "
            f"layers={self._num_layers}, {levels_str})"
        )
