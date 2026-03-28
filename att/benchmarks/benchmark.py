"""Coupling benchmark framework with sweep and normalization."""

import numpy as np
import pandas as pd

from att.config.seed import get_rng
from att.benchmarks.methods import transfer_entropy, pac, crqa


class CouplingBenchmark:
    """Benchmark multiple coupling measures on the same system pairs.

    Parameters
    ----------
    methods : list of method names, or None for all built-in methods
        Built-in: "binding_score", "transfer_entropy", "pac", "crqa"
    normalization : str
        "rank" (default), "minmax", "zscore", or "none"
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        normalization: str = "rank",
    ):
        if normalization not in ("rank", "minmax", "zscore", "none"):
            raise ValueError(f"Unknown normalization: {normalization}")

        self.normalization = normalization
        self._methods: dict[str, callable] = {}

        if methods is None:
            methods = ["binding_score", "transfer_entropy", "pac", "crqa"]

        for name in methods:
            if name == "binding_score":
                self._methods["binding_score"] = self._compute_binding_score
            elif name == "transfer_entropy":
                self._methods["transfer_entropy"] = lambda X, Y: transfer_entropy(X, Y)
            elif name == "pac":
                self._methods["pac"] = lambda X, Y: pac(X, Y)
            elif name == "crqa":
                self._methods["crqa"] = lambda X, Y: crqa(X, Y)
            else:
                raise ValueError(f"Unknown method: {name}")

    def register_method(self, name: str, fn: callable) -> None:
        """Register a custom coupling method.

        Parameters
        ----------
        name : method name (appears in output)
        fn : callable(X, Y) -> float
        """
        self._methods[name] = fn

    def run(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> dict:
        """Compute all registered methods on a single pair.

        Returns
        -------
        dict : {method_name: score}
        """
        X = np.asarray(X).ravel()
        Y = np.asarray(Y).ravel()
        results = {}
        for name, fn in self._methods.items():
            try:
                results[name] = float(fn(X, Y))
            except Exception as e:
                results[name] = float("nan")
        return results

    def sweep(
        self,
        generator_fn: callable,
        coupling_values: list[float] | np.ndarray,
        seed: int | None = None,
        transient_discard: int = 1000,
    ) -> pd.DataFrame:
        """Run all methods across a range of coupling values.

        Parameters
        ----------
        generator_fn : callable(coupling, seed) -> (X, Y) tuple of arrays
        coupling_values : coupling strengths to sweep
        seed : random seed (same for all coupling values)
        transient_discard : samples to discard from start of each time series

        Returns
        -------
        DataFrame with columns: coupling, method, score, score_normalized
        """
        rows = []

        for c in coupling_values:
            ts_x, ts_y = generator_fn(c, seed)
            # Use first column if multi-dimensional
            X = np.asarray(ts_x).ravel() if ts_x.ndim == 1 else ts_x[:, 0]
            Y = np.asarray(ts_y).ravel() if ts_y.ndim == 1 else ts_y[:, 0]

            # Discard transient
            X = X[transient_discard:]
            Y = Y[transient_discard:]

            scores = self.run(X, Y)
            for method, score in scores.items():
                rows.append({"coupling": float(c), "method": method, "score": score})

        df = pd.DataFrame(rows)

        # Apply normalization per method
        df["score_normalized"] = df["score"]
        if self.normalization != "none" and len(df) > 0:
            for method in df["method"].unique():
                mask = df["method"] == method
                vals = df.loc[mask, "score"].values

                if self.normalization == "rank":
                    ranked = pd.Series(vals).rank().values
                    n = len(ranked)
                    df.loc[mask, "score_normalized"] = (ranked - 1) / max(n - 1, 1)
                elif self.normalization == "minmax":
                    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                    rng = vmax - vmin
                    if rng > 1e-15:
                        df.loc[mask, "score_normalized"] = (vals - vmin) / rng
                    else:
                        df.loc[mask, "score_normalized"] = 0.0
                elif self.normalization == "zscore":
                    mean, std = np.nanmean(vals), np.nanstd(vals)
                    if std > 1e-15:
                        df.loc[mask, "score_normalized"] = (vals - mean) / std
                    else:
                        df.loc[mask, "score_normalized"] = 0.0

        return df

    @staticmethod
    def _compute_binding_score(X: np.ndarray, Y: np.ndarray) -> float:
        """Compute binding score using BindingDetector with default params."""
        from att.binding import BindingDetector

        det = BindingDetector(max_dim=1, baseline="max", embedding_quality_gate=False)
        det.fit(X, Y, subsample=500, seed=42)
        return det.binding_score()
