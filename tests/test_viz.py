"""Tests for att.viz — plotting and JSON export."""

import os
import tempfile
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from att.config import set_seed
from att.synthetic import lorenz_system
from att.embedding import TakensEmbedder
from att.topology import PersistenceAnalyzer
from att.viz import (
    plot_persistence_diagram,
    plot_persistence_image,
    plot_barcode,
    plot_betti_curve,
    plot_attractor_3d,
    plot_binding_comparison,
    plot_binding_image,
    export_to_json,
    load_from_json,
)


@pytest.fixture
def lorenz_results():
    set_seed(42)
    ts = lorenz_system(n_steps=5000)
    cloud = TakensEmbedder(delay=15, dimension=3).fit_transform(ts[:, 0])
    analyzer = PersistenceAnalyzer(max_dim=1)
    return analyzer.fit_transform(cloud, subsample=500, seed=42)


class TestPlots:
    def test_persistence_diagram(self, lorenz_results):
        fig = plot_persistence_diagram(lorenz_results["diagrams"])
        assert fig is not None

    def test_persistence_image(self, lorenz_results):
        fig = plot_persistence_image(lorenz_results["persistence_images"])
        assert fig is not None

    def test_barcode(self, lorenz_results):
        fig = plot_barcode(lorenz_results["diagrams"])
        assert fig is not None

    def test_betti_curve(self, lorenz_results):
        fig = plot_betti_curve(lorenz_results["betti_curves"])
        assert fig is not None

    def test_attractor_3d_matplotlib(self):
        set_seed(42)
        ts = lorenz_system(n_steps=1000)
        fig = plot_attractor_3d(ts, backend="matplotlib")
        assert fig is not None


class TestBindingPlots:
    @pytest.fixture
    def binding_detector(self):
        from att.synthetic import coupled_lorenz
        from att.binding import BindingDetector
        set_seed(42)
        ts_x, ts_y = coupled_lorenz(n_steps=5000, coupling=0.5, seed=42)
        X, Y = ts_x[1000:, 0], ts_y[1000:, 0]
        det = BindingDetector(max_dim=1)
        det.fit(X, Y, subsample=500, seed=42)
        return det

    def test_plot_binding_comparison(self, binding_detector):
        fig = plot_binding_comparison(binding_detector)
        assert fig is not None

    def test_plot_binding_image(self, binding_detector):
        fig = plot_binding_image(binding_detector.binding_image())
        assert fig is not None


class TestJsonExport:
    def test_round_trip(self, lorenz_results):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            export_to_json(lorenz_results, path)
            loaded = load_from_json(path)
            assert "diagrams" in loaded
            assert "persistence_entropy" in loaded
        finally:
            os.unlink(path)

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "results.json")
            export_to_json({"test": np.array([1, 2, 3])}, path)
            loaded = load_from_json(path)
            assert loaded["test"] == [1, 2, 3]


class TestTransitionTimeline:
    def test_plot_transition_timeline_returns_figure(self):
        """plot_transition_timeline produces a figure from a mock detector."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from att.viz.plotting import plot_transition_timeline

        # Create a minimal mock detector
        class MockDetector:
            def __init__(self):
                self._result = {
                    "window_centers": np.arange(0, 1000, 50),
                    "image_distances": np.random.default_rng(42).random(19),
                    "topology_timeseries": [
                        {"persistence_entropy": [0.5, np.random.default_rng(i).random()]}
                        for i in range(20)
                    ],
                }
            def detect_changepoints(self):
                return [5, 10]

        fig = plot_transition_timeline(MockDetector(), ground_truth=[250, 750])
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)


class TestVizEdgeCases:
    """Tests for untested plotting functions."""

    def test_plot_surrogate_distribution(self):
        """plot_surrogate_distribution should return a Figure."""
        from att.viz.plotting import plot_surrogate_distribution
        import matplotlib.figure
        surrogates = np.random.default_rng(42).standard_normal(100)
        fig = plot_surrogate_distribution(observed=5.0, surrogates=surrogates)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_benchmark_sweep(self):
        """plot_benchmark_sweep should return a Figure."""
        import pandas as pd
        from att.viz.plotting import plot_benchmark_sweep
        import matplotlib.figure
        df = pd.DataFrame({
            "coupling": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
            "method": ["binding"] * 3 + ["te"] * 3,
            "score": [1.0, 2.0, 3.0, 0.5, 1.5, 2.5],
            "score_normalized": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        })
        fig = plot_benchmark_sweep(df)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_surrogate_distribution_with_ax(self):
        """plot_surrogate_distribution should work with explicit ax."""
        from att.viz.plotting import plot_surrogate_distribution
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        surrogates = np.random.default_rng(42).standard_normal(50)
        returned_fig = plot_surrogate_distribution(observed=2.0, surrogates=surrogates, ax=ax)
        assert returned_fig is fig
        plt.close(fig)

    def test_plot_benchmark_sweep_raw_score(self):
        """plot_benchmark_sweep should work without score_normalized column."""
        import pandas as pd
        from att.viz.plotting import plot_benchmark_sweep
        import matplotlib.pyplot as plt
        df = pd.DataFrame({
            "coupling": [0.0, 0.1, 0.0, 0.1],
            "method": ["binding", "binding", "te", "te"],
            "score": [1.0, 2.0, 0.5, 1.5],
        })
        fig = plot_benchmark_sweep(df)
        assert fig is not None
        plt.close(fig)
