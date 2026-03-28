"""Tests for sliding-window transition detection."""

import numpy as np
import pytest
from att.transitions import TransitionDetector


class TestTransitionDetector:
    def test_output_structure(self):
        """fit_transform returns dict with expected keys."""
        from att.synthetic import coupled_lorenz
        ts_x, _ = coupled_lorenz(n_steps=3000, coupling=0.1, seed=42)
        cloud = ts_x[500:, :3]  # 3D Lorenz, 2500 points

        det = TransitionDetector(window_size=300, step_size=100, max_dim=1)
        result = det.fit_transform(cloud, seed=42)

        expected_keys = {"topology_timeseries", "distances", "image_distances",
                         "window_centers", "transition_scores"}
        assert expected_keys.issubset(result.keys())
        assert len(result["topology_timeseries"]) == len(result["window_centers"])
        assert len(result["distances"]) == len(result["window_centers"]) - 1
        assert len(result["image_distances"]) == len(result["window_centers"]) - 1

    def test_distances_positive(self):
        """All distances should be non-negative."""
        from att.synthetic import coupled_lorenz
        ts_x, _ = coupled_lorenz(n_steps=3000, coupling=0.1, seed=42)
        cloud = ts_x[500:, :3]

        det = TransitionDetector(window_size=300, step_size=100, max_dim=1)
        result = det.fit_transform(cloud, seed=42)

        assert np.all(result["distances"] >= 0)
        assert np.all(result["image_distances"] >= 0)

    @pytest.mark.slow
    def test_changepoints_near_ground_truth(self):
        """TransitionDetector finds at least 1 of 3 regime switches in switching Rossler."""
        from att.synthetic import switching_rossler
        from att.embedding.takens import TakensEmbedder

        ts = switching_rossler(n_steps=20000, switch_every=5000, seed=42)
        x = ts[:, 0]

        # Embed the full series
        embedder = TakensEmbedder(delay=10, dimension=4)
        embedder.fit(x)
        cloud = embedder.transform(x)

        det = TransitionDetector(window_size=500, step_size=100, max_dim=1)
        result = det.fit_transform(cloud, seed=42)
        changepoints = det.detect_changepoints(method="cusum")

        # Ground truth transitions at samples 5000, 10000, 15000
        # (adjusted for embedding offset: (dim-1)*delay = 30 samples lost)
        true_transitions = [5000, 10000, 15000]
        tolerance = 500

        # Convert changepoint indices to sample positions
        window_centers = result["window_centers"]
        dist_x = (window_centers[:-1] + window_centers[1:]) / 2

        detected_samples = [dist_x[cp] for cp in changepoints if cp < len(dist_x)]

        # At least 1 of 3 should be within tolerance
        hits = 0
        for true_t in true_transitions:
            for det_t in detected_samples:
                if abs(det_t - true_t) < tolerance:
                    hits += 1
                    break

        assert hits >= 1, (
            f"Expected at least 1 hit within {tolerance} samples of true transitions "
            f"{true_transitions}, got {hits}. Detected at: {detected_samples}"
        )

    def test_per_window_embedding(self):
        """1D input path with per-window embedding produces valid results."""
        from att.synthetic import switching_rossler
        ts = switching_rossler(n_steps=5000, seed=42)
        x = ts[:, 0]

        det = TransitionDetector(window_size=1000, step_size=500, max_dim=1)
        result = det.fit_transform(x, embedding_dim=4, embedding_delay=10, seed=42)

        assert len(result["window_centers"]) >= 2
        assert len(result["image_distances"]) >= 1

    def test_not_fitted_raises(self):
        """detect_changepoints before fit_transform raises RuntimeError."""
        det = TransitionDetector()
        with pytest.raises(RuntimeError, match="fit_transform"):
            det.detect_changepoints()
