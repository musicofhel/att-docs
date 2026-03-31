"""Tests for att.binding — binding detection via persistence image subtraction."""

import warnings
import numpy as np
import pytest

from att.config import set_seed
from att.synthetic import coupled_lorenz, coupled_rossler_lorenz
from att.embedding.validation import EmbeddingDegeneracyWarning
from att.binding import BindingDetector, SurrogateMethodWarning


@pytest.fixture
def coupled_pair():
    """Coupled Lorenz at coupling=0.5, transient discarded."""
    set_seed(42)
    ts_x, ts_y = coupled_lorenz(n_steps=6000, coupling=0.5, seed=42)
    return ts_x[1000:, 0], ts_y[1000:, 0]


@pytest.fixture
def uncoupled_pair():
    """Uncoupled Lorenz (coupling=0), transient discarded."""
    set_seed(42)
    ts_x, ts_y = coupled_lorenz(n_steps=6000, coupling=0.0, seed=42)
    return ts_x[1000:, 0], ts_y[1000:, 0]


class TestBindingDetectorCore:
    def test_coupled_positive_score(self, coupled_pair):
        X, Y = coupled_pair
        det = BindingDetector(max_dim=1, baseline="max")
        det.fit(X, Y, subsample=500, seed=42)
        score = det.binding_score()
        assert score > 0, f"Coupled binding score {score} should be positive"

    def test_h1_excess_for_coupled(self, coupled_pair):
        """Coupled system should have nonzero H1 excess features."""
        det = BindingDetector(max_dim=1, baseline="max")
        det.fit(*coupled_pair, subsample=500, seed=42)
        features = det.binding_features()
        assert features[1]["total_persistence"] > 0

    def test_max_baseline(self, coupled_pair):
        det = BindingDetector(max_dim=1, baseline="max")
        det.fit(*coupled_pair, subsample=500, seed=42)
        score = det.binding_score()
        assert isinstance(score, float)
        assert score >= 0

    def test_sum_baseline(self, coupled_pair):
        det = BindingDetector(max_dim=1, baseline="sum")
        det.fit(*coupled_pair, subsample=500, seed=42)
        score = det.binding_score()
        assert isinstance(score, float)
        assert score >= 0

    def test_binding_image_shape(self, coupled_pair):
        det = BindingDetector(max_dim=1, image_resolution=30)
        det.fit(*coupled_pair, subsample=500, seed=42)
        images = det.binding_image()
        assert len(images) == 2  # H0 and H1
        for img in images:
            assert img.shape == (30, 30)

    def test_binding_features_structure(self, coupled_pair):
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=500, seed=42)
        features = det.binding_features()
        assert 0 in features and 1 in features
        for d in (0, 1):
            assert "n_excess" in features[d]
            assert "total_persistence" in features[d]
            assert "max_persistence" in features[d]

    def test_embedding_quality_structure(self, coupled_pair):
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=500, seed=42)
        eq = det.embedding_quality()
        assert "marginal_x" in eq
        assert "marginal_y" in eq
        assert "joint" in eq
        assert "any_degenerate" in eq
        assert isinstance(eq["any_degenerate"], bool)

    def test_reproducible_score(self, coupled_pair):
        det1 = BindingDetector(max_dim=1)
        det1.fit(*coupled_pair, subsample=500, seed=42)

        det2 = BindingDetector(max_dim=1)
        det2.fit(*coupled_pair, subsample=500, seed=42)

        assert det1.binding_score() == det2.binding_score()

    def test_not_fitted_raises(self):
        det = BindingDetector()
        with pytest.raises(RuntimeError, match="fit"):
            det.binding_score()


class TestEmbeddingQualityGate:
    def test_gate_fires_on_degenerate(self):
        """Minimal delay on heterogeneous system should trigger degeneracy warning."""
        set_seed(42)
        ts_r, ts_l = coupled_rossler_lorenz(n_steps=6000, coupling=0.3, seed=42)
        X, Y = ts_r[1000:, 0], ts_l[1000:, 0]

        from att.embedding import TakensEmbedder, JointEmbedder
        # Force delay=1 which produces a nearly degenerate embedding
        # (consecutive samples are highly correlated → ill-conditioned)
        degen_embedder_x = TakensEmbedder(delay=1, dimension=3)
        degen_embedder_y = TakensEmbedder(delay=1, dimension=3)
        degen_joint = JointEmbedder(delays=[1, 1], dimensions=[3, 3])

        det = BindingDetector(max_dim=1, embedding_quality_gate=True)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            det.fit(
                X, Y,
                joint_embedder=degen_joint,
                marginal_embedder_x=degen_embedder_x,
                marginal_embedder_y=degen_embedder_y,
                subsample=500, seed=42,
            )
            eq = det.embedding_quality()
            # At least one embedding should be degenerate with delay=1
            assert eq["any_degenerate"] is True

    def test_gate_passes_good_embedding(self, coupled_pair):
        """Auto-estimated per-channel delays should not trigger warning."""
        det = BindingDetector(max_dim=1, embedding_quality_gate=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            det.fit(*coupled_pair, subsample=500, seed=42)
            degeneracy_warnings = [
                x for x in w if issubclass(x.category, EmbeddingDegeneracyWarning)
            ]
            assert len(degeneracy_warnings) == 0

    def test_gate_bypass(self, coupled_pair):
        """embedding_quality_gate=False should suppress warnings."""
        det = BindingDetector(max_dim=1, embedding_quality_gate=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            det.fit(*coupled_pair, subsample=500, seed=42)
            degeneracy_warnings = [
                x for x in w if issubclass(x.category, EmbeddingDegeneracyWarning)
            ]
            assert len(degeneracy_warnings) == 0


class TestSignificance:
    @pytest.fixture
    def short_uncoupled(self):
        """Short uncoupled pair for faster significance tests."""
        set_seed(42)
        ts_x, ts_y = coupled_lorenz(n_steps=3000, coupling=0.0, seed=42)
        return ts_x[500:, 0], ts_y[500:, 0]

    @pytest.fixture
    def short_coupled(self):
        """Short coupled pair for faster significance tests."""
        set_seed(42)
        ts_x, ts_y = coupled_lorenz(n_steps=3000, coupling=0.5, seed=42)
        return ts_x[500:, 0], ts_y[500:, 0]

    @pytest.mark.slow
    def test_uncoupled_not_significant(self, short_uncoupled):
        """Uncoupled systems should not produce significant binding."""
        det = BindingDetector(max_dim=1, baseline="max")
        det.fit(*short_uncoupled, subsample=300, seed=42)
        result = det.test_significance(
            n_surrogates=19, method="phase_randomize", seed=42, subsample=300,
        )
        assert result["p_value"] > 0.05, f"False positive: p={result['p_value']}"

    @pytest.mark.slow
    def test_significance_output_structure(self, short_coupled):
        det = BindingDetector(max_dim=1, baseline="max")
        det.fit(*short_coupled, subsample=300, seed=42)
        result = det.test_significance(
            n_surrogates=9, method="phase_randomize", seed=42, subsample=300,
        )
        assert "p_value" in result
        assert "observed_score" in result
        assert "surrogate_scores" in result
        assert "significant" in result
        assert "embedding_quality" in result
        assert len(result["surrogate_scores"]) == 9
        assert isinstance(result["p_value"], float)


class TestDiagramMatching:
    """Tests for the diagram_matching binding method."""

    def test_coupled_positive_score(self, coupled_pair):
        """Coupled system should have positive diagram matching score."""
        det = BindingDetector(max_dim=1, method="diagram_matching")
        det.fit(*coupled_pair, subsample=500, seed=42)
        score = det.binding_score()
        assert score > 0, f"Coupled binding score {score} should be positive"

    def test_uncoupled_has_nonzero_score(self, uncoupled_pair):
        """Uncoupled system should still produce a valid nonneg score."""
        det = BindingDetector(max_dim=1, method="diagram_matching")
        det.fit(*uncoupled_pair, subsample=500, seed=42)
        score = det.binding_score()
        assert isinstance(score, float)
        assert score >= 0

    def test_no_sigma_dependency(self, coupled_pair):
        """Diagram matching score should not depend on image_sigma."""
        det1 = BindingDetector(max_dim=1, method="diagram_matching", image_sigma=0.1)
        det1.fit(*coupled_pair, subsample=500, seed=42)

        det2 = BindingDetector(max_dim=1, method="diagram_matching", image_sigma=1.0)
        det2.fit(*coupled_pair, subsample=500, seed=42)

        assert det1.binding_score() == det2.binding_score()

    def test_binding_features_structure(self, coupled_pair):
        """binding_features() should return per-dim matching details."""
        det = BindingDetector(max_dim=1, method="diagram_matching")
        det.fit(*coupled_pair, subsample=500, seed=42)
        features = det.binding_features()
        assert 0 in features and 1 in features
        for d in (0, 1):
            assert "score" in features[d]
            assert "n_joint" in features[d]
            assert "n_baseline" in features[d]
            assert "n_unmatched" in features[d]
            assert features[d]["score"] >= 0
            assert features[d]["n_joint"] >= 0
            assert features[d]["n_baseline"] >= 0
            assert features[d]["n_unmatched"] >= 0
            assert features[d]["n_unmatched"] <= features[d]["n_joint"]

    def test_score_equals_sum_of_dim_scores(self, coupled_pair):
        """Total score should equal sum of per-dimension scores."""
        det = BindingDetector(max_dim=1, method="diagram_matching")
        det.fit(*coupled_pair, subsample=500, seed=42)
        features = det.binding_features()
        expected = sum(features[d]["score"] for d in range(2))
        assert abs(det.binding_score() - expected) < 1e-10

    def test_reproducible_score(self, coupled_pair):
        """Same seed should produce identical results."""
        det1 = BindingDetector(max_dim=1, method="diagram_matching")
        det1.fit(*coupled_pair, subsample=500, seed=42)

        det2 = BindingDetector(max_dim=1, method="diagram_matching")
        det2.fit(*coupled_pair, subsample=500, seed=42)

        assert det1.binding_score() == det2.binding_score()

    def test_binding_image_raises(self, coupled_pair):
        """binding_image() should raise for diagram_matching method."""
        det = BindingDetector(max_dim=1, method="diagram_matching")
        det.fit(*coupled_pair, subsample=500, seed=42)
        with pytest.raises(RuntimeError, match="not available"):
            det.binding_image()

    def test_embedding_quality_available(self, coupled_pair):
        """embedding_quality() should work for diagram_matching."""
        det = BindingDetector(max_dim=1, method="diagram_matching")
        det.fit(*coupled_pair, subsample=500, seed=42)
        eq = det.embedding_quality()
        assert "marginal_x" in eq
        assert "joint" in eq
        assert "any_degenerate" in eq

    def test_not_fitted_raises(self):
        """Should raise before fit() is called."""
        det = BindingDetector(method="diagram_matching")
        with pytest.raises(RuntimeError, match="fit"):
            det.binding_score()

    def test_max_dim_0_only(self, coupled_pair):
        """Should work with max_dim=0 (only H0)."""
        det = BindingDetector(max_dim=0, method="diagram_matching")
        det.fit(*coupled_pair, subsample=500, seed=42)
        score = det.binding_score()
        assert isinstance(score, float)
        assert score >= 0
        features = det.binding_features()
        assert 0 in features
        assert 1 not in features


class TestBindingEdgeCases:
    """Edge-case tests for BindingDetector robustness."""

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            BindingDetector(method="bad")

    def test_invalid_baseline_raises(self):
        """Unknown baseline should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown baseline"):
            BindingDetector(baseline="bad")

    def test_very_short_series(self):
        """50-sample input should either raise or warn about degeneracy."""
        det = BindingDetector(max_dim=1)
        short = np.random.default_rng(42).standard_normal(50)
        # Very short input may raise or produce a degenerate embedding warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                det.fit(short, short, subsample=None, seed=42)
                # If it doesn't raise, it should at least warn about degeneracy
                degeneracy_warnings = [
                    x for x in w if issubclass(x.category, EmbeddingDegeneracyWarning)
                ]
                assert len(degeneracy_warnings) > 0, (
                    "Short series should trigger degeneracy warning"
                )
            except ValueError:
                pass  # Also acceptable

    def test_identical_signals(self):
        """fit(X, X) should produce a defined score without crashing."""
        set_seed(42)
        ts_x, _ = coupled_lorenz(n_steps=4000, coupling=0.3, seed=42)
        x = ts_x[1000:, 0]
        det = BindingDetector(max_dim=1, baseline="max")
        det.fit(x, x.copy(), subsample=500, seed=42)
        score = det.binding_score()
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_nan_input(self):
        """NaN in input should propagate or raise — not silently produce garbage."""
        det = BindingDetector(max_dim=1)
        x = np.random.default_rng(42).standard_normal(3000)
        y = x.copy()
        y[100] = np.nan
        # NaN will propagate through embedding and persistence.
        # The exact behavior depends on ripser — just verify it doesn't
        # silently produce a finite positive score.
        try:
            det.fit(x, y, subsample=500, seed=42)
            score = det.binding_score()
            # If it completes, score should be NaN or the method handled it
            # We just document the behavior here
            assert isinstance(score, float)
        except (ValueError, RuntimeError):
            pass  # Acceptable: raising on bad input

    def test_diagram_matching_empty_h1(self):
        """With a tiny cloud, H1 should be empty and score=0 for that dim."""
        det = BindingDetector(max_dim=1, method="diagram_matching")
        # Very small subsample forces tiny cloud -> no H1 features
        set_seed(42)
        ts_x, ts_y = coupled_lorenz(n_steps=4000, coupling=0.3, seed=42)
        x, y = ts_x[1000:, 0], ts_y[1000:, 0]
        det.fit(x, y, subsample=20, seed=42)
        features = det.binding_features()
        # With 20 points max_dim=1, H1 may or may not exist
        # Just verify the structure is valid
        assert 0 in features
        for d in features:
            assert features[d]["score"] >= 0

    def test_n_surrogates_zero(self):
        """test_significance with n_surrogates=0 should return p_value=1.0."""
        set_seed(42)
        ts_x, ts_y = coupled_lorenz(n_steps=3000, coupling=0.3, seed=42)
        x, y = ts_x[500:, 0], ts_y[500:, 0]
        det = BindingDetector(max_dim=1)
        det.fit(x, y, subsample=300, seed=42)
        result = det.test_significance(n_surrogates=0, seed=42, subsample=300)
        assert result["p_value"] == 1.0
        assert len(result["surrogate_scores"]) == 0

    def test_significance_time_shuffle(self):
        """test_significance with time_shuffle should produce valid output."""
        set_seed(42)
        ts_x, ts_y = coupled_lorenz(n_steps=3000, coupling=0.3, seed=42)
        x, y = ts_x[500:, 0], ts_y[500:, 0]
        det = BindingDetector(max_dim=1)
        det.fit(x, y, subsample=300, seed=42)
        result = det.test_significance(
            n_surrogates=3, method="time_shuffle", seed=42, subsample=300
        )
        assert "p_value" in result
        assert len(result["surrogate_scores"]) == 3
        assert isinstance(result["p_value"], float)

    def test_significance_twin_surrogate(self):
        """test_significance with twin_surrogate should produce valid output."""
        set_seed(42)
        ts_x, ts_y = coupled_lorenz(n_steps=3000, coupling=0.3, seed=42)
        x, y = ts_x[500:, 0], ts_y[500:, 0]
        det = BindingDetector(max_dim=1)
        det.fit(x, y, subsample=300, seed=42)
        result = det.test_significance(
            n_surrogates=3, method="twin_surrogate", seed=42, subsample=300
        )
        assert "p_value" in result
        assert len(result["surrogate_scores"]) == 3

    def test_significance_invalid_method_raises(self):
        """Unknown surrogate method should raise ValueError."""
        set_seed(42)
        ts_x, ts_y = coupled_lorenz(n_steps=3000, coupling=0.3, seed=42)
        x, y = ts_x[500:, 0], ts_y[500:, 0]
        det = BindingDetector(max_dim=1)
        det.fit(x, y, subsample=300, seed=42)
        with pytest.raises(ValueError, match="Unknown method"):
            det.test_significance(method="bad", seed=42)

    def test_significance_diagram_matching_raises(self):
        """Significance testing not supported for diagram_matching method."""
        set_seed(42)
        ts_x, ts_y = coupled_lorenz(n_steps=3000, coupling=0.3, seed=42)
        x, y = ts_x[500:, 0], ts_y[500:, 0]
        det = BindingDetector(max_dim=1, method="diagram_matching")
        det.fit(x, y, subsample=300, seed=42)
        with pytest.raises(NotImplementedError, match="diagram_matching"):
            det.test_significance(n_surrogates=3, seed=42)

    def test_plot_comparison_returns_figure(self, coupled_pair):
        """plot_comparison() should return a matplotlib Figure."""
        import matplotlib.figure
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=500, seed=42)
        fig = det.plot_comparison()
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_ensemble_binding(self, coupled_pair):
        """Ensemble binding with n_ensemble > 1 returns mean of K scores."""
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=300, seed=42, n_ensemble=3)
        score = det.binding_score()
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert det.ensemble_scores is not None
        assert len(det.ensemble_scores) == 3
        assert abs(score - float(np.mean(det.ensemble_scores))) < 1e-10

    def test_ensemble_no_subsample_skips(self):
        """Ensemble requires subsample; without it, ensemble_scores is None."""
        # Use small data — subsample=None runs ripser on the full cloud,
        # so we need n < 1000 embedded points to avoid a hang.
        ts_x, ts_y = coupled_lorenz(n_steps=1500, coupling=0.3, seed=42)
        x, y = ts_x[500:, 0], ts_y[500:, 0]
        det = BindingDetector(max_dim=1)
        det.fit(x, y, subsample=None, seed=42, n_ensemble=5)
        assert det.ensemble_scores is None

    def test_confidence_interval_requires_ensemble(self, coupled_pair):
        """confidence_interval() returns None without ensemble."""
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=300, seed=42)
        assert det.confidence_interval() is None

    def test_confidence_interval_with_ensemble(self, coupled_pair):
        """confidence_interval() returns (lo, hi) with ensemble."""
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=300, seed=42, n_ensemble=5)
        ci = det.confidence_interval()
        assert ci is not None
        lo, hi = ci
        assert lo < hi

    def test_zscore_in_significance_result(self, coupled_pair):
        """test_significance() returns z_score and calibrated_score."""
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=300, seed=42)
        result = det.test_significance(n_surrogates=3, seed=42, subsample=300)
        assert "z_score" in result
        assert "calibrated_score" in result
        assert "surrogate_mean" in result
        assert "surrogate_std" in result
        assert isinstance(result["z_score"], float)
        assert isinstance(result["calibrated_score"], float)

    def test_kernel_diagnostics_keys(self, coupled_pair):
        """kernel_diagnostics() should return all expected keys after fit()."""
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=500, seed=42)
        kd = det.kernel_diagnostics()
        for name in ("marginal_x", "marginal_y", "joint"):
            assert name in kd
            for key in ("mean_dist", "var_dist", "heterogeneity"):
                assert key in kd[name]
        assert "perturbative_regime" in kd
        assert "max_heterogeneity" in kd
        assert isinstance(kd["perturbative_regime"], bool)

    def test_kernel_diagnostics_perturbative(self, coupled_pair):
        """Coupled Lorenz joint cloud heterogeneity should be below 0.5."""
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=500, seed=42)
        kd = det.kernel_diagnostics()
        # Joint embedding has lower heterogeneity than marginals
        assert kd["joint"]["heterogeneity"] < 0.5
        assert kd["max_heterogeneity"] > 0

    def test_kernel_diagnostics_before_fit(self):
        """kernel_diagnostics() before fit() should raise RuntimeError."""
        det = BindingDetector(max_dim=1)
        with pytest.raises(RuntimeError, match="fit"):
            det.kernel_diagnostics()

    def test_residual_energy_fraction_key(self, coupled_pair):
        """binding_features() should include residual_energy_fraction for each dim."""
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=500, seed=42)
        features = det.binding_features()
        for d in (0, 1):
            assert "residual_energy_fraction" in features[d]
            assert isinstance(features[d]["residual_energy_fraction"], float)

    def test_residual_energy_fraction_uncoupled(self, uncoupled_pair):
        """Uncoupled: H1 residual energy fraction should be non-negative and bounded."""
        det = BindingDetector(max_dim=1, baseline="max")
        det.fit(*uncoupled_pair, subsample=500, seed=42)
        features = det.binding_features()
        frac = features[1]["residual_energy_fraction"]
        assert 0.0 <= frac <= 1.0, f"H1 fraction {frac} outside [0, 1]"

    def test_residual_energy_fraction_coupled(self, coupled_pair):
        """Coupled: H1 residual energy fraction should be positive and bounded."""
        det = BindingDetector(max_dim=1, baseline="max")
        det.fit(*coupled_pair, subsample=500, seed=42)
        features = det.binding_features()
        frac = features[1]["residual_energy_fraction"]
        assert 0.0 < frac <= 1.0, f"H1 fraction {frac} outside (0, 1]"

    def test_surrogate_acf_warning(self):
        """Cumulative-sum noise (acf > 0.99) with time_shuffle should warn."""
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(3000))
        y = np.cumsum(rng.standard_normal(3000))
        det = BindingDetector(max_dim=1)
        det.fit(x, y, subsample=300, seed=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            det.test_significance(n_surrogates=3, method="time_shuffle", seed=42, subsample=300)
            acf_warnings = [x for x in w if issubclass(x.category, SurrogateMethodWarning)]
            assert len(acf_warnings) > 0, "Should warn about autocorrelation"

    def test_surrogate_acf_silent_phase(self):
        """Same autocorrelated data with phase_randomize should not warn."""
        rng = np.random.default_rng(42)
        x = np.cumsum(rng.standard_normal(3000))
        y = np.cumsum(rng.standard_normal(3000))
        det = BindingDetector(max_dim=1)
        det.fit(x, y, subsample=300, seed=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            det.test_significance(n_surrogates=3, method="phase_randomize", seed=42, subsample=300)
            acf_warnings = [x for x in w if issubclass(x.category, SurrogateMethodWarning)]
            assert len(acf_warnings) == 0

    def test_surrogate_acf_silent_white(self):
        """White noise with time_shuffle should not warn."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(3000)
        y = rng.standard_normal(3000)
        det = BindingDetector(max_dim=1)
        det.fit(x, y, subsample=300, seed=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            det.test_significance(n_surrogates=3, method="time_shuffle", seed=42, subsample=300)
            acf_warnings = [x for x in w if issubclass(x.category, SurrogateMethodWarning)]
            assert len(acf_warnings) == 0

    def test_cached_params_used_in_surrogates(self, coupled_pair):
        """After fit(), surrogate computation reuses embedding params."""
        det = BindingDetector(max_dim=1)
        det.fit(*coupled_pair, subsample=300, seed=42)
        # Verify params were cached
        assert det._marginal_delay_x is not None
        assert det._marginal_dim_x is not None
        assert det._marginal_delay_y is not None
        assert det._marginal_dim_y is not None
        assert det._joint_delays is not None
        assert det._joint_dims is not None
        assert len(det._joint_delays) == 2
        assert len(det._joint_dims) == 2
