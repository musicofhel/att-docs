"""Tests for att.binding — binding detection via persistence image subtraction."""

import warnings
import pytest

from att.config import set_seed
from att.synthetic import coupled_lorenz, coupled_rossler_lorenz
from att.embedding.validation import EmbeddingDegeneracyWarning
from att.binding import BindingDetector


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
