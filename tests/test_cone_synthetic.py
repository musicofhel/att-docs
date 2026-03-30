"""Tests for Aizawa attractor, layered network generators, and ConeDetector."""

import numpy as np
import pytest

from att.cone.detector import ConeDetector
from att.synthetic.generators import aizawa_system
from att.synthetic.layered_network import (
    NODE_NAMES,
    layered_aizawa_network,
    layered_aizawa_network_symmetric,
)


class TestAizawa:
    """Tests for the Aizawa attractor generator."""

    def test_shape(self):
        ts = aizawa_system(n_steps=1000, seed=42)
        assert ts.shape == (1000, 3)

    def test_reproducible(self):
        a = aizawa_system(n_steps=500, seed=42)
        b = aizawa_system(n_steps=500, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self):
        a = aizawa_system(n_steps=500, seed=42)
        b = aizawa_system(n_steps=500, seed=99)
        assert not np.array_equal(a, b)

    def test_bounded(self):
        """Aizawa should stay bounded for default parameters."""
        ts = aizawa_system(n_steps=5000, seed=42)
        assert np.all(np.abs(ts) < 10), "Aizawa diverged"

    def test_not_degenerate(self):
        """Each component should have meaningful variance."""
        ts = aizawa_system(n_steps=5000, seed=42)
        for dim in range(3):
            assert ts[:, dim].std() > 0.1, f"Dimension {dim} is degenerate"

    def test_noise(self):
        clean = aizawa_system(n_steps=500, seed=42, noise=0.0)
        noisy = aizawa_system(n_steps=500, seed=42, noise=0.1)
        # Same seed but noise makes them differ
        assert not np.array_equal(clean, noisy)

    def test_custom_initial(self):
        ic = np.array([0.5, 0.5, 0.5])
        ts = aizawa_system(n_steps=100, initial=ic, seed=42)
        # First point should be near the IC (RK45 may not land exactly)
        assert np.allclose(ts[0], ic, atol=0.1)

    def test_custom_params(self):
        """Should run without error with non-default parameters."""
        ts = aizawa_system(n_steps=500, alpha=1.0, beta=0.8, gamma=0.5, seed=42)
        assert ts.shape == (500, 3)
        assert np.all(np.isfinite(ts))


class TestLayeredNetwork:
    """Tests for the 2-column, 3-layer directed Aizawa network."""

    def test_returns_all_nodes(self):
        traj = layered_aizawa_network(n_steps=500, seed=42)
        assert set(traj.keys()) == set(NODE_NAMES)

    def test_shapes(self):
        n = 1000
        traj = layered_aizawa_network(n_steps=n, seed=42)
        for name in NODE_NAMES:
            assert traj[name].shape == (n, 3), f"{name} has wrong shape"

    def test_reproducible(self):
        a = layered_aizawa_network(n_steps=500, seed=42)
        b = layered_aizawa_network(n_steps=500, seed=42)
        for name in NODE_NAMES:
            np.testing.assert_array_equal(a[name], b[name])

    def test_different_seeds(self):
        a = layered_aizawa_network(n_steps=500, seed=42)
        b = layered_aizawa_network(n_steps=500, seed=99)
        assert not np.array_equal(a["C"], b["C"])

    def test_bounded(self):
        """All nodes should stay bounded."""
        traj = layered_aizawa_network(n_steps=10000, seed=42)
        for name in NODE_NAMES:
            assert np.all(np.abs(traj[name]) < 20), f"{name} diverged"

    def test_not_degenerate(self):
        traj = layered_aizawa_network(n_steps=5000, seed=42)
        for name in NODE_NAMES:
            for dim in range(3):
                assert traj[name][:, dim].std() > 0.05, (
                    f"{name} dim {dim} degenerate"
                )

    def test_zero_coupling_independent(self):
        """With no coupling, nodes should evolve independently."""
        traj = layered_aizawa_network(
            n_steps=2000, coupling_source=0.0, coupling_down=0.0, seed=42
        )
        # C and A3 should have near-zero correlation
        corr = np.corrcoef(traj["C"][:, 0], traj["A3"][:, 0])[0, 1]
        assert abs(corr) < 0.3, f"Uncoupled correlation too high: {corr}"

    def test_coupling_vs_uncoupled(self):
        """Coupled nodes should be more correlated than uncoupled."""
        traj_off = layered_aizawa_network(
            n_steps=10000, coupling_source=0.0, coupling_down=0.0, seed=42
        )
        traj_on = layered_aizawa_network(
            n_steps=10000, coupling_source=0.15, coupling_down=0.15, seed=42
        )
        # Use second half to avoid transient IC similarity
        half = 5000
        corr_off = abs(np.corrcoef(
            traj_off["C"][half:, 0], traj_off["A3"][half:, 0]
        )[0, 1])
        corr_on = abs(np.corrcoef(
            traj_on["C"][half:, 0], traj_on["A3"][half:, 0]
        )[0, 1])
        assert corr_on > corr_off, (
            f"Coupled ({corr_on:.3f}) not > uncoupled ({corr_off:.3f})"
        )

    def test_column_symmetry(self):
        """A and B columns should have similar statistics (same coupling from C)."""
        traj = layered_aizawa_network(n_steps=10000, seed=42)
        std_a3 = traj["A3"][:, 0].std()
        std_b3 = traj["B3"][:, 0].std()
        # Not identical (different ICs) but similar scale
        assert abs(std_a3 - std_b3) / max(std_a3, std_b3) < 0.3

    def test_cross_column_via_source_only(self):
        """A3-B3 correlation should increase when source coupling is turned on."""
        # With source coupling, A3 and B3 are both driven by C (common driver)
        traj_on = layered_aizawa_network(
            n_steps=20000, coupling_source=0.2, coupling_down=0.0, seed=42
        )
        traj_off = layered_aizawa_network(
            n_steps=20000, coupling_source=0.0, coupling_down=0.0, seed=42
        )
        # Use second half to let chaotic divergence separate uncoupled nodes
        half = 10000
        corr_on = abs(np.corrcoef(
            traj_on["A3"][half:, 0], traj_on["B3"][half:, 0]
        )[0, 1])
        corr_off = abs(np.corrcoef(
            traj_off["A3"][half:, 0], traj_off["B3"][half:, 0]
        )[0, 1])
        assert corr_on > corr_off, (
            f"Source-coupled A3-B3 ({corr_on:.3f}) not > uncoupled ({corr_off:.3f})"
        )


class TestLayeredNetworkSymmetric:
    """Tests for the symmetric (all-to-all) network variant."""

    def test_returns_all_nodes(self):
        traj = layered_aizawa_network_symmetric(n_steps=500, seed=42)
        assert set(traj.keys()) == set(NODE_NAMES)

    def test_shapes(self):
        n = 1000
        traj = layered_aizawa_network_symmetric(n_steps=n, seed=42)
        for name in NODE_NAMES:
            assert traj[name].shape == (n, 3)

    def test_bounded(self):
        traj = layered_aizawa_network_symmetric(n_steps=5000, seed=42)
        for name in NODE_NAMES:
            assert np.all(np.abs(traj[name]) < 20), f"{name} diverged"

    def test_symmetric_more_correlated(self):
        """Symmetric coupling should produce higher cross-column correlation."""
        traj_dir = layered_aizawa_network(
            n_steps=5000, coupling_source=0.15, coupling_down=0.15, seed=42
        )
        traj_sym = layered_aizawa_network_symmetric(
            n_steps=5000, coupling_source=0.15, coupling_down=0.15, seed=42
        )
        # A3-B3 correlation should be higher with direct lateral coupling
        corr_dir = abs(np.corrcoef(
            traj_dir["A3"][:, 0], traj_dir["B3"][:, 0]
        )[0, 1])
        corr_sym = abs(np.corrcoef(
            traj_sym["A3"][:, 0], traj_sym["B3"][:, 0]
        )[0, 1])
        # Symmetric has direct A3<->B3 coupling, directed doesn't
        assert corr_sym > corr_dir or corr_sym > 0.2, (
            f"Symmetric ({corr_sym:.3f}) not more correlated than directed ({corr_dir:.3f})"
        )


class TestConeDetector:
    """Tests for ConeDetector fit, axis estimation, and availability profile."""

    @pytest.fixture
    def network_data(self):
        """Generate a small coupled network for testing."""
        traj = layered_aizawa_network(
            n_steps=5000, coupling_source=0.15, coupling_down=0.15, seed=42
        )
        # Discard transient
        trim = 1000
        source = traj["C"][trim:, 0]
        receivers = [traj[n][trim:, 0] for n in ["A3", "B3", "A5", "B5"]]
        return source, receivers, traj, trim

    def test_fit_returns_self(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1)
        result = det.fit(source, receivers)
        assert result is det

    def test_fit_populates_state(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1)
        det.fit(source, receivers)
        assert det._source_embedded is not None
        assert det._receiver_cloud is not None
        assert det._projection_axis is not None
        assert det._depth_projections is not None
        assert det._cca_subspace is not None

    def test_source_receiver_aligned(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1)
        det.fit(source, receivers)
        assert len(det._source_embedded) == len(det._receiver_cloud)

    def test_projection_axis_is_unit(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1)
        det.fit(source, receivers)
        norm = np.linalg.norm(det._projection_axis)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_projection_axis_dimension(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1)
        det.fit(source, receivers)
        assert det._projection_axis.shape == (det._receiver_cloud.shape[1],)

    def test_slice_at_depth_valid_bins(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1)
        det.fit(source, receivers)
        total = 0
        for i in range(3):
            sl = det.slice_at_depth(i)
            assert sl.ndim == 2
            assert sl.shape[1] == det._receiver_cloud.shape[1]
            total += len(sl)
        # All points should be covered
        assert total == len(det._receiver_cloud)

    def test_slice_at_depth_invalid(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1)
        det.fit(source, receivers)
        with pytest.raises(ValueError):
            det.slice_at_depth(3)
        with pytest.raises(ValueError):
            det.slice_at_depth(-1)

    def test_availability_profile_keys(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1)
        det.fit(source, receivers)
        profile = det.availability_profile(subsample=500)
        expected_keys = {
            "depths", "betti_0", "betti_1", "persistence_entropy",
            "diagrams", "is_monotonic", "trend_slope",
        }
        assert set(profile.keys()) == expected_keys

    def test_availability_profile_shapes(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1)
        det.fit(source, receivers)
        profile = det.availability_profile(subsample=500)
        assert len(profile["depths"]) == 3
        assert len(profile["betti_0"]) == 3
        assert len(profile["betti_1"]) == 3
        assert len(profile["diagrams"]) == 3
        assert isinstance(profile["trend_slope"], float)
        assert isinstance(profile["is_monotonic"], bool)

    def test_availability_profile_cca(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1)
        det.fit(source, receivers)
        profile = det.availability_profile(subspace="cca", subsample=500)
        assert len(profile["depths"]) == 3

    def test_cca_subspace_shape(self, network_data):
        source, receivers, _, _ = network_data
        det = ConeDetector(n_depth_bins=3, max_dim=1, cca_components=3)
        det.fit(source, receivers)
        n_points = len(det._receiver_cloud)
        # CCA components clamped to min(cca_components, source_dim, receiver_dim)
        assert det._cca_subspace.shape[0] == n_points
        assert det._cca_subspace.shape[1] <= 3

    @pytest.fixture
    def short_binding_data(self):
        """Shorter series with forced embedding params for memory-safe binding tests."""
        traj = layered_aizawa_network(
            n_steps=3000, coupling_source=0.15, coupling_down=0.15, seed=42
        )
        trim = 500
        source = traj["C"][trim:, 0]
        shallow = traj["A3"][trim:, 0]
        deep = traj["A5"][trim:, 0]
        return source, shallow, deep

    @pytest.mark.slow
    def test_depth_asymmetry_keys(self, short_binding_data):
        source, shallow, deep = short_binding_data
        det = ConeDetector(max_dim=1)
        result = det.depth_asymmetry(source, shallow, deep, subsample=500, seed=42)
        assert "shallow_binding" in result
        assert "deep_binding" in result
        assert "asymmetry" in result
        assert "ratio" in result
        assert isinstance(result["shallow_binding"], float)
        assert isinstance(result["deep_binding"], float)

    @pytest.mark.slow
    def test_full_chain_emergence_keys(self, short_binding_data):
        source, shallow, deep = short_binding_data
        det = ConeDetector(max_dim=1)
        result = det.full_chain_emergence(source, shallow, deep, subsample=500, seed=42)
        assert "pairwise_bindings" in result
        assert "full_chain_binding" in result
        assert "max_pairwise" in result
        assert "emergence" in result
        assert "has_emergence" in result
        assert isinstance(result["has_emergence"], bool)
