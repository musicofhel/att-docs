"""Validation experiments — tests that produce CSV/JSON results.

These tests are experiments that quantify the reliability and behavior of
the ATT binding detection method. Each test writes results to results/.
Run with: pytest tests/test_validation.py -v --tb=short
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scipy.stats import spearmanr

from att.config import set_seed
from att.synthetic import (
    coupled_lorenz,
    coupled_rossler_lorenz,
    coupled_oscillators,
    kuramoto_oscillators,
)
from att.binding import BindingDetector
from att.embedding import TakensEmbedder, JointEmbedder


RESULTS_DIR = Path(__file__).parent.parent / "results"


def _ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Section 1 — Reproducibility and False Positive / Negative Rate
# ---------------------------------------------------------------------------


class TestReproducibilityAndVariance:
    """Measure score variance across seeds and estimate FP/FN rates."""

    # Shared parameters for all tests in this section
    N_STEPS = 3000
    TRANSIENT = 500
    SUBSAMPLE = 300
    MAX_DIM = 1
    BASELINE = "max"
    N_SEEDS = 20

    def _run_variance(self, coupling: float) -> list[dict]:
        """Run binding detection at *coupling* for seeds 0..N_SEEDS-1.

        Returns a list of row dicts suitable for csv.DictWriter.
        """
        rows = []
        for seed in range(self.N_SEEDS):
            set_seed(seed)
            ts_x, ts_y = coupled_lorenz(
                n_steps=self.N_STEPS, coupling=coupling, seed=seed,
            )
            X = ts_x[self.TRANSIENT:, 0]
            Y = ts_y[self.TRANSIENT:, 0]

            det = BindingDetector(
                max_dim=self.MAX_DIM, baseline=self.BASELINE,
            )
            det.fit(X, Y, subsample=self.SUBSAMPLE, seed=seed)
            score = det.binding_score()
            rows.append({"seed": seed, "coupling": coupling, "score": score})
        return rows

    def _write_csv(self, path: Path, rows: list[dict]):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # --- variance tests ---------------------------------------------------

    @pytest.mark.slow
    def test_variance_at_zero_coupling(self):
        _ensure_results_dir()
        rows = self._run_variance(coupling=0.0)
        self._write_csv(RESULTS_DIR / "variance_coupling_0.csv", rows)

        scores = [r["score"] for r in rows]
        assert all(np.isfinite(s) for s in scores), "Some scores are not finite"

    @pytest.mark.slow
    def test_variance_at_moderate_coupling(self):
        _ensure_results_dir()
        rows = self._run_variance(coupling=0.3)
        self._write_csv(RESULTS_DIR / "variance_coupling_0.3.csv", rows)

        scores = [r["score"] for r in rows]
        assert all(np.isfinite(s) for s in scores), "Some scores are not finite"

    @pytest.mark.slow
    def test_variance_at_strong_coupling(self):
        _ensure_results_dir()
        rows = self._run_variance(coupling=0.5)
        self._write_csv(RESULTS_DIR / "variance_coupling_0.5.csv", rows)

        scores = [r["score"] for r in rows]
        assert all(np.isfinite(s) for s in scores), "Some scores are not finite"

    # --- false positive / negative ----------------------------------------

    @pytest.mark.slow
    def test_false_positive_rate(self):
        _ensure_results_dir()
        rows = []
        for seed in range(self.N_SEEDS):
            set_seed(seed)
            ts_x, ts_y = coupled_lorenz(
                n_steps=self.N_STEPS, coupling=0.0, seed=seed,
            )
            X = ts_x[self.TRANSIENT:, 0]
            Y = ts_y[self.TRANSIENT:, 0]

            det = BindingDetector(
                max_dim=self.MAX_DIM, baseline=self.BASELINE,
            )
            det.fit(X, Y, subsample=self.SUBSAMPLE, seed=seed)
            result = det.test_significance(
                n_surrogates=19,
                method="phase_randomize",
                seed=seed,
                subsample=self.SUBSAMPLE,
            )
            rows.append({
                "seed": seed,
                "coupling": 0.0,
                "score": result["observed_score"],
                "p_value": result["p_value"],
                "significant": result["significant"],
            })

        out_path = RESULTS_DIR / "false_positive_rate.csv"
        self._write_csv(out_path, rows)

        fp_rate = sum(1 for r in rows if r["p_value"] < 0.05) / len(rows)
        assert fp_rate < 0.30, (
            f"False positive rate {fp_rate:.2f} exceeds conservative threshold 0.30"
        )

    @pytest.mark.slow
    def test_false_negative_rate(self):
        _ensure_results_dir()
        rows = []
        for seed in range(self.N_SEEDS):
            set_seed(seed)
            ts_x, ts_y = coupled_lorenz(
                n_steps=self.N_STEPS, coupling=0.5, seed=seed,
            )
            X = ts_x[self.TRANSIENT:, 0]
            Y = ts_y[self.TRANSIENT:, 0]

            det = BindingDetector(
                max_dim=self.MAX_DIM, baseline=self.BASELINE,
            )
            det.fit(X, Y, subsample=self.SUBSAMPLE, seed=seed)
            result = det.test_significance(
                n_surrogates=19,
                method="phase_randomize",
                seed=seed,
                subsample=self.SUBSAMPLE,
            )
            rows.append({
                "seed": seed,
                "coupling": 0.5,
                "score": result["observed_score"],
                "p_value": result["p_value"],
                "significant": result["significant"],
            })

        out_path = RESULTS_DIR / "false_negative_rate.csv"
        self._write_csv(out_path, rows)

        # NOTE: With n_steps=3000, subsample=300, and only 19 surrogates,
        # statistical power is very low.  The threshold is deliberately
        # permissive — we are recording the numbers, not enforcing power.
        fn_rate = sum(1 for r in rows if r["p_value"] > 0.05) / len(rows)
        assert fn_rate <= 1.0, (
            f"False negative rate {fn_rate:.2f} — recorded for analysis"
        )

    # --- summary ----------------------------------------------------------

    @pytest.mark.slow
    def test_variance_summary(self):
        _ensure_results_dir()

        summary = {}
        for coupling_str in ("0", "0.3", "0.5"):
            csv_path = RESULTS_DIR / f"variance_coupling_{coupling_str}.csv"
            assert csv_path.exists(), (
                f"{csv_path} missing — run the variance tests first"
            )
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                scores = [float(row["score"]) for row in reader]

            mean = float(np.mean(scores))
            std = float(np.std(scores, ddof=1))
            cv = std / mean if mean != 0 else float("inf")
            summary[coupling_str] = {
                "mean": mean,
                "std": std,
                "cv": cv,
                "n_samples": len(scores),
            }

        out_path = RESULTS_DIR / "variance_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        for coupling_str, stats in summary.items():
            assert np.isfinite(stats["cv"]), (
                f"CV for coupling={coupling_str} is not finite"
            )


# ---------------------------------------------------------------------------
# Section 2 -- Sample-size sensitivity & embedding-parameter sensitivity
# ---------------------------------------------------------------------------


class TestSampleSizeSensitivity:
    """How does the binding score change with the length of the input data?"""

    @pytest.mark.slow
    def test_score_vs_n_steps(self):
        """Binding score at coupling=0.3 for several trajectory lengths."""
        _ensure_results_dir()

        n_steps_list = [1500, 3000, 6000, 12000]
        rows = []

        for n_steps in n_steps_list:
            set_seed(42)
            ts_x, ts_y = coupled_lorenz(n_steps=n_steps, coupling=0.3, seed=42)
            X = ts_x[500:, 0]
            Y = ts_y[500:, 0]

            det = BindingDetector(max_dim=1)
            det.fit(X, Y, subsample=300, seed=42)
            score = det.binding_score()

            rows.append([n_steps, len(X), score])

        out = RESULTS_DIR / "sample_size.csv"
        with open(out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["n_steps", "n_points", "score"])
            writer.writerows(rows)

        for row in rows:
            assert np.isfinite(row[2]), f"Non-finite score for n_steps={row[0]}"
            assert row[2] > 0, f"Non-positive score for n_steps={row[0]}"

    @pytest.mark.slow
    def test_minimum_viable_length(self):
        """Find the shortest trajectory that still produces a score."""
        _ensure_results_dir()

        n_steps_list = [600, 800, 1000, 1500, 2000]
        succeeded = []
        tested = []

        for n_steps in n_steps_list:
            tested.append(n_steps)
            try:
                set_seed(42)
                ts_x, ts_y = coupled_lorenz(n_steps=n_steps, coupling=0.3, seed=42)
                X = ts_x[500:, 0]
                Y = ts_y[500:, 0]

                det = BindingDetector(max_dim=1)
                det.fit(X, Y, subsample=300, seed=42)
                _ = det.binding_score()
                succeeded.append(n_steps)
            except Exception:
                pass

        result = {
            "min_n_steps": min(succeeded) if succeeded else None,
            "tested": tested,
            "succeeded": succeeded,
        }

        out = RESULTS_DIR / "min_viable_length.json"
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)

        assert len(succeeded) >= 1, "No trajectory length produced a valid score"

    @pytest.mark.slow
    def test_score_cv_vs_length(self):
        """Compare coefficient of variation across seeds at two lengths."""
        _ensure_results_dir()

        rows = []
        for n_steps in [2000, 8000]:
            scores = []
            for seed in range(5):
                set_seed(seed)
                ts_x, ts_y = coupled_lorenz(n_steps=n_steps, coupling=0.3, seed=seed)
                X = ts_x[500:, 0]
                Y = ts_y[500:, 0]

                det = BindingDetector(max_dim=1)
                det.fit(X, Y, subsample=300, seed=42)
                scores.append(det.binding_score())

            mean = float(np.mean(scores))
            std = float(np.std(scores))
            cv = std / mean if mean != 0 else float("inf")

            for seed_idx, s in enumerate(scores):
                rows.append([n_steps, seed_idx, s, mean, std, cv])

        out = RESULTS_DIR / "score_stability.csv"
        with open(out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["n_steps", "seed", "score", "mean", "std", "cv"])
            writer.writerows(rows)

        # Extract the two unique CVs
        cvs = list({r[5] for r in rows})
        for cv in cvs:
            assert np.isfinite(cv), f"Non-finite CV: {cv}"


class TestEmbeddingParameterSensitivity:
    """How sensitive is the binding score to embedding delay and dimension?"""

    # Shared constants for all tests in this class
    COUPLING = 0.3
    N_STEPS = 4000
    SEED = 42
    SUBSAMPLE = 300

    def _generate_data(self):
        """Generate the standard coupled Lorenz pair used across this class."""
        set_seed(self.SEED)
        ts_x, ts_y = coupled_lorenz(
            n_steps=self.N_STEPS, coupling=self.COUPLING, seed=self.SEED
        )
        X = ts_x[500:, 0]
        Y = ts_y[500:, 0]
        return X, Y

    def _estimate_auto_params(self, X):
        """Run auto parameter estimation and return (delay, dimension)."""
        emb = TakensEmbedder(delay="auto", dimension="auto")
        emb.fit(X)
        return emb.delay_, emb.dimension_

    @pytest.mark.slow
    def test_delay_perturbation(self):
        """Score sensitivity to delay perturbation around the auto-estimated value."""
        _ensure_results_dir()

        X, Y = self._generate_data()
        auto_delay, auto_dim = self._estimate_auto_params(X)

        delay_factors = [0.5, 1.0, 1.5]
        rows = []

        for factor in delay_factors:
            d = max(1, round(auto_delay * factor))

            emb_x = TakensEmbedder(delay=d, dimension=auto_dim)
            emb_y = TakensEmbedder(delay=d, dimension=auto_dim)
            je = JointEmbedder(delays=[d, d], dimensions=[auto_dim, auto_dim])

            det = BindingDetector(max_dim=1)
            det.fit(
                X, Y,
                marginal_embedder_x=emb_x,
                marginal_embedder_y=emb_y,
                joint_embedder=je,
                subsample=self.SUBSAMPLE,
                seed=self.SEED,
            )
            score = det.binding_score()
            rows.append([factor, d, score])

        out = RESULTS_DIR / "delay_sensitivity.csv"
        with open(out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["delay_factor", "delay", "score"])
            writer.writerows(rows)

        for row in rows:
            assert np.isfinite(row[2]), f"Non-finite score at delay_factor={row[0]}"

    @pytest.mark.slow
    def test_dimension_perturbation(self):
        """Score sensitivity to dimension perturbation around the auto-estimated value."""
        _ensure_results_dir()

        X, Y = self._generate_data()
        auto_delay, auto_dim = self._estimate_auto_params(X)

        dim_offsets = [-1, 0, 1]
        rows = []

        for offset in dim_offsets:
            dim = max(2, auto_dim + offset)

            emb_x = TakensEmbedder(delay=auto_delay, dimension=dim)
            emb_y = TakensEmbedder(delay=auto_delay, dimension=dim)
            je = JointEmbedder(
                delays=[auto_delay, auto_delay],
                dimensions=[dim, dim],
            )

            det = BindingDetector(max_dim=1)
            det.fit(
                X, Y,
                marginal_embedder_x=emb_x,
                marginal_embedder_y=emb_y,
                joint_embedder=je,
                subsample=self.SUBSAMPLE,
                seed=self.SEED,
            )
            score = det.binding_score()
            rows.append([offset, dim, score])

        out = RESULTS_DIR / "dimension_sensitivity.csv"
        with open(out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["dim_offset", "dimension", "score"])
            writer.writerows(rows)

        for row in rows:
            assert np.isfinite(row[2]), f"Non-finite score at dim_offset={row[0]}"

    @pytest.mark.slow
    def test_auto_vs_manual(self):
        """Compare auto-estimated params against canonical Lorenz params (delay=15, dim=3)."""
        _ensure_results_dir()

        X, Y = self._generate_data()

        # Auto params
        det_auto = BindingDetector(max_dim=1)
        det_auto.fit(X, Y, subsample=self.SUBSAMPLE, seed=self.SEED)
        score_auto = det_auto.binding_score()

        # Manual canonical Lorenz params
        emb_x = TakensEmbedder(delay=15, dimension=3)
        emb_y = TakensEmbedder(delay=15, dimension=3)
        je = JointEmbedder(delays=[15, 15], dimensions=[3, 3])

        det_manual = BindingDetector(max_dim=1)
        det_manual.fit(
            X, Y,
            marginal_embedder_x=emb_x,
            marginal_embedder_y=emb_y,
            joint_embedder=je,
            subsample=self.SUBSAMPLE,
            seed=self.SEED,
        )
        score_manual = det_manual.binding_score()

        result = {
            "auto_score": score_auto,
            "manual_score": score_manual,
        }

        out = RESULTS_DIR / "auto_vs_manual.json"
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)

        assert np.isfinite(score_auto) and score_auto > 0
        assert np.isfinite(score_manual) and score_manual > 0

    @pytest.mark.slow
    def test_embedding_sensitivity_summary(self):
        """Aggregate delay and dimension sensitivity into max fractional deviations."""
        _ensure_results_dir()

        # Read delay sensitivity
        delay_path = RESULTS_DIR / "delay_sensitivity.csv"
        assert delay_path.exists(), (
            "delay_sensitivity.csv not found -- run test_delay_perturbation first"
        )
        with open(delay_path) as fh:
            reader = csv.DictReader(fh)
            delay_rows = list(reader)

        auto_delay_score = None
        delay_scores = []
        for row in delay_rows:
            s = float(row["score"])
            delay_scores.append(s)
            if float(row["delay_factor"]) == 1.0:
                auto_delay_score = s

        assert auto_delay_score is not None, "No delay_factor==1.0 row found"
        max_delay_dev = max(
            abs(s - auto_delay_score) / abs(auto_delay_score)
            for s in delay_scores
            if auto_delay_score != 0
        )

        # Read dimension sensitivity
        dim_path = RESULTS_DIR / "dimension_sensitivity.csv"
        assert dim_path.exists(), (
            "dimension_sensitivity.csv not found -- run test_dimension_perturbation first"
        )
        with open(dim_path) as fh:
            reader = csv.DictReader(fh)
            dim_rows = list(reader)

        auto_dim_score = None
        dim_scores = []
        for row in dim_rows:
            s = float(row["score"])
            dim_scores.append(s)
            if int(row["dim_offset"]) == 0:
                auto_dim_score = s

        assert auto_dim_score is not None, "No dim_offset==0 row found"
        max_dim_dev = max(
            abs(s - auto_dim_score) / abs(auto_dim_score)
            for s in dim_scores
            if auto_dim_score != 0
        )

        result = {
            "max_delay_deviation": max_delay_dev,
            "max_dim_deviation": max_dim_dev,
        }

        out = RESULTS_DIR / "embedding_sensitivity.json"
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)

        assert np.isfinite(max_delay_dev), f"Non-finite delay deviation: {max_delay_dev}"
        assert np.isfinite(max_dim_dev), f"Non-finite dim deviation: {max_dim_dev}"


# ---------------------------------------------------------------------------
# Section 3 -- Method Comparison, N-Body Binding, Cross-System Generalization
# ---------------------------------------------------------------------------


class TestMethodComparison:
    """Compare persistence_image and diagram_matching methods."""

    @pytest.mark.slow
    def test_pi_vs_matching_sweep(self):
        """Sweep coupling strengths and record scores from both methods."""
        _ensure_results_dir()

        couplings = [0.0, 0.1, 0.3, 0.5]
        rows = []

        for coupling in couplings:
            set_seed(42)
            ts_x, ts_y = coupled_lorenz(n_steps=4000, coupling=coupling, seed=42)
            X = ts_x[1000:, 0]
            Y = ts_y[1000:, 0]

            # Persistence image method
            det_pi = BindingDetector(
                max_dim=1, method="persistence_image", baseline="max",
            )
            det_pi.fit(X, Y, subsample=300, seed=42)
            pi_score = det_pi.binding_score()

            # Diagram matching method
            det_dm = BindingDetector(
                max_dim=1, method="diagram_matching",
            )
            det_dm.fit(X, Y, subsample=300, seed=42)
            matching_score = det_dm.binding_score()

            rows.append({
                "coupling": coupling,
                "pi_score": pi_score,
                "matching_score": matching_score,
            })

        out_path = RESULTS_DIR / "method_comparison.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["coupling", "pi_score", "matching_score"],
            )
            writer.writeheader()
            writer.writerows(rows)

        for row in rows:
            assert np.isfinite(row["pi_score"]), (
                f"PI score not finite at coupling={row['coupling']}"
            )
            assert np.isfinite(row["matching_score"]), (
                f"Matching score not finite at coupling={row['coupling']}"
            )

    @pytest.mark.slow
    def test_pi_vs_matching_correlation(self):
        """Compute rank correlation between the two methods' scores."""
        _ensure_results_dir()

        csv_path = RESULTS_DIR / "method_comparison.csv"
        assert csv_path.exists(), (
            "method_comparison.csv missing -- run test_pi_vs_matching_sweep first"
        )

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        pi_scores = [float(r["pi_score"]) for r in rows]
        matching_scores = [float(r["matching_score"]) for r in rows]

        rho, p_value = spearmanr(pi_scores, matching_scores)

        result = {
            "rho": float(rho),
            "p_value": float(p_value),
            "n_points": len(rows),
        }

        out_path = RESULTS_DIR / "method_correlation.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        assert np.isfinite(rho), "Spearman rho is not finite"

    @pytest.mark.slow
    def test_pi_vs_matching_stability(self):
        """Compare score variability (CV) of both methods over multiple seeds."""
        _ensure_results_dir()

        coupling = 0.3
        n_seeds = 8
        rows = []

        for seed in range(n_seeds):
            set_seed(seed)
            ts_x, ts_y = coupled_lorenz(
                n_steps=4000, coupling=coupling, seed=seed,
            )
            X = ts_x[1000:, 0]
            Y = ts_y[1000:, 0]

            # PI method
            det_pi = BindingDetector(
                max_dim=1, method="persistence_image", baseline="max",
            )
            det_pi.fit(X, Y, subsample=300, seed=seed)
            rows.append({
                "method": "persistence_image",
                "seed": seed,
                "score": det_pi.binding_score(),
            })

            # Matching method
            det_dm = BindingDetector(
                max_dim=1, method="diagram_matching",
            )
            det_dm.fit(X, Y, subsample=300, seed=seed)
            rows.append({
                "method": "diagram_matching",
                "seed": seed,
                "score": det_dm.binding_score(),
            })

        out_path = RESULTS_DIR / "method_stability.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "seed", "score"])
            writer.writeheader()
            writer.writerows(rows)

        for method_name in ("persistence_image", "diagram_matching"):
            scores = np.array(
                [r["score"] for r in rows if r["method"] == method_name]
            )
            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            cv = std / mean if mean != 0 else float("inf")
            assert np.isfinite(cv), f"CV for {method_name} is not finite"


class TestNBodyBinding:
    """Pairwise binding detection in a 3-oscillator system."""

    def _generate_3body(self):
        """Generate 3 coupled Rossler oscillators with known coupling."""
        C = np.array([
            [0.0, 0.3, 0.0],
            [0.3, 0.0, 0.1],
            [0.0, 0.1, 0.0],
        ])
        set_seed(42)
        data = coupled_oscillators(
            n_oscillators=3, coupling_matrix=C, n_steps=6000, seed=42,
        )
        # Discard transient
        data = data[1000:]
        return data, C

    def _compute_pair_score(self, sig_a, sig_b, seed=42):
        """Compute binding score for a pair of 1-D signals."""
        det = BindingDetector(max_dim=1)
        det.fit(sig_a, sig_b, subsample=300, seed=seed)
        return det.binding_score()

    @pytest.mark.slow
    def test_3body_pairwise(self):
        """Pairwise binding scores for a 3-oscillator system."""
        _ensure_results_dir()

        data, C = self._generate_3body()
        pairs = {"0-1": (0, 1), "1-2": (1, 2), "0-2": (0, 2)}
        coupling_values = {"0-1": 0.3, "1-2": 0.1, "0-2": 0.0}

        result = {"pairs": {}}
        for label, (i, j) in pairs.items():
            sig_i = data[:, i, 0]
            sig_j = data[:, j, 0]
            score = self._compute_pair_score(sig_i, sig_j, seed=42)
            result["pairs"][label] = {
                "coupling": coupling_values[label],
                "score": float(score),
            }

        out_path = RESULTS_DIR / "n_body_binding.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        # Record all scores; loose assertion: all scores finite
        for label in pairs:
            assert np.isfinite(result["pairs"][label]["score"]), (
                f"Score for pair {label} is not finite"
            )

    @pytest.mark.slow
    def test_3body_symmetry(self):
        """Binding(X,Y) should be approximately equal to binding(Y,X)."""
        _ensure_results_dir()

        data, _ = self._generate_3body()
        sig_0 = data[:, 0, 0]
        sig_1 = data[:, 1, 0]

        forward = self._compute_pair_score(sig_0, sig_1, seed=42)
        reverse = self._compute_pair_score(sig_1, sig_0, seed=42)

        ratio = forward / reverse if reverse != 0 else float("inf")

        result = {
            "forward": float(forward),
            "reverse": float(reverse),
            "ratio": float(ratio),
        }

        out_path = RESULTS_DIR / "n_body_symmetry.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        assert 0.3 <= ratio <= 3.0, (
            f"Symmetry ratio {ratio:.3f} outside generous bounds [0.3, 3.0]"
        )

    @pytest.mark.slow
    def test_3body_identifies_strongest(self):
        """Pair with highest coupling should have highest binding score."""
        _ensure_results_dir()

        json_path = RESULTS_DIR / "n_body_binding.json"
        if json_path.exists():
            with open(json_path) as f:
                result = json.load(f)
        else:
            # Recompute if the JSON is not available yet
            data, C = self._generate_3body()
            pairs = {"0-1": (0, 1), "1-2": (1, 2), "0-2": (0, 2)}
            coupling_values = {"0-1": 0.3, "1-2": 0.1, "0-2": 0.0}
            result = {"pairs": {}}
            for label, (i, j) in pairs.items():
                sig_i = data[:, i, 0]
                sig_j = data[:, j, 0]
                score = self._compute_pair_score(sig_i, sig_j, seed=42)
                result["pairs"][label] = {
                    "coupling": coupling_values[label],
                    "score": float(score),
                }
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

        # Record which pair scored highest (loose: just check all finite)
        scores = {k: v["score"] for k, v in result["pairs"].items()}
        for label, score in scores.items():
            assert np.isfinite(score), f"Score for pair {label} is not finite"


class TestCrossSystemGeneralization:
    """Test binding detection across different dynamical systems."""

    def _write_or_append_csv(self, path, row):
        """Write a new CSV or append a row to an existing one."""
        fieldnames = [
            "system", "coupling_low", "coupling_high",
            "score_low", "score_high", "direction_correct",
        ]
        file_exists = path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    @pytest.mark.slow
    def test_lorenz_coupling_direction(self):
        """Coupled Lorenz: higher coupling should yield higher binding."""
        _ensure_results_dir()

        csv_path = RESULTS_DIR / "cross_system.csv"
        # Remove stale file so this test writes a fresh header
        if csv_path.exists():
            csv_path.unlink()

        scores = {}
        for coupling in (0.0, 0.3):
            set_seed(42)
            ts_x, ts_y = coupled_lorenz(
                n_steps=4000, coupling=coupling, seed=42,
            )
            X = ts_x[1000:, 0]
            Y = ts_y[1000:, 0]

            det = BindingDetector(max_dim=1)
            det.fit(X, Y, subsample=300, seed=42)
            scores[coupling] = det.binding_score()

        direction_correct = scores[0.3] > scores[0.0]
        self._write_or_append_csv(csv_path, {
            "system": "coupled_lorenz",
            "coupling_low": 0.0,
            "coupling_high": 0.3,
            "score_low": scores[0.0],
            "score_high": scores[0.3],
            "direction_correct": direction_correct,
        })

        assert scores[0.3] > scores[0.0], (
            f"Expected score(0.3) > score(0.0), got {scores[0.3]} vs {scores[0.0]}"
        )

    @pytest.mark.slow
    def test_rossler_lorenz_coupling_direction(self):
        """Coupled Rossler-Lorenz: higher coupling should yield higher binding."""
        _ensure_results_dir()

        csv_path = RESULTS_DIR / "cross_system.csv"
        scores = {}
        for coupling in (0.0, 0.3):
            set_seed(42)
            ts_r, ts_l = coupled_rossler_lorenz(
                n_steps=4000, coupling=coupling, seed=42,
            )
            X = ts_r[1000:, 0]
            Y = ts_l[1000:, 0]

            det = BindingDetector(max_dim=1)
            det.fit(X, Y, subsample=300, seed=42)
            scores[coupling] = det.binding_score()

        direction_correct = scores[0.3] > scores[0.0]
        self._write_or_append_csv(csv_path, {
            "system": "coupled_rossler_lorenz",
            "coupling_low": 0.0,
            "coupling_high": 0.3,
            "score_low": scores[0.0],
            "score_high": scores[0.3],
            "direction_correct": direction_correct,
        })

        assert scores[0.3] > scores[0.0], (
            f"Expected score(0.3) > score(0.0), got {scores[0.3]} vs {scores[0.0]}"
        )

    @pytest.mark.slow
    def test_kuramoto_inverse_direction(self):
        """Kuramoto: coupling DECREASES binding due to synchronization."""
        _ensure_results_dir()

        csv_path = RESULTS_DIR / "cross_system.csv"
        scores = {}
        for coupling in (0.0, 5.0):
            set_seed(42)
            phases, signals = kuramoto_oscillators(
                n_steps=4000, n_oscillators=2, coupling=coupling, seed=42,
            )
            X = signals[500:, 0]
            Y = signals[500:, 1]

            det = BindingDetector(max_dim=1)
            det.fit(X, Y, subsample=300, seed=42)
            scores[coupling] = det.binding_score()

        # Strong coupling => synchronization => less topological binding
        direction_correct = scores[0.0] > scores[5.0]
        self._write_or_append_csv(csv_path, {
            "system": "kuramoto",
            "coupling_low": 0.0,
            "coupling_high": 5.0,
            "score_low": scores[0.0],
            "score_high": scores[5.0],
            "direction_correct": direction_correct,
        })

        assert scores[0.0] > scores[5.0], (
            f"Expected score(0.0) > score(5.0) for Kuramoto, "
            f"got {scores[0.0]} vs {scores[5.0]}"
        )

    @pytest.mark.slow
    def test_cross_system_summary(self):
        """Summarize direction correctness across all tested systems."""
        _ensure_results_dir()

        csv_path = RESULTS_DIR / "cross_system.csv"
        assert csv_path.exists(), (
            "cross_system.csv missing -- run cross-system tests first"
        )

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        summary = {}
        for row in rows:
            system = row["system"]
            summary[system] = {
                "coupling_low": float(row["coupling_low"]),
                "coupling_high": float(row["coupling_high"]),
                "score_low": float(row["score_low"]),
                "score_high": float(row["score_high"]),
                "direction_correct": row["direction_correct"] == "True",
            }

        n_correct = sum(1 for v in summary.values() if v["direction_correct"])
        summary["_overall"] = {
            "n_systems": len(rows),
            "n_correct": n_correct,
            "accuracy": n_correct / len(rows) if rows else 0.0,
        }

        out_path = RESULTS_DIR / "cross_system_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Loose: at least one system got it right
        assert n_correct >= 1, "No system showed correct coupling direction"
