#!/usr/bin/env python3
"""Cross-frequency topological binding screening.

Tests whether ATT's BindingDetector finds surrogate-significant coupling
between different EEG frequency bands from the same channel. Exploits ATT's
proven sensitivity to heterogeneous-timescale coupling.

Protocol per pair:
  1. Load one subject's preprocessed EEG (rivalry SSVEP dataset)
  2. Pick occipital channel Oz (or nearest available)
  3. Bandpass filter into theta (4-8 Hz), alpha (8-13 Hz), gamma (30-45 Hz)
  4. Run BindingDetector(max_dim=1, embedding_quality_gate=False) on each
     cross-frequency pair: theta-alpha, theta-gamma, alpha-gamma
  5. Surrogate test: phase-randomize one band, recompute binding, 15 surrogates
  6. Report z-score and p-value for 3 seeds per pair
  7. Same-band null control: theta-vs-theta from Oz vs O1

Pass criterion: z > 2 AND p < 0.05 on at least 2/3 seeds.

If EEG data is unavailable, falls back to synthetic Rossler systems at
different timescales, bandpassed to theta and gamma ranges.

Usage:
    python scripts/cross_freq_binding_screen.py
    python scripts/cross_freq_binding_screen.py --data-dir data/eeg/rivalry_ssvep
    python scripts/cross_freq_binding_screen.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

# ATT imports — reuse patterns from batch_eeg.py
from att.binding import BindingDetector
from att.surrogates import phase_randomize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBSAMPLE = 400
N_SURROGATES = 15
TRANSIENT_TRIM = 2000  # samples to discard after filtering
SEEDS = [0, 1, 2]

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "gamma": (30.0, 45.0),
}

CROSS_FREQ_PAIRS = [
    ("theta", "alpha"),
    ("theta", "gamma"),
    ("alpha", "gamma"),
]

# Occipital channels — Oz preferred, O1/O2 as fallback
OCCIPITAL_PRIORITY = ["Oz", "O1", "O2", "POz", "PO3", "PO4"]


# ---------------------------------------------------------------------------
# Data loading — reuses batch_eeg.py pattern
# ---------------------------------------------------------------------------

def load_first_subject_epoch(data_dir: Path) -> tuple[np.ndarray, list[str], int] | None:
    """Load the first available subject's rivalry epoch.

    Mirrors batch_eeg.py: discover_subjects → load_rivalry_epoch.
    Returns (epoch_data, channel_names, sfreq) or None.
    """
    import scipy.io

    if not data_dir.exists():
        return None

    for p in sorted(data_dir.iterdir()):
        if not p.is_dir():
            continue
        epochs_dir = p / "Epochs"
        if not epochs_dir.exists():
            continue

        mat_name = "csd_rejevs_icacomprem_gaprem_filt_rivindiff_riv_12.mat"
        mat_path = epochs_dir / mat_name
        if not mat_path.exists():
            continue

        eeg = scipy.io.loadmat(str(mat_path), simplify_cells=True)
        ch_names = [ch["labels"] for ch in eeg["chanlocs"]]
        sfreq = int(eeg["fs"])
        epochs = eeg["epochs"]

        if isinstance(epochs, np.ndarray) and epochs.ndim == 2:
            epoch_data = epochs
        elif isinstance(epochs, (list, np.ndarray)):
            epoch_data = epochs[0]
        else:
            continue

        print(f"Loaded subject: {p.name}")
        print(f"  Channels: {len(ch_names)}, Samples: {epoch_data.shape[1]}, sfreq: {sfreq} Hz")
        return epoch_data.astype(np.float64), ch_names, sfreq

    return None


def find_channel(ch_names: list[str], target: str) -> int | None:
    """Find channel index by name, case-insensitive."""
    target_lower = target.lower()
    for i, name in enumerate(ch_names):
        if name.lower() == target_lower:
            return i
    return None


def pick_occipital(ch_names: list[str], preferred: str = "Oz") -> tuple[int, str]:
    """Pick the best available occipital channel."""
    priority = [preferred] + [c for c in OCCIPITAL_PRIORITY if c != preferred]
    for name in priority:
        idx = find_channel(ch_names, name)
        if idx is not None:
            return idx, name
    raise ValueError(f"No occipital channel found. Available: {ch_names}")


# ---------------------------------------------------------------------------
# Signal processing — matches batch_eeg.py bandpass_filter
# ---------------------------------------------------------------------------

def bandpass_filter(signal: np.ndarray, low: float, high: float, sfreq: float,
                    order: int = 4) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, signal)


def extract_band(signal: np.ndarray, band_name: str, sfreq: float) -> np.ndarray:
    """Bandpass filter and trim transient."""
    low, high = BANDS[band_name]
    filtered = bandpass_filter(signal, low, high, sfreq)
    return filtered[TRANSIENT_TRIM:]


# ---------------------------------------------------------------------------
# Synthetic fallback — two Rossler systems at different timescales
# ---------------------------------------------------------------------------

def generate_synthetic_bands(seed: int = 42) -> dict:
    """Generate synthetic cross-frequency signals from Rossler systems.

    Two independent Rossler attractors at different dt values, bandpassed
    to theta and gamma ranges. Simulates heterogeneous-timescale coupling
    without actual EEG data.
    """
    from att.synthetic import rossler_system

    # Slow Rossler → theta-like (dt=0.05, lower frequency content)
    slow = rossler_system(n_steps=50000, dt=0.05, seed=seed)
    # Fast Rossler → gamma-like (dt=0.005, higher frequency content)
    fast = rossler_system(n_steps=50000, dt=0.005, seed=seed + 100)

    # Use x-component, normalize
    slow_x = slow[:, 0]
    fast_x = fast[:, 0]
    slow_x = (slow_x - slow_x.mean()) / slow_x.std()
    fast_x = (fast_x - fast_x.mean()) / fast_x.std()

    # Trim to common length and discard transient
    n = min(len(slow_x), len(fast_x))
    slow_x = slow_x[TRANSIENT_TRIM:n]
    fast_x = fast_x[TRANSIENT_TRIM:n]

    return {
        "theta": slow_x,
        "gamma": fast_x,
        "sfreq": None,  # synthetic, no real sfreq
        "mode": "synthetic",
    }


# ---------------------------------------------------------------------------
# Core: binding + surrogate test for one pair
# ---------------------------------------------------------------------------

def run_binding_surrogate(
    band_a: np.ndarray,
    band_b: np.ndarray,
    seed: int,
) -> dict:
    """Run BindingDetector + surrogate test on a single pair/seed.

    Returns dict with observed score, z-score, p-value, surrogate stats.
    """
    bd = BindingDetector(max_dim=1, embedding_quality_gate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bd.fit(band_a, band_b, subsample=SUBSAMPLE, seed=seed)

    result = bd.test_significance(
        n_surrogates=N_SURROGATES,
        method="phase_randomize",
        seed=seed,
        subsample=SUBSAMPLE,
    )

    return {
        "observed": result["observed_score"],
        "surr_mean": result["surrogate_mean"],
        "surr_std": result["surrogate_std"],
        "z": result["z_score"],
        "p": result["p_value"],
    }


# ---------------------------------------------------------------------------
# Main screening logic
# ---------------------------------------------------------------------------

def run_screen(bands_data: dict[str, np.ndarray], label: str = "EEG") -> list[dict]:
    """Run cross-frequency screening on extracted band signals.

    Returns list of result rows for the table.
    """
    rows = []

    for band_a_name, band_b_name in CROSS_FREQ_PAIRS:
        if band_a_name not in bands_data or band_b_name not in bands_data:
            continue

        band_a = bands_data[band_a_name]
        band_b = bands_data[band_b_name]

        # Trim to common length
        n = min(len(band_a), len(band_b))
        band_a = band_a[:n]
        band_b = band_b[:n]

        pair_label = f"{band_a_name}-{band_b_name}"
        print(f"\n  {pair_label} (n={n})...", end="", flush=True)

        for seed in SEEDS:
            t0 = time.time()
            result = run_binding_surrogate(band_a, band_b, seed)
            elapsed = time.time() - t0
            print(f" s{seed}={elapsed:.1f}s", end="", flush=True)

            rows.append({
                "pair": pair_label,
                "type": "cross-freq",
                "source": label,
                "seed": seed,
                **result,
            })

    return rows


def run_null_control(
    epoch_data: np.ndarray,
    ch_names: list[str],
    sfreq: float,
) -> list[dict]:
    """Same-band null control: theta(Oz) vs theta(O1).

    Same timescale from nearby electrodes — should FAIL, matching
    the Lorenz-Lorenz zero-power finding.
    """
    idx_oz = find_channel(ch_names, "Oz")
    idx_o1 = find_channel(ch_names, "O1")

    if idx_oz is None or idx_o1 is None:
        print("\n  [SKIP] Null control: need both Oz and O1")
        return []

    theta_oz = extract_band(epoch_data[idx_oz], "theta", sfreq)
    theta_o1 = extract_band(epoch_data[idx_o1], "theta", sfreq)
    n = min(len(theta_oz), len(theta_o1))
    theta_oz = theta_oz[:n]
    theta_o1 = theta_o1[:n]

    rows = []
    pair_label = "theta(Oz)-theta(O1)"
    print(f"\n  {pair_label} (n={n})...", end="", flush=True)

    for seed in SEEDS:
        t0 = time.time()
        result = run_binding_surrogate(theta_oz, theta_o1, seed)
        elapsed = time.time() - t0
        print(f" s{seed}={elapsed:.1f}s", end="", flush=True)

        rows.append({
            "pair": pair_label,
            "type": "null-control",
            "source": "EEG",
            "seed": seed,
            **result,
        })

    return rows


# ---------------------------------------------------------------------------
# Output formatting — matches cone_diag_p1.py table style
# ---------------------------------------------------------------------------

def print_results_table(rows: list[dict]) -> None:
    """Print results as a formatted table."""
    print("\n")
    print("=" * 95)
    print("CROSS-FREQUENCY BINDING SCREEN — RESULTS")
    print("=" * 95)
    print(f"{'Pair':<22} {'Type':<14} {'Seed':>4}  {'Observed':>10} "
          f"{'Surr Mean':>10} {'Surr Std':>9} {'z':>7} {'p':>7} {'Sig':>4}")
    print("-" * 95)

    for r in rows:
        sig = "*" if (r["z"] > 2 and r["p"] < 0.05) else ""
        print(f"{r['pair']:<22} {r['type']:<14} {r['seed']:>4}  "
              f"{r['observed']:>10.1f} {r['surr_mean']:>10.1f} {r['surr_std']:>9.1f} "
              f"{r['z']:>7.2f} {r['p']:>7.3f} {sig:>4}")

    print("-" * 95)


def print_verdict(rows: list[dict]) -> None:
    """Print VERDICT section: which pairs passed, which failed."""
    print("\n" + "=" * 95)
    print("VERDICT")
    print("=" * 95)

    # Group by pair
    pairs = {}
    for r in rows:
        key = r["pair"]
        if key not in pairs:
            pairs[key] = {"type": r["type"], "results": []}
        pairs[key]["results"].append(r)

    passed_pairs = []
    failed_pairs = []

    for pair_name, info in pairs.items():
        n_pass = sum(1 for r in info["results"] if r["z"] > 2 and r["p"] < 0.05)
        n_total = len(info["results"])
        verdict = "PASS" if n_pass >= 2 else "FAIL"

        status_line = f"  {pair_name:<25} {n_pass}/{n_total} seeds passed → {verdict}"
        print(status_line)

        if verdict == "PASS":
            passed_pairs.append((pair_name, info["type"]))
        else:
            failed_pairs.append((pair_name, info["type"]))

    print()

    # Summary
    cross_freq_passed = [p for p, t in passed_pairs if t == "cross-freq"]
    null_passed = [p for p, t in passed_pairs if t == "null-control"]
    cross_freq_failed = [p for p, t in failed_pairs if t == "cross-freq"]
    null_failed = [p for p, t in failed_pairs if t == "null-control"]

    if cross_freq_passed:
        print(f"CROSS-FREQUENCY PAIRS THAT PASSED: {', '.join(cross_freq_passed)}")
        print("  → These show surrogate-significant topological binding between")
        print("    frequency bands with different timescales. Consistent with ATT's")
        print("    heterogeneous-timescale sensitivity.")
    else:
        print("NO cross-frequency pairs passed surrogate testing.")
        print("  → BindingDetector does not find significant cross-frequency coupling")
        print("    at this sample size / surrogate count. May need longer epochs,")
        print("    more surrogates, or a different measurement approach.")

    if null_failed:
        print(f"\nNULL CONTROL (same-band): {', '.join(null_failed)} → FAIL (as expected)")
        print("  → Same-timescale signals from nearby electrodes show no excess binding.")
        print("    Validates that any cross-frequency signal is not just spatial proximity.")
    elif null_passed:
        print(f"\nWARNING: Null control PASSED: {', '.join(null_passed)}")
        print("  → Same-band same-timescale shows significant binding. This undermines")
        print("    cross-frequency specificity — the signal may be spatial, not spectral.")

    # Return passed cross-freq pairs for timecourse script decision
    return cross_freq_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-frequency binding screen")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/eeg/rivalry_ssvep"),
        help="Path to rivalry SSVEP dataset",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Force synthetic mode (skip EEG data)",
    )
    args = parser.parse_args()

    print("=" * 95)
    print("CROSS-FREQUENCY TOPOLOGICAL BINDING SCREEN")
    print(f"  subsample={SUBSAMPLE}, n_surrogates={N_SURROGATES}, "
          f"seeds={SEEDS}, transient_trim={TRANSIENT_TRIM}")
    print("=" * 95)

    t_start = time.time()
    all_rows = []
    used_eeg = False

    # --- Try real EEG data first ---
    if not args.synthetic:
        result = load_first_subject_epoch(args.data_dir)
        if result is not None:
            epoch_data, ch_names, sfreq = result
            used_eeg = True

            # Pick Oz
            oz_idx, oz_name = pick_occipital(ch_names)
            print(f"  Using channel: {oz_name} (index {oz_idx})")

            # Extract bands from Oz
            raw_oz = epoch_data[oz_idx]
            bands_data = {}
            for band_name in BANDS:
                bands_data[band_name] = extract_band(raw_oz, band_name, sfreq)
                print(f"  {band_name}: {len(bands_data[band_name])} samples "
                      f"({BANDS[band_name][0]}-{BANDS[band_name][1]} Hz)")

            # Cross-frequency pairs
            print("\n--- Cross-Frequency Pairs ---")
            all_rows.extend(run_screen(bands_data, label="EEG"))

            # Null control
            print("\n\n--- Same-Band Null Control ---")
            all_rows.extend(run_null_control(epoch_data, ch_names, sfreq))

    # --- Synthetic fallback ---
    if not used_eeg:
        print("\nEEG data not available. Using synthetic Rossler fallback.")
        synth = generate_synthetic_bands(seed=42)
        bands_data = {k: v for k, v in synth.items() if isinstance(v, np.ndarray)}
        print(f"  theta (slow Rossler): {len(bands_data['theta'])} samples")
        print(f"  gamma (fast Rossler): {len(bands_data['gamma'])} samples")

        print("\n--- Synthetic Cross-Frequency Pairs ---")
        # Only theta-gamma available in synthetic mode
        all_rows.extend(run_screen(bands_data, label="synthetic"))

    # --- Print results ---
    print_results_table(all_rows)
    passed = print_verdict(all_rows)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")

    if passed:
        print(f"\nPassing pairs: {passed}")
        print("→ Run scripts/cross_freq_binding_timecourse.py for sliding-window analysis")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
