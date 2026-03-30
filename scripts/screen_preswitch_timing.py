#!/usr/bin/env python3
"""Screen: How far in advance does topology change before a perceptual switch?

Runs sliding-window PH on rivalry EEG and measures the temporal relationship
between topology change peaks (image_distance maxima) and behavioral switch
events. If topology genuinely tracks perceptual state, the peak should precede
the button press.

Protocol:
  1. Load rivalry EEG (Oz) + behavioral switch events
  2. Bandpass 4-13 Hz, embed via embed_channel
  3. Parallel sliding-window PH (window=500, step=50, max_dim=1, subsample=200)
  4. For each switch: find peak image_distance in [-3s, +1s] window
  5. Compute lag = peak_time - switch_time (negative = topology precedes)
  6. One-sample t-test on lags against zero
  7. Null control: same analysis with random timepoints (should center at zero)

Pass criterion: mean lag < 0, t-test p < 0.05, AND real lags significantly
different from random-timepoint lags (two-sample t-test p < 0.05).

Usage:
    python scripts/screen_preswitch_timing.py
    python scripts/screen_preswitch_timing.py --data-dir data/eeg/rivalry_ssvep
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import butter, sosfilt
from scipy.stats import ttest_1samp, ttest_ind
from tqdm import tqdm

from att.neuro.embedding import embed_channel
from att.topology.persistence import PersistenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TD_WINDOW_SIZE = 500
TD_STEP_SIZE = 50
TD_MAX_DIM = 1
TD_SUBSAMPLE = 200       # 200 not 300: ripser 0.16s vs 0.35s per call
BANDPASS_LOW = 4
BANDPASS_HIGH = 13
BANDPASS_ORDER = 4
PRE_WINDOW_S = 3.0
POST_WINDOW_S = 1.0
N_NULL_REPEATS = 3
SEED = 42
MAX_SUBJECTS = 3
CONDITION_SUFFIX = "riv_12"
PARAM_SET = 2
EPOCH_INDEX = 0
DEFAULT_DATA_DIR = Path("data/eeg/rivalry_ssvep")
N_JOBS = min(16, mp.cpu_count())


# ---------------------------------------------------------------------------
# Data loading (from batch_eeg.py patterns)
# ---------------------------------------------------------------------------

def discover_subjects(data_dir: Path) -> list[dict]:
    subjects = []
    if not data_dir.exists():
        return subjects
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir():
            continue
        epochs_dir = p / "Epochs"
        behavior_dir = p / "Behavior"
        if epochs_dir.exists():
            subjects.append({
                "name": p.name,
                "path": p,
                "epochs_dir": epochs_dir,
                "behavior_dir": behavior_dir,
            })
    return subjects


def load_rivalry_epoch(
    epochs_dir: Path, condition_suffix: str, epoch_index: int = 0,
) -> tuple[np.ndarray, list[str], int] | None:
    prefix = "csd_rejevs_icacomprem_gaprem_filt_rivindiff_"
    mat_path = epochs_dir / f"{prefix}{condition_suffix}.mat"
    if not mat_path.exists():
        return None
    eeg = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    ch_names = [ch["labels"] for ch in eeg["chanlocs"]]
    sfreq = int(eeg["fs"])
    epochs = eeg["epochs"]
    if isinstance(epochs, np.ndarray) and epochs.ndim == 2:
        epoch_data = epochs
    elif isinstance(epochs, (list, np.ndarray)):
        if epoch_index >= len(epochs):
            return None
        epoch_data = epochs[epoch_index]
    else:
        return None
    return epoch_data.astype(np.float64), ch_names, sfreq


def load_behavioral_switches(
    behavior_dir: Path, param_set: int, sfreq: int,
) -> list[dict] | None:
    if not behavior_dir.exists():
        return None
    beh_files = [f for f in behavior_dir.glob("BR_Rivalry_*.mat")
                 if "PRACT" not in f.name]
    if not beh_files:
        return None
    try:
        beh = scipy.io.loadmat(str(beh_files[0]), simplify_cells=True)
        results_beh = beh["results"]
    except Exception:
        return None
    matching = []
    for i, r in enumerate(results_beh):
        try:
            if r["params"]["paramSet"] == param_set:
                matching.append(i)
        except (KeyError, TypeError):
            continue
    if not matching:
        return None
    r = results_beh[matching[0]]
    psycho = r["psycho"]
    t_key = psycho["tKeyPress"]
    resp_key = psycho["responseKey"]
    switches = []
    for i in range(1, len(resp_key)):
        if resp_key[i] != resp_key[i - 1]:
            switches.append({
                "time": float(t_key[i]),
                "sample": int(t_key[i] * sfreq),
            })
    return switches


def bandpass_filter(signal, low, high, sfreq, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, signal)


def find_channel(ch_names, target="Oz"):
    for i, name in enumerate(ch_names):
        if name == target:
            return i
    for i, name in enumerate(ch_names):
        if name.startswith(target):
            return i
    return None


# ---------------------------------------------------------------------------
# Parallel sliding-window PH
# ---------------------------------------------------------------------------

def _ph_worker(args):
    """Worker: compute PH on one pre-embedded window."""
    cloud, max_dim, subsample, seed = args
    pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser")
    pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return pa.diagrams_


def parallel_windowed_ph_2d(
    cloud: np.ndarray,
    window_size: int,
    step_size: int,
    max_dim: int,
    subsample: int,
    seed: int,
    n_jobs: int = N_JOBS,
) -> dict:
    """Sliding-window PH on pre-embedded cloud, parallel ripser.

    Returns dict with image_distances, window_centers.
    """
    n_points = len(cloud)
    window_starts = list(range(0, n_points - window_size + 1, step_size))
    window_centers = np.array([s + window_size // 2 for s in window_starts])

    # Slice windows (fast)
    windows = [cloud[s : s + window_size] for s in window_starts]

    # Parallel PH
    args = [(w, max_dim, subsample, seed) for w in windows]
    with mp.Pool(n_jobs) as pool:
        all_diagrams = list(tqdm(
            pool.imap(_ph_worker, args),
            total=len(args),
            desc="  PH windows",
        ))

    # Shared birth/persistence ranges
    all_births = []
    all_persistences = []
    for dgms in all_diagrams:
        for dgm in dgms:
            if len(dgm) > 0:
                all_births.extend(dgm[:, 0].tolist())
                pers = dgm[:, 1] - dgm[:, 0]
                all_persistences.extend(pers[pers > 1e-10].tolist())

    if all_births and all_persistences:
        birth_range = (min(all_births), max(all_births))
        persistence_range = (0.0, max(all_persistences))
    else:
        birth_range = (0.0, 1.0)
        persistence_range = (0.0, 1.0)

    # Shared images (fast, sequential)
    shared_images = []
    for dgms in all_diagrams:
        pa = PersistenceAnalyzer(max_dim=max_dim)
        pa.diagrams_ = dgms
        imgs = pa.to_image(birth_range=birth_range, persistence_range=persistence_range)
        shared_images.append(imgs)

    # Consecutive L2 image distances
    image_distances = []
    for i in range(len(shared_images) - 1):
        dist = sum(
            float(np.sqrt(np.sum((shared_images[i][d] - shared_images[i + 1][d]) ** 2)))
            for d in range(max_dim + 1)
        )
        image_distances.append(dist)

    return {
        "image_distances": np.array(image_distances),
        "window_centers": window_centers,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_timing(signal: np.ndarray, switches: list[dict], sfreq: int) -> dict:
    """Run pre-switch timing analysis on one subject."""
    # Bandpass
    filtered = bandpass_filter(signal, BANDPASS_LOW, BANDPASS_HIGH, sfreq, BANDPASS_ORDER)

    # Embed full signal
    cloud, meta = embed_channel(filtered, band="theta_alpha", sfreq=float(sfreq))
    delay = meta["delay"]
    dim = meta["dimension"]
    print(f"  embed: delay={delay}, dim={dim}, cloud={cloud.shape}")

    # Parallel sliding-window PH on pre-embedded cloud
    result = parallel_windowed_ph_2d(
        cloud, TD_WINDOW_SIZE, TD_STEP_SIZE, TD_MAX_DIM, TD_SUBSAMPLE, SEED,
    )

    image_distances = result["image_distances"]
    window_centers = result["window_centers"]

    # Distance positions (midpoints between consecutive windows)
    dist_x = (window_centers[:-1] + window_centers[1:]) / 2.0
    # cloud index ≈ signal sample (offset is (dim-1)*delay/2 ≈ 42ms, negligible)
    dist_times = dist_x / sfreq

    # For each switch, find peak image_distance in [-3s, +1s]
    real_lags = _compute_lags(
        image_distances, dist_times, switches, sfreq,
        PRE_WINDOW_S, POST_WINDOW_S,
    )

    if len(real_lags) < 3:
        return {"error": f"too few usable events ({len(real_lags)})"}

    # One-sample t-test: are lags < 0?
    t_stat, p_onesamp = ttest_1samp(real_lags, 0)
    p_onesided = p_onesamp / 2 if t_stat < 0 else 1 - p_onesamp / 2

    # Null control: random timepoints
    rng = np.random.default_rng(SEED)
    valid_lo = PRE_WINDOW_S
    valid_hi = dist_times[-1] - POST_WINDOW_S
    null_lags_all = []
    for _ in range(N_NULL_REPEATS):
        fake_switches = [
            {"time": t, "sample": int(t * sfreq)}
            for t in rng.uniform(valid_lo, valid_hi, size=len(switches))
        ]
        null_lags = _compute_lags(
            image_distances, dist_times, fake_switches, sfreq,
            PRE_WINDOW_S, POST_WINDOW_S,
        )
        null_lags_all.extend(null_lags)
    null_lags_all = np.array(null_lags_all)

    null_mean = float(null_lags_all.mean()) if len(null_lags_all) > 0 else 0.0

    # Two-sample test: real vs null lags
    if len(null_lags_all) >= 3:
        _, p_twosamp = ttest_ind(real_lags, null_lags_all)
    else:
        p_twosamp = 1.0

    n_preceding = int(np.sum(real_lags < 0))
    frac_preceding = n_preceding / len(real_lags)

    return {
        "n_events": len(real_lags),
        "n_switches": len(switches),
        "mean_lag_ms": float(real_lags.mean()) * 1000,
        "median_lag_ms": float(np.median(real_lags)) * 1000,
        "std_lag_ms": float(real_lags.std()) * 1000,
        "frac_preceding": frac_preceding,
        "ttest_p": float(p_onesided),
        "null_mean_lag_ms": null_mean * 1000,
        "null_p": float(p_twosamp),
        "embedding_delay": delay,
        "embedding_dim": dim,
        "n_windows": len(window_centers),
    }


def _compute_lags(
    image_distances: np.ndarray,
    dist_times: np.ndarray,
    switches: list[dict],
    sfreq: int,
    pre_s: float,
    post_s: float,
) -> np.ndarray:
    lags = []
    for sw in switches:
        sw_time = sw["time"]
        lo = sw_time - pre_s
        hi = sw_time + post_s
        mask = (dist_times >= lo) & (dist_times <= hi)
        if mask.sum() == 0:
            continue
        window_dists = image_distances[mask]
        window_times = dist_times[mask]
        peak_idx = np.argmax(window_dists)
        peak_time = window_times[peak_idx]
        lag = peak_time - sw_time
        lags.append(lag)
    return np.array(lags) if lags else np.array([])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results_table(rows: list[dict]) -> None:
    hdr = (f"{'Subject':<12} {'n_evt':>5} {'mean_ms':>8} {'med_ms':>8} "
           f"{'frac<0':>7} {'ttest_p':>8} {'null_ms':>8} {'null_p':>8} {'verdict':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['subject']:<12} {'---':>5} {'---':>8} {'---':>8} "
                  f"{'---':>7} {'---':>8} {'---':>8} {'---':>8} {'SKIP':>7}")
            continue
        passed = (r["mean_lag_ms"] < 0 and r["ttest_p"] < 0.05 and r["null_p"] < 0.05)
        v = "PASS" if passed else "FAIL"
        print(f"{r['subject']:<12} {r['n_events']:>5} {r['mean_lag_ms']:>8.0f} "
              f"{r['median_lag_ms']:>8.0f} {r['frac_preceding']:>7.2f} "
              f"{r['ttest_p']:>8.3f} {r['null_mean_lag_ms']:>8.0f} "
              f"{r['null_p']:>8.3f} {v:>7}")


def print_verdict(rows: list[dict]) -> None:
    valid = [r for r in rows if "error" not in r]
    if not valid:
        print("\nVERDICT: NO DATA")
        return

    all_mean = np.mean([r["mean_lag_ms"] for r in valid])
    all_frac = np.mean([r["frac_preceding"] for r in valid])
    n_pass = sum(1 for r in valid
                 if r["mean_lag_ms"] < 0 and r["ttest_p"] < 0.05 and r["null_p"] < 0.05)

    overall = "PASS" if n_pass >= max(1, int(np.ceil(len(valid) * 2 / 3))) else "FAIL"

    print()
    print("=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"Pre-switch timing: {overall}")
    print(f"  Subjects passing: {n_pass}/{len(valid)}")
    print(f"  Criterion: mean_lag<0 AND ttest_p<0.05 AND null_p<0.05")
    print(f"  Grand mean lag: {all_mean:.0f} ms")
    print(f"  Grand fraction preceding: {all_frac:.2f}")
    if all_mean < 0:
        print(f"  Topology peaks precede button press by ~{abs(all_mean):.0f} ms on average")
    else:
        print(f"  Topology peaks FOLLOW button press by ~{all_mean:.0f} ms — no predictive signal")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Screen: pre-switch timing")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Data dir: {data_dir}")
    print(f"Parallel workers: {N_JOBS}")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    subjects = discover_subjects(data_dir)
    if not subjects:
        print("ERROR: No subjects found.")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects, using first {MAX_SUBJECTS}")
    subjects = subjects[:MAX_SUBJECTS]

    rows = []
    for si, subj in enumerate(subjects):
        t0 = time.time()
        name = subj["name"][:12]
        print(f"\n--- Subject {si + 1}/{len(subjects)}: {name} ---")

        result = load_rivalry_epoch(subj["epochs_dir"], CONDITION_SUFFIX, EPOCH_INDEX)
        if result is None:
            rows.append({"subject": name, "error": "no epoch"})
            continue
        epoch_data, ch_names, sfreq = result

        ch_idx = find_channel(ch_names, "Oz")
        if ch_idx is None:
            rows.append({"subject": name, "error": "no Oz"})
            continue
        signal = epoch_data[ch_idx]
        print(f"  Channel: {ch_names[ch_idx]}, {len(signal)} samples @ {sfreq} Hz")

        switches = load_behavioral_switches(subj["behavior_dir"], PARAM_SET, sfreq)
        if switches is None:
            rows.append({"subject": name, "error": "no behavior"})
            continue
        print(f"  Switches: {len(switches)}")

        r = run_timing(signal, switches, sfreq)
        r["subject"] = name
        if "error" in r:
            print(f"  Skipping: {r['error']}")
        else:
            print(f"  mean_lag={r['mean_lag_ms']:.0f}ms, frac<0={r['frac_preceding']:.2f}, "
                  f"p={r['ttest_p']:.3f}")
        rows.append(r)
        print(f"  Time: {time.time() - t0:.1f}s")

    print("\n")
    print_results_table(rows)
    print_verdict(rows)

    valid = [r for r in rows if "error" not in r]
    n_pass = sum(1 for r in valid
                 if r["mean_lag_ms"] < 0 and r["ttest_p"] < 0.05 and r["null_p"] < 0.05)
    threshold = max(1, int(np.ceil(len(valid) * 2 / 3)))
    sys.exit(0 if n_pass >= threshold else 1)


if __name__ == "__main__":
    main()
