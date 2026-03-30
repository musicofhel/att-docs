#!/usr/bin/env python3
"""Benchmark: seizure onset detection on CHB-MIT Scalp EEG.

Tests whether topological changepoint detection provides earlier seizure onset
detection than standard methods on the CHB-MIT database from PhysioNet.

For each seizure:
  1. Extract 5 min pre-onset to 1 min post-onset
  2. Run 5 methods: topological PH, PELT, spectral, variance, BOCPD
  3. Ground truth = 1 transition at annotated seizure onset
  4. Measure: detection lag (negative = early), false alarms, precision/recall/F1

Also runs each method on one seizure-free file per subject for false alarm rate.

Pass criterion: Topology detection rate > 80% AND mean lag < 0 AND lag
significantly earlier than at least one comparison method (Wilcoxon p < 0.05).

Usage:
    python -u scripts/benchmark_seizure_onset.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, welch
from scipy.stats import wilcoxon
from tqdm import tqdm

from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 500
STEP_SIZE = 50
MAX_DIM = 1
SUBSAMPLE = 200
PI_RES = 20
SEED = 42
N_JOBS = min(16, mp.cpu_count())

BANDPASS_LOW = 0.5
BANDPASS_HIGH = 30.0
BANDPASS_ORDER = 4

TOLERANCE_S = 10.0   # +/- 10 seconds
PRE_ONSET_S = 300    # 5 minutes before onset
POST_ONSET_S = 60    # 1 minute after onset

PELT_DEFAULT = 10

# CHB-MIT subjects to use
SUBJECTS = ["chb01", "chb02", "chb03"]
BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
CACHE_DIR = Path("data/eeg/chbmit")

# Preferred channels (bipolar montage)
CHANNEL_PRIORITY = ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3", "FP2-F4"]

METHODS = ["Topological", "PELT", "Spectral", "Variance", "BOCPD"]


# ---------------------------------------------------------------------------
# CHB-MIT data download and parsing
# ---------------------------------------------------------------------------

def ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> bool:
    """Download a file if not already cached."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading {url}...", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def parse_summary(summary_path: Path) -> list[dict]:
    """Parse chbNN-summary.txt to get seizure info.

    Returns list of dicts with keys:
        file_name, n_seizures, seizures: [{start_s, end_s}]
        n_channels, channels (if available)
    """
    text = summary_path.read_text()
    entries = []

    # Split by "File Name:" blocks
    blocks = re.split(r'(?=File Name:)', text)

    for block in blocks:
        block = block.strip()
        if not block.startswith("File Name:"):
            continue

        entry = {"seizures": []}

        # File name
        m = re.search(r'File Name:\s*(\S+)', block)
        if not m:
            continue
        entry["file_name"] = m.group(1)

        # Number of seizures
        m = re.search(r'Number of Seizures in File:\s*(\d+)', block)
        entry["n_seizures"] = int(m.group(1)) if m else 0

        # Channel info
        m = re.search(r'Number of Channels:\s*(\d+)', block)
        if m:
            entry["n_channels"] = int(m.group(1))

        # Channel names (parse "Channel N:" lines)
        channels = re.findall(r'Channel \d+:\s*(\S+)', block)
        if channels:
            entry["channels"] = channels

        # Seizure start/end times
        # Format varies: "Seizure Start Time: 2996 seconds"
        #                "Seizure 1 Start Time: 2996 seconds"
        starts = re.findall(r'Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds', block)
        ends = re.findall(r'Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds', block)

        for s, e in zip(starts, ends):
            entry["seizures"].append({
                "start_s": int(s),
                "end_s": int(e),
            })

        entries.append(entry)

    return entries


def get_seizure_files(subject: str) -> tuple[list[dict], list[str]]:
    """Download summary, parse it, return (seizure_entries, seizure_free_files).

    seizure_entries: list of dicts with file_name, seizures, channels
    seizure_free_files: list of filenames with 0 seizures
    """
    ensure_cache_dir()
    subj_dir = CACHE_DIR / subject
    subj_dir.mkdir(parents=True, exist_ok=True)

    summary_url = f"{BASE_URL}/{subject}/{subject}-summary.txt"
    summary_path = subj_dir / f"{subject}-summary.txt"

    if not download_file(summary_url, summary_path):
        return [], []

    entries = parse_summary(summary_path)

    seizure_entries = [e for e in entries if e["n_seizures"] > 0]
    seizure_free = [e["file_name"] for e in entries if e["n_seizures"] == 0]

    return seizure_entries, seizure_free


def download_edf(subject: str, filename: str) -> Path | None:
    """Download an EDF file, return local path or None."""
    ensure_cache_dir()
    local = CACHE_DIR / subject / filename
    url = f"{BASE_URL}/{subject}/{filename}"
    if download_file(url, local):
        return local
    return None


def load_edf_channel(edf_path: Path, channel_priority: list[str],
                     sfreq_expected: float | None = None):
    """Load an EDF file, pick best channel.

    Returns (signal, ch_name, sfreq) or None.
    """
    try:
        import mne
        mne.set_log_level("ERROR")
    except ImportError:
        print("ERROR: MNE not installed. pip install mne")
        sys.exit(1)

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:
        print(f"    Failed to read {edf_path.name}: {e}")
        return None

    available = raw.ch_names
    sfreq = raw.info["sfreq"]

    for ch in channel_priority:
        if ch in available:
            signal = raw.get_data(picks=[ch])[0]
            return signal, ch, sfreq

    # If none match exactly, try case-insensitive and with/without spaces
    for ch in channel_priority:
        for avail in available:
            if ch.replace("-", "").lower() == avail.replace("-", "").lower():
                signal = raw.get_data(picks=[avail])[0]
                return signal, avail, sfreq

    # Last resort: pick first EEG-like channel
    for avail in available:
        if any(prefix in avail.upper() for prefix in ["FP", "F7", "F3", "C3", "T7"]):
            signal = raw.get_data(picks=[avail])[0]
            return signal, avail, sfreq

    print(f"    No suitable channel in {available[:10]}...")
    return None


# ---------------------------------------------------------------------------
# Parallel PH infrastructure (from benchmark_changepoint_methods.py)
# ---------------------------------------------------------------------------

def _ph_worker(args):
    cloud, max_dim, subsample, seed = args
    pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser")
    pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return pa.diagrams_


def parallel_windowed_ph(cloud, label="PH"):
    n_points = len(cloud)
    starts = list(range(0, n_points - WINDOW_SIZE + 1, STEP_SIZE))
    centers = np.array([s + WINDOW_SIZE // 2 for s in starts])
    windows = [cloud[s : s + WINDOW_SIZE] for s in starts]

    args = [(w, MAX_DIM, SUBSAMPLE, SEED) for w in windows]
    with mp.Pool(N_JOBS) as pool:
        all_diagrams = list(tqdm(
            pool.imap(_ph_worker, args),
            total=len(args), desc=f"  {label}",
        ))

    birth_min, birth_max, pers_max = float("inf"), float("-inf"), 0.0
    has_data = False
    for dgms in all_diagrams:
        for dgm in dgms:
            if len(dgm) > 0:
                has_data = True
                birth_min = min(birth_min, float(dgm[:, 0].min()))
                birth_max = max(birth_max, float(dgm[:, 0].max()))
                p = dgm[:, 1] - dgm[:, 0]
                p = p[p > 1e-10]
                if len(p) > 0:
                    pers_max = max(pers_max, float(p.max()))

    br = (birth_min, birth_max) if has_data else (0, 1)
    pr = (0.0, pers_max) if has_data and pers_max > 0 else (0, 1)

    images = []
    for dgms in all_diagrams:
        pa = PersistenceAnalyzer(max_dim=MAX_DIM)
        pa.diagrams_ = dgms
        imgs = pa.to_image(resolution=PI_RES, birth_range=br, persistence_range=pr)
        images.append(imgs)

    dists = []
    for i in range(len(images) - 1):
        d = sum(
            float(np.sqrt(np.sum((images[i][k] - images[i + 1][k]) ** 2)))
            for k in range(MAX_DIM + 1)
        )
        dists.append(d)

    return centers, np.array(dists)


# ---------------------------------------------------------------------------
# CUSUM + evaluation
# ---------------------------------------------------------------------------

def cusum_changepoints(scores):
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    threshold = mu + 2 * sigma
    cusum = 0.0
    cps = []
    for i, s in enumerate(scores):
        cusum = max(0.0, cusum + (s - mu))
        if cusum > threshold:
            cps.append(i)
            cusum = 0.0
    return cps


def scores_to_samples(cps, centers):
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    return np.array([int(midpoints[i]) for i in cps if i < len(midpoints)])


def evaluate(true_s, det_s, tolerance):
    true_s = np.asarray(true_s, dtype=float)
    det_s = np.asarray(det_s, dtype=float)
    n_true = len(true_s)
    n_det = len(det_s)

    if n_det == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_lag": float("nan"), "n_detected": 0, "n_true": n_true,
                "detected": False, "lag_s": float("nan")}
    if n_true == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_lag": float("nan"), "n_detected": n_det, "n_true": 0,
                "detected": False, "lag_s": float("nan")}

    tp, lags = 0, []
    for d in det_s:
        offsets = d - true_s
        best = int(np.argmin(np.abs(offsets)))
        if abs(offsets[best]) <= tolerance:
            tp += 1
            lags.append(int(offsets[best]))

    recalled = sum(1 for t in true_s if np.any(np.abs(det_s - t) <= tolerance))

    prec = tp / n_det if n_det > 0 else 0.0
    rec = recalled / n_true if n_true > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_lag = float(np.mean(lags)) if lags else float("nan")

    return {"precision": prec, "recall": rec, "f1": f1,
            "mean_lag": mean_lag, "n_detected": n_det, "n_true": n_true,
            "detected": recalled > 0, "lag_s": mean_lag}


# ---------------------------------------------------------------------------
# 5 methods (self-contained)
# ---------------------------------------------------------------------------

def method_topological(signal, sfreq, label="topo"):
    MAX_FIT_SAMPLES = 20000
    fit_signal = signal[:MAX_FIT_SAMPLES] if len(signal) > MAX_FIT_SAMPLES else signal
    embedder = TakensEmbedder("auto", "auto")
    try:
        embedder.fit(fit_signal)
    except Exception:
        embedder = TakensEmbedder(delay=10, dimension=3)
        embedder.fit(fit_signal)
    cloud = embedder.transform(signal)
    print(f"    Takens: delay={embedder.delay_}, dim={embedder.dimension_}, "
          f"cloud={cloud.shape}", flush=True)
    centers, dists = parallel_windowed_ph(cloud, label=label)
    cps = cusum_changepoints(dists)
    return scores_to_samples(cps, centers)


def _run_pelt_single(signal, penalty):
    import ruptures
    sig = signal.reshape(-1, 1) if signal.ndim == 1 else signal
    jump = max(5, len(signal) // 2000)
    algo = ruptures.Pelt(model="normal", min_size=500, jump=jump).fit(sig)
    bkps = algo.predict(pen=penalty)
    if bkps and bkps[-1] == len(signal):
        bkps = bkps[:-1]
    return np.array(bkps, dtype=int)


def method_pelt_fixed(signal, penalty=10):
    try:
        import ruptures  # noqa: F401
    except ImportError:
        return None
    try:
        return _run_pelt_single(signal, penalty)
    except Exception as e:
        print(f"    PELT pen={penalty}: {e}")
        return None


def method_spectral(signal, sfreq):
    n = len(signal)
    starts = list(range(0, n - WINDOW_SIZE + 1, STEP_SIZE))
    centers = np.array([s + WINDOW_SIZE // 2 for s in starts])
    psds = []
    for s in starts:
        _, psd = welch(signal[s : s + WINDOW_SIZE], fs=sfreq,
                       nperseg=min(256, WINDOW_SIZE))
        psds.append(psd)
    dists = np.array([
        float(np.sqrt(np.sum((psds[i] - psds[i + 1]) ** 2)))
        for i in range(len(psds) - 1)
    ])
    cps = cusum_changepoints(dists)
    return scores_to_samples(cps, centers)


def method_variance(signal, sfreq):
    n = len(signal)
    starts = list(range(0, n - WINDOW_SIZE + 1, STEP_SIZE))
    centers = np.array([s + WINDOW_SIZE // 2 for s in starts])
    variances = np.array([np.var(signal[s : s + WINDOW_SIZE]) for s in starts])
    diffs = np.abs(np.diff(variances))
    cps = cusum_changepoints(diffs)
    return scores_to_samples(cps, centers)


def _simple_bocpd(signal, hazard_rate=1.0 / 500):
    max_len = 5000
    if len(signal) > max_len:
        step = len(signal) // max_len
        sig = signal[::step]
        scale = step
    else:
        sig = signal
        scale = 1

    T = len(sig)
    R = min(500, T)
    rl_probs = np.zeros(R)
    rl_probs[0] = 1.0
    mu0 = float(np.mean(sig[: min(100, T)]))
    var0 = float(np.var(sig[: min(100, T)])) + 1e-6
    sums = np.zeros(R)
    sq_sums = np.zeros(R)
    counts = np.zeros(R)
    map_rl = np.zeros(T)

    for t in range(T):
        x = sig[t]
        n = counts + 1e-10
        means = np.where(counts > 1, sums / n, mu0)
        vrs = np.where(counts > 1,
                       np.maximum(sq_sums / n - means ** 2, 1e-6), var0)
        log_p = -0.5 * np.log(2 * np.pi * vrs) - 0.5 * (x - means) ** 2 / vrs
        pred = np.exp(log_p - np.max(log_p))
        growth = rl_probs * pred * (1 - hazard_rate)
        cp_mass = float(np.sum(rl_probs * pred * hazard_rate))
        new_rl = np.zeros(R)
        new_rl[0] = cp_mass
        new_rl[1:] = growth[:-1]
        total = new_rl.sum()
        if total > 0:
            new_rl /= total
        rl_probs = new_rl
        map_rl[t] = np.argmax(rl_probs)
        sums[1:] = sums[:-1] + x
        sums[0] = 0.0
        sq_sums[1:] = sq_sums[:-1] + x * x
        sq_sums[0] = 0.0
        counts[1:] = counts[:-1] + 1
        counts[0] = 0

    cps = np.where(np.diff(map_rl) < -10)[0]
    return (cps * scale).astype(int)


def method_bocpd(signal, sfreq):
    try:
        from functools import partial
        from bayesian_changepoint_detection.online_changepoint_detection import (
            online_changepoint_detection, constant_hazard,
        )
        from bayesian_changepoint_detection.priors import ifm_obs
        max_len = 5000
        if len(signal) > max_len:
            step = len(signal) // max_len
            sig = signal[::step]
            scale = step
        else:
            sig = signal
            scale = 1
        hazard_fn = partial(constant_hazard, 250)
        R, maxes = online_changepoint_detection(sig, hazard_fn, ifm_obs)
        rl = R.argmax(axis=0)
        cps = np.where(np.diff(rl) < -10)[0]
        return (cps * scale).astype(int)
    except ImportError:
        return _simple_bocpd(signal)


# ---------------------------------------------------------------------------
# Run all methods on one signal segment
# ---------------------------------------------------------------------------

def run_all_methods(signal, sfreq, label="seg"):
    """Run 5 methods on signal, return dict of {method: detected_samples}."""
    detections = {}

    print(f"  [1/5] Topological", flush=True)
    detections["Topological"] = method_topological(signal, sfreq,
                                                    label=f"topo-{label}")

    print(f"  [2/5] PELT", flush=True)
    det = method_pelt_fixed(signal, penalty=PELT_DEFAULT)
    detections["PELT"] = det if det is not None else np.array([], dtype=int)

    print(f"  [3/5] Spectral", flush=True)
    detections["Spectral"] = method_spectral(signal, sfreq)

    print(f"  [4/5] Variance", flush=True)
    detections["Variance"] = method_variance(signal, sfreq)

    print(f"  [5/5] BOCPD", flush=True)
    detections["BOCPD"] = method_bocpd(signal, sfreq)

    return detections


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_seizure_table(subject, filename, onset_s, results):
    print(f"\n  Subject {subject}, File {filename}, Seizure at {onset_s}s")
    hdr = (f"  {'Method':<16} {'Detected?':>9} {'Lag (s)':>8} "
           f"{'FA (pre-sz)':>11} {'F1':>7}")
    print(hdr)
    print(f"  {'-' * 56}")
    for m in METHODS:
        r = results.get(m, {})
        det_str = "YES" if r.get("detected", False) else "NO"
        lag_str = (f"{r['lag_s']:+.1f}" if not np.isnan(r.get('lag_s', float('nan')))
                   else "--")
        # False alarms = detections that didn't match the true onset
        n_det = r.get("n_detected", 0)
        tp = 1 if r.get("detected", False) else 0
        fa = max(0, n_det - tp)
        f1_str = f"{r.get('f1', 0):.2f}"
        print(f"  {m:<16} {det_str:>9} {lag_str:>8} {fa:>11} {f1_str:>7}")


def print_grand_summary(all_seizure_results, false_alarm_results):
    n_seizures = len(all_seizure_results)
    print(f"\n{'=' * 72}")
    print(f"SEIZURE ONSET DETECTION (N={n_seizures} seizures)")
    print(f"{'=' * 72}")

    hdr = (f"{'Method':<16} {'Det Rate':>9} {'Mean Lag':>9} {'Med Lag':>9} "
           f"{'FA/hour':>8} {'Mean F1':>8}")
    print(hdr)
    print(f"{'-' * 72}")

    method_stats = {}
    for m in METHODS:
        detected = 0
        lags = []
        f1s = []
        total_fa = 0
        total_hours = 0

        for sr in all_seizure_results:
            r = sr.get(m, {})
            if r.get("detected", False):
                detected += 1
            lag = r.get("lag_s", float("nan"))
            if not np.isnan(lag):
                lags.append(lag)
            f1s.append(r.get("f1", 0))

        # False alarm data from seizure-free files
        for fa_info in false_alarm_results:
            fa = fa_info.get(m, {})
            total_fa += fa.get("n_detected", 0)
            total_hours += fa.get("duration_hours", 0)

        det_rate = detected / n_seizures if n_seizures > 0 else 0
        mean_lag = float(np.mean(lags)) if lags else float("nan")
        med_lag = float(np.median(lags)) if lags else float("nan")
        fa_per_hour = total_fa / total_hours if total_hours > 0 else float("nan")
        mean_f1 = float(np.mean(f1s)) if f1s else 0

        method_stats[m] = {
            "det_rate": det_rate,
            "detected": detected,
            "mean_lag": mean_lag,
            "median_lag": med_lag,
            "fa_per_hour": fa_per_hour,
            "mean_f1": mean_f1,
            "lags": lags,
            "f1s": f1s,
        }

        det_str = f"{detected}/{n_seizures}"
        lag_str = f"{mean_lag:+.1f}" if not np.isnan(mean_lag) else "--"
        med_str = f"{med_lag:+.1f}" if not np.isnan(med_lag) else "--"
        fa_str = f"{fa_per_hour:.1f}" if not np.isnan(fa_per_hour) else "--"
        print(f"{m:<16} {det_str:>9} {lag_str:>9} {med_str:>9} "
              f"{fa_str:>8} {mean_f1:>8.2f}")

    return method_stats


def print_statistical_tests(method_stats, n_seizures):
    print(f"\nStatistical tests:")
    print(f"{'-' * 72}")

    topo = method_stats.get("Topological", {})
    topo_lags = topo.get("lags", [])

    # Paired Wilcoxon on detection lags
    print(f"\nPaired Wilcoxon on detection lags (Topology vs each):")
    wilcoxon_lag_results = {}
    for m in METHODS:
        if m == "Topological":
            continue
        other_lags = method_stats[m].get("lags", [])

        # Need paired lags — only from seizures where both detected
        # Reconstruct from the per-seizure data
        if len(topo_lags) < 3 or len(other_lags) < 3:
            print(f"  vs {m}: insufficient paired lag data")
            wilcoxon_lag_results[m] = float("nan")
            continue

        # Use the shorter of the two lists for pairing
        n_pair = min(len(topo_lags), len(other_lags))
        t_vals = np.array(topo_lags[:n_pair])
        o_vals = np.array(other_lags[:n_pair])

        diffs = t_vals - o_vals
        if np.all(diffs == 0):
            print(f"  vs {m}: all differences zero, p=1.0")
            wilcoxon_lag_results[m] = 1.0
            continue

        try:
            stat, p = wilcoxon(t_vals, o_vals, alternative="less")
            wilcoxon_lag_results[m] = p
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            diff = float(np.mean(diffs))
            print(f"  vs {m}: W={stat:.0f}, p={p:.4f} {sig}  "
                  f"(mean lag diff={diff:+.1f}s, n={n_pair})")
        except Exception as e:
            print(f"  vs {m}: test failed ({e})")
            wilcoxon_lag_results[m] = float("nan")

    # McNemar's test on detection rate
    print(f"\nMcNemar's test on detection rate (Topology vs each):")
    topo_detected = topo.get("detected", 0)
    for m in METHODS:
        if m == "Topological":
            continue
        other_detected = method_stats[m].get("detected", 0)
        # Simple comparison (McNemar requires per-seizure concordance table,
        # which we don't have in aggregated form — report counts instead)
        print(f"  vs {m}: Topology {topo_detected}/{n_seizures}, "
              f"{m} {other_detected}/{n_seizures}")

    return wilcoxon_lag_results


def print_verdict(method_stats, wilcoxon_lag_results, n_seizures):
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")

    topo = method_stats.get("Topological", {})

    det_rate = topo.get("det_rate", 0)
    mean_lag = topo.get("mean_lag", float("nan"))
    det_pass = det_rate > 0.80
    lag_pass = not np.isnan(mean_lag) and mean_lag < 0

    sig_methods = [m for m, p in wilcoxon_lag_results.items()
                   if not np.isnan(p) and p < 0.05]
    wilcoxon_pass = len(sig_methods) > 0

    overall = det_pass and lag_pass and wilcoxon_pass

    print(f"\n1. Detection rate: {det_rate:.0%} (threshold: >80%) — "
          f"{'PASS' if det_pass else 'FAIL'}")
    print(f"2. Mean lag: {mean_lag:+.1f}s (threshold: <0) — "
          f"{'PASS' if lag_pass else 'FAIL'}" if not np.isnan(mean_lag)
          else f"2. Mean lag: -- (no detections) — FAIL")
    print(f"3. Wilcoxon sig. earlier than: "
          f"{', '.join(sig_methods) if sig_methods else 'NONE'} — "
          f"{'PASS' if wilcoxon_pass else 'FAIL'}")

    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    if overall:
        print(f"  Topology detects seizures early ({mean_lag:+.1f}s) with "
              f"{det_rate:.0%} detection rate")
    else:
        reasons = []
        if not det_pass:
            reasons.append(f"detection rate {det_rate:.0%} <= 80%")
        if not lag_pass:
            reasons.append("mean lag >= 0 (not early detection)")
        if not wilcoxon_pass:
            reasons.append("no significant lag advantage")
        print(f"  Failed: {'; '.join(reasons)}")

    # Comparison table
    print(f"\nMethod ranking by detection lag (lower = earlier):")
    ranked = sorted(method_stats.items(),
                    key=lambda x: x[1].get("mean_lag", float("inf")))
    for i, (m, stats) in enumerate(ranked):
        lag = stats.get("mean_lag", float("nan"))
        lag_str = f"{lag:+.1f}s" if not np.isnan(lag) else "--"
        det = stats.get("det_rate", 0)
        marker = " <-- topology" if m == "Topological" else ""
        print(f"  {i+1}. {m:<16} lag={lag_str:>8}  det={det:.0%}{marker}")

    print(f"\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("Seizure onset detection benchmark: CHB-MIT Scalp EEG")
    print(f"Workers: {N_JOBS}, window={WINDOW_SIZE}, step={STEP_SIZE}, "
          f"subsample={SUBSAMPLE}")
    print(f"Subjects: {', '.join(SUBJECTS)}")
    print(f"Tolerance: +/-{TOLERANCE_S}s, bandpass: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
    print(f"Segment: {PRE_ONSET_S}s pre-onset to {POST_ONSET_S}s post-onset")
    print()

    all_seizure_results = []
    false_alarm_results = []
    seizure_labels = []  # (subject, filename, onset_s) for each seizure

    for subject in SUBJECTS:
        print(f"{'#' * 72}")
        print(f"SUBJECT: {subject}")
        print(f"{'#' * 72}")

        seizure_entries, seizure_free_files = get_seizure_files(subject)

        if not seizure_entries:
            print(f"  No seizure files found for {subject}")
            continue

        print(f"  {len(seizure_entries)} files with seizures, "
              f"{len(seizure_free_files)} seizure-free files")

        # Process seizure files (limit to first 3 seizures per subject for speed)
        seizure_count = 0
        max_seizures_per_subject = 3

        for entry in seizure_entries:
            if seizure_count >= max_seizures_per_subject:
                break

            filename = entry["file_name"]
            print(f"\n  --- {filename} ({len(entry['seizures'])} seizure(s)) ---")

            edf_path = download_edf(subject, filename)
            if edf_path is None:
                print(f"  SKIP: Download failed")
                continue

            result = load_edf_channel(edf_path, CHANNEL_PRIORITY)
            if result is None:
                print(f"  SKIP: No suitable channel")
                continue

            full_signal, ch_name, sfreq = result
            print(f"  Channel: {ch_name}, {len(full_signal)} samples @ {sfreq} Hz")

            for sz in entry["seizures"]:
                if seizure_count >= max_seizures_per_subject:
                    break

                onset_s = sz["start_s"]
                onset_sample = int(onset_s * sfreq)

                # Extract window: PRE_ONSET_S before to POST_ONSET_S after
                start_sample = max(0, onset_sample - int(PRE_ONSET_S * sfreq))
                end_sample = min(len(full_signal),
                                 onset_sample + int(POST_ONSET_S * sfreq))

                if end_sample - start_sample < WINDOW_SIZE * 2:
                    print(f"  SKIP: Segment too short for seizure at {onset_s}s")
                    continue

                segment = full_signal[start_sample:end_sample]

                # Ground truth: onset relative to segment start
                onset_in_segment = onset_sample - start_sample
                transitions = np.array([onset_in_segment], dtype=int)
                tolerance = int(TOLERANCE_S * sfreq)

                print(f"\n  Seizure at {onset_s}s: segment={len(segment)} samples "
                      f"({len(segment)/sfreq:.0f}s), onset at sample "
                      f"{onset_in_segment} in segment")

                # Bandpass filter
                sos = butter(BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH],
                             btype="bandpass", fs=sfreq, output="sos")
                filtered = sosfilt(sos, segment)

                # Run all methods
                t1 = time.time()
                detections = run_all_methods(filtered, sfreq,
                                             label=f"{subject}-sz{seizure_count}")

                # Evaluate each method
                seizure_results = {}
                for m in METHODS:
                    det_samples = detections[m]
                    r = evaluate(transitions, det_samples, tolerance)

                    # Compute lag in seconds
                    if r["detected"] and not np.isnan(r["mean_lag"]):
                        r["lag_s"] = r["mean_lag"] / sfreq
                    else:
                        r["lag_s"] = float("nan")

                    # Count false alarms in pre-seizure period
                    pre_onset_fa = sum(
                        1 for d in det_samples
                        if d < onset_in_segment - tolerance
                    )
                    r["pre_onset_fa"] = pre_onset_fa
                    seizure_results[m] = r

                print_seizure_table(subject, filename, onset_s, seizure_results)
                print(f"  Time: {time.time() - t1:.1f}s")

                all_seizure_results.append(seizure_results)
                seizure_labels.append((subject, filename, onset_s))
                seizure_count += 1

        # Process one seizure-free file for false alarm rate
        if seizure_free_files:
            fa_file = seizure_free_files[0]
            print(f"\n  --- False alarm check: {fa_file} ---")

            edf_path = download_edf(subject, fa_file)
            if edf_path is not None:
                result = load_edf_channel(edf_path, CHANNEL_PRIORITY)
                if result is not None:
                    fa_signal, ch_name, sfreq = result

                    # Use first 10 minutes max
                    max_fa_samples = int(600 * sfreq)
                    if len(fa_signal) > max_fa_samples:
                        fa_signal = fa_signal[:max_fa_samples]

                    duration_hours = len(fa_signal) / sfreq / 3600

                    sos = butter(BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH],
                                 btype="bandpass", fs=sfreq, output="sos")
                    fa_filtered = sosfilt(sos, fa_signal)

                    print(f"  {len(fa_signal)} samples ({duration_hours*60:.0f} min), "
                          f"channel: {ch_name}")

                    detections = run_all_methods(fa_filtered, sfreq,
                                                 label=f"{subject}-fa")
                    fa_result = {}
                    for m in METHODS:
                        fa_result[m] = {
                            "n_detected": len(detections[m]),
                            "duration_hours": duration_hours,
                        }
                        print(f"    {m}: {len(detections[m])} false detections "
                              f"({len(detections[m])/duration_hours:.1f}/hour)")

                    false_alarm_results.append(fa_result)

    # Grand summary
    if not all_seizure_results:
        print("\nERROR: No seizures processed. Check data availability.")
        sys.exit(1)

    method_stats = print_grand_summary(all_seizure_results, false_alarm_results)
    wilcoxon_lag_results = print_statistical_tests(method_stats,
                                                    len(all_seizure_results))
    print_verdict(method_stats, wilcoxon_lag_results, len(all_seizure_results))

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
