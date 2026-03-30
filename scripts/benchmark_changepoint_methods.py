#!/usr/bin/env python3
"""Benchmark: topological changepoint detection vs standard methods.

Compares sliding-window persistent homology against PELT, spectral,
variance, and BOCPD changepoint detectors on three data sources:
  A) Synthetic switching Rossler (known ground truth)
  B) PhysioNet Sleep-EDF (hypnogram transitions)
  C) Rivalry EEG (behavioral switch reports, if available)

Evaluation: precision, recall, F1, mean detection lag per method x source.

Usage:
    python scripts/benchmark_changepoint_methods.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, welch
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
PI_RES = 20          # persistence image resolution (50 default is too slow)
SEED = 42
N_JOBS = min(16, mp.cpu_count())

TOLERANCE_SYNTHETIC = 1000       # +/-1000 samples
TOLERANCE_SLEEP_S = 30.0         # +/-30 seconds (converted at runtime)
TOLERANCE_RIVALRY_S = 5.0        # +/-5 seconds

PELT_SWEEP = [1, 5, 10, 20, 50]
PELT_DEFAULT = 10
PELT_SENSITIVITY = [5, 10, 20]


# ---------------------------------------------------------------------------
# Parallel PH infrastructure
# ---------------------------------------------------------------------------

def _ph_worker(args):
    """Compute PH on one window cloud."""
    cloud, max_dim, subsample, seed = args
    pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser")
    pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return pa.diagrams_


def parallel_windowed_ph(cloud, label="PH"):
    """Sliding-window PH -> persistence image L2 distances."""
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
    # Shared birth/persistence ranges (min/max only — no list accumulation)
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

    # Persistence images on shared grid
    images = []
    for dgms in all_diagrams:
        pa = PersistenceAnalyzer(max_dim=MAX_DIM)
        pa.diagrams_ = dgms
        imgs = pa.to_image(resolution=PI_RES, birth_range=br, persistence_range=pr)
        images.append(imgs)
    # Consecutive L2 distances
    dists = []
    for i in range(len(images) - 1):
        d = sum(
            float(np.sqrt(np.sum((images[i][k] - images[i + 1][k]) ** 2)))
            for k in range(MAX_DIM + 1)
        )
        dists.append(d)

    return centers, np.array(dists)


# ---------------------------------------------------------------------------
# CUSUM changepoint detection (same threshold logic as TransitionDetector)
# ---------------------------------------------------------------------------

def cusum_changepoints(scores):
    """Forward CUSUM, threshold = mean + 2*std. Returns indices into scores."""
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
    """Convert score-array indices to sample positions."""
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    return np.array([int(midpoints[i]) for i in cps if i < len(midpoints)])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(true_s, det_s, tolerance):
    """Precision, recall, F1, mean detection lag."""
    true_s = np.asarray(true_s, dtype=float)
    det_s = np.asarray(det_s, dtype=float)
    n_true = len(true_s)
    n_det = len(det_s)

    if n_det == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_lag": float("nan"), "n_detected": 0, "n_true": n_true}
    if n_true == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_lag": float("nan"), "n_detected": n_det, "n_true": 0}

    # True positives + lag
    tp, lags = 0, []
    for d in det_s:
        offsets = d - true_s
        best = int(np.argmin(np.abs(offsets)))
        if abs(offsets[best]) <= tolerance:
            tp += 1
            lags.append(int(offsets[best]))

    # Recall: fraction of true transitions with >=1 detection nearby
    recalled = sum(1 for t in true_s if np.any(np.abs(det_s - t) <= tolerance))

    prec = tp / n_det if n_det > 0 else 0.0
    rec = recalled / n_true if n_true > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_lag = float(np.mean(lags)) if lags else float("nan")

    return {"precision": prec, "recall": rec, "f1": f1,
            "mean_lag": mean_lag, "n_detected": n_det, "n_true": n_true}


# ---------------------------------------------------------------------------
# Method 1: Topological (Takens -> sliding-window PH -> PI L2 -> CUSUM)
# ---------------------------------------------------------------------------

def method_topological(signal, sfreq, label="topo"):
    t_embed = time.time()
    # Cap signal length for auto estimation (AMI + FNN are O(n log n))
    MAX_FIT_SAMPLES = 20000
    if len(signal) > MAX_FIT_SAMPLES:
        fit_signal = signal[:MAX_FIT_SAMPLES]
    else:
        fit_signal = signal
    embedder = TakensEmbedder("auto", "auto")
    try:
        embedder.fit(fit_signal)
    except Exception:
        embedder = TakensEmbedder(delay=10, dimension=3)
        embedder.fit(fit_signal)
    cloud = embedder.transform(signal)
    print(f"    Takens: delay={embedder.delay_}, dim={embedder.dimension_}, "
          f"cloud={cloud.shape} ({time.time()-t_embed:.1f}s)", flush=True)

    centers, dists = parallel_windowed_ph(cloud, label=label)
    cps = cusum_changepoints(dists)
    return scores_to_samples(cps, centers)


# ---------------------------------------------------------------------------
# Method 2: PELT (ruptures)
# ---------------------------------------------------------------------------

def _run_pelt_single(signal, penalty):
    """One PELT run -> array of changepoint sample indices.
    Uses model='normal' (mean+variance change detection, O(n) cost)."""
    import ruptures
    sig = signal.reshape(-1, 1) if signal.ndim == 1 else signal
    # Adaptive jump: default 5 for short signals, larger for long ones
    jump = max(5, len(signal) // 2000)
    algo = ruptures.Pelt(model="normal", min_size=500, jump=jump).fit(sig)
    bkps = algo.predict(pen=penalty)
    if bkps and bkps[-1] == len(signal):
        bkps = bkps[:-1]
    return np.array(bkps, dtype=int)


def method_pelt_best(signal, n_true):
    """Sweep penalties, pick closest to n_true changepoints."""
    try:
        import ruptures  # noqa: F401
    except ImportError:
        print("    ruptures not installed -- skipping PELT")
        return None, None

    best, best_pen, best_diff = None, None, float("inf")
    for pen in PELT_SWEEP:
        try:
            bkps = _run_pelt_single(signal, pen)
            diff = abs(len(bkps) - n_true)
            if diff < best_diff:
                best_diff = diff
                best = bkps
                best_pen = pen
        except Exception as e:
            print(f"    PELT pen={pen}: {e}")
    if best is not None:
        print(f"    PELT best: pen={best_pen}, {len(best)} breakpoints")
    return best, best_pen


def method_pelt_fixed(signal, penalty=10):
    """Single penalty run."""
    try:
        import ruptures  # noqa: F401
    except ImportError:
        return None
    try:
        return _run_pelt_single(signal, penalty)
    except Exception as e:
        print(f"    PELT pen={penalty}: {e}")
        return None


# ---------------------------------------------------------------------------
# Method 3: Window-based spectral (Welch PSD L2 -> CUSUM)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Method 4: Window-based variance (abs diff -> CUSUM)
# ---------------------------------------------------------------------------

def method_variance(signal, sfreq):
    n = len(signal)
    starts = list(range(0, n - WINDOW_SIZE + 1, STEP_SIZE))
    centers = np.array([s + WINDOW_SIZE // 2 for s in starts])

    variances = np.array([np.var(signal[s : s + WINDOW_SIZE]) for s in starts])
    diffs = np.abs(np.diff(variances))

    cps = cusum_changepoints(diffs)
    return scores_to_samples(cps, centers)


# ---------------------------------------------------------------------------
# Method 5: BOCPD (Bayesian Online Changepoint Detection)
# ---------------------------------------------------------------------------

def method_bocpd(signal, sfreq):
    """Try bayesian_changepoint_detection package, fall back to simple Gaussian."""
    try:
        from functools import partial
        from bayesian_changepoint_detection.online_changepoint_detection import (
            online_changepoint_detection,
            constant_hazard,
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
        print("    BOCPD: using simple Gaussian implementation")
        return _simple_bocpd(signal)


def _simple_bocpd(signal, hazard_rate=1.0 / 500):
    """Gaussian BOCPD (Adams & MacKay 2007), truncated run length."""
    max_len = 5000
    if len(signal) > max_len:
        step = len(signal) // max_len
        sig = signal[::step]
        scale = step
    else:
        sig = signal
        scale = 1

    T = len(sig)
    R = min(500, T)  # max run length

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

        # Predictive probability per run length (Gaussian)
        n = counts + 1e-10
        means = np.where(counts > 1, sums / n, mu0)
        vrs = np.where(counts > 1,
                       np.maximum(sq_sums / n - means ** 2, 1e-6), var0)

        log_p = -0.5 * np.log(2 * np.pi * vrs) - 0.5 * (x - means) ** 2 / vrs
        pred = np.exp(log_p - np.max(log_p))  # numerical stability

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

        # Update sufficient statistics (shift right + reset position 0)
        sums[1:] = sums[:-1] + x
        sums[0] = 0.0
        sq_sums[1:] = sq_sums[:-1] + x * x
        sq_sums[0] = 0.0
        counts[1:] = counts[:-1] + 1
        counts[0] = 0

    # Changepoints: where MAP run length drops sharply
    cps = np.where(np.diff(map_rl) < -10)[0]
    return (cps * scale).astype(int)


# ---------------------------------------------------------------------------
# Data source loaders
# ---------------------------------------------------------------------------

def load_synthetic():
    """Switching Rossler -> (signal, transitions, sfreq, name)."""
    from att.synthetic.generators import switching_rossler

    ts = switching_rossler(n_steps=20000, switch_every=5000, seed=SEED)
    signal = ts[:, 0]  # x-component
    transitions = np.array([5000, 10000, 15000])
    sfreq = 1.0 / 0.01  # 100 Hz effective
    return signal, transitions, sfreq, "Synthetic Rossler"


def load_sleep():
    """Sleep-EDF via MNE -> (signal, transitions, sfreq, name) or None."""
    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError:
        print("  MNE not installed -- skipping sleep data")
        return None

    print("  Fetching Sleep-EDF data...")
    try:
        paths = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=[0], recording=[1],
        )
    except Exception as e:
        print(f"  Failed to download sleep data: {e}")
        return None

    raw_fname, annot_fname = paths[0]
    raw = mne.io.read_raw_edf(raw_fname, preload=True)
    annots = mne.read_annotations(annot_fname)
    raw.set_annotations(annots)

    sfreq = raw.info["sfreq"]
    ch_name = None
    for ch in ["EEG Fpz-Cz", "EEG Pz-Oz"]:
        if ch in raw.ch_names:
            signal = raw.get_data(picks=[ch])[0]
            ch_name = ch
            break
    if ch_name is None:
        print("  No EEG channel found")
        return None

    # Extract stage transitions
    stages = [(int(o * sfreq), d)
              for d, o in zip(annots.description, annots.onset)
              if d.startswith("Sleep stage")]
    transitions = np.array(
        [stages[i][0] for i in range(1, len(stages))
         if stages[i][1] != stages[i - 1][1]],
        dtype=int,
    )
    if len(transitions) == 0:
        print("  No sleep stage transitions found")
        return None

    # 30-min window starting 2 min before first transition
    max_s = int(1800 * sfreq)
    start = max(0, transitions[0] - int(120 * sfreq))
    end = min(len(signal), start + max_s)
    signal = signal[start:end]
    transitions = transitions[
        (transitions >= start) & (transitions < end)
    ] - start

    # Bandpass 0.5-30 Hz
    sos = butter(4, [0.5, 30.0], btype="bandpass", fs=sfreq, output="sos")
    signal = sosfilt(sos, signal)

    print(f"  {ch_name}: {len(signal)} samples @ {sfreq} Hz, "
          f"{len(transitions)} transitions")
    return signal, transitions, sfreq, "Sleep-EDF"


def load_rivalry():
    """Rivalry EEG -> (signal, transitions, sfreq, name) or None."""
    data_dir = Path("data/eeg/rivalry_ssvep")
    if not data_dir.exists():
        print("  Rivalry data not at data/eeg/rivalry_ssvep -- skipping")
        return None

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from batch_eeg import (
            discover_subjects, load_rivalry_epoch,
            load_behavioral_switches, bandpass_filter,
        )
    except ImportError:
        print("  Could not import batch_eeg -- skipping rivalry")
        return None

    subjects = discover_subjects(data_dir)
    if not subjects:
        print("  No subjects found")
        return None

    subj = subjects[0]
    result = load_rivalry_epoch(subj["epochs_dir"], "riv_12", 0)
    if result is None:
        print("  Could not load rivalry epoch")
        return None

    epoch_data, ch_names, sfreq = result
    if "Oz" not in ch_names:
        print("  Oz channel not found")
        return None

    signal = epoch_data[ch_names.index("Oz")]
    switches = load_behavioral_switches(
        subj["behavior_dir"], param_set=2, sfreq=sfreq,
    )
    if not switches:
        print("  No behavioral switches")
        return None

    transitions = np.array([s["sample"] for s in switches], dtype=int)
    signal = bandpass_filter(signal, 4, 13, sfreq, order=4)

    print(f"  Oz: {len(signal)} samples @ {sfreq} Hz, "
          f"{len(transitions)} switches ({subj['name']})")
    return signal, transitions, sfreq, "Rivalry EEG"


# ---------------------------------------------------------------------------
# Run all methods on one data source
# ---------------------------------------------------------------------------

def run_source(signal, transitions, sfreq, name, tolerance,
               use_pelt_sweep=False):
    """Run all 5 methods, return list of (method_name, eval_dict)."""
    results = []

    # 1. Topological
    print("  [1/5] Topological")
    t1 = time.time()
    det = method_topological(signal, sfreq, label=f"topo-{name[:5].lower()}")
    r = evaluate(transitions, det, tolerance)
    results.append(("Topological", r))
    print(f"    -> {r['n_detected']} det, F1={r['f1']:.2f} ({time.time()-t1:.1f}s)")

    # 2. PELT
    print("  [2/5] PELT")
    t1 = time.time()
    if use_pelt_sweep:
        det, pen = method_pelt_best(signal, n_true=len(transitions))
    else:
        det = method_pelt_fixed(signal, penalty=PELT_DEFAULT)
        pen = PELT_DEFAULT
    if det is not None:
        r = evaluate(transitions, det, tolerance)
        results.append(("PELT", r))
        print(f"    -> {r['n_detected']} det, F1={r['f1']:.2f} ({time.time()-t1:.1f}s)")
        # Sensitivity report for EEG
        if not use_pelt_sweep:
            for p in PELT_SENSITIVITY:
                if p == PELT_DEFAULT:
                    continue
                d2 = method_pelt_fixed(signal, penalty=p)
                if d2 is not None:
                    r2 = evaluate(transitions, d2, tolerance)
                    print(f"    PELT pen={p}: {len(d2)} det, "
                          f"P={r2['precision']:.2f} R={r2['recall']:.2f} "
                          f"F1={r2['f1']:.2f}")
    else:
        results.append(("PELT", None))
        print(f"    -> skipped ({time.time()-t1:.1f}s)")

    # 3. Spectral
    print("  [3/5] Spectral")
    t1 = time.time()
    det = method_spectral(signal, sfreq)
    r = evaluate(transitions, det, tolerance)
    results.append(("Spectral", r))
    print(f"    -> {r['n_detected']} det, F1={r['f1']:.2f} ({time.time()-t1:.1f}s)")

    # 4. Variance
    print("  [4/5] Variance")
    t1 = time.time()
    det = method_variance(signal, sfreq)
    r = evaluate(transitions, det, tolerance)
    results.append(("Variance", r))
    print(f"    -> {r['n_detected']} det, F1={r['f1']:.2f} ({time.time()-t1:.1f}s)")

    # 5. BOCPD
    print("  [5/5] BOCPD")
    t1 = time.time()
    det = method_bocpd(signal, sfreq)
    if det is not None:
        r = evaluate(transitions, det, tolerance)
        results.append(("BOCPD", r))
        print(f"    -> {r['n_detected']} det, F1={r['f1']:.2f} ({time.time()-t1:.1f}s)")
    else:
        results.append(("BOCPD", None))
        print(f"    -> skipped ({time.time()-t1:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(name, n_true, results):
    print(f"\n{'=' * 72}")
    print(f"{name.upper()} ({n_true} ground-truth transitions)")
    print(f"{'=' * 72}")
    hdr = (f"{'Method':<16} {'Precision':>9} {'Recall':>9} {'F1':>9} "
           f"{'Mean Lag':>9} {'Detections':>10}")
    print(hdr)
    print(f"{'-' * 72}")
    for method, r in results:
        if r is None:
            print(f"{method:<16} {'--':>9} {'--':>9} {'--':>9} "
                  f"{'--':>9} {'--':>10}")
        else:
            lag = (f"{r['mean_lag']:+.0f}"
                   if not np.isnan(r['mean_lag']) else "--")
            print(f"{method:<16} {r['precision']:>9.2f} {r['recall']:>9.2f} "
                  f"{r['f1']:>9.2f} {lag:>9} {r['n_detected']:>10}")


def print_summary(all_results):
    methods = ["Topological", "PELT", "Spectral", "Variance", "BOCPD"]
    sources = list(all_results.keys())

    print(f"\n{'=' * 72}")
    print("SUMMARY ACROSS ALL DATA SOURCES")
    print(f"{'=' * 72}")

    hdr = f"{'':16}"
    for src in sources:
        hdr += f" {src[:12]:>12}"
    hdr += f" {'Mean F1':>9}"
    print(hdr)
    print(f"{'-' * 72}")

    f1_map = {}
    for method in methods:
        row = f"{method:<16}"
        f1s = []
        for src in sources:
            r = all_results[src].get(method)
            if r is not None:
                row += f" {r['f1']:>12.2f}"
                f1s.append(r["f1"])
            else:
                row += f" {'--':>12}"
        mean = float(np.mean(f1s)) if f1s else float("nan")
        f1_map[method] = mean
        if not np.isnan(mean):
            row += f" {mean:>9.2f}"
        else:
            row += f" {'--':>9}"
        print(row)

    return f1_map


def print_verdict(f1_map, all_results):
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")

    valid = {k: v for k, v in f1_map.items() if not np.isnan(v)}
    if not valid:
        print("No results to compare.")
        return

    best = max(valid, key=valid.get)
    print(f"\n1. Best mean F1: {best} ({valid[best]:.2f})")

    topo = f1_map.get("Topological", float("nan"))
    spec = f1_map.get("Spectral", float("nan"))
    var_ = f1_map.get("Variance", float("nan"))
    pelt = f1_map.get("PELT", float("nan"))

    print()
    if not np.isnan(topo) and not np.isnan(spec):
        if topo > spec + 0.02:
            print(f"2. Topology BEATS spectral: {topo:.2f} vs {spec:.2f}")
            print("   -> PH captures info beyond frequency-domain summaries")
        elif spec > topo + 0.02:
            print(f"2. Topology LOSES to spectral: {topo:.2f} vs {spec:.2f}")
            print("   -> PH not justified over simpler PSD comparison")
        else:
            print(f"2. Topology ~ spectral: {topo:.2f} vs {spec:.2f}")

    if not np.isnan(topo) and not np.isnan(var_):
        if topo > var_ + 0.02:
            print(f"3. Topology BEATS variance: {topo:.2f} vs {var_:.2f}")
            print("   -> Topology captures structure beyond simple statistics")
        elif var_ > topo + 0.02:
            print(f"3. Topology LOSES to variance: {topo:.2f} vs {var_:.2f}")
            print("   -> Simple variance sufficient; PH is wasted compute")
        else:
            print(f"3. Topology ~ variance: {topo:.2f} vs {var_:.2f}")

    if not np.isnan(topo) and not np.isnan(pelt):
        if topo > pelt + 0.02:
            print(f"4. Topology BEATS PELT: {topo:.2f} vs {pelt:.2f}")
            print("   -> Distribution-free advantage; topology justified")
        elif pelt > topo + 0.02:
            print(f"4. Topology LOSES to PELT: {topo:.2f} vs {pelt:.2f}")
            print("   -> PELT is parametric, faster, and better")
        else:
            print(f"4. Topology ~ PELT: {topo:.2f} vs {pelt:.2f}")

    # Per-source topology wins/losses
    print()
    print("Per-source breakdown:")
    for src, results in all_results.items():
        topo_r = results.get("Topological")
        if topo_r is None:
            continue
        wins, losses, ties = [], [], []
        for method, r in results.items():
            if method == "Topological" or r is None:
                continue
            if topo_r["f1"] > r["f1"] + 0.05:
                wins.append(f"{method}({r['f1']:.2f})")
            elif topo_r["f1"] < r["f1"] - 0.05:
                losses.append(f"{method}({r['f1']:.2f})")
            else:
                ties.append(method)

        line = f"  {src} (topo F1={topo_r['f1']:.2f}): "
        parts = []
        if wins:
            parts.append(f"WINS vs {', '.join(wins)}")
        if losses:
            parts.append(f"LOSES to {', '.join(losses)}")
        if ties:
            parts.append(f"TIED with {', '.join(ties)}")
        print(line + "; ".join(parts) if parts else line + "no comparison")

        # Detection timing advantage
        topo_lag = topo_r.get("mean_lag", float("nan"))
        if not np.isnan(topo_lag) and topo_lag < 0:
            later = [m for m, r in results.items()
                     if m != "Topological" and r is not None
                     and not np.isnan(r.get("mean_lag", float("nan")))
                     and r["mean_lag"] > topo_lag + 50]
            if later:
                lag_strs = [f"{m}={results[m]['mean_lag']:+.0f}" for m in later]
                print(f"    Early detection: lag={topo_lag:+.0f} vs "
                      f"{', '.join(lag_strs)}")

    print(f"\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("Benchmark: topological changepoints vs standard methods")
    print(f"Workers: {N_JOBS}, window={WINDOW_SIZE}, step={STEP_SIZE}, "
          f"subsample={SUBSAMPLE}")
    print()

    all_results = {}

    # -- Source A: Synthetic --
    print(f"{'#' * 72}")
    print("SOURCE A: SYNTHETIC SWITCHING ROSSLER")
    print(f"{'#' * 72}")
    signal_a, trans_a, sfreq_a, name_a = load_synthetic()
    print(f"  {len(signal_a)} samples, {len(trans_a)} transitions, "
          f"tolerance=+/-{TOLERANCE_SYNTHETIC}")

    results_a = run_source(signal_a, trans_a, sfreq_a, name_a,
                           TOLERANCE_SYNTHETIC, use_pelt_sweep=True)
    print_table(name_a, len(trans_a), results_a)
    all_results[name_a] = {m: r for m, r in results_a}

    # -- Source B: Sleep-EDF --
    print(f"\n{'#' * 72}")
    print("SOURCE B: SLEEP-EDF")
    print(f"{'#' * 72}")
    sleep = load_sleep()
    if sleep is not None:
        signal_b, trans_b, sfreq_b, name_b = sleep
        tol_b = int(TOLERANCE_SLEEP_S * sfreq_b)
        print(f"  tolerance=+/-{TOLERANCE_SLEEP_S}s = +/-{tol_b} samples")

        results_b = run_source(signal_b, trans_b, sfreq_b, name_b, tol_b)
        print_table(name_b, len(trans_b), results_b)
        all_results[name_b] = {m: r for m, r in results_b}

    # -- Source C: Rivalry EEG --
    print(f"\n{'#' * 72}")
    print("SOURCE C: RIVALRY EEG")
    print(f"{'#' * 72}")
    rivalry = load_rivalry()
    if rivalry is not None:
        signal_c, trans_c, sfreq_c, name_c = rivalry
        tol_c = int(TOLERANCE_RIVALRY_S * sfreq_c)
        print(f"  tolerance=+/-{TOLERANCE_RIVALRY_S}s = +/-{tol_c} samples")

        results_c = run_source(signal_c, trans_c, sfreq_c, name_c, tol_c)
        print_table(name_c, len(trans_c), results_c)
        all_results[name_c] = {m: r for m, r in results_c}

    # -- Summary & Verdict --
    f1_map = print_summary(all_results)
    print_verdict(f1_map, all_results)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
