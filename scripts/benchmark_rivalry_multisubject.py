#!/usr/bin/env python3
"""Multi-subject validation: Does topology's F1 advantage on rivalry EEG hold?

Runs the same 5-method changepoint comparison from benchmark_changepoint_methods.py
on up to 10 subjects from the rivalry SSVEP dataset. Reports per-subject tables,
grand summary with mean/std, win counts, and paired Wilcoxon signed-rank tests.

Pass criterion: Topology has highest mean F1 AND Wilcoxon p < 0.05 vs at least
one comparison method.

Usage:
    python -u scripts/benchmark_rivalry_multisubject.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
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

TOLERANCE_RIVALRY_S = 5.0
MAX_SUBJECTS = 10

BANDPASS_LOW = 4
BANDPASS_HIGH = 13
BANDPASS_ORDER = 4
CHANNEL = "Oz"
EPOCH_CONDITION = "riv_12"
EPOCH_PARAM_SET = 2
EPOCH_INDEX = 0

PELT_DEFAULT = 10
PELT_SENSITIVITY = [5, 10, 20]

DATA_DIR = Path("data/eeg/rivalry_ssvep")

METHODS = ["Topological", "PELT", "Spectral", "Variance", "BOCPD"]


# ---------------------------------------------------------------------------
# Subject discovery + data loading (from batch_eeg.py patterns)
# ---------------------------------------------------------------------------

def discover_subjects(data_dir: Path) -> list[dict]:
    """Discover subject directories with Epochs/ and Behavior/ subdirs."""
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


def load_rivalry_epoch(epochs_dir, condition_suffix, epoch_index):
    """Load rivalry epoch -> (epoch_data, ch_names, sfreq) or None."""
    import scipy.io
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


def load_behavioral_switches(behavior_dir, param_set, sfreq=360):
    """Load behavioral switch times -> list of dicts or None."""
    import scipy.io
    if not behavior_dir.exists():
        return None
    beh_files = [f for f in behavior_dir.glob("BR_Rivalry_*.mat")
                 if "PRACT" not in f.name]
    if not beh_files:
        return None
    beh = scipy.io.loadmat(str(beh_files[0]), simplify_cells=True)
    results_beh = beh["results"]
    matching = [i for i, r in enumerate(results_beh)
                if r.get("params", {}).get("paramSet") == param_set]
    if not matching:
        return None
    r = results_beh[matching[0]]
    psycho = r["psycho"]
    t_key_press = psycho["tKeyPress"]
    response_key = psycho["responseKey"]
    switches = []
    for i in range(1, len(response_key)):
        if response_key[i] != response_key[i - 1]:
            switches.append({
                "time": float(t_key_press[i]),
                "sample": int(t_key_press[i] * sfreq),
            })
    return switches


def bandpass_filter(signal, low, high, sfreq, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, signal)


def load_subject(subj):
    """Load one subject -> (signal, transitions, sfreq) or None."""
    result = load_rivalry_epoch(subj["epochs_dir"], EPOCH_CONDITION, EPOCH_INDEX)
    if result is None:
        return None
    epoch_data, ch_names, sfreq = result
    if CHANNEL not in ch_names:
        return None
    signal = epoch_data[ch_names.index(CHANNEL)]
    switches = load_behavioral_switches(
        subj["behavior_dir"], param_set=EPOCH_PARAM_SET, sfreq=sfreq,
    )
    if not switches or len(switches) < 2:
        return None
    transitions = np.array([s["sample"] for s in switches], dtype=int)
    signal = bandpass_filter(signal, BANDPASS_LOW, BANDPASS_HIGH, sfreq,
                             order=BANDPASS_ORDER)
    return signal, transitions, sfreq


# ---------------------------------------------------------------------------
# Parallel PH infrastructure
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
                "mean_lag": float("nan"), "n_detected": 0, "n_true": n_true}
    if n_true == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_lag": float("nan"), "n_detected": n_det, "n_true": 0}

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
            "mean_lag": mean_lag, "n_detected": n_det, "n_true": n_true}


# ---------------------------------------------------------------------------
# 5 methods (self-contained copies from benchmark_changepoint_methods.py)
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
# Run all methods on one subject
# ---------------------------------------------------------------------------

def run_subject(signal, transitions, sfreq, subj_name, tolerance):
    """Run 5 methods, return dict of {method_name: eval_dict}."""
    results = {}

    # 1. Topological
    print(f"  [1/5] Topological", flush=True)
    det = method_topological(signal, sfreq, label=f"topo-{subj_name[:8]}")
    results["Topological"] = evaluate(transitions, det, tolerance)

    # 2. PELT
    print(f"  [2/5] PELT", flush=True)
    det = method_pelt_fixed(signal, penalty=PELT_DEFAULT)
    if det is not None:
        results["PELT"] = evaluate(transitions, det, tolerance)
    else:
        results["PELT"] = None

    # 3. Spectral
    print(f"  [3/5] Spectral", flush=True)
    det = method_spectral(signal, sfreq)
    results["Spectral"] = evaluate(transitions, det, tolerance)

    # 4. Variance
    print(f"  [4/5] Variance", flush=True)
    det = method_variance(signal, sfreq)
    results["Variance"] = evaluate(transitions, det, tolerance)

    # 5. BOCPD
    print(f"  [5/5] BOCPD", flush=True)
    det = method_bocpd(signal, sfreq)
    results["BOCPD"] = evaluate(transitions, det, tolerance)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_subject_table(subj_name, n_true, results):
    print(f"\n  {subj_name} ({n_true} switches)")
    hdr = (f"  {'Method':<16} {'Prec':>7} {'Recall':>7} {'F1':>7} "
           f"{'Lag':>7} {'Det':>5}")
    print(hdr)
    print(f"  {'-' * 56}")
    for m in METHODS:
        r = results.get(m)
        if r is None:
            print(f"  {m:<16} {'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>5}")
        else:
            lag = f"{r['mean_lag']:+.0f}" if not np.isnan(r['mean_lag']) else "--"
            print(f"  {m:<16} {r['precision']:>7.2f} {r['recall']:>7.2f} "
                  f"{r['f1']:>7.2f} {lag:>7} {r['n_detected']:>5}")


def print_grand_summary(all_subject_results):
    n_subjects = len(all_subject_results)
    print(f"\n{'=' * 72}")
    print(f"MULTI-SUBJECT SUMMARY (N={n_subjects} subjects)")
    print(f"{'=' * 72}")

    # Collect per-method F1 arrays
    method_f1s = {m: [] for m in METHODS}
    method_precs = {m: [] for m in METHODS}
    method_recalls = {m: [] for m in METHODS}
    method_lags = {m: [] for m in METHODS}
    method_wins = {m: 0 for m in METHODS}

    for subj_results in all_subject_results:
        best_f1 = -1
        best_method = None
        for m in METHODS:
            r = subj_results.get(m)
            if r is not None:
                method_f1s[m].append(r["f1"])
                method_precs[m].append(r["precision"])
                method_recalls[m].append(r["recall"])
                if not np.isnan(r["mean_lag"]):
                    method_lags[m].append(r["mean_lag"])
                if r["f1"] > best_f1:
                    best_f1 = r["f1"]
                    best_method = m
            else:
                method_f1s[m].append(float("nan"))
                method_precs[m].append(float("nan"))
                method_recalls[m].append(float("nan"))
        if best_method:
            method_wins[best_method] += 1

    hdr = (f"{'Method':<16} {'Mean F1 +/- Std':>16} {'Mean Prec':>10} "
           f"{'Mean Recall':>12} {'Mean Lag':>9} {'Wins':>6}")
    print(hdr)
    print(f"{'-' * 72}")

    for m in METHODS:
        f1arr = np.array(method_f1s[m])
        valid = f1arr[~np.isnan(f1arr)]
        if len(valid) > 0:
            mean_f1 = float(np.mean(valid))
            std_f1 = float(np.std(valid))
            mean_prec = float(np.nanmean(method_precs[m]))
            mean_rec = float(np.nanmean(method_recalls[m]))
            f1_str = f"{mean_f1:.2f} +/- {std_f1:.2f}"
            lag_str = (f"{np.mean(method_lags[m]):+.0f}"
                       if method_lags[m] else "--")
        else:
            f1_str = "--"
            mean_prec = float("nan")
            mean_rec = float("nan")
            lag_str = "--"

        prec_str = f"{mean_prec:.2f}" if not np.isnan(mean_prec) else "--"
        rec_str = f"{mean_rec:.2f}" if not np.isnan(mean_rec) else "--"
        wins_str = f"{method_wins[m]}/{n_subjects}"

        print(f"{m:<16} {f1_str:>16} {prec_str:>10} {rec_str:>12} "
              f"{lag_str:>9} {wins_str:>6}")

    # Wilcoxon signed-rank tests: topology F1 vs each other method
    print(f"\nPaired Wilcoxon signed-rank tests (Topology F1 vs each method):")
    print(f"{'-' * 72}")

    topo_f1 = np.array(method_f1s["Topological"])
    wilcoxon_results = {}

    for m in METHODS:
        if m == "Topological":
            continue
        other_f1 = np.array(method_f1s[m])
        # Need paired samples — drop NaNs from either
        mask = ~np.isnan(topo_f1) & ~np.isnan(other_f1)
        t_valid = topo_f1[mask]
        o_valid = other_f1[mask]

        if len(t_valid) < 3:
            print(f"  Topology vs {m}: insufficient paired samples (n={len(t_valid)})")
            wilcoxon_results[m] = float("nan")
            continue

        # Check if all differences are zero
        diffs = t_valid - o_valid
        if np.all(diffs == 0):
            print(f"  Topology vs {m}: all differences zero, p=1.0")
            wilcoxon_results[m] = 1.0
            continue

        try:
            stat, p = wilcoxon(t_valid, o_valid, alternative="greater")
            wilcoxon_results[m] = p
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            diff = float(np.mean(t_valid - o_valid))
            print(f"  Topology vs {m}: W={stat:.0f}, p={p:.4f} {sig}  "
                  f"(mean diff={diff:+.3f}, n={len(t_valid)})")
        except Exception as e:
            print(f"  Topology vs {m}: test failed ({e})")
            wilcoxon_results[m] = float("nan")

    return method_f1s, method_wins, wilcoxon_results


def print_verdict(method_f1s, method_wins, wilcoxon_results, n_subjects):
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")

    # Highest mean F1?
    means = {}
    for m in METHODS:
        valid = [f for f in method_f1s[m] if not np.isnan(f)]
        means[m] = float(np.mean(valid)) if valid else float("nan")

    valid_means = {k: v for k, v in means.items() if not np.isnan(v)}
    if not valid_means:
        print("No valid results.")
        return

    best = max(valid_means, key=valid_means.get)
    topo_mean = means.get("Topological", float("nan"))
    topo_best = best == "Topological"

    print(f"\n1. Highest mean F1: {best} ({valid_means[best]:.3f})")
    if topo_best:
        print(f"   Topology IS the best method across {n_subjects} subjects")
    else:
        print(f"   Topology ({topo_mean:.3f}) is NOT the best — "
              f"{best} ({valid_means[best]:.3f}) wins")

    # Any significant Wilcoxon?
    sig_methods = [m for m, p in wilcoxon_results.items()
                   if not np.isnan(p) and p < 0.05]
    any_sig = len(sig_methods) > 0

    print(f"\n2. Wilcoxon p < 0.05 vs: {', '.join(sig_methods) if sig_methods else 'NONE'}")
    if any_sig:
        print(f"   Topology significantly better than {len(sig_methods)} method(s)")
    else:
        print(f"   No statistically significant advantage")

    # Overall pass
    passed = topo_best and any_sig
    print(f"\n3. Overall: {'PASS' if passed else 'FAIL'}")
    if passed:
        print(f"   Topology's rivalry advantage replicates across {n_subjects} subjects")
    else:
        reasons = []
        if not topo_best:
            reasons.append(f"not highest mean F1 ({best} wins)")
        if not any_sig:
            reasons.append("no significant Wilcoxon test")
        print(f"   Failed because: {'; '.join(reasons)}")

    # Win distribution
    print(f"\n4. Win distribution:")
    for m in METHODS:
        bar = "#" * method_wins[m]
        print(f"   {m:<16} {method_wins[m]:>2}/{n_subjects} {bar}")

    print(f"\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("Multi-subject rivalry validation: topology vs 4 standard methods")
    print(f"Workers: {N_JOBS}, window={WINDOW_SIZE}, step={STEP_SIZE}, "
          f"subsample={SUBSAMPLE}")
    print(f"Tolerance: +/-{TOLERANCE_RIVALRY_S}s, channel: {CHANNEL}, "
          f"bandpass: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
    print()

    if not DATA_DIR.exists():
        print(f"ERROR: Rivalry data not found at {DATA_DIR}")
        print("This script requires the Nie/Katyal/Engel (2023) rivalry SSVEP dataset.")
        sys.exit(1)

    subjects = discover_subjects(DATA_DIR)
    if not subjects:
        print(f"ERROR: No subjects found in {DATA_DIR}")
        sys.exit(1)

    subjects = subjects[:MAX_SUBJECTS]
    print(f"Found {len(subjects)} subjects (using up to {MAX_SUBJECTS})")
    print()

    all_subject_results = []
    subject_names = []

    for i, subj in enumerate(subjects):
        print(f"{'#' * 72}")
        print(f"SUBJECT {i+1}/{len(subjects)}: {subj['name']}")
        print(f"{'#' * 72}")

        try:
            data = load_subject(subj)
        except Exception as e:
            print(f"  SKIP: Failed to load ({e})")
            continue

        if data is None:
            print(f"  SKIP: Missing data (no epoch, channel, or switches)")
            continue

        signal, transitions, sfreq = data
        tolerance = int(TOLERANCE_RIVALRY_S * sfreq)
        print(f"  {len(signal)} samples @ {sfreq} Hz, "
              f"{len(transitions)} switches, "
              f"tolerance=+/-{tolerance} samples")

        t_subj = time.time()
        results = run_subject(signal, transitions, sfreq, subj["name"], tolerance)
        print_subject_table(subj["name"], len(transitions), results)
        print(f"  Time: {time.time() - t_subj:.1f}s")

        all_subject_results.append(results)
        subject_names.append(subj["name"])

    if len(all_subject_results) < 2:
        print(f"\nERROR: Only {len(all_subject_results)} subjects loaded. "
              f"Need at least 2 for statistical comparison.")
        sys.exit(1)

    method_f1s, method_wins, wilcoxon_results = print_grand_summary(
        all_subject_results
    )
    print_verdict(method_f1s, method_wins, wilcoxon_results,
                  len(all_subject_results))

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
