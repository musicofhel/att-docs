#!/usr/bin/env python3
"""Reservoir regime detection: topological changepoints on echo state networks.

Tests whether sliding-window persistent homology detects dynamical regime
transitions in reservoir computers (echo state networks). The spectral radius
of the recurrent weight matrix controls the dynamical regime:
  - SR < 0.8: ordered (echo state property, signals decay)
  - SR 0.9-1.1: edge of chaos (maximum computational power)
  - SR > 1.2: chaotic (signals explode, unstable)

Four experiments:
  Part 1: Static topological features vs spectral radius
  Part 2: Continuous ramp detection (SR 0.5→1.3 over 20k steps)
  Part 3: Discrete regime switches (SR 0.7↔1.1, 4 switches)
  Part 4: Coupled reservoir binding (if Parts 1-3 show signal)

Usage:
    python scripts/reservoir_regime_detection.py
"""

from __future__ import annotations

import multiprocessing as mp
import time
import warnings

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr
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
N_SEEDS = 3
TOLERANCE = 500  # ±500 samples for Parts 2-3

# ESN parameters
N_NEURONS = 100
SPARSITY = 0.9       # 90% zeros = 10% connectivity
INPUT_SCALING = 0.1
WASHOUT = 1000
N_STEPS_STATIC = 5000
N_STEPS_RAMP = 20000
N_STEPS_SWITCH = 15000
SWITCH_EVERY = 3000

SR_VALUES = [0.5, 0.7, 0.9, 0.95, 1.0, 1.1, 1.3]
SR_LOW = 0.7
SR_HIGH = 1.1
SR_RAMP_START = 0.5
SR_RAMP_END = 1.3


# ---------------------------------------------------------------------------
# Minimal Echo State Network
# ---------------------------------------------------------------------------

class MinimalESN:
    """Echo state network with adjustable spectral radius."""

    def __init__(self, n_neurons=N_NEURONS, spectral_radius=0.9,
                 input_scaling=INPUT_SCALING, sparsity=SPARSITY, seed=42):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n_neurons, n_neurons))
        W[rng.random((n_neurons, n_neurons)) < sparsity] = 0
        sr = np.max(np.abs(np.linalg.eigvals(W)))
        if sr < 1e-10:
            sr = 1.0
        self.W_base = W / sr  # normalised to SR=1
        self.W = self.W_base * spectral_radius
        self.W_in = rng.standard_normal((n_neurons, 1)) * input_scaling
        self.state = np.zeros(n_neurons)
        self.spectral_radius = spectral_radius

    def run(self, inputs):
        """inputs: (n_steps,) -> states: (n_steps, n_neurons)"""
        states = []
        for u in inputs:
            self.state = np.tanh(self.W @ self.state + self.W_in.ravel() * u)
            states.append(self.state.copy())
        return np.array(states)

    def run_dynamic(self, inputs, spectral_radii):
        """Run with time-varying spectral radius."""
        states = []
        for u, sr in zip(inputs, spectral_radii):
            W = self.W_base * sr
            self.state = np.tanh(W @ self.state + self.W_in.ravel() * u)
            states.append(self.state.copy())
        return np.array(states)

    def run_switching(self, inputs, sr_schedule):
        """Run with piecewise-constant SR. schedule: [(n_steps, sr), ...]"""
        states = []
        idx = 0
        for n, sr in sr_schedule:
            W = self.W_base * sr
            for i in range(n):
                u = inputs[idx + i]
                self.state = np.tanh(W @ self.state + self.W_in.ravel() * u)
                states.append(self.state.copy())
            idx += n
        return np.array(states)


# ---------------------------------------------------------------------------
# Parallel PH infrastructure (from benchmark_changepoint_methods.py)
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

    # Shared birth/persistence ranges
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
# CUSUM changepoint detection
# ---------------------------------------------------------------------------

def cusum_changepoints(scores):
    """Forward CUSUM, threshold = mean + 2*std."""
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
# Method 1: Topological (Takens -> sliding-window PH -> PI L2 -> CUSUM)
# ---------------------------------------------------------------------------

def method_topological(signal, label="topo"):
    """Topological changepoint detection."""
    t_embed = time.time()
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
          f"cloud={cloud.shape} ({time.time()-t_embed:.1f}s)", flush=True)

    centers, dists = parallel_windowed_ph(cloud, label=label)
    cps = cusum_changepoints(dists)
    return scores_to_samples(cps, centers)


# ---------------------------------------------------------------------------
# Method 2: Spectral (Welch PSD L2 -> CUSUM)
# ---------------------------------------------------------------------------

def method_spectral(signal):
    """Window-based spectral changepoint detection."""
    n = len(signal)
    starts = list(range(0, n - WINDOW_SIZE + 1, STEP_SIZE))
    centers = np.array([s + WINDOW_SIZE // 2 for s in starts])

    psds = []
    for s in starts:
        _, psd = welch(signal[s : s + WINDOW_SIZE], fs=1.0,
                       nperseg=min(256, WINDOW_SIZE))
        psds.append(psd)

    dists = np.array([
        float(np.sqrt(np.sum((psds[i] - psds[i + 1]) ** 2)))
        for i in range(len(psds) - 1)
    ])

    cps = cusum_changepoints(dists)
    return scores_to_samples(cps, centers)


# ---------------------------------------------------------------------------
# Method 3: Variance (abs diff -> CUSUM)
# ---------------------------------------------------------------------------

def method_variance(signal):
    """Window-based variance changepoint detection."""
    n = len(signal)
    starts = list(range(0, n - WINDOW_SIZE + 1, STEP_SIZE))
    centers = np.array([s + WINDOW_SIZE // 2 for s in starts])

    variances = np.array([np.var(signal[s : s + WINDOW_SIZE]) for s in starts])
    diffs = np.abs(np.diff(variances))

    cps = cusum_changepoints(diffs)
    return scores_to_samples(cps, centers)


# ---------------------------------------------------------------------------
# Method 4: BOCPD (Bayesian Online Changepoint Detection)
# ---------------------------------------------------------------------------

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
    R = min(500, T)

    rl_probs = np.zeros(R)
    rl_probs[0] = 1.0

    mu0 = float(np.mean(sig[:min(100, T)]))
    var0 = float(np.var(sig[:min(100, T)])) + 1e-6

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


def method_bocpd(signal):
    """BOCPD with package fallback to simple Gaussian."""
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
        return _simple_bocpd(signal)


# ---------------------------------------------------------------------------
# Helper: run all 4 methods on a signal
# ---------------------------------------------------------------------------

def run_all_methods(signal, transitions, tolerance, seed_label=""):
    """Run topological, spectral, variance, BOCPD. Return dict of eval dicts."""
    results = {}

    print(f"    [Topological]", flush=True)
    t0 = time.time()
    det = method_topological(signal, label=f"topo{seed_label}")
    results["Topological"] = evaluate(transitions, det, tolerance)
    results["Topological"]["detections"] = det
    lag = results["Topological"]["mean_lag"]
    lag_s = f"{lag:+.0f}" if not np.isnan(lag) else "--"
    print(f"      det={results['Topological']['n_detected']}, "
          f"F1={results['Topological']['f1']:.2f}, lag={lag_s} "
          f"({time.time()-t0:.1f}s)")

    print(f"    [Spectral]", flush=True)
    t0 = time.time()
    det = method_spectral(signal)
    results["Spectral"] = evaluate(transitions, det, tolerance)
    results["Spectral"]["detections"] = det
    lag = results["Spectral"]["mean_lag"]
    lag_s = f"{lag:+.0f}" if not np.isnan(lag) else "--"
    print(f"      det={results['Spectral']['n_detected']}, "
          f"F1={results['Spectral']['f1']:.2f}, lag={lag_s} "
          f"({time.time()-t0:.1f}s)")

    print(f"    [Variance]", flush=True)
    t0 = time.time()
    det = method_variance(signal)
    results["Variance"] = evaluate(transitions, det, tolerance)
    results["Variance"]["detections"] = det
    lag = results["Variance"]["mean_lag"]
    lag_s = f"{lag:+.0f}" if not np.isnan(lag) else "--"
    print(f"      det={results['Variance']['n_detected']}, "
          f"F1={results['Variance']['f1']:.2f}, lag={lag_s} "
          f"({time.time()-t0:.1f}s)")

    print(f"    [BOCPD]", flush=True)
    t0 = time.time()
    det = method_bocpd(signal)
    results["BOCPD"] = evaluate(transitions, det, tolerance)
    results["BOCPD"]["detections"] = det
    lag = results["BOCPD"]["mean_lag"]
    lag_s = f"{lag:+.0f}" if not np.isnan(lag) else "--"
    print(f"      det={results['BOCPD']['n_detected']}, "
          f"F1={results['BOCPD']['f1']:.2f}, lag={lag_s} "
          f"({time.time()-t0:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Part 1: Static regime classification
# ---------------------------------------------------------------------------

def part1_static_classification():
    """Topological features vs spectral radius at 7 static values."""
    print(f"\n{'#' * 72}")
    print("PART 1: STATIC REGIME CLASSIFICATION")
    print(f"{'#' * 72}")
    print(f"ESN: {N_NEURONS} neurons, {int((1-SPARSITY)*100)}% connectivity, "
          f"washout={WASHOUT}")
    print(f"Input: white noise + sine, {N_STEPS_STATIC} steps, {N_SEEDS} seeds\n")

    all_features = {}  # input_type -> sr -> feature_dict

    for input_type in ["noise", "sine"]:
        print(f"  Input: {input_type}")
        all_features[input_type] = {}

        for sr in SR_VALUES:
            feats = {k: [] for k in [
                "betti_0", "betti_1",
                "entropy_0", "entropy_1",
                "total_pers_0", "total_pers_1",
                "max_pers_0", "max_pers_1",
            ]}

            for seed in range(SEED, SEED + N_SEEDS):
                rng = np.random.default_rng(seed)
                esn = MinimalESN(spectral_radius=sr, seed=seed)

                if input_type == "noise":
                    inputs = rng.standard_normal(N_STEPS_STATIC) * 0.1
                else:
                    inputs = 0.1 * np.sin(
                        2 * np.pi * np.arange(N_STEPS_STATIC) / 50
                    )

                states = esn.run(inputs)
                signal = states[WASHOUT:, 0]

                # Takens embedding
                try:
                    embedder = TakensEmbedder("auto", "auto")
                    embedder.fit(signal[:min(len(signal), 20000)])
                except Exception:
                    embedder = TakensEmbedder(delay=10, dimension=3)
                    embedder.fit(signal)
                cloud = embedder.transform(signal)

                pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
                pa.fit_transform(cloud, subsample=400, seed=seed)

                for dim in [0, 1]:
                    dgm = (pa.diagrams_[dim]
                           if dim < len(pa.diagrams_) else np.array([]))
                    if len(dgm) > 0:
                        lifetimes = dgm[:, 1] - dgm[:, 0]
                        lifetimes = lifetimes[lifetimes > 1e-10]
                        feats[f"betti_{dim}"].append(len(lifetimes))
                        if len(lifetimes) > 0:
                            total = lifetimes.sum()
                            p = lifetimes / total if total > 0 else lifetimes
                            entropy = float(
                                -np.sum(p * np.log(p + 1e-15))
                            ) if total > 0 else 0.0
                            feats[f"entropy_{dim}"].append(entropy)
                            feats[f"total_pers_{dim}"].append(float(total))
                            feats[f"max_pers_{dim}"].append(
                                float(lifetimes.max()))
                        else:
                            for k in [f"entropy_{dim}", f"total_pers_{dim}",
                                      f"max_pers_{dim}"]:
                                feats[k].append(0.0)
                    else:
                        feats[f"betti_{dim}"].append(0)
                        for k in [f"entropy_{dim}", f"total_pers_{dim}",
                                  f"max_pers_{dim}"]:
                            feats[k].append(0.0)

            all_features[input_type][sr] = feats

    # Print tables
    for input_type in ["noise", "sine"]:
        print(f"\n  --- {input_type.upper()} input ---")
        header = (f"  {'SR':>6} | {'Betti-0':>10} | {'Betti-1':>10} | "
                  f"{'Ent-H0':>10} | {'Ent-H1':>10} | "
                  f"{'TotP-H0':>10} | {'TotP-H1':>10} | "
                  f"{'MaxP-H0':>10} | {'MaxP-H1':>10}")
        print(header)
        print(f"  {'-' * 100}")

        for sr in SR_VALUES:
            f = all_features[input_type][sr]

            def fmt(vals):
                if not vals:
                    return "--"
                return f"{np.mean(vals):.2f}±{np.std(vals):.2f}"

            print(f"  {sr:>6.2f} | "
                  f"{fmt(f['betti_0']):>10} | {fmt(f['betti_1']):>10} | "
                  f"{fmt(f['entropy_0']):>10} | {fmt(f['entropy_1']):>10} | "
                  f"{fmt(f['total_pers_0']):>10} | {fmt(f['total_pers_1']):>10} | "
                  f"{fmt(f['max_pers_0']):>10} | {fmt(f['max_pers_1']):>10}")

    # Trend analysis — check all features for monotonic trend with SR
    print(f"\n  Trend analysis (Spearman correlation with SR):")
    has_signal = False
    for input_type in ["noise", "sine"]:
        for feat_name in ["entropy_1", "total_pers_1", "max_pers_1",
                          "betti_1", "entropy_0", "total_pers_0"]:
            vals = [np.mean(all_features[input_type][sr][feat_name])
                    for sr in SR_VALUES]
            rho, p = spearmanr(SR_VALUES, vals)
            sig = "*" if p < 0.05 else " "
            print(f"    {input_type:>5} {feat_name:>12}: rho={rho:+.3f}, "
                  f"p={p:.4f} {sig}")
            if abs(rho) > 0.7 and p < 0.05:
                has_signal = True

    if has_signal:
        print(f"\n  -> CLEAR systematic trend: topological features change with SR")
    else:
        print(f"\n  -> NO CLEAR systematic trend detected (p>0.05 or |rho|<0.7)")

    return all_features, has_signal


# ---------------------------------------------------------------------------
# Part 2: Continuous regime ramp
# ---------------------------------------------------------------------------

def part2_ramp_detection():
    """Ramp SR from 0.5 to 1.3 over 20k steps, detect transitions."""
    print(f"\n{'#' * 72}")
    print("PART 2: CONTINUOUS REGIME RAMP (SR 0.5 -> 1.3)")
    print(f"{'#' * 72}")
    print(f"  {N_STEPS_RAMP} timesteps, ground truth at SR=0.9 and SR=1.1")
    print(f"  Tolerance: +/-{TOLERANCE} samples\n")

    method_names = ["Topological", "Spectral", "Variance", "BOCPD"]
    all_results = {m: [] for m in method_names}
    all_sr_at_det = {m: [] for m in method_names}

    for seed in range(SEED, SEED + N_SEEDS):
        print(f"  --- Seed {seed} ---")
        rng = np.random.default_rng(seed)
        esn = MinimalESN(spectral_radius=SR_RAMP_START, seed=seed)

        inputs = rng.standard_normal(N_STEPS_RAMP) * 0.1
        sr_schedule = np.linspace(SR_RAMP_START, SR_RAMP_END, N_STEPS_RAMP)

        states = esn.run_dynamic(inputs, sr_schedule)
        signal = states[WASHOUT:, 0]
        sr_after = sr_schedule[WASHOUT:]

        # Ground truth: where SR crosses 0.9 and 1.1
        gt_09 = int(np.argmin(np.abs(sr_after - 0.9)))
        gt_11 = int(np.argmin(np.abs(sr_after - 1.1)))
        transitions = np.array([gt_09, gt_11])
        print(f"    Signal: {len(signal)} samples, transitions at "
              f"{gt_09} (SR=0.9), {gt_11} (SR=1.1)")

        results = run_all_methods(signal, transitions, TOLERANCE,
                                  seed_label=f"-ramp-s{seed}")

        for name in method_names:
            all_results[name].append(results[name])
            # Record SR at each detection
            for d in results[name]["detections"]:
                if 0 <= d < len(sr_after):
                    all_sr_at_det[name].append(float(sr_after[d]))

    # Summary table
    print(f"\n  {'=' * 72}")
    print(f"  PART 2 RESULTS (mean +/- std over {N_SEEDS} seeds)")
    print(f"  {'=' * 72}")
    print(f"  {'Method':<16} {'Precision':>12} {'Recall':>12} {'F1':>12} "
          f"{'Mean Lag':>12} {'SR@det':>12}")
    print(f"  {'-' * 72}")

    for name in method_names:
        rs = all_results[name]
        precs = [r["precision"] for r in rs]
        recs = [r["recall"] for r in rs]
        f1s = [r["f1"] for r in rs]
        lags = [r["mean_lag"] for r in rs if not np.isnan(r["mean_lag"])]
        srs = all_sr_at_det[name]

        p_s = f"{np.mean(precs):.2f}+/-{np.std(precs):.2f}"
        r_s = f"{np.mean(recs):.2f}+/-{np.std(recs):.2f}"
        f_s = f"{np.mean(f1s):.2f}+/-{np.std(f1s):.2f}"
        l_s = (f"{np.mean(lags):+.0f}+/-{np.std(lags):.0f}"
               if lags else "--")
        sr_s = (f"{np.mean(srs):.2f}+/-{np.std(srs):.2f}"
                if srs else "--")

        print(f"  {name:<16} {p_s:>12} {r_s:>12} {f_s:>12} "
              f"{l_s:>12} {sr_s:>12}")

    return all_results


# ---------------------------------------------------------------------------
# Part 3: Discrete regime switches
# ---------------------------------------------------------------------------

def part3_discrete_switches():
    """Switch abruptly between SR=0.7 and SR=1.1 every 3000 steps."""
    print(f"\n{'#' * 72}")
    print(f"PART 3: DISCRETE REGIME SWITCHES (SR {SR_LOW} <-> {SR_HIGH})")
    print(f"{'#' * 72}")
    print(f"  {N_STEPS_SWITCH} timesteps, switch every {SWITCH_EVERY}, "
          f"tolerance=+/-{TOLERANCE}\n")

    method_names = ["Topological", "Spectral", "Variance", "BOCPD"]
    all_results = {m: [] for m in method_names}

    for seed in range(SEED, SEED + N_SEEDS):
        print(f"  --- Seed {seed} ---")
        rng = np.random.default_rng(seed)
        esn = MinimalESN(spectral_radius=SR_LOW, seed=seed)

        inputs = rng.standard_normal(N_STEPS_SWITCH) * 0.1

        # Build switching schedule: 0.7, 1.1, 0.7, 1.1, 0.7
        n_segments = N_STEPS_SWITCH // SWITCH_EVERY
        schedule = []
        sr_current = SR_LOW
        for i in range(n_segments):
            schedule.append((SWITCH_EVERY, sr_current))
            sr_current = SR_HIGH if sr_current == SR_LOW else SR_LOW

        states = esn.run_switching(inputs, schedule)
        signal = states[:, 0]

        # Ground truth at switch boundaries
        transitions = np.array(
            [SWITCH_EVERY * i for i in range(1, n_segments)]
        )
        print(f"    Signal: {len(signal)} samples, {len(transitions)} "
              f"transitions at {transitions.tolist()}")

        results = run_all_methods(signal, transitions, TOLERANCE,
                                  seed_label=f"-sw-s{seed}")
        for name in method_names:
            all_results[name].append(results[name])

    # Summary table
    print(f"\n  {'=' * 72}")
    print(f"  PART 3 RESULTS (mean +/- std over {N_SEEDS} seeds)")
    print(f"  {'=' * 72}")
    print(f"  {'Method':<16} {'Precision':>12} {'Recall':>12} {'F1':>12} "
          f"{'Mean Lag':>12} {'Detections':>12}")
    print(f"  {'-' * 72}")

    best_f1 = 0.0
    best_method = None

    for name in method_names:
        rs = all_results[name]
        precs = [r["precision"] for r in rs]
        recs = [r["recall"] for r in rs]
        f1s = [r["f1"] for r in rs]
        lags = [r["mean_lag"] for r in rs if not np.isnan(r["mean_lag"])]
        dets = [r["n_detected"] for r in rs]

        mean_f1 = float(np.mean(f1s))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_method = name

        p_s = f"{np.mean(precs):.2f}+/-{np.std(precs):.2f}"
        r_s = f"{np.mean(recs):.2f}+/-{np.std(recs):.2f}"
        f_s = f"{np.mean(f1s):.2f}+/-{np.std(f1s):.2f}"
        l_s = (f"{np.mean(lags):+.0f}+/-{np.std(lags):.0f}"
               if lags else "--")
        d_s = f"{np.mean(dets):.1f}+/-{np.std(dets):.1f}"

        print(f"  {name:<16} {p_s:>12} {r_s:>12} {f_s:>12} "
              f"{l_s:>12} {d_s:>12}")

    return all_results, best_method


# ---------------------------------------------------------------------------
# Part 4: Coupled reservoirs
# ---------------------------------------------------------------------------

def part4_coupled_reservoirs():
    """Test binding detection between two ESNs with shared vs independent input."""
    print(f"\n{'#' * 72}")
    print("PART 4: COUPLED RESERVOIR BINDING")
    print(f"{'#' * 72}")
    print(f"  Two ESNs: SR={SR_LOW} (ordered) + SR={SR_HIGH} (chaotic)")
    print(f"  Coupling: shared input (coupled) vs independent input (uncoupled)\n")

    from att.binding.detector import BindingDetector

    coupled_scores = []
    uncoupled_scores = []
    coupled_pvals = []
    uncoupled_pvals = []

    n_steps = 5000

    for seed in range(SEED, SEED + N_SEEDS):
        print(f"  --- Seed {seed} ---")
        rng = np.random.default_rng(seed)

        shared_input = rng.standard_normal(n_steps) * 0.1
        indep_input1 = rng.standard_normal(n_steps) * 0.1
        indep_input2 = rng.standard_normal(n_steps) * 0.1

        # Coupled: same input to both reservoirs
        esn1 = MinimalESN(spectral_radius=SR_LOW, seed=seed)
        esn2 = MinimalESN(spectral_radius=SR_HIGH, seed=seed + 1000)
        s1_c = esn1.run(shared_input)[WASHOUT:, 0]
        s2_c = esn2.run(shared_input)[WASHOUT:, 0]

        # Uncoupled: independent inputs
        esn3 = MinimalESN(spectral_radius=SR_LOW, seed=seed)
        esn4 = MinimalESN(spectral_radius=SR_HIGH, seed=seed + 1000)
        s1_u = esn3.run(indep_input1)[WASHOUT:, 0]
        s2_u = esn4.run(indep_input2)[WASHOUT:, 0]

        for label, x, y, scores, pvals in [
            ("Coupled", s1_c, s2_c, coupled_scores, coupled_pvals),
            ("Uncoupled", s1_u, s2_u, uncoupled_scores, uncoupled_pvals),
        ]:
            try:
                bd = BindingDetector(max_dim=1, image_resolution=PI_RES)
                bd.fit(x, y, subsample=200, seed=seed)
                result = bd.test_significance(
                    n_surrogates=19, seed=seed, subsample=200
                )
                score = result["observed_score"]
                p_val = result["p_value"]
                z = result["z_score"]
                scores.append(score)
                pvals.append(p_val)
                print(f"    {label}: score={score:.4f}, z={z:.2f}, "
                      f"p={p_val:.4f}")
            except Exception as e:
                print(f"    {label}: FAILED ({e})")
                scores.append(0.0)
                pvals.append(1.0)

    # Summary
    print(f"\n  Binding scores (mean +/- std):")
    print(f"    Coupled:   {np.mean(coupled_scores):.4f} +/- "
          f"{np.std(coupled_scores):.4f}  "
          f"(p: {[f'{p:.3f}' for p in coupled_pvals]})")
    print(f"    Uncoupled: {np.mean(uncoupled_scores):.4f} +/- "
          f"{np.std(uncoupled_scores):.4f}  "
          f"(p: {[f'{p:.3f}' for p in uncoupled_pvals]})")

    binding_detected = False
    if len(coupled_scores) >= 2 and len(uncoupled_scores) >= 2:
        from scipy.stats import mannwhitneyu
        try:
            u, p = mannwhitneyu(coupled_scores, uncoupled_scores,
                                alternative="greater")
            print(f"    Mann-Whitney U={u:.0f}, p={p:.4f}")
            if (np.mean(coupled_scores) > np.mean(uncoupled_scores)
                    and p < 0.1):
                print(f"    -> Binding detector DETECTS inter-reservoir "
                      f"coupling (trend)")
                binding_detected = True
            else:
                print(f"    -> Binding detector does NOT reliably detect "
                      f"coupling")
        except Exception as e:
            print(f"    -> Statistical test failed: {e}")

    return coupled_scores, uncoupled_scores, binding_detected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 72)
    print("RESERVOIR REGIME DETECTION: TOPOLOGICAL CHANGEPOINTS ON ESNs")
    print("=" * 72)
    print(f"Workers: {N_JOBS}, window={WINDOW_SIZE}, step={STEP_SIZE}, "
          f"subsample={SUBSAMPLE}, seeds={N_SEEDS}\n")

    # Part 1
    t1 = time.time()
    _, has_trend = part1_static_classification()
    print(f"\n  Part 1 time: {time.time()-t1:.1f}s")

    # Part 2
    t2 = time.time()
    ramp_results = part2_ramp_detection()
    print(f"\n  Part 2 time: {time.time()-t2:.1f}s")

    # Part 3
    t3 = time.time()
    switch_results, best_switch_method = part3_discrete_switches()
    print(f"\n  Part 3 time: {time.time()-t3:.1f}s")

    # Part 4: only if at least one part shows signal
    method_names = ["Topological", "Spectral", "Variance", "BOCPD"]
    topo_f1_ramp = float(np.mean(
        [r["f1"] for r in ramp_results["Topological"]]
    ))
    topo_f1_switch = float(np.mean(
        [r["f1"] for r in switch_results["Topological"]]
    ))
    any_signal = has_trend or topo_f1_ramp > 0.3 or topo_f1_switch > 0.3

    binding_detected = False
    if any_signal:
        t4 = time.time()
        _, _, binding_detected = part4_coupled_reservoirs()
        print(f"\n  Part 4 time: {time.time()-t4:.1f}s")
    else:
        print(f"\n{'#' * 72}")
        print("PART 4: SKIPPED (no clear signal in Parts 1-3)")
        print(f"{'#' * 72}")

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")

    # Part 1
    print(f"\n  Part 1 (Static classification):")
    if has_trend:
        print(f"    PASS -- Topological features change systematically "
              f"with spectral radius")
    else:
        print(f"    FAIL -- No systematic trend in topological features "
              f"vs spectral radius")

    # Part 2
    print(f"\n  Part 2 (Continuous ramp):")
    for name in method_names:
        f1 = np.mean([r["f1"] for r in ramp_results[name]])
        print(f"    {name}: mean F1={f1:.2f}")
    if topo_f1_ramp > 0.3:
        print(f"    -> Topology DETECTS regime ramp (F1={topo_f1_ramp:.2f})")
    else:
        print(f"    -> Topology does NOT detect regime ramp "
              f"(F1={topo_f1_ramp:.2f})")

    # Part 3
    print(f"\n  Part 3 (Discrete switches):")
    for name in method_names:
        f1 = np.mean([r["f1"] for r in switch_results[name]])
        lag_vals = [r["mean_lag"] for r in switch_results[name]
                    if not np.isnan(r["mean_lag"])]
        lag_s = f", lag={np.mean(lag_vals):+.0f}" if lag_vals else ""
        print(f"    {name}: mean F1={f1:.2f}{lag_s}")

    other_f1s = [np.mean([r["f1"] for r in switch_results[m]])
                 for m in ["Spectral", "Variance", "BOCPD"]]
    topo_wins = topo_f1_switch > max(other_f1s) if other_f1s else False

    if topo_wins:
        print(f"    -> Topology WINS on discrete switches")
    else:
        print(f"    -> Topology does NOT win (best: {best_switch_method})")

    # Part 4
    if any_signal:
        print(f"\n  Part 4 (Coupled reservoirs):")
        if binding_detected:
            print(f"    -> Binding detector DETECTS inter-reservoir coupling")
        else:
            print(f"    -> Binding detector does NOT detect coupling")

    # Overall
    print(f"\n  {'=' * 40}")
    print(f"  OVERALL ASSESSMENT")
    print(f"  {'=' * 40}")

    positives = []
    if has_trend:
        positives.append("static features trend with SR")
    if topo_f1_ramp > 0.3:
        positives.append(f"ramp detection F1={topo_f1_ramp:.2f}")
    if topo_f1_switch > 0.3:
        positives.append(f"switch detection F1={topo_f1_switch:.2f}")
    if topo_wins:
        positives.append("beats alternatives on discrete switches")
    if binding_detected:
        positives.append("inter-reservoir binding detected")

    if positives:
        print(f"\n  TOPOLOGY DETECTS RESERVOIR REGIME TRANSITIONS")
        for p in positives:
            print(f"    + {p}")
        useful = topo_f1_switch > 0.5 or topo_f1_ramp > 0.5
        if useful:
            print(f"\n  Practical utility: YES -- viable for neuromorphic "
                  f"system monitoring")
        else:
            print(f"\n  Practical utility: MARGINAL -- detects but "
                  f"performance is modest")
    else:
        print(f"\n  TOPOLOGY DOES NOT RELIABLY DETECT RESERVOIR REGIME "
              f"TRANSITIONS")
        print(f"  The spectral radius transition does not produce "
              f"sufficient topological")
        print(f"  change in the attractor structure for PH to capture.")

    # Early detection comparison
    if topo_f1_switch > 0.3:
        topo_lags = [r["mean_lag"] for r in switch_results["Topological"]
                     if not np.isnan(r["mean_lag"])]
        if topo_lags:
            tl = np.mean(topo_lags)
            print(f"\n  Early detection (Part 3, topology lag={tl:+.0f}):")
            for name in ["Spectral", "Variance", "BOCPD"]:
                olags = [r["mean_lag"] for r in switch_results[name]
                         if not np.isnan(r["mean_lag"])]
                if olags:
                    ol = np.mean(olags)
                    diff = ol - tl
                    earlier = "EARLIER" if diff > 0 else "LATER"
                    print(f"    vs {name}: {ol:+.0f} (topology {diff:+.0f} "
                          f"samples {earlier})")

    print(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
