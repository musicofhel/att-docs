#!/usr/bin/env python3
"""Reservoir degradation monitoring: topology as early warning for hardware drift.

Tests whether topological features of reservoir dynamics can detect structural
degradation before standard task-performance metrics (MC, NARMA error).

Five parts:
  1. Simulated weight degradation (3 modes x 3 seeds)
  2. Detection lag analysis (topology vs MC vs NARMA, paired Wilcoxon)
  3. Feature-degradation specificity (which topo features detect which mode)
  4. Comparison with simple monitors (variance, SR estimate, Lyapunov)
  5. Online monitoring cost (computation time per window)

Usage:
    python scripts/reservoir_degradation_monitor.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
from numpy.linalg import lstsq, svd, eigvals
from scipy.stats import spearmanr, wilcoxon

from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_NEURONS = 200
SR_INIT = 0.95
SPARSITY = 0.9
INPUT_SCALING = 0.1
N_HEALTHY = 5000
N_DEGRADE = 10000
N_TOTAL = N_HEALTHY + N_DEGRADE  # 15000
WINDOW = 2000
STRIDE = 200
MAX_LAG_MC = 30
N_SEEDS = 3
BASE_SEED = 42
SUBSAMPLE = 400
SIGMA_ALARM = 2.0

MODES = ["noise", "sr_drift", "connection_death"]
MODE_LABELS = {
    "noise": "Weight Noise",
    "sr_drift": "Spectral Radius Drift",
    "connection_death": "Connection Death",
}
TOPO_FEATURES = ["persistence_entropy", "betti_1", "max_H1_pers"]


# ---------------------------------------------------------------------------
# Minimal Echo State Network (from snn_topology_deepen.py)
# ---------------------------------------------------------------------------

class MinimalESN:
    """ESN with adjustable spectral radius. Stores W_base for fast rescaling."""

    def __init__(self, n_neurons=N_NEURONS, spectral_radius=0.9,
                 input_scaling=INPUT_SCALING, sparsity=SPARSITY, seed=42):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n_neurons, n_neurons))
        W[rng.random((n_neurons, n_neurons)) < sparsity] = 0
        sr = np.max(np.abs(eigvals(W)))
        if sr < 1e-10:
            sr = 1.0
        self.W_base = W / sr
        self.W = self.W_base * spectral_radius
        self.W_in = rng.standard_normal((n_neurons, 1)) * input_scaling
        self.state = np.zeros(n_neurons)
        self.n_neurons = n_neurons

    def run(self, inputs):
        """inputs: (n_steps,) -> states: (n_steps, n_neurons)"""
        states = np.zeros((len(inputs), self.n_neurons))
        for i, u in enumerate(inputs):
            self.state = np.tanh(self.W @ self.state + self.W_in.ravel() * u)
            states[i] = self.state
        return states

    def reset(self):
        self.state = np.zeros(self.n_neurons)


# ---------------------------------------------------------------------------
# NARMA-10
# ---------------------------------------------------------------------------

def generate_narma10(n_steps, seed=None):
    """Generate NARMA-10 time series."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n_steps)
    y = np.zeros(n_steps)
    for t in range(10, n_steps):
        y[t] = (0.3 * y[t - 1]
                + 0.05 * y[t - 1] * np.sum(y[t - 10:t])
                + 1.5 * u[t - 1] * u[t - 10]
                + 0.1)
        y[t] = np.clip(y[t], -10, 10)
    return u, y


# ---------------------------------------------------------------------------
# Topological feature extraction
# ---------------------------------------------------------------------------

ZERO_FEATURES = {k: 0.0 for k in TOPO_FEATURES}


def _features_from_pa(pa):
    """Extract persistence_entropy, betti_1, max_H1_pers from PersistenceAnalyzer."""
    features = {}
    dgm1 = pa.diagrams_[1] if len(pa.diagrams_) > 1 else np.array([])
    if len(dgm1) > 0:
        lifetimes = dgm1[:, 1] - dgm1[:, 0]
        lifetimes = lifetimes[lifetimes > 1e-10]
        features["betti_1"] = len(lifetimes)
        if len(lifetimes) > 0:
            total = float(lifetimes.sum())
            features["max_H1_pers"] = float(lifetimes.max())
            p = lifetimes / total if total > 0 else lifetimes
            features["persistence_entropy"] = float(
                -np.sum(p * np.log(p + 1e-15))
            ) if total > 0 else 0.0
        else:
            features["max_H1_pers"] = 0.0
            features["persistence_entropy"] = 0.0
    else:
        features["betti_1"] = 0
        features["max_H1_pers"] = 0.0
        features["persistence_entropy"] = 0.0
    return features


def extract_topo_cloud(cloud, subsample=SUBSAMPLE, seed=BASE_SEED):
    """PH on PCA-reduced 3D point cloud (no Takens)."""
    if np.std(cloud) < 1e-15:
        return dict(ZERO_FEATURES)
    pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return _features_from_pa(pa)


def extract_topo_1d(signal, subsample=SUBSAMPLE, seed=BASE_SEED):
    """Takens embedding -> PH -> features (single neuron time series)."""
    if np.std(signal) < 1e-15:
        return dict(ZERO_FEATURES)
    try:
        embedder = TakensEmbedder("auto", "auto")
        embedder.fit(signal[:min(len(signal), 20000)])
    except Exception:
        embedder = TakensEmbedder(delay=10, dimension=3)
        embedder.fit(signal)
    cloud = embedder.transform(signal)
    pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return _features_from_pa(pa)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca_reduce(states, n_components=3):
    """Project states to first n PCA components (no washout — window is clean)."""
    X_c = states - states.mean(axis=0)
    if np.std(X_c) < 1e-15:
        return X_c[:, :n_components]
    _, _, Vt = svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


# ---------------------------------------------------------------------------
# Degradation simulation
# ---------------------------------------------------------------------------

def simulate_degradation(mode, seed):
    """Run ESN step-by-step with gradual weight degradation.

    Returns (states, inputs, narma_targets) — all shape (N_TOTAL,) or (N_TOTAL, N_NEURONS).
    """
    esn = MinimalESN(n_neurons=N_NEURONS, spectral_radius=SR_INIT, seed=seed)
    rng = np.random.default_rng(seed + 7777)

    u, y_narma = generate_narma10(N_TOTAL, seed=seed)

    W_original = esn.W.copy()
    original_fro = np.linalg.norm(W_original, "fro")

    # Pre-compute for connection_death: fixed random thresholds per weight
    death_random = rng.random(esn.W.shape) if mode == "connection_death" else None

    states = np.zeros((N_TOTAL, N_NEURONS))

    for t in range(N_TOTAL):
        if t >= N_HEALTHY:
            progress = (t - N_HEALTHY) / N_DEGRADE  # 0 → 1

            if mode == "noise":
                sigma = 0.05 * progress
                noise = rng.normal(0, max(sigma, 1e-15), W_original.shape)
                W_noisy = W_original + noise
                noisy_fro = np.linalg.norm(W_noisy, "fro")
                if noisy_fro > 1e-10:
                    esn.W = W_noisy * (original_fro / noisy_fro)

            elif mode == "sr_drift":
                new_sr = SR_INIT + (1.4 - SR_INIT) * progress
                esn.W = esn.W_base * new_sr

            elif mode == "connection_death":
                death_prob = 0.3 * progress
                mask = death_random > death_prob  # cumulative: once dead, stays dead
                esn.W = W_original * mask

        esn.state = np.tanh(esn.W @ esn.state + esn.W_in.ravel() * u[t])
        states[t] = esn.state

    return states, u, y_narma


# ---------------------------------------------------------------------------
# Windowed monitors
# ---------------------------------------------------------------------------

def window_mc(states, inputs, max_lag=MAX_LAG_MC):
    """Memory capacity on a window: train on first 1500, test on rest."""
    n = len(states)
    n_train = 1500
    mc = 0.0
    for k in range(1, max_lag + 1):
        target = inputs[:-k]
        X = states[k:]
        if len(X) < n_train + 10:
            continue
        X_tr, X_te = X[:n_train], X[n_train:]
        t_tr, t_te = target[:n_train], target[n_train:]
        if len(t_te) < 10:
            continue
        try:
            W_out, _, _, _ = lstsq(X_tr, t_tr, rcond=None)
            pred = X_te @ W_out
            ss_res = np.sum((t_te - pred) ** 2)
            ss_tot = np.sum((t_te - np.mean(t_te)) ** 2)
            r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
            mc += r2
        except Exception:
            pass
    return mc


def window_narma_error(states, targets):
    """NRMSE on a window: train on first 1500, test on rest."""
    n_train = 1500
    if len(states) < n_train + 50:
        return float("inf")
    X_tr, X_te = states[:n_train], states[n_train:]
    y_tr, y_te = targets[:n_train], targets[n_train:]
    try:
        W_out, _, _, _ = lstsq(X_tr, y_tr, rcond=None)
        pred = X_te @ W_out
        std_test = np.std(y_te)
        if std_test < 1e-10:
            return float("inf")
        return float(np.sqrt(np.mean((y_te - pred) ** 2)) / std_test)
    except Exception:
        return float("inf")


def window_variance(states):
    """Mean variance across neurons in the window."""
    return float(np.mean(np.var(states, axis=0)))


def window_sr_estimate(states):
    """Top eigenvalue of state correlation matrix (effective spectral radius)."""
    X = states - states.mean(axis=0)
    C = X.T @ X / len(X)
    try:
        return float(np.max(np.real(eigvals(C))))
    except Exception:
        return 0.0


def window_lyapunov(states, n_pairs=100, max_steps=50, min_sep=50, seed=42):
    """Largest Lyapunov exponent via divergence of nearby trajectories."""
    rng = np.random.default_rng(seed)
    n = len(states)
    if n < max_steps + min_sep + 10:
        return 0.0
    lyap_vals = []
    for _ in range(n_pairs):
        i = rng.integers(0, n - max_steps)
        candidates = rng.integers(0, n - max_steps, size=50)
        best_j, best_d = -1, np.inf
        for j in candidates:
            if abs(i - j) < min_sep:
                continue
            d = np.linalg.norm(states[i] - states[j])
            if 1e-10 < d < best_d:
                best_d = d
                best_j = j
        if best_j < 0:
            continue
        d_final = np.linalg.norm(states[i + max_steps] - states[best_j + max_steps])
        if d_final < 1e-10:
            lyap_vals.append(-10.0)
            continue
        lyap_vals.append(np.log(d_final / best_d) / max_steps)
    return float(np.median(lyap_vals)) if lyap_vals else 0.0


# ---------------------------------------------------------------------------
# Compute all monitors across the timecourse
# ---------------------------------------------------------------------------

def compute_all_monitors(states, inputs, narma_targets):
    """Sliding-window monitors for a full run.

    Returns (results_dict, measurement_steps).
    results_dict maps monitor_name -> list of values at each measurement step.
    """
    measurement_steps = list(range(WINDOW, N_TOTAL + 1, STRIDE))

    keys = (["mc", "narma_error", "variance", "sr_estimate", "lyapunov"]
            + [f"topo_pop_{f}" for f in TOPO_FEATURES]
            + [f"topo_1n_{f}" for f in TOPO_FEATURES])
    results = {k: [] for k in keys}

    for step in measurement_steps:
        ws = states[step - WINDOW:step]
        wi = inputs[step - WINDOW:step]
        wn = narma_targets[step - WINDOW:step]

        results["mc"].append(window_mc(ws, wi))
        results["narma_error"].append(window_narma_error(ws, wn))
        results["variance"].append(window_variance(ws))
        results["sr_estimate"].append(window_sr_estimate(ws))
        results["lyapunov"].append(window_lyapunov(ws, seed=step))

        cloud = pca_reduce(ws, n_components=3)
        topo_pop = extract_topo_cloud(cloud, seed=step)
        for f in TOPO_FEATURES:
            results[f"topo_pop_{f}"].append(topo_pop[f])

        topo_1n = extract_topo_1d(ws[:, 0], seed=step)
        for f in TOPO_FEATURES:
            results[f"topo_1n_{f}"].append(topo_1n[f])

    return results, measurement_steps


# ---------------------------------------------------------------------------
# Alarm detection
# ---------------------------------------------------------------------------

def find_alarm_step(series, measurement_steps, healthy_end=N_HEALTHY):
    """First step where signal exceeds 2-sigma from healthy baseline.

    Returns (alarm_step_or_None, baseline_mean, baseline_std).
    """
    healthy_vals = [v for v, s in zip(series, measurement_steps) if s <= healthy_end]
    if len(healthy_vals) < 3:
        return None, 0.0, 0.0

    mu = np.mean(healthy_vals)
    sd = np.std(healthy_vals)
    if sd < 1e-15:
        sd = 1e-15  # constant baseline — any fluctuation counts

    for step, val in zip(measurement_steps, series):
        if step <= healthy_end:
            continue
        if abs(val - mu) > SIGMA_ALARM * sd:
            return step, mu, sd

    return None, mu, sd


def find_composite_topo_alarm(results, measurement_steps, prefix="topo_pop"):
    """Earliest alarm across all topology features (composite detector)."""
    earliest, earliest_feat = None, None
    for feat in TOPO_FEATURES:
        alarm, _, _ = find_alarm_step(results[f"{prefix}_{feat}"], measurement_steps)
        if alarm is not None and (earliest is None or alarm < earliest):
            earliest = alarm
            earliest_feat = feat
    return earliest, earliest_feat


# ---------------------------------------------------------------------------
# Part 5: cost timing
# ---------------------------------------------------------------------------

def time_monitors(states_window):
    """Time each monitor on one window. Returns dict of seconds."""
    times = {}

    t0 = time.perf_counter()
    cloud = pca_reduce(states_window, n_components=3)
    _ = extract_topo_cloud(cloud)
    times["topo_pop"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = window_variance(states_window)
    times["variance"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = window_sr_estimate(states_window)
    times["sr_estimate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = window_lyapunov(states_window)
    times["lyapunov"] = time.perf_counter() - t0

    return times


# =========================================================================
# Main
# =========================================================================

def main():
    t_global = time.time()

    print("=" * 72)
    print("RESERVOIR DEGRADATION MONITORING")
    print("Topology as early warning for hardware drift")
    print("=" * 72)
    print(f"  ESN: {N_NEURONS} neurons, SR={SR_INIT}, sparsity={SPARSITY}")
    print(f"  Timeline: {N_HEALTHY} healthy + {N_DEGRADE} degradation = {N_TOTAL} steps")
    print(f"  Window: {WINDOW} steps, stride: {STRIDE}")
    print(f"  Modes: {', '.join(MODE_LABELS.values())}")
    print(f"  Seeds: {N_SEEDS} per mode")
    print(f"  Alarm threshold: {SIGMA_ALARM}-sigma from healthy baseline")
    print()

    # Storage
    all_results = {}   # (mode, seed) -> (results, msteps)
    all_alarms = {}    # (mode, seed) -> {monitor: alarm_step}

    # =================================================================
    # PART 1: Simulated weight degradation
    # =================================================================
    print("=" * 72)
    print("PART 1: SIMULATED WEIGHT DEGRADATION")
    print("=" * 72)

    for mode in MODES:
        print(f"\n  --- {MODE_LABELS[mode]} ---")
        for si in range(N_SEEDS):
            seed = BASE_SEED + si
            t0 = time.time()
            print(f"  seed {seed}: simulate...", end="", flush=True)
            states, inputs, narma = simulate_degradation(mode, seed)
            t1 = time.time()
            print(f" {t1-t0:.1f}s → monitors...", end="", flush=True)
            results, msteps = compute_all_monitors(states, inputs, narma)
            print(f" {time.time()-t1:.1f}s", flush=True)

            all_results[(mode, seed)] = (results, msteps)

            alarms = {}
            alarms["mc"], _, _ = find_alarm_step(results["mc"], msteps)
            alarms["narma_error"], _, _ = find_alarm_step(
                results["narma_error"], msteps)
            alarms["topo_pop"], alarms["topo_pop_feat"] = \
                find_composite_topo_alarm(results, msteps, "topo_pop")
            alarms["topo_1n"], alarms["topo_1n_feat"] = \
                find_composite_topo_alarm(results, msteps, "topo_1n")
            alarms["variance"], _, _ = find_alarm_step(
                results["variance"], msteps)
            alarms["sr_estimate"], _, _ = find_alarm_step(
                results["sr_estimate"], msteps)
            alarms["lyapunov"], _, _ = find_alarm_step(
                results["lyapunov"], msteps)
            all_alarms[(mode, seed)] = alarms

    # =================================================================
    # PARTS 1 & 4: Alarm-time tables per degradation mode
    # =================================================================
    print(f"\n{'=' * 72}")
    print("PARTS 1 & 4: ALARM TIMES PER DEGRADATION MODE")
    print(f"{'=' * 72}")

    monitor_order = ["topo_pop", "topo_1n", "variance", "sr_estimate", "lyapunov"]
    monitor_nice = {
        "topo_pop": "Topology (PCA pop)",
        "topo_1n": "Topology (single neuron)",
        "variance": "Variance",
        "sr_estimate": "SR estimate",
        "lyapunov": "Lyapunov",
    }

    for mode in MODES:
        print(f"\n  DEGRADATION MODE: {MODE_LABELS[mode]} ({N_SEEDS} seeds)")
        print(f"  {'Monitor':<26} {'Alarm Step':>18} "
              f"{'Lag vs MC':>12} {'Lag vs NARMA':>14}")
        print(f"  {'-' * 72}")

        for mon in monitor_order:
            steps_list, lags_mc, lags_nr = [], [], []
            feat_set = set()
            for si in range(N_SEEDS):
                seed = BASE_SEED + si
                a = all_alarms[(mode, seed)]
                s = a[mon]
                if s is not None:
                    steps_list.append(s)
                    if a["mc"] is not None:
                        lags_mc.append(s - a["mc"])
                    if a["narma_error"] is not None:
                        lags_nr.append(s - a["narma_error"])
                f = a.get(f"{mon}_feat")
                if f:
                    feat_set.add(f)

            s_str = (f"{np.mean(steps_list):.0f}±{np.std(steps_list):.0f}"
                     if steps_list else "never")
            mc_str = (f"{np.mean(lags_mc):+.0f}±{np.std(lags_mc):.0f}"
                      if lags_mc else "n/a")
            nr_str = (f"{np.mean(lags_nr):+.0f}±{np.std(lags_nr):.0f}"
                      if lags_nr else "n/a")
            feat_note = f"  [{', '.join(sorted(feat_set))}]" if feat_set else ""
            print(f"  {monitor_nice.get(mon, mon):<26} {s_str:>18} "
                  f"{mc_str:>12} {nr_str:>14}{feat_note}")

        # Reference: MC and NARMA alarm times
        for ref, label in [("mc", "Memory Capacity"),
                           ("narma_error", "NARMA Error")]:
            vals = [all_alarms[(mode, BASE_SEED + i)][ref]
                    for i in range(N_SEEDS)]
            valid = [v for v in vals if v is not None]
            s = (f"{np.mean(valid):.0f}±{np.std(valid):.0f}"
                 if valid else "never")
            print(f"  {'(ref) ' + label:<26} {s:>18}")

    # =================================================================
    # PART 2: Detection lag analysis (paired Wilcoxon across all modes)
    # =================================================================
    print(f"\n{'=' * 72}")
    print("PART 2: DETECTION LAG ANALYSIS (paired Wilcoxon, all mode×seed)")
    print(f"{'=' * 72}")

    test_monitors = ["topo_pop", "variance", "sr_estimate", "lyapunov"]

    for mon in test_monitors:
        lags_mc, lags_nr = [], []
        for mode in MODES:
            for si in range(N_SEEDS):
                seed = BASE_SEED + si
                a = all_alarms[(mode, seed)]
                s = a[mon]
                if s is not None and a["mc"] is not None:
                    lags_mc.append(s - a["mc"])
                if s is not None and a["narma_error"] is not None:
                    lags_nr.append(s - a["narma_error"])

        print(f"\n  {monitor_nice.get(mon, mon)}:")
        for label, lags in [("MC", lags_mc), ("NARMA", lags_nr)]:
            if len(lags) >= 3:
                try:
                    _, p = wilcoxon(lags)
                except ValueError:
                    p = 1.0
                direction = "EARLIER" if np.mean(lags) < 0 else "LATER"
                print(f"    vs {label:>5}: mean lag = {np.mean(lags):+.0f} steps  "
                      f"({direction}, n={len(lags)}, Wilcoxon p={p:.4f})")
            else:
                print(f"    vs {label:>5}: insufficient data (n={len(lags)})")

    # =================================================================
    # PART 3: Which features detect which degradation?
    # =================================================================
    print(f"\n{'=' * 72}")
    print("PART 3: WHICH FEATURES DETECT WHICH DEGRADATION?")
    print("  Spearman rho: topology feature vs MC / NARMA error (degradation portion)")
    print(f"{'=' * 72}")

    for mode in MODES:
        print(f"\n  --- {MODE_LABELS[mode]} ---")
        all_mc, all_nr = [], []
        feat_pool = {f: [] for f in TOPO_FEATURES}

        for si in range(N_SEEDS):
            seed = BASE_SEED + si
            results, msteps = all_results[(mode, seed)]
            for i, step in enumerate(msteps):
                if step > N_HEALTHY:
                    all_mc.append(results["mc"][i])
                    all_nr.append(results["narma_error"][i])
                    for f in TOPO_FEATURES:
                        feat_pool[f].append(results[f"topo_pop_{f}"][i])

        arr_mc, arr_nr = np.array(all_mc), np.array(all_nr)

        print(f"  {'Feature':<25} {'rho(MC)':>10} {'p':>8} "
              f"{'rho(NARMA)':>12} {'p':>8}")
        print(f"  {'-' * 66}")

        for f in TOPO_FEATURES:
            fa = np.array(feat_pool[f])
            if np.std(fa) < 1e-15:
                rho_m, p_m, rho_n, p_n = 0., 1., 0., 1.
            else:
                rho_m, p_m = spearmanr(fa, arr_mc) if np.std(arr_mc) > 1e-15 else (0., 1.)
                rho_n, p_n = spearmanr(fa, arr_nr) if np.std(arr_nr) > 1e-15 else (0., 1.)
            sig_m = "***" if p_m < .001 else "**" if p_m < .01 else "*" if p_m < .05 else ""
            sig_n = "***" if p_n < .001 else "**" if p_n < .01 else "*" if p_n < .05 else ""
            print(f"  {f:<25} {rho_m:>+8.3f}{sig_m:<3} {p_m:>7.4f} "
                  f"{rho_n:>+10.3f}{sig_n:<3} {p_n:>7.4f}")

    # =================================================================
    # PART 5: Online monitoring cost
    # =================================================================
    print(f"\n{'=' * 72}")
    print("PART 5: ONLINE MONITORING COST")
    print(f"{'=' * 72}")

    esn_t = MinimalESN(n_neurons=N_NEURONS, spectral_radius=SR_INIT, seed=99)
    rng_t = np.random.default_rng(99)
    u_t = rng_t.uniform(0, 0.5, WINDOW + 1000)
    st_t = esn_t.run(u_t)
    w_t = st_t[1000:]  # representative post-warmup window

    n_reps = 5
    cost_all = {k: [] for k in test_monitors}
    for _ in range(n_reps):
        t = time_monitors(w_t)
        for k in test_monitors:
            cost_all[k].append(t[k])

    print(f"\n  {'Monitor':<26} {'Time / window':>16}")
    print(f"  {'-' * 44}")
    for mon in test_monitors:
        v = np.mean(cost_all[mon])
        if v < 0.001:
            print(f"  {monitor_nice[mon]:<26} {v*1000:>13.2f} ms")
        else:
            print(f"  {monitor_nice[mon]:<26} {v:>13.3f} s")

    # =================================================================
    # GRAND SUMMARY
    # =================================================================
    print(f"\n{'=' * 72}")
    print("GRAND SUMMARY: EARLY WARNING PERFORMANCE (across all modes)")
    print(f"{'=' * 72}")

    print(f"\n  {'Monitor':<26} {'Mean lag MC':>12} {'Mean lag NARMA':>15} "
          f"{'Wilcox p(MC)':>13} {'Wilcox p(NR)':>13} {'Cost':>10}")
    print(f"  {'-' * 91}")

    for mon in test_monitors:
        lags_mc, lags_nr = [], []
        for mode in MODES:
            for si in range(N_SEEDS):
                seed = BASE_SEED + si
                a = all_alarms[(mode, seed)]
                s = a[mon]
                if s is not None and a["mc"] is not None:
                    lags_mc.append(s - a["mc"])
                if s is not None and a["narma_error"] is not None:
                    lags_nr.append(s - a["narma_error"])

        ml_mc = f"{np.mean(lags_mc):+.0f}" if lags_mc else "n/a"
        ml_nr = f"{np.mean(lags_nr):+.0f}" if lags_nr else "n/a"

        def _wilcox_p(vals):
            if len(vals) < 3:
                return "n/a"
            try:
                _, p = wilcoxon(vals)
                return f"{p:.4f}"
            except ValueError:
                return "n/a"

        p_mc = _wilcox_p(lags_mc)
        p_nr = _wilcox_p(lags_nr)

        c = np.mean(cost_all[mon])
        c_str = f"{c*1000:.1f}ms" if c < 0.001 else f"{c:.3f}s"

        print(f"  {monitor_nice[mon]:<26} {ml_mc:>12} {ml_nr:>15} "
              f"{p_mc:>13} {p_nr:>13} {c_str:>10}")

    # Verdict
    print(f"\n  VERDICT:")

    topo_lags_mc, topo_lags_nr = [], []
    for mode in MODES:
        for si in range(N_SEEDS):
            seed = BASE_SEED + si
            a = all_alarms[(mode, seed)]
            s = a["topo_pop"]
            if s is not None and a["mc"] is not None:
                topo_lags_mc.append(s - a["mc"])
            if s is not None and a["narma_error"] is not None:
                topo_lags_nr.append(s - a["narma_error"])

    n_early_mc = sum(1 for l in topo_lags_mc if l < 0)
    n_early_nr = sum(1 for l in topo_lags_nr if l < 0)

    print(f"    Topology alarms before MC:    {n_early_mc}/{len(topo_lags_mc)} cases"
          f" (mean lead = {-np.mean(topo_lags_mc):.0f} steps)" if topo_lags_mc else
          "    Topology alarms before MC:    no data")
    print(f"    Topology alarms before NARMA: {n_early_nr}/{len(topo_lags_nr)} cases"
          f" (mean lead = {-np.mean(topo_lags_nr):.0f} steps)" if topo_lags_nr else
          "    Topology alarms before NARMA: no data")

    if len(topo_lags_mc) > 0 and n_early_mc > len(topo_lags_mc) / 2:
        print(f"    → Topology IS an early warning system for reservoir degradation.")
        for mode in MODES:
            ml = []
            for si in range(N_SEEDS):
                seed = BASE_SEED + si
                a = all_alarms[(mode, seed)]
                s, r = a["topo_pop"], a["mc"]
                if s is not None and r is not None:
                    ml.append(s - r)
            if ml:
                print(f"      {MODE_LABELS[mode]}: "
                      f"mean lead = {-np.mean(ml):.0f} steps")
        cost_pop = np.mean(cost_all["topo_pop"])
        if cost_pop < 1.0:
            print(f"    → Compute cost ({cost_pop:.3f}s/window) is acceptable "
                  "for real-time monitoring.")
        else:
            print(f"    → Compute cost ({cost_pop:.1f}s/window) may limit "
                  "real-time use.")
    elif not topo_lags_mc:
        print(f"    → KILL: Neither topology nor performance metrics alarmed.")
        print(f"      Degradation may be too gradual for {SIGMA_ALARM}-sigma "
              "detection in this timeline.")
    else:
        print(f"    → KILL: Topology does NOT consistently detect degradation "
              "before MC/NARMA.")

    # Best feature per mode
    print(f"\n  BEST TOPOLOGY FEATURE PER DEGRADATION MODE:")
    for mode in MODES:
        fc = {}
        for si in range(N_SEEDS):
            f = all_alarms[(mode, BASE_SEED + si)].get("topo_pop_feat")
            if f:
                fc[f] = fc.get(f, 0) + 1
        if fc:
            best = max(fc, key=fc.get)
            print(f"    {MODE_LABELS[mode]:<25} {best} ({fc[best]}/{N_SEEDS} seeds)")
        else:
            print(f"    {MODE_LABELS[mode]:<25} no alarm triggered")

    elapsed = time.time() - t_global
    print(f"\n  Total runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
