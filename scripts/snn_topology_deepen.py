#!/usr/bin/env python3
"""Deepen SNN topology: stability, scaling, spiking fixes, comparison with standard metrics.

Builds on screen_snn_topology.py (Parts 1-2 PASS, Part 3 weak). Five parts:
  1. Stability: 20 seeds x 11 SR, are the rho values stable?
  2. Prediction model: LOO-CV R^2 for MC and NARMA using topology features
  3. Network size scaling: 50/100/200/500 neurons, does the correlation hold?
  4. Fix spiking: (a) tau sweep, (b) PCA pop. rates, (c) subthreshold voltage
  5. Comparison: topology vs Lyapunov, kernel quality, spectral radius

Usage:
    python scripts/snn_topology_deepen.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
from numpy.linalg import lstsq, svd, eigvals
from scipy.stats import spearmanr

from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_SEED = 42
N_SEEDS_STABILITY = 20
N_SEEDS_SCALING = 5
N_SEEDS_SPIKING = 5
N_NEURONS = 100
SPARSITY = 0.9
INPUT_SCALING = 0.1
WASHOUT = 1000
N_STEPS = 5000
MAX_LAG = 50
SUBSAMPLE = 400

SR_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
LIF_CS_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
TAU_VALUES = [5, 10, 20, 50, 100]
SIZE_VALUES = [50, 100, 200, 500]

TOPO_FEATURES = [
    "persistence_entropy",
    "total_H1_pers",
    "max_H1_pers",
    "betti_0",
    "betti_1",
]

RHO_GATE = 0.7


# ---------------------------------------------------------------------------
# Minimal Echo State Network
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

    def set_sr(self, spectral_radius):
        """Rescale to new spectral radius without recomputing eigvals."""
        self.W = self.W_base * spectral_radius
        self.reset()

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
# Minimal LIF Spiking Network
# ---------------------------------------------------------------------------

class MinimalLIF:
    """Leaky integrate-and-fire population with spike recording."""

    def __init__(self, n_neurons=100, sparsity=0.9, tau=20.0, v_thresh=1.0,
                 v_reset=0.0, dt=1.0, input_scaling=0.5, bias=0.8, seed=None):
        rng = np.random.default_rng(seed)
        self.n = n_neurons
        self.tau = tau
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.dt = dt
        self.bias = bias
        W = rng.standard_normal((n_neurons, n_neurons))
        mask = rng.random((n_neurons, n_neurons)) > sparsity
        W *= mask
        n_conn = mask.sum()
        if n_conn > 0:
            W *= np.sqrt(n_neurons) / np.sqrt(n_conn)
        self.W = W
        self.W_in = rng.standard_normal(n_neurons) * input_scaling

    def run_with_spikes(self, inputs, connection_strength=1.0):
        """Run LIF returning (voltages, spike_indicators)."""
        n_steps = len(inputs)
        W_scaled = self.W * connection_strength
        v = np.full(self.n, self.bias * 0.5)
        all_v = np.zeros((n_steps, self.n))
        all_spikes = np.zeros((n_steps, self.n), dtype=bool)
        for t in range(n_steps):
            dv = self.dt / self.tau * (
                -v + self.bias + W_scaled @ np.clip(v, 0, self.v_thresh)
                + self.W_in * inputs[t]
            )
            v = v + dv
            spikes = v > self.v_thresh
            v[spikes] = self.v_reset
            all_spikes[t] = spikes
            all_v[t] = v
        return all_v, all_spikes

    def run_no_reset(self, inputs, connection_strength=1.0):
        """Run LIF WITHOUT spike reset — continuous subthreshold dynamics."""
        n_steps = len(inputs)
        W_scaled = self.W * connection_strength
        v = np.full(self.n, self.bias * 0.5)
        all_v = np.zeros((n_steps, self.n))
        for t in range(n_steps):
            dv = self.dt / self.tau * (
                -v + self.bias + W_scaled @ np.clip(v, 0, self.v_thresh)
                + self.W_in * inputs[t]
            )
            v = v + dv
            all_v[t] = v
        return all_v


# ---------------------------------------------------------------------------
# Spike smoothing and PCA
# ---------------------------------------------------------------------------

def smooth_spikes(spikes, tau_filter, dt=1.0):
    """Exponential smoothing of spike trains -> firing rates."""
    decay = np.exp(-dt / tau_filter)
    n_steps, n_neurons = spikes.shape
    rates = np.zeros((n_steps, n_neurons))
    rate = np.zeros(n_neurons)
    for t in range(n_steps):
        rate = rate * decay + spikes[t].astype(float)
        rates[t] = rate
    return rates


def pca_reduce(states, n_components=3, washout=WASHOUT):
    """Project states to first n PCA components after washout."""
    X = states[washout:]
    X_c = X - X.mean(axis=0)
    if np.std(X_c) < 1e-15:
        return X_c[:, :n_components]
    _, _, Vt = svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


# ---------------------------------------------------------------------------
# Memory Capacity
# ---------------------------------------------------------------------------

def compute_memory_capacity(states, inputs, washout=WASHOUT, max_lag=MAX_LAG):
    """MC = sum of R^2 across lags 1..max_lag (Jaeger 2001)."""
    states = states[washout:]
    inputs_trimmed = inputs[washout:]
    mc = 0.0
    for k in range(1, max_lag + 1):
        target = inputs_trimmed[:-k]
        X = states[k:]
        W_out, _, _, _ = lstsq(X, target, rcond=None)
        pred = X @ W_out
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
        mc += r2
    return mc


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


def evaluate_narma10(states, u, y_target, washout=WASHOUT):
    """NRMSE on 80/20 train/test split."""
    states = states[washout:]
    y_target = y_target[washout:]
    split = int(0.8 * len(states))
    X_train, X_test = states[:split], states[split:]
    y_train, y_test = y_target[:split], y_target[split:]
    W_out, _, _, _ = lstsq(X_train, y_train, rcond=None)
    pred = X_test @ W_out
    std_test = np.std(y_test)
    if std_test < 1e-10:
        return float("inf")
    return np.sqrt(np.mean((y_test - pred) ** 2)) / std_test


# ---------------------------------------------------------------------------
# Topological Feature Extraction
# ---------------------------------------------------------------------------

def _features_from_pa(pa):
    """Extract feature dict from fitted PersistenceAnalyzer."""
    features = {}
    dgm0 = pa.diagrams_[0] if len(pa.diagrams_) > 0 else np.array([])
    if len(dgm0) > 0:
        lifetimes0 = dgm0[:, 1] - dgm0[:, 0]
        lifetimes0 = lifetimes0[lifetimes0 > 1e-10]
        features["betti_0"] = len(lifetimes0)
    else:
        features["betti_0"] = 0

    dgm1 = pa.diagrams_[1] if len(pa.diagrams_) > 1 else np.array([])
    if len(dgm1) > 0:
        lifetimes1 = dgm1[:, 1] - dgm1[:, 0]
        lifetimes1 = lifetimes1[lifetimes1 > 1e-10]
        features["betti_1"] = len(lifetimes1)
        if len(lifetimes1) > 0:
            total = float(lifetimes1.sum())
            features["total_H1_pers"] = total
            features["max_H1_pers"] = float(lifetimes1.max())
            p = lifetimes1 / total if total > 0 else lifetimes1
            features["persistence_entropy"] = float(
                -np.sum(p * np.log(p + 1e-15))
            ) if total > 0 else 0.0
        else:
            features["total_H1_pers"] = 0.0
            features["max_H1_pers"] = 0.0
            features["persistence_entropy"] = 0.0
    else:
        features["betti_1"] = 0
        features["total_H1_pers"] = 0.0
        features["max_H1_pers"] = 0.0
        features["persistence_entropy"] = 0.0
    return features


ZERO_FEATURES = {k: 0.0 for k in TOPO_FEATURES}


def extract_topo_features(signal, subsample=SUBSAMPLE, seed=BASE_SEED):
    """Takens embedding -> PH -> feature dict."""
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


def extract_topo_features_cloud(cloud, subsample=SUBSAMPLE, seed=BASE_SEED):
    """PH directly on multi-D point cloud (no Takens)."""
    if np.std(cloud) < 1e-15:
        return dict(ZERO_FEATURES)
    pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return _features_from_pa(pa)


# ---------------------------------------------------------------------------
# Standard Reservoir Metrics
# ---------------------------------------------------------------------------

def estimate_lyapunov(states, washout=WASHOUT, n_pairs=200, max_steps=50,
                      min_sep=50, seed=42):
    """Largest Lyapunov exponent via divergence of nearby trajectories."""
    rng = np.random.default_rng(seed)
    X = states[washout:]
    n = len(X)
    if n < max_steps + min_sep + 10:
        return 0.0
    lyap_values = []
    for _ in range(n_pairs):
        i = rng.integers(0, n - max_steps)
        candidates = rng.integers(0, n - max_steps, size=min(100, n - max_steps))
        best_j, best_dist = -1, np.inf
        for j in candidates:
            if abs(i - j) < min_sep:
                continue
            d = np.linalg.norm(X[i] - X[j])
            if 1e-10 < d < best_dist:
                best_dist = d
                best_j = j
        if best_j < 0:
            continue
        d_final = np.linalg.norm(X[i + max_steps] - X[best_j + max_steps])
        if d_final < 1e-10:
            lyap_values.append(-10.0)
            continue
        lyap_values.append(np.log(d_final / best_dist) / max_steps)
    return float(np.median(lyap_values)) if lyap_values else 0.0


def compute_kernel_quality(states, washout=WASHOUT, threshold_ratio=0.01):
    """Numerical rank of state matrix (effective dimension)."""
    X = states[washout:]
    s = svd(X, compute_uv=False)
    if len(s) == 0 or s[0] < 1e-15:
        return 0
    threshold = threshold_ratio * s[0]
    return int(np.sum(s > threshold))


def estimate_effective_sr(states, washout=WASHOUT):
    """Effective spectral radius from linear state-to-state model x_{t+1} ~ A x_t."""
    X = states[washout:]
    if len(X) < 3:
        return 0.0
    try:
        A, _, _, _ = lstsq(X[:-1], X[1:], rcond=None)
        return float(np.max(np.abs(eigvals(A))))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# LOO Cross-Validation (hat-matrix method, O(np^2))
# ---------------------------------------------------------------------------

def loo_r2(X, y):
    """Leave-one-out CV R^2 for linear regression with intercept."""
    n = len(y)
    if n < 3:
        return 0.0
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_aug = np.column_stack([X, np.ones(n)])
    try:
        XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)
        H_diag = np.sum((X_aug @ XtX_inv) * X_aug, axis=1)
        W_ols, _, _, _ = lstsq(X_aug, y, rcond=None)
        resid = y - X_aug @ W_ols
        loo_resid = resid / np.maximum(1 - H_diag, 1e-15)
        ss_res = np.sum(loo_resid ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def compute_correlations(metric_values, feature_matrix):
    """Spearman correlations between each topology feature and metric."""
    correlations = {}
    metric_arr = np.array(metric_values)
    for feat_name in TOPO_FEATURES:
        feat_arr = np.array([f[feat_name] for f in feature_matrix])
        if np.std(feat_arr) < 1e-15 or np.std(metric_arr) < 1e-15:
            correlations[feat_name] = (0.0, 1.0)
            continue
        rho, p = spearmanr(feat_arr, metric_arr)
        correlations[feat_name] = (float(rho), float(p))
    return correlations


def print_corr_table(correlations, metric_name):
    print(f"\n  {'Feature':<25} {metric_name + ' rho':>20} {'p-value':>12} {'Sig':>5}")
    print(f"  {'-' * 65}")
    for feat_name in TOPO_FEATURES:
        rho, p = correlations[feat_name]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat_name:<25} {rho:>+20.4f} {p:>12.4f} {sig:>5}")


def best_rho(correlations):
    """Return (feature_name, rho) for highest |rho|."""
    best = max(correlations, key=lambda k: abs(correlations[k][0]))
    return best, correlations[best][0]


# =========================================================================
# PART 1: STABILITY ACROSS 20 NETWORK REALIZATIONS
# =========================================================================

def part1_stability():
    print(f"\n{'=' * 72}")
    print("PART 1: STABILITY ACROSS 20 NETWORK REALIZATIONS")
    print(f"{'=' * 72}")
    print(f"  {N_NEURONS} neurons, {N_SEEDS_STABILITY} seeds x "
          f"{len(SR_VALUES)} SR = {N_SEEDS_STABILITY * len(SR_VALUES)} points")

    records = []
    t0 = time.time()

    for sr in SR_VALUES:
        for seed_offset in range(N_SEEDS_STABILITY):
            seed = BASE_SEED + seed_offset

            esn = MinimalESN(n_neurons=N_NEURONS, spectral_radius=sr, seed=seed)

            # MC run
            rng = np.random.default_rng(seed)
            inputs_noise = rng.standard_normal(N_STEPS)
            states_mc = esn.run(inputs_noise)
            mc = compute_memory_capacity(states_mc, inputs_noise)
            topo_mc = extract_topo_features(states_mc[WASHOUT:, 0], seed=seed)
            lyap = estimate_lyapunov(states_mc, seed=seed)
            kq = compute_kernel_quality(states_mc)
            eff_sr = estimate_effective_sr(states_mc)

            # NARMA run (same weight matrix, different input)
            esn.reset()
            u_narma, y_narma = generate_narma10(N_STEPS, seed=seed)
            states_narma = esn.run(u_narma)
            nrmse = evaluate_narma10(states_narma, u_narma, y_narma)
            topo_narma = extract_topo_features(states_narma[WASHOUT:, 0], seed=seed)

            records.append({
                "sr": sr, "seed": seed,
                "mc": mc, "nrmse": nrmse,
                "topo_mc": topo_mc, "topo_narma": topo_narma,
                "lyapunov": lyap, "kernel_quality": kq,
                "effective_sr": eff_sr,
            })

        sr_recs = [r for r in records if r["sr"] == sr]
        mc_vals = [r["mc"] for r in sr_recs]
        nr_vals = [r["nrmse"] for r in sr_recs]
        print(f"  SR={sr:<5.2f}: MC={np.mean(mc_vals):6.2f}+/-{np.std(mc_vals):5.2f}  "
              f"NRMSE={np.mean(nr_vals):6.3f}+/-{np.std(nr_vals):5.3f}", flush=True)

    print(f"\n  Data collection: {time.time() - t0:.1f}s ({len(records)} samples)")

    # Per-SR summary
    print(f"\n  Per-SR summary (mean+/-std across {N_SEEDS_STABILITY} seeds):")
    print(f"  {'SR':>5} | {'MC':>14} | {'NRMSE':>14} | "
          f"{'H1_ent':>14} | {'betti_1':>10} | {'Lyapunov':>14} | {'KQ':>8}")
    print(f"  {'-' * 95}")

    for sr in SR_VALUES:
        recs = [r for r in records if r["sr"] == sr]
        mc_v = [r["mc"] for r in recs]
        nr_v = [r["nrmse"] for r in recs]
        h1e = [r["topo_mc"]["persistence_entropy"] for r in recs]
        b1 = [r["topo_mc"]["betti_1"] for r in recs]
        ly = [r["lyapunov"] for r in recs]
        kq = [r["kernel_quality"] for r in recs]
        print(f"  {sr:>5.2f} | {np.mean(mc_v):6.2f}+/-{np.std(mc_v):5.2f} | "
              f"{np.mean(nr_v):6.3f}+/-{np.std(nr_v):5.3f} | "
              f"{np.mean(h1e):6.3f}+/-{np.std(h1e):5.3f} | "
              f"{np.mean(b1):5.1f}+/-{np.std(b1):4.1f} | "
              f"{np.mean(ly):7.4f}+/-{np.std(ly):5.4f} | "
              f"{np.mean(kq):4.1f}+/-{np.std(kq):3.1f}")

    # Correlations on full dataset
    all_mc = [r["mc"] for r in records]
    all_nrmse = [r["nrmse"] for r in records]
    corr_mc = compute_correlations(all_mc, [r["topo_mc"] for r in records])
    corr_narma = compute_correlations(all_nrmse, [r["topo_narma"] for r in records])

    print_corr_table(corr_mc, "MC Spearman")
    print_corr_table(corr_narma, "NARMA Spearman")

    feat_mc, rho_mc = best_rho(corr_mc)
    feat_narma, rho_narma = best_rho(corr_narma)
    print(f"\n  MC best: {feat_mc} (rho={rho_mc:+.4f})")
    print(f"  NARMA best: {feat_narma} (rho={rho_narma:+.4f})")
    stable = abs(rho_mc) > 0.5 and abs(rho_narma) > 0.5
    print(f"  Stability: {'STABLE' if stable else 'UNSTABLE'} at {N_SEEDS_STABILITY} seeds")

    print(f"\n  Part 1 total: {time.time() - t0:.1f}s")
    return records, corr_mc, corr_narma


# =========================================================================
# PART 2: PREDICTION MODEL (LOO-CV)
# =========================================================================

def part2_prediction(records):
    print(f"\n{'=' * 72}")
    print("PART 2: PREDICTION MODEL (LOO CROSS-VALIDATION)")
    print(f"{'=' * 72}")

    all_mc = np.array([r["mc"] for r in records])
    all_nrmse = np.array([r["nrmse"] for r in records])

    # MC = f(persistence_entropy, betti_1)
    X_mc_2 = np.column_stack([
        [r["topo_mc"]["persistence_entropy"] for r in records],
        [r["topo_mc"]["betti_1"] for r in records],
    ])
    mc_r2_2feat = loo_r2(X_mc_2, all_mc)

    # NARMA = f(max_H1_pers, persistence_entropy)
    X_narma_2 = np.column_stack([
        [r["topo_narma"]["max_H1_pers"] for r in records],
        [r["topo_narma"]["persistence_entropy"] for r in records],
    ])
    narma_r2_2feat = loo_r2(X_narma_2, all_nrmse)

    print(f"\n  MC = f(persistence_entropy, betti_1)")
    print(f"    LOO-CV R^2 = {mc_r2_2feat:.4f}  "
          f"{'PASS (>0.5)' if mc_r2_2feat > 0.5 else 'FAIL (<0.5)'}")

    print(f"\n  NARMA = f(max_H1_pers, persistence_entropy)")
    print(f"    LOO-CV R^2 = {narma_r2_2feat:.4f}  "
          f"{'PASS (>0.5)' if narma_r2_2feat > 0.5 else 'FAIL (<0.5)'}")

    # All 5 features
    X_mc_all = np.column_stack([
        [r["topo_mc"][f] for r in records] for f in TOPO_FEATURES
    ])
    mc_r2_all = loo_r2(X_mc_all, all_mc)

    X_narma_all = np.column_stack([
        [r["topo_narma"][f] for r in records] for f in TOPO_FEATURES
    ])
    narma_r2_all = loo_r2(X_narma_all, all_nrmse)

    print(f"\n  All 5 topology features:")
    print(f"    MC LOO-CV R^2 = {mc_r2_all:.4f}")
    print(f"    NARMA LOO-CV R^2 = {narma_r2_all:.4f}")

    # Best single feature (for fair comparison with 1D standard metrics)
    best_mc_r2_1, best_mc_feat_1 = -np.inf, ""
    best_narma_r2_1, best_narma_feat_1 = -np.inf, ""
    for f in TOPO_FEATURES:
        x_mc = np.array([r["topo_mc"][f] for r in records])
        r2 = loo_r2(x_mc, all_mc)
        if r2 > best_mc_r2_1:
            best_mc_r2_1, best_mc_feat_1 = r2, f
        x_nr = np.array([r["topo_narma"][f] for r in records])
        r2 = loo_r2(x_nr, all_nrmse)
        if r2 > best_narma_r2_1:
            best_narma_r2_1, best_narma_feat_1 = r2, f

    print(f"\n  Best single topology feature:")
    print(f"    MC: {best_mc_feat_1} R^2={best_mc_r2_1:.4f}")
    print(f"    NARMA: {best_narma_feat_1} R^2={best_narma_r2_1:.4f}")

    return {
        "mc_2feat": mc_r2_2feat, "narma_2feat": narma_r2_2feat,
        "mc_all": mc_r2_all, "narma_all": narma_r2_all,
        "mc_1feat": best_mc_r2_1, "mc_1feat_name": best_mc_feat_1,
        "narma_1feat": best_narma_r2_1, "narma_1feat_name": best_narma_feat_1,
    }


# =========================================================================
# PART 3: NETWORK SIZE SCALING
# =========================================================================

def part3_scaling():
    print(f"\n{'=' * 72}")
    print("PART 3: NETWORK SIZE SCALING")
    print(f"{'=' * 72}")
    print(f"  Sizes: {SIZE_VALUES}, {N_SEEDS_SCALING} seeds x {len(SR_VALUES)} SR")

    t0 = time.time()
    results = {}

    for n_neur in SIZE_VALUES:
        all_mc, all_feats = [], []

        for seed_offset in range(N_SEEDS_SCALING):
            seed = BASE_SEED + seed_offset
            # Build ESN once per seed, rescale for each SR
            esn = MinimalESN(n_neurons=n_neur, spectral_radius=SR_VALUES[0], seed=seed)

            for sr in SR_VALUES:
                esn.set_sr(sr)
                rng = np.random.default_rng(seed + int(sr * 1000))
                inputs = rng.standard_normal(N_STEPS)
                states = esn.run(inputs)
                mc = compute_memory_capacity(states, inputs)
                feats = extract_topo_features(states[WASHOUT:, 0], seed=seed)
                all_mc.append(mc)
                all_feats.append(feats)

        corr = compute_correlations(all_mc, all_feats)
        feat, rho = best_rho(corr)
        results[n_neur] = corr
        print(f"  N={n_neur:>4}: best |rho|={abs(rho):.4f} ({feat}), "
              f"n={len(all_mc)}", flush=True)

    # Summary table
    print(f"\n  {'Feature':<25}", end="")
    for n in SIZE_VALUES:
        print(f" {'N=' + str(n):>10}", end="")
    print()
    print(f"  {'-' * (25 + 11 * len(SIZE_VALUES))}")
    for feat_name in TOPO_FEATURES:
        print(f"  {feat_name:<25}", end="")
        for n in SIZE_VALUES:
            rho, p = results[n][feat_name]
            sig = "*" if p < 0.05 else " "
            print(f" {rho:>+8.3f}{sig}", end="")
        print()

    rho_50 = abs(best_rho(results[50])[1])
    rho_500 = abs(best_rho(results[500])[1])
    print(f"\n  N=50 best |rho|={rho_50:.3f}, N=500 best |rho|={rho_500:.3f}")
    print(f"  Scaling: {'YES — holds at 500' if rho_500 > 0.5 else 'NO — breaks at 500'}")
    print(f"\n  Part 3 total: {time.time() - t0:.1f}s")
    return results


# =========================================================================
# PART 4: FIX SPIKING
# =========================================================================

def part4_spiking():
    print(f"\n{'=' * 72}")
    print("PART 4: FIX SPIKING")
    print(f"{'=' * 72}")
    print(f"  LIF: {N_NEURONS} neurons, {N_SEEDS_SPIKING} seeds x "
          f"{len(LIF_CS_VALUES)} CS = {N_SEEDS_SPIKING * len(LIF_CS_VALUES)} points")

    t0 = time.time()

    # ---- Approach (a): Vary smoothing kernel tau ----
    print(f"\n  --- 4a: Smoothed spike rates, tau = {TAU_VALUES} ---")

    # Collect per-tau results efficiently: run LIF once, filter multiple times
    tau_results = {tau: {"mc": [], "mc_feats": [], "nrmse": [], "narma_feats": []}
                   for tau in TAU_VALUES}

    for cs in LIF_CS_VALUES:
        for seed_offset in range(N_SEEDS_SPIKING):
            seed = BASE_SEED + seed_offset
            rng = np.random.default_rng(seed)
            lif = MinimalLIF(n_neurons=N_NEURONS, sparsity=SPARSITY, seed=seed)

            # MC run
            inputs_noise = rng.standard_normal(N_STEPS)
            voltages, spikes = lif.run_with_spikes(inputs_noise, connection_strength=cs)
            mc = compute_memory_capacity(voltages, inputs_noise)

            # NARMA run (new LIF with same weights)
            lif2 = MinimalLIF(n_neurons=N_NEURONS, sparsity=SPARSITY, seed=seed)
            u_narma, y_narma = generate_narma10(N_STEPS, seed=seed)
            v_narma, sp_narma = lif2.run_with_spikes(u_narma, connection_strength=cs)
            nrmse = evaluate_narma10(v_narma, u_narma, y_narma)

            for tau in TAU_VALUES:
                rates_mc = smooth_spikes(spikes, tau_filter=tau)
                feats_mc = extract_topo_features(rates_mc[WASHOUT:, 0], seed=seed)
                tau_results[tau]["mc"].append(mc)
                tau_results[tau]["mc_feats"].append(feats_mc)

                rates_nr = smooth_spikes(sp_narma, tau_filter=tau)
                feats_nr = extract_topo_features(rates_nr[WASHOUT:, 0], seed=seed)
                tau_results[tau]["nrmse"].append(nrmse)
                tau_results[tau]["narma_feats"].append(feats_nr)

    best_tau_mc = (None, 0.0, None)
    best_tau_narma = (None, 0.0, None)

    for tau in TAU_VALUES:
        d = tau_results[tau]
        corr_mc = compute_correlations(d["mc"], d["mc_feats"])
        corr_nr = compute_correlations(d["nrmse"], d["narma_feats"])
        f_mc, r_mc = best_rho(corr_mc)
        f_nr, r_nr = best_rho(corr_nr)
        print(f"    tau={tau:>3}: MC best rho={r_mc:+.3f} ({f_mc}), "
              f"NARMA best rho={r_nr:+.3f} ({f_nr})")
        if abs(r_mc) > abs(best_tau_mc[1]):
            best_tau_mc = (tau, r_mc, corr_mc)
        if abs(r_nr) > abs(best_tau_narma[1]):
            best_tau_narma = (tau, r_nr, corr_nr)

    print(f"\n    Best MC tau: {best_tau_mc[0]} (rho={best_tau_mc[1]:+.3f})")
    print(f"    Best NARMA tau: {best_tau_narma[0]} (rho={best_tau_narma[1]:+.3f})")

    # ---- Approach (b): PCA of population rates (3D cloud) ----
    print(f"\n  --- 4b: PCA of population smoothed rates (3D cloud) ---")

    all_mc_b, all_feats_b = [], []
    all_nrmse_b, all_feats_narma_b = [], []

    for cs in LIF_CS_VALUES:
        for seed_offset in range(N_SEEDS_SPIKING):
            seed = BASE_SEED + seed_offset
            rng = np.random.default_rng(seed)
            lif = MinimalLIF(n_neurons=N_NEURONS, sparsity=SPARSITY, seed=seed)

            # MC
            inputs_noise = rng.standard_normal(N_STEPS)
            voltages, spikes = lif.run_with_spikes(inputs_noise, connection_strength=cs)
            mc = compute_memory_capacity(voltages, inputs_noise)
            rates = smooth_spikes(spikes, tau_filter=20.0)
            cloud = pca_reduce(rates, n_components=3)
            feats = extract_topo_features_cloud(cloud, seed=seed)
            all_mc_b.append(mc)
            all_feats_b.append(feats)

            # NARMA
            lif2 = MinimalLIF(n_neurons=N_NEURONS, sparsity=SPARSITY, seed=seed)
            u_narma, y_narma = generate_narma10(N_STEPS, seed=seed)
            v_nr, sp_nr = lif2.run_with_spikes(u_narma, connection_strength=cs)
            nrmse = evaluate_narma10(v_nr, u_narma, y_narma)
            rates_nr = smooth_spikes(sp_nr, tau_filter=20.0)
            cloud_nr = pca_reduce(rates_nr, n_components=3)
            feats_nr = extract_topo_features_cloud(cloud_nr, seed=seed)
            all_nrmse_b.append(nrmse)
            all_feats_narma_b.append(feats_nr)

    corr_mc_b = compute_correlations(all_mc_b, all_feats_b)
    corr_narma_b = compute_correlations(all_nrmse_b, all_feats_narma_b)
    f_mc_b, rho_mc_b = best_rho(corr_mc_b)
    f_nr_b, rho_narma_b = best_rho(corr_narma_b)
    print(f"    MC best rho={rho_mc_b:+.3f} ({f_mc_b})")
    print(f"    NARMA best rho={rho_narma_b:+.3f} ({f_nr_b})")

    # ---- Approach (c): No-reset subthreshold voltage ----
    print(f"\n  --- 4c: Subthreshold voltage (no spike reset) ---")

    all_mc_c, all_feats_c = [], []
    all_nrmse_c, all_feats_narma_c = [], []

    for cs in LIF_CS_VALUES:
        for seed_offset in range(N_SEEDS_SPIKING):
            seed = BASE_SEED + seed_offset
            rng = np.random.default_rng(seed)
            lif = MinimalLIF(n_neurons=N_NEURONS, sparsity=SPARSITY, seed=seed)

            # MC (no-reset dynamics)
            inputs_noise = rng.standard_normal(N_STEPS)
            states_c = lif.run_no_reset(inputs_noise, connection_strength=cs)
            mc = compute_memory_capacity(states_c, inputs_noise)
            feats = extract_topo_features(states_c[WASHOUT:, 0], seed=seed)
            all_mc_c.append(mc)
            all_feats_c.append(feats)

            # NARMA (no-reset dynamics)
            lif2 = MinimalLIF(n_neurons=N_NEURONS, sparsity=SPARSITY, seed=seed)
            u_narma, y_narma = generate_narma10(N_STEPS, seed=seed)
            states_cn = lif2.run_no_reset(u_narma, connection_strength=cs)
            nrmse = evaluate_narma10(states_cn, u_narma, y_narma)
            feats_nr = extract_topo_features(states_cn[WASHOUT:, 0], seed=seed)
            all_nrmse_c.append(nrmse)
            all_feats_narma_c.append(feats_nr)

    corr_mc_c = compute_correlations(all_mc_c, all_feats_c)
    corr_narma_c = compute_correlations(all_nrmse_c, all_feats_narma_c)
    f_mc_c, rho_mc_c = best_rho(corr_mc_c)
    f_nr_c, rho_narma_c = best_rho(corr_narma_c)
    print(f"    MC best rho={rho_mc_c:+.3f} ({f_mc_c})")
    print(f"    NARMA best rho={rho_narma_c:+.3f} ({f_nr_c})")

    # ---- Summary ----
    print(f"\n  Part 4 Summary:")
    print(f"  {'Approach':<40} {'MC |rho|':>10} {'NARMA |rho|':>12}")
    print(f"  {'-' * 65}")
    print(f"  {'Original (membrane voltage, screen P3)':<40} {'~0.52':>10} {'N/A':>12}")
    print(f"  {'4a: Smoothed rates (best tau)':.<40} {abs(best_tau_mc[1]):>10.3f} "
          f"{abs(best_tau_narma[1]):>12.3f}")
    print(f"  {'4b: PCA pop. rates (3D cloud)':.<40} {abs(rho_mc_b):>10.3f} "
          f"{abs(rho_narma_b):>12.3f}")
    print(f"  {'4c: Subthreshold (no reset)':.<40} {abs(rho_mc_c):>10.3f} "
          f"{abs(rho_narma_c):>12.3f}")

    spiking_best_mc = max(abs(best_tau_mc[1]), abs(rho_mc_b), abs(rho_mc_c))
    spiking_best_narma = max(abs(best_tau_narma[1]), abs(rho_narma_b), abs(rho_narma_c))
    print(f"\n  Best spiking MC |rho|={spiking_best_mc:.3f} "
          f"({'PASS' if spiking_best_mc > RHO_GATE else 'FAIL'} gate={RHO_GATE})")
    print(f"  Best spiking NARMA |rho|={spiking_best_narma:.3f} "
          f"({'PASS' if spiking_best_narma > RHO_GATE else 'FAIL'} gate={RHO_GATE})")

    print(f"\n  Part 4 total: {time.time() - t0:.1f}s")

    return {
        "4a_mc_rho": best_tau_mc[1], "4a_mc_tau": best_tau_mc[0],
        "4a_narma_rho": best_tau_narma[1], "4a_narma_tau": best_tau_narma[0],
        "4b_mc_rho": rho_mc_b, "4b_narma_rho": rho_narma_b,
        "4c_mc_rho": rho_mc_c, "4c_narma_rho": rho_narma_c,
    }


# =========================================================================
# PART 5: COMPARISON WITH STANDARD RESERVOIR METRICS
# =========================================================================

def part5_comparison(records, pred_results):
    print(f"\n{'=' * 72}")
    print("PART 5: COMPARISON WITH STANDARD RESERVOIR METRICS")
    print(f"{'=' * 72}")

    all_mc = np.array([r["mc"] for r in records])
    all_nrmse = np.array([r["nrmse"] for r in records])
    all_sr = np.array([r["sr"] for r in records])
    all_lyap = np.array([r["lyapunov"] for r in records])
    all_kq = np.array([r["kernel_quality"] for r in records], dtype=float)
    all_esr = np.array([r["effective_sr"] for r in records])

    # Spearman correlations
    print(f"\n  Spearman correlations with MC:")
    for name, vals in [("Lyapunov", all_lyap), ("Kernel quality", all_kq),
                       ("Effective SR", all_esr), ("SR (known)", all_sr)]:
        if np.std(vals) > 1e-15:
            rho, p = spearmanr(vals, all_mc)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {name:<20} rho={rho:+.4f}  p={p:.4f} {sig}")
        else:
            print(f"    {name:<20} (no variance)")

    print(f"\n  Spearman correlations with NARMA NRMSE:")
    for name, vals in [("Lyapunov", all_lyap), ("Kernel quality", all_kq),
                       ("Effective SR", all_esr), ("SR (known)", all_sr)]:
        if np.std(vals) > 1e-15:
            rho, p = spearmanr(vals, all_nrmse)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {name:<20} rho={rho:+.4f}  p={p:.4f} {sig}")
        else:
            print(f"    {name:<20} (no variance)")

    # LOO-CV R^2 for each predictor
    mc_r2_lyap = loo_r2(all_lyap, all_mc)
    narma_r2_lyap = loo_r2(all_lyap, all_nrmse)

    mc_r2_kq = loo_r2(all_kq, all_mc)
    narma_r2_kq = loo_r2(all_kq, all_nrmse)

    mc_r2_esr = loo_r2(all_esr, all_mc)
    narma_r2_esr = loo_r2(all_esr, all_nrmse)

    mc_r2_sr = loo_r2(all_sr, all_mc)
    narma_r2_sr = loo_r2(all_sr, all_nrmse)

    # Time representative calls
    rng = np.random.default_rng(0)
    esn_tmp = MinimalESN(spectral_radius=0.9, seed=0)
    states_tmp = esn_tmp.run(rng.standard_normal(N_STEPS))
    sig_tmp = states_tmp[WASHOUT:, 0]

    t = time.time()
    _ = extract_topo_features(sig_tmp, seed=0)
    t_ph = (time.time() - t) * 1000

    t = time.time()
    _ = estimate_lyapunov(states_tmp, seed=0)
    t_lyap = (time.time() - t) * 1000

    t = time.time()
    _ = compute_kernel_quality(states_tmp)
    t_kq = (time.time() - t) * 1000

    t = time.time()
    _ = estimate_effective_sr(states_tmp)
    t_esr = (time.time() - t) * 1000

    # Topology R^2 from Part 2
    mc_r2_topo = pred_results["mc_2feat"]
    narma_r2_topo = pred_results["narma_2feat"]
    mc_r2_topo_1 = pred_results["mc_1feat"]
    narma_r2_topo_1 = pred_results["narma_1feat"]

    # --- PREDICTION COMPARISON TABLE ---
    print(f"\n  {'=' * 68}")
    print(f"  PREDICTION COMPARISON")
    print(f"  {'=' * 68}")
    print(f"  {'Predictor':<30} {'MC R^2':>10} {'NARMA R^2':>12} {'Time':>10}")
    print(f"  {'-' * 68}")
    print(f"  {'Topology (2 feat)':<30} {mc_r2_topo:>10.4f} {narma_r2_topo:>12.4f} "
          f"{t_ph:>8.0f}ms")
    print(f"  {'Topology (best 1 feat)':<30} {mc_r2_topo_1:>10.4f} {narma_r2_topo_1:>12.4f} "
          f"{t_ph:>8.0f}ms")
    print(f"  {'Lyapunov exponent':<30} {mc_r2_lyap:>10.4f} {narma_r2_lyap:>12.4f} "
          f"{t_lyap:>8.0f}ms")
    print(f"  {'Effective SR':<30} {mc_r2_esr:>10.4f} {narma_r2_esr:>12.4f} "
          f"{t_esr:>8.0f}ms")
    print(f"  {'Kernel quality':<30} {mc_r2_kq:>10.4f} {narma_r2_kq:>12.4f} "
          f"{t_kq:>8.0f}ms")
    print(f"  {'Spectral radius alone':<30} {mc_r2_sr:>10.4f} {narma_r2_sr:>12.4f} "
          f"{'0':>8}ms")
    print(f"  {'-' * 68}")

    return {
        "topo_2": (mc_r2_topo, narma_r2_topo),
        "topo_1": (mc_r2_topo_1, narma_r2_topo_1),
        "lyap": (mc_r2_lyap, narma_r2_lyap),
        "esr": (mc_r2_esr, narma_r2_esr),
        "kq": (mc_r2_kq, narma_r2_kq),
        "sr": (mc_r2_sr, narma_r2_sr),
        "times": {"ph": t_ph, "lyap": t_lyap, "kq": t_kq, "esr": t_esr},
    }


# =========================================================================
# VERDICT
# =========================================================================

def print_verdict(scaling_results, spiking_results, comparison):
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")

    topo_mc, topo_narma = comparison["topo_2"]
    topo1_mc, topo1_narma = comparison["topo_1"]
    sr_mc, sr_narma = comparison["sr"]
    lyap_mc, lyap_narma = comparison["lyap"]
    kq_mc, kq_narma = comparison["kq"]
    esr_mc, esr_narma = comparison["esr"]

    best_std_mc = max(sr_mc, lyap_mc, kq_mc, esr_mc)
    best_std_narma = max(sr_narma, lyap_narma, kq_narma, esr_narma)

    print(f"\n  1. Is topology a better predictor than standard metrics?")
    print(f"     MC:    Topology R^2={topo_mc:.4f} vs best standard R^2={best_std_mc:.4f}")
    topo_wins_mc = topo_mc > best_std_mc
    print(f"            -> {'YES, topology wins' if topo_wins_mc else 'NO, standard metrics win'}")
    print(f"     NARMA: Topology R^2={topo_narma:.4f} vs best standard R^2={best_std_narma:.4f}")
    topo_wins_narma = topo_narma > best_std_narma
    print(f"            -> {'YES, topology wins' if topo_wins_narma else 'NO, standard metrics win'}")

    # Fair comparison: single-feature topology vs single-feature standard
    print(f"\n     Fair 1-vs-1: Topo best single R^2 (MC)={topo1_mc:.4f} vs SR alone={sr_mc:.4f}")
    print(f"     Fair 1-vs-1: Topo best single R^2 (NARMA)={topo1_narma:.4f} vs SR alone={sr_narma:.4f}")

    # 2. Does it scale?
    rho_50 = abs(best_rho(scaling_results[50])[1])
    rho_100 = abs(best_rho(scaling_results[100])[1])
    rho_200 = abs(best_rho(scaling_results[200])[1])
    rho_500 = abs(best_rho(scaling_results[500])[1])
    print(f"\n  2. Does it scale?")
    print(f"     N=50: |rho|={rho_50:.3f}, N=100: |rho|={rho_100:.3f}, "
          f"N=200: |rho|={rho_200:.3f}, N=500: |rho|={rho_500:.3f}")
    scales = rho_500 > 0.5
    print(f"     -> {'YES' if scales else 'NO'}")

    # 3. Does it work on spiking?
    best_spike_mc = max(
        abs(spiking_results["4a_mc_rho"]),
        abs(spiking_results["4b_mc_rho"]),
        abs(spiking_results["4c_mc_rho"]),
    )
    best_spike_narma = max(
        abs(spiking_results["4a_narma_rho"]),
        abs(spiking_results["4b_narma_rho"]),
        abs(spiking_results["4c_narma_rho"]),
    )
    print(f"\n  3. Does it work on spiking?")
    print(f"     Best MC |rho|={best_spike_mc:.3f} "
          f"(4a tau={spiking_results['4a_mc_tau']}: {abs(spiking_results['4a_mc_rho']):.3f}, "
          f"4b: {abs(spiking_results['4b_mc_rho']):.3f}, "
          f"4c: {abs(spiking_results['4c_mc_rho']):.3f})")
    print(f"     Best NARMA |rho|={best_spike_narma:.3f} "
          f"(4a tau={spiking_results['4a_narma_tau']}: {abs(spiking_results['4a_narma_rho']):.3f}, "
          f"4b: {abs(spiking_results['4b_narma_rho']):.3f}, "
          f"4c: {abs(spiking_results['4c_narma_rho']):.3f})")
    if best_spike_mc > RHO_GATE:
        print(f"     -> YES, passes rho > {RHO_GATE} gate")
    elif best_spike_mc > 0.5:
        print(f"     -> PARTIAL, moderate correlation but below {RHO_GATE} gate")
    else:
        print(f"     -> NO, weak correlation on spiking networks")

    # 4. Use case
    print(f"\n  4. What's the actual use case?")
    if topo_wins_mc or topo_wins_narma:
        print(f"     Topology provides information BEYOND standard metrics.")
        print(f"     Use case: training-free reservoir quality probe — compute PH")
        print(f"     on a single neuron trajectory, get MC/NARMA prediction without")
        print(f"     ever training a readout. Valuable for neuromorphic hardware")
        print(f"     where readout training is expensive or impossible in real time.")
        if not scales:
            print(f"     CAVEAT: does not scale to large networks (N>500).")
        if best_spike_mc < RHO_GATE:
            print(f"     CAVEAT: limited to rate-coded networks, not spiking.")
    else:
        print(f"     Topology is a REDUNDANT proxy for spectral radius.")
        print(f"     Everything correlates with SR. Topology captures the same")
        print(f"     information as simpler, faster metrics (Lyapunov, kernel rank).")
        print(f"     Use case: only when SR/Lyapunov/KQ are unknown — e.g.,")
        print(f"     biological circuits or opaque neuromorphic hardware where")
        print(f"     you can observe one neuron but not the weight matrix.")

    print(f"\n  {'=' * 60}")


# =========================================================================
# Main
# =========================================================================

def main():
    t_start = time.time()
    print("Deepen SNN Topology: Stability, Scaling, Spiking Fixes, "
          "Standard Metric Comparison")
    print(f"{'=' * 72}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {BASE_SEED}..{BASE_SEED + N_SEEDS_STABILITY - 1} (stability), "
          f"{BASE_SEED}..{BASE_SEED + N_SEEDS_SCALING - 1} (scaling/spiking)")
    print(f"N_STEPS={N_STEPS}, WASHOUT={WASHOUT}, SUBSAMPLE={SUBSAMPLE}")
    print()

    # Part 1: Stability
    records, corr_mc, corr_narma = part1_stability()

    # Part 2: Prediction model
    pred_results = part2_prediction(records)

    # Part 3: Scaling
    scaling_results = part3_scaling()

    # Part 4: Spiking
    spiking_results = part4_spiking()

    # Part 5: Comparison
    comparison = part5_comparison(records, pred_results)

    # Verdict
    print_verdict(scaling_results, spiking_results, comparison)

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
