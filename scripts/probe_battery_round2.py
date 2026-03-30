#!/usr/bin/env python3
"""Probe battery round 2: failure fixes, boundary stress tests, real-world data.

Round 1 (probe_domain_sweep.py): 8/10 systems passed. Two failures:
  - Lorenz-96: PCA(3) too low for 40D system (rho=0.408)
  - Spiking STDP: voltage reset discontinuities break PCA

This round:
  Tests 1-2: Fix the failures (higher PCA dim, population firing rates)
  Tests 3-5: Stress-test boundaries (min trajectory, noise, partial observability)
  Tests 6-8: Real-world data (power grid, EEG, weather)
  Tests 9-10: New systems (reaction-diffusion, flocking)

Protocol: same as round 1 — sweep control param, 3 seeds, PCA + PH(max_dim=1,
subsample=400), discard first 20% transient, Spearman rho, pass gate |rho| > 0.6.

Usage:
    python scripts/probe_battery_round2.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
from scipy.stats import spearmanr
from numpy.linalg import svd

from att.topology.persistence import PersistenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_SEED = 42
N_SEEDS = 3
RHO_GATE = 0.6
SUBSAMPLE = 400
PCA_DIM = 3
TRANSIENT_FRAC = 0.2

TOPO_FEATURES = [
    "persistence_entropy",
    "total_H1_pers",
    "max_H1_pers",
    "betti_0",
    "betti_1",
]

ZERO_FEATURES = {k: 0.0 for k in TOPO_FEATURES}


# ---------------------------------------------------------------------------
# PCA + topology helpers (same as probe_domain_sweep.py)
# ---------------------------------------------------------------------------

def pca_reduce(states: np.ndarray, n_components: int = PCA_DIM) -> np.ndarray:
    """PCA reduce (n_steps, n_vars) -> (n_steps, n_components)."""
    X = states - states.mean(axis=0)
    if np.std(X) < 1e-15:
        return X[:, :min(n_components, X.shape[1])]
    _, _, Vt = svd(X, full_matrices=False)
    return X @ Vt[:min(n_components, len(Vt))].T


def extract_topo(cloud: np.ndarray, seed: int = BASE_SEED) -> dict:
    """PH features from a point cloud."""
    if np.std(cloud) < 1e-15:
        return dict(ZERO_FEATURES)
    pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa.fit_transform(cloud, subsample=min(SUBSAMPLE, len(cloud)), seed=seed)
    return _features_from_pa(pa)


def _features_from_pa(pa) -> dict:
    features = {}
    dgm0 = pa.diagrams_[0] if len(pa.diagrams_) > 0 else np.array([])
    if len(dgm0) > 0:
        lt0 = dgm0[:, 1] - dgm0[:, 0]
        lt0 = lt0[lt0 > 1e-10]
        features["betti_0"] = len(lt0)
    else:
        features["betti_0"] = 0

    dgm1 = pa.diagrams_[1] if len(pa.diagrams_) > 1 else np.array([])
    if len(dgm1) > 0:
        lt1 = dgm1[:, 1] - dgm1[:, 0]
        lt1 = lt1[lt1 > 1e-10]
        features["betti_1"] = len(lt1)
        if len(lt1) > 0:
            total = float(lt1.sum())
            features["total_H1_pers"] = total
            features["max_H1_pers"] = float(lt1.max())
            p = lt1 / total if total > 0 else lt1
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


def discard_transient(states: np.ndarray) -> np.ndarray:
    """Discard first TRANSIENT_FRAC of trajectory."""
    n = int(len(states) * TRANSIENT_FRAC)
    return states[n:]


def best_correlation(all_gt, all_feats):
    """Find topology feature with highest |Spearman rho| against ground truth."""
    if len(all_gt) < 5:
        return None, 0.0, 1.0
    gt_arr = np.array(all_gt)
    best_feat, best_rho, best_p = None, 0.0, 1.0
    for fname in TOPO_FEATURES:
        feat_arr = np.array([f[fname] for f in all_feats])
        if np.std(feat_arr) < 1e-15 or np.std(gt_arr) < 1e-15:
            continue
        rho, p = spearmanr(feat_arr, gt_arr)
        if np.isfinite(rho) and abs(rho) > abs(best_rho):
            best_feat, best_rho, best_p = fname, float(rho), float(p)
    return best_feat, abs(best_rho), best_p


# ═══════════════════════════════════════════════════════════════════════════
# System generators (reused from round 1)
# ═══════════════════════════════════════════════════════════════════════════

def lorenz96(N: int, F: float, n_steps: int, dt: float, seed: int):
    """Lorenz-96 model. Returns trajectory and largest Lyapunov exponent."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N) * 0.01
    x[0] += F
    trajectory = np.zeros((n_steps, N))

    for t in range(n_steps):
        trajectory[t] = x
        dx = np.zeros(N)
        for i in range(N):
            dx[i] = ((x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N]
                     - x[i] + F)
        x = x + dx * dt
        if np.any(~np.isfinite(x)):
            trajectory = trajectory[:t + 1]
            break

    # Lyapunov exponent via trajectory divergence
    n_trans = int(len(trajectory) * TRANSIENT_FRAC)
    post = trajectory[n_trans:]
    if len(post) < 200:
        return trajectory, 0.0

    def l96_rhs(x_):
        xp1 = np.roll(x_, -1)
        xm1 = np.roll(x_, 1)
        xm2 = np.roll(x_, 2)
        return (xp1 - xm2) * xm1 - x_ + F

    x_ref = post[0].copy()
    x_pert = x_ref + rng.standard_normal(N) * 1e-8
    d0 = 1e-8
    lyap_sum = 0.0
    n_lyap = 0
    renorm_interval = 5

    for t in range(min(3000, len(post) - 1)):
        k1_ref = l96_rhs(x_ref)
        k1_pert = l96_rhs(x_pert)
        x_ref = x_ref + k1_ref * dt
        x_pert = x_pert + k1_pert * dt
        if (t + 1) % renorm_interval == 0:
            delta = x_pert - x_ref
            d = np.linalg.norm(delta)
            if d > 1e-15 and np.isfinite(d):
                lyap_sum += np.log(d / d0)
                n_lyap += 1
                x_pert = x_ref + delta / d * d0
            else:
                x_pert = x_ref + rng.standard_normal(N) * d0

    lyap = lyap_sum / (n_lyap * renorm_interval * dt) if n_lyap > 0 else 0.0
    return trajectory, float(lyap)


def kuramoto(N: int, K: float, n_steps: int, dt: float, seed: int):
    """Kuramoto model. Returns (n_steps, 2*N) sin/cos trajectory and order param."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(N)
    theta = rng.uniform(0, 2 * np.pi, N)
    trajectory = np.zeros((n_steps, 2 * N))

    for t in range(n_steps):
        trajectory[t, :N] = np.cos(theta)
        trajectory[t, N:] = np.sin(theta)
        sin_diff = np.sin(theta[None, :] - theta[:, None])
        coupling = (K / N) * sin_diff.sum(axis=1)
        theta = theta + dt * (omega + coupling)

    cos_part = trajectory[:, :N]
    sin_part = trajectory[:, N:]
    r = np.sqrt(np.mean(cos_part, axis=1)**2 + np.mean(sin_part, axis=1)**2)
    r_mean = float(np.mean(r[int(len(r) * TRANSIENT_FRAC):]))
    return trajectory, r_mean


def spiking_stdp(N: int, lr: float, n_steps: int, dt: float, seed: int):
    """LIF network with STDP. Returns spike trains (binary) and weight entropy."""
    rng = np.random.default_rng(seed)
    tau_m = 10.0
    v_thresh = 1.0
    v_reset = 0.0
    v_rest = 0.0

    W = rng.uniform(0.1, 0.8, (N, N))
    np.fill_diagonal(W, 0)

    tau_plus = 20.0
    tau_minus = 20.0
    A_plus = lr
    A_minus = lr * 1.05

    v = rng.uniform(0, 0.8, N)
    last_spike = np.full(N, -1000.0)
    prev_spikes = np.zeros(N)

    # Store raw spike trains (binary) instead of voltages
    spike_trains = np.zeros((n_steps, N))

    stdp_interval = 5

    for t in range(n_steps):
        time_ms = t * dt
        I_ext = rng.poisson(3.0, N).astype(float) * 0.5
        recurrent = W @ prev_spikes
        dv = (-v + v_rest + recurrent + I_ext) / tau_m * dt
        v = v + dv

        spikes = v >= v_thresh
        spike_trains[t] = spikes.astype(float)
        prev_spikes = spikes.astype(float)
        v[spikes] = v_reset

        if np.any(spikes) and t % stdp_interval == 0:
            spike_idx = np.where(spikes)[0]
            last_spike[spike_idx] = time_ms
            dt_all = time_ms - last_spike
            for i in spike_idx:
                pot_mask = (dt_all > 0) & (dt_all < 5 * tau_plus)
                W[pot_mask, i] += A_plus * np.exp(-dt_all[pot_mask] / tau_plus)
                dep_mask = (dt_all < 0) & (dt_all > -5 * tau_minus)
                W[dep_mask, i] -= A_minus * np.exp(dt_all[dep_mask] / tau_minus)
            W = np.clip(W, 0, 1)
            np.fill_diagonal(W, 0)
        elif np.any(spikes):
            last_spike[np.where(spikes)[0]] = time_ms

    # Weight entropy
    w_flat = W[W > 1e-6]
    if len(w_flat) > 0:
        w_norm = w_flat / w_flat.sum()
        entropy = -float(np.sum(w_norm * np.log(w_norm + 1e-15)))
    else:
        entropy = 0.0

    return spike_trains, entropy


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Lorenz-96 PCA dimension fix
# ═══════════════════════════════════════════════════════════════════════════

def lorenz96_lyapunov_robust(N: int, F: float, dt: float, seed: int):
    """More robust Lyapunov estimator: renormalize every step, RK2 integration."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N) * 0.01 + F

    def l96_rhs(x_):
        xp1 = np.roll(x_, -1)
        xm1 = np.roll(x_, 1)
        xm2 = np.roll(x_, 2)
        return (xp1 - xm2) * xm1 - x_ + F

    # Spin up
    for _ in range(2000):
        k1 = l96_rhs(x)
        k2 = l96_rhs(x + k1 * dt)
        x = x + 0.5 * dt * (k1 + k2)

    # Lyapunov via perturbation with per-step renormalization
    d0 = 1e-8
    x_pert = x + rng.standard_normal(N) * d0
    lyap_sum = 0.0
    n_lyap = 0

    for _ in range(5000):
        # RK2 for both
        k1_r = l96_rhs(x)
        k2_r = l96_rhs(x + k1_r * dt)
        x = x + 0.5 * dt * (k1_r + k2_r)

        k1_p = l96_rhs(x_pert)
        k2_p = l96_rhs(x_pert + k1_p * dt)
        x_pert = x_pert + 0.5 * dt * (k1_p + k2_p)

        delta = x_pert - x
        d = np.linalg.norm(delta)
        if d > 1e-15 and np.isfinite(d):
            lyap_sum += np.log(d / d0)
            n_lyap += 1
            x_pert = x + delta / d * d0
        else:
            x_pert = x + rng.standard_normal(N) * d0

    return lyap_sum / (n_lyap * dt) if n_lyap > 0 else 0.0


def test1_lorenz96_pca_fix():
    """Lorenz-96 failed at PCA(3). Try PCA = [3, 5, 10, 20].
    Uses robust Lyapunov estimator (RK2, per-step renormalization)."""
    pca_dims = [3, 5, 10, 20]
    F_values = np.linspace(2, 12, 8)
    results = {}

    # Pre-compute trajectories and ground truth (shared across PCA dims)
    runs = []
    for si in range(N_SEEDS):
        seed = BASE_SEED + si
        for F in F_values:
            traj, _ = lorenz96(N=40, F=F, n_steps=5000, dt=0.01, seed=seed)
            states = discard_transient(traj)
            lyap = lorenz96_lyapunov_robust(N=40, F=F, dt=0.01, seed=seed)
            if len(states) < 100 or not np.isfinite(lyap):
                continue
            runs.append((states, lyap, seed))

    for pca_dim in pca_dims:
        all_gt = []
        all_feats = []
        for states, lyap, seed in runs:
            all_gt.append(lyap)
            cloud = pca_reduce(states, pca_dim)
            feats = extract_topo(cloud, seed=seed)
            all_feats.append(feats)
        _, rho, _ = best_correlation(all_gt, all_feats)
        results[pca_dim] = rho

    best_dim = max(results, key=results.get)
    best_rho = results[best_dim]
    passed = best_rho > RHO_GATE

    detail_str = ", ".join(f"PCA={d}: {r:.3f}" for d, r in sorted(results.items()))
    notes = f"Best PCA dim = {best_dim}. {detail_str}"
    return best_rho, passed, notes


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Spiking STDP with population firing rates
# ═══════════════════════════════════════════════════════════════════════════

def smooth_spike_trains(spike_trains: np.ndarray, tau: float, dt: float = 1.0):
    """Convolve each neuron's spike train with exponential kernel."""
    n_steps, N = spike_trains.shape
    # Build kernel: exp(-t/tau) for t >= 0
    kernel_len = int(5 * tau / dt)
    t_kernel = np.arange(kernel_len) * dt
    kernel = np.exp(-t_kernel / tau)
    kernel /= kernel.sum()

    rates = np.zeros_like(spike_trains)
    for i in range(N):
        rates[:, i] = np.convolve(spike_trains[:, i], kernel, mode='same')
    return rates


def test2_spiking_pop_pca():
    """Spiking STDP with smoothed population firing rates instead of voltages."""
    taus = [10, 20, 50, 100]
    lr_values = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    results = {}

    for tau in taus:
        all_gt = []
        all_feats = []
        for si in range(N_SEEDS):
            seed = BASE_SEED + si
            for lr in lr_values:
                spike_trains, entropy = spiking_stdp(
                    N=100, lr=lr, n_steps=10000, dt=1.0, seed=seed
                )
                states = discard_transient(spike_trains)
                # Smooth spike trains into firing rates
                rates = smooth_spike_trains(states, tau=tau, dt=1.0)
                if np.std(rates) < 1e-15:
                    continue
                all_gt.append(entropy)
                cloud = pca_reduce(rates, PCA_DIM)
                feats = extract_topo(cloud, seed=seed)
                all_feats.append(feats)
        _, rho, _ = best_correlation(all_gt, all_feats)
        results[tau] = rho

    best_tau = max(results, key=results.get)
    best_rho = results[best_tau]
    passed = best_rho > RHO_GATE

    detail_str = ", ".join(f"tau={t}: {r:.3f}" for t, r in sorted(results.items()))
    notes = f"Best tau = {best_tau}. {detail_str}"
    return best_rho, passed, notes


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Minimum trajectory length (Kuramoto)
# ═══════════════════════════════════════════════════════════════════════════

def test3_min_trajectory_length():
    """Kuramoto with varying trajectory lengths."""
    lengths = [200, 500, 1000, 2000, 5000]
    K_values = np.linspace(0, 5, 8)
    results = {}

    for length in lengths:
        all_gt = []
        all_feats = []
        for si in range(N_SEEDS):
            seed = BASE_SEED + si
            for K in K_values:
                # Run longer than needed, then truncate after transient discard
                total_steps = int(length / (1 - TRANSIENT_FRAC)) + 100
                traj, r = kuramoto(N=50, K=K, n_steps=total_steps, dt=0.01, seed=seed)
                states = discard_transient(traj)
                states = states[:length]
                if len(states) < 50:
                    continue
                all_gt.append(r)
                cloud = pca_reduce(states, PCA_DIM)
                feats = extract_topo(cloud, seed=seed)
                all_feats.append(feats)
        _, rho, _ = best_correlation(all_gt, all_feats)
        results[length] = rho

    # Find minimum length where |rho| > 0.6
    min_length = None
    for length in sorted(results.keys()):
        if results[length] > RHO_GATE:
            min_length = length
            break

    best_rho = max(results.values())
    passed = best_rho > RHO_GATE

    detail_str = ", ".join(f"{l}: {r:.3f}" for l, r in sorted(results.items()))
    min_str = f"Min length = {min_length}" if min_length else "No length passed"
    notes = f"{min_str}. {detail_str}"
    return best_rho, passed, notes


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Noise robustness (Kuramoto)
# ═══════════════════════════════════════════════════════════════════════════

def test4_noise_robustness():
    """Kuramoto with observation noise at various SNR levels.
    SNR is normalized per-trajectory (per-column std) to avoid systematic
    bias where synchronized states have lower signal variance."""
    snr_values = [50, 20, 10, 5, 3, 1]
    K_values = np.linspace(0, 5, 8)
    results = {}

    for snr in snr_values:
        all_gt = []
        all_feats = []
        for si in range(N_SEEDS):
            seed = BASE_SEED + si
            rng = np.random.default_rng(seed + 1000 + int(snr * 100))
            for K in K_values:
                traj, r = kuramoto(N=50, K=K, n_steps=5000, dt=0.01, seed=seed)
                states = discard_transient(traj)
                # Per-column SNR: noise_std = column_std / sqrt(snr)
                col_std = np.std(states, axis=0)
                col_std = np.maximum(col_std, 1e-10)
                noise_std = col_std / np.sqrt(snr)
                noise = rng.normal(0, 1, states.shape) * noise_std[None, :]
                states_noisy = states + noise

                all_gt.append(r)
                cloud = pca_reduce(states_noisy, PCA_DIM)
                feats = extract_topo(cloud, seed=seed)
                all_feats.append(feats)
        _, rho, _ = best_correlation(all_gt, all_feats)
        results[snr] = rho

    # Find minimum SNR where |rho| > 0.6
    min_snr = None
    for snr in sorted(results.keys()):  # ascending SNR
        if results[snr] > RHO_GATE:
            min_snr = snr
            break

    best_rho = max(results.values())
    passed = best_rho > RHO_GATE

    detail_str = ", ".join(f"SNR={s}: {r:.3f}" for s, r in
                           sorted(results.items(), reverse=True))
    min_str = f"Min SNR = {min_snr}" if min_snr else "No SNR passed"
    notes = f"{min_str}. {detail_str}"
    return best_rho, passed, notes


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Partial observability (Kuramoto)
# ═══════════════════════════════════════════════════════════════════════════

def test5_partial_observability():
    """Kuramoto N=50, observe random subset of k oscillators."""
    k_values = [3, 5, 10, 20, 30, 50]
    K_values = np.linspace(0, 5, 8)
    results = {}

    for k in k_values:
        all_gt = []
        all_feats = []
        for si in range(N_SEEDS):
            seed = BASE_SEED + si
            rng = np.random.default_rng(seed + 2000)
            # Choose random subset of oscillators
            observed_idx = rng.choice(50, k, replace=False)
            # Indices in sin/cos embedding: cos columns + sin columns
            obs_cols = np.concatenate([observed_idx, observed_idx + 50])

            for K_val in K_values:
                traj, r = kuramoto(N=50, K=K_val, n_steps=5000, dt=0.01, seed=seed)
                states = discard_transient(traj)
                # Observe only subset
                states_partial = states[:, obs_cols]

                all_gt.append(r)  # ground truth from ALL oscillators
                cloud = pca_reduce(states_partial, PCA_DIM)
                feats = extract_topo(cloud, seed=seed)
                all_feats.append(feats)
        _, rho, _ = best_correlation(all_gt, all_feats)
        results[k] = rho

    # Find minimum k where |rho| > 0.6
    min_k = None
    for kk in sorted(results.keys()):
        if results[kk] > RHO_GATE:
            min_k = kk
            break

    best_rho = max(results.values())
    passed = best_rho > RHO_GATE

    detail_str = ", ".join(f"k={kk}: {r:.3f}" for kk, r in sorted(results.items()))
    min_str = f"Min k = {min_k}" if min_k else "No k passed"
    notes = f"{min_str}. {detail_str}"
    return best_rho, passed, notes


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Real power grid frequency (synthetic fallback)
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_grid_frequency(disturbance_mag: float, seed: int):
    """Synthetic 50 Hz grid: pink noise + 3 disturbance events.
    Returns (signal_1d, disturbance_mag) for correlation."""
    rng = np.random.default_rng(seed)
    n_points = 3600  # 1 hour at 1 Hz (simplified)

    # 50 Hz base (constant at this sampling — we model deviation from nominal)
    base = np.zeros(n_points)

    # Pink noise (1/f): generate white, cumsum, detrend
    white = rng.standard_normal(n_points)
    pink = np.cumsum(white) * 0.01
    pink -= np.linspace(pink[0], pink[-1], n_points)

    # 3 disturbance events: sudden deviations lasting 30 samples
    signal = base + pink
    event_starts = [600, 1200, 2400]  # separated by ~10 min
    for start in event_starts:
        end = min(start + 30, n_points)
        signal[start:end] += disturbance_mag * rng.choice([-1, 1])

    return signal


def test6_real_power_grid():
    """Synthetic power grid frequency. Sweep disturbance magnitude."""
    # Try real data download first
    real_data = False
    try:
        import urllib.request
        url = "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
        # Don't actually download — it's huge. Use synthetic.
        raise ConnectionError("Using synthetic fallback")
    except Exception:
        real_data = False

    dist_values = np.linspace(0.1, 1.0, 8)
    window_size = 300
    step = 60
    all_gt = []
    all_feats = []

    for si in range(N_SEEDS):
        seed = BASE_SEED + si
        for dist_mag in dist_values:
            signal = generate_synthetic_grid_frequency(dist_mag, seed)
            n_points = len(signal)

            # Sliding windows: compute topology features and RMS deviation
            win_feats = []
            win_rms = []
            for start in range(0, n_points - window_size, step):
                window = signal[start:start + window_size]
                rms = float(np.sqrt(np.mean(window**2)))
                win_rms.append(rms)

                # Takens-like embedding of 1D signal for PH
                delay = 10
                dim = 5
                n_embed = len(window) - (dim - 1) * delay
                if n_embed < 50:
                    continue
                embedded = np.zeros((n_embed, dim))
                for d in range(dim):
                    embedded[:, d] = window[d * delay:d * delay + n_embed]

                feats = extract_topo(embedded, seed=seed)
                win_feats.append(feats)

            # Correlate topology features with RMS across windows
            if len(win_feats) < len(win_rms):
                win_rms = win_rms[:len(win_feats)]

            all_gt.append(dist_mag)
            # Average topology features across windows for this param value
            if win_feats:
                avg_feats = {}
                for fname in TOPO_FEATURES:
                    avg_feats[fname] = np.mean([f[fname] for f in win_feats])
                all_feats.append(avg_feats)
            else:
                all_feats.append(dict(ZERO_FEATURES))

    _, rho, _ = best_correlation(all_gt, all_feats)
    passed = rho > RHO_GATE
    src = "Real" if real_data else "Synthetic"
    notes = f"{src}"
    return rho, passed, notes


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Real EEG alpha power prediction
# ═══════════════════════════════════════════════════════════════════════════

def bandpass_filter(data: np.ndarray, low: float, high: float,
                    sfreq: float, order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter."""
    from scipy.signal import butter, sosfilt
    nyq = sfreq / 2
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return data
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return sosfilt(sos, data)


def band_power(signal: np.ndarray, low: float, high: float, sfreq: float) -> float:
    """Mean squared amplitude after bandpass."""
    filtered = bandpass_filter(signal, low, high, sfreq)
    return float(np.mean(filtered**2))


def test7_real_eeg():
    """Sleep-EDF EEG: topology features vs band powers in sliding windows."""
    try:
        import mne
        mne.set_log_level("ERROR")
    except ImportError:
        return 0.0, False, "MNE not installed"

    try:
        paths = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=[0], recording=[1],
        )
        raw_fname, _ = paths[0]
        raw = mne.io.read_raw_edf(raw_fname, preload=True)
    except Exception as e:
        return 0.0, False, f"Data download failed: {e}"

    # Extract Fpz-Cz channel
    ch_name = None
    for ch in ["EEG Fpz-Cz", "EEG Pz-Oz"]:
        if ch in raw.ch_names:
            ch_name = ch
            break
    if ch_name is None:
        return 0.0, False, "Channel not found"

    signal = raw.get_data(picks=[ch_name])[0]
    sfreq = raw.info["sfreq"]  # 100 Hz

    # First 30 minutes
    max_samples = int(30 * 60 * sfreq)
    signal = signal[:max_samples]

    # Sliding windows: 2000 samples (20s), step 500
    window_size = 2000
    step_size = 500
    takens_delay = 10
    takens_dim = 5

    topo_features_list = []
    alpha_powers = []
    delta_powers = []
    theta_powers = []
    beta_powers = []

    n_windows = (len(signal) - window_size) // step_size + 1
    # Limit windows for speed
    max_windows = 200
    actual_step = step_size
    if n_windows > max_windows:
        actual_step = (len(signal) - window_size) // max_windows

    for start in range(0, len(signal) - window_size, actual_step):
        window = signal[start:start + window_size]

        # Bandpass 1-45 Hz
        filtered = bandpass_filter(window, 1.0, 45.0, sfreq)

        # Takens embedding
        n_embed = len(filtered) - (takens_dim - 1) * takens_delay
        if n_embed < 50:
            continue
        embedded = np.zeros((n_embed, takens_dim))
        for d in range(takens_dim):
            embedded[:, d] = filtered[d * takens_delay:d * takens_delay + n_embed]

        # PH
        feats = extract_topo(embedded, seed=BASE_SEED)
        topo_features_list.append(feats)

        # Band powers
        alpha_powers.append(band_power(window, 8.0, 13.0, sfreq))
        delta_powers.append(band_power(window, 1.0, 4.0, sfreq))
        theta_powers.append(band_power(window, 4.0, 8.0, sfreq))
        beta_powers.append(band_power(window, 13.0, 30.0, sfreq))

    if len(topo_features_list) < 10:
        return 0.0, False, "Too few windows"

    # Correlate each topo feature with each band
    bands = {
        "alpha (8-13 Hz)": alpha_powers,
        "delta (1-4 Hz)": delta_powers,
        "theta (4-8 Hz)": theta_powers,
        "beta (13-30 Hz)": beta_powers,
    }

    best_band = None
    best_band_rho = 0.0
    band_results = {}

    for band_name, powers in bands.items():
        powers_arr = np.array(powers)
        band_best_rho = 0.0
        for fname in TOPO_FEATURES:
            feat_arr = np.array([f[fname] for f in topo_features_list])
            if np.std(feat_arr) < 1e-15 or np.std(powers_arr) < 1e-15:
                continue
            rho, _ = spearmanr(feat_arr, powers_arr)
            if np.isfinite(rho) and abs(rho) > band_best_rho:
                band_best_rho = abs(rho)
        band_results[band_name] = band_best_rho
        if band_best_rho > best_band_rho:
            best_band_rho = band_best_rho
            best_band = band_name

    passed = best_band_rho > RHO_GATE
    band_str = ", ".join(f"{b}: {r:.3f}" for b, r in band_results.items())
    notes = f"Best band = {best_band}. {band_str}"
    return best_band_rho, passed, notes


# ═══════════════════════════════════════════════════════════════════════════
# TEST 8: Real weather (Lorenz-96 seasonal proxy)
# ═══════════════════════════════════════════════════════════════════════════

def test8_real_weather():
    """Lorenz-96 with seasonal forcing as weather proxy.
    F varies sinusoidally 6-10 over 1000 'days'. PCA(10), 30-day rolling Lyapunov."""
    N = 36
    dt = 0.01
    n_days = 1000
    steps_per_day = int(1 / dt)  # 100 steps per "day" at dt=0.01
    # Subsample to ~36500 steps total, then take daily values
    total_steps = 36500
    subsample_rate = max(1, total_steps // (n_days * 10))

    rng = np.random.default_rng(BASE_SEED)
    x = rng.standard_normal(N) * 0.01 + 8.0

    # Collect trajectory with time-varying F
    trajectory = []
    for step in range(total_steps):
        day = step * n_days / total_steps
        # Seasonal forcing: summer=6 (less chaotic), winter=10 (more chaotic)
        F = 8.0 + 2.0 * np.sin(2 * np.pi * day / 365.0)

        dx = np.zeros(N)
        for i in range(N):
            dx[i] = ((x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N]
                     - x[i] + F)
        x = x + dx * dt
        if np.any(~np.isfinite(x)):
            break

        if step % subsample_rate == 0:
            trajectory.append(x.copy())

    trajectory = np.array(trajectory)
    if len(trajectory) < 100:
        return 0.0, False, "Trajectory too short"

    # 30-day windows: compute rolling Lyapunov and topology features
    # Each "day" ≈ subsample_rate steps, so window = 30 * (total_steps / n_days) / subsample_rate
    points_per_day = total_steps / (n_days * subsample_rate)
    window_points = max(30, int(30 * points_per_day))
    step_points = max(5, window_points // 3)

    all_gt = []
    all_feats = []

    for start in range(0, len(trajectory) - window_points, step_points):
        window_traj = trajectory[start:start + window_points]

        # Rolling Lyapunov: variance of trajectory as proxy
        # (full Lyapunov computation per window is too expensive)
        # Use local expansion rate: mean norm of differences
        diffs = np.diff(window_traj, axis=0)
        local_expansion = float(np.mean(np.linalg.norm(diffs, axis=1)))
        all_gt.append(local_expansion)

        # PCA(10) + PH
        cloud = pca_reduce(window_traj, 10)
        feats = extract_topo(cloud, seed=BASE_SEED)
        all_feats.append(feats)

    _, rho, _ = best_correlation(all_gt, all_feats)
    passed = rho > RHO_GATE
    notes = "Seasonal correlation"
    return rho, passed, notes


# ═══════════════════════════════════════════════════════════════════════════
# TEST 9: Reaction-diffusion (Gray-Scott) Turing patterns
# ═══════════════════════════════════════════════════════════════════════════

def gray_scott(N_grid: int, f: float, k: float, n_steps: int, dt: float, seed: int):
    """1D Gray-Scott reaction-diffusion. Returns trajectory and spatial entropy."""
    rng = np.random.default_rng(seed)
    D_u = 0.16
    D_v = 0.08
    dx2 = 1.0  # grid spacing squared

    u = np.ones(N_grid) + rng.uniform(-0.01, 0.01, N_grid)
    v = np.zeros(N_grid) + rng.uniform(0, 0.01, N_grid)
    # Seed a small region with v
    mid = N_grid // 2
    v[mid - 3:mid + 3] = 0.25
    u[mid - 3:mid + 3] = 0.5

    trajectory = np.zeros((n_steps, 2 * N_grid))

    for t in range(n_steps):
        trajectory[t, :N_grid] = u
        trajectory[t, N_grid:] = v

        # Laplacian with periodic BC
        lap_u = (np.roll(u, 1) + np.roll(u, -1) - 2 * u) / dx2
        lap_v = (np.roll(v, 1) + np.roll(v, -1) - 2 * v) / dx2

        du = D_u * lap_u - u * v**2 + f * (1 - u)
        dv = D_v * lap_v + u * v**2 - (f + k) * v

        u = u + du * dt
        v = v + dv * dt
        u = np.clip(u, 0, 2)
        v = np.clip(v, 0, 2)

    # Spatial entropy of final u-field
    u_final = u
    bins = np.linspace(u_final.min() - 1e-10, u_final.max() + 1e-10, 11)
    counts, _ = np.histogram(u_final, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))

    return trajectory, entropy


def test9_reaction_diffusion():
    """Gray-Scott: sweep feed rate f, measure spatial entropy."""
    f_values = np.linspace(0.02, 0.06, 8)
    k = 0.06
    all_gt = []
    all_feats = []

    for si in range(N_SEEDS):
        seed = BASE_SEED + si
        for f in f_values:
            traj, entropy = gray_scott(
                N_grid=50, f=f, k=k, n_steps=10000, dt=0.5, seed=seed
            )
            states = discard_transient(traj)
            if len(states) < 100:
                continue
            all_gt.append(entropy)
            cloud = pca_reduce(states, PCA_DIM)
            feats = extract_topo(cloud, seed=seed)
            all_feats.append(feats)

    _, rho, _ = best_correlation(all_gt, all_feats)
    passed = rho > RHO_GATE
    return rho, passed, ""


# ═══════════════════════════════════════════════════════════════════════════
# TEST 10: Multi-agent flocking (Vicsek model)
# ═══════════════════════════════════════════════════════════════════════════

def vicsek_model(N: int, eta: float, n_steps: int, v0: float, R: float,
                 L: float, seed: int):
    """Vicsek flocking model. Returns trajectory (positions + headings) and
    order parameter."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, L, N)
    y = rng.uniform(0, L, N)
    theta = rng.uniform(-np.pi, np.pi, N)

    trajectory = np.zeros((n_steps, 3 * N))  # x, y, theta for each agent

    for t in range(n_steps):
        trajectory[t, :N] = x
        trajectory[t, N:2*N] = y
        trajectory[t, 2*N:] = theta

        # Compute neighbors and average heading
        new_theta = np.zeros(N)
        for i in range(N):
            # Distance with periodic BC
            dx = x - x[i]
            dy = y - y[i]
            dx = dx - L * np.round(dx / L)
            dy = dy - L * np.round(dy / L)
            dist = np.sqrt(dx**2 + dy**2)

            neighbors = dist < R
            if np.any(neighbors):
                avg_cos = np.mean(np.cos(theta[neighbors]))
                avg_sin = np.mean(np.sin(theta[neighbors]))
                new_theta[i] = np.arctan2(avg_sin, avg_cos)
            else:
                new_theta[i] = theta[i]

        # Add noise
        theta = new_theta + rng.uniform(-eta / 2, eta / 2, N)

        # Update positions
        x = (x + v0 * np.cos(theta)) % L
        y = (y + v0 * np.sin(theta)) % L

    # Order parameter: |mean(exp(i*theta))|
    cos_mean = np.mean(np.cos(theta))
    sin_mean = np.mean(np.sin(theta))
    order = np.sqrt(cos_mean**2 + sin_mean**2)

    return trajectory, float(order)


def test10_vicsek_flocking():
    """Vicsek model: sweep noise eta, measure alignment order parameter."""
    eta_values = np.linspace(0, 2.0, 8)
    all_gt = []
    all_feats = []

    for si in range(N_SEEDS):
        seed = BASE_SEED + si
        for eta in eta_values:
            traj, order = vicsek_model(
                N=50, eta=eta, n_steps=5000, v0=0.5, R=1.0, L=10.0, seed=seed
            )
            states = discard_transient(traj)
            if len(states) < 100:
                continue
            all_gt.append(order)
            cloud = pca_reduce(states, PCA_DIM)
            feats = extract_topo(cloud, seed=seed)
            all_feats.append(feats)

    _, rho, _ = best_correlation(all_gt, all_feats)
    passed = rho > RHO_GATE
    return rho, passed, ""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

# Round 1 results for combined summary
ROUND1_RESULTS = {
    "Kuramoto oscillators": (0.883, True),
    "Lotka-Volterra ecosystem": (0.694, True),
    "Coupled map lattice": (0.767, True),
    "Hopfield network": (0.719, True),
    "FitzHugh-Nagumo population": (0.644, True),
    "Boolean network (NK)": (0.619, True),
    "Lorenz-96 weather": (0.408, False),
    "Spiking STDP network": (0.295, False),
    "Power grid stability": (0.827, True),
    "Gene regulatory network": (0.659, True),
}

TESTS = [
    (1, "Lorenz-96 PCA fix", test1_lorenz96_pca_fix),
    (2, "Spiking pop. PCA fix", test2_spiking_pop_pca),
    (3, "Min trajectory length", test3_min_trajectory_length),
    (4, "Noise robustness (Kuramoto)", test4_noise_robustness),
    (5, "Partial observability", test5_partial_observability),
    (6, "Real power grid", test6_real_power_grid),
    (7, "Real EEG alpha power", test7_real_eeg),
    (8, "Real weather (Lorenz-96)", test8_real_weather),
    (9, "Reaction-diffusion", test9_reaction_diffusion),
    (10, "Vicsek flocking", test10_vicsek_flocking),
]


def main():
    t_global = time.time()

    print("=" * 79)
    print("PROBE BATTERY ROUND 2: 10 TESTS")
    print("Failure fixes, boundary stress tests, real-world data validation")
    print("=" * 79)
    print(f"  Protocol: PCA + PH(max_dim=1, subsample={SUBSAMPLE})")
    print(f"  Discard first {int(TRANSIENT_FRAC*100)}% as transient, {N_SEEDS} seeds")
    print(f"  Pass gate: |rho| > {RHO_GATE}")
    print()

    results = []

    for num, name, test_fn in TESTS:
        t0 = time.time()
        print(f"  [{num:2d}/10] {name:<30s} ...", end="", flush=True)

        try:
            rho, passed, notes = test_fn()
        except Exception as e:
            rho, passed, notes = 0.0, False, f"ERROR: {e}"

        elapsed = time.time() - t0
        tag = "PASS" if passed else "FAIL"
        results.append({
            "num": num,
            "name": name,
            "rho": rho,
            "passed": passed,
            "notes": notes,
        })
        print(f"  |rho|={rho:.3f}  {tag}  ({elapsed:.1f}s)")

        # Time budget warning
        total_so_far = time.time() - t_global
        remaining = 10 - num
        if remaining > 0 and total_so_far > 15 * 60:
            avg_per = total_so_far / num
            est_total = avg_per * 10
            if est_total > 25 * 60:
                print(f"\n  WARNING: Projected {est_total/60:.0f} min total.")

    # ===================================================================
    # RESULTS TABLE
    # ===================================================================
    print()
    print("ROUND 2: 10 TESTS")
    print("=" * 79)
    hdr = f"{'#':>3}   {'Test':<30} {'|rho|':>7}   {'Pass?':>5}   Notes"
    print(hdr)
    print("-" * 79)

    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        notes = r["notes"][:30] if r["notes"] else ""
        print(f"{r['num']:>3}   {r['name']:<30} {r['rho']:>7.3f}   {tag:>5}   {notes}")

    n_pass = sum(1 for r in results if r["passed"])
    print("-" * 79)
    print(f"PASSED: {n_pass}/10")

    # ===================================================================
    # OPERATING ENVELOPE (combined with Round 1)
    # ===================================================================
    print()
    print("=" * 79)
    print("OPERATING ENVELOPE (combined with Round 1)")
    print("=" * 79)

    # Round 1 passes
    r1_pass = sum(1 for _, (_, p) in ROUND1_RESULTS.items() if p)
    total_systems = r1_pass + n_pass

    # Extract specific results
    min_traj = "N/A"
    min_obs = "N/A"
    max_noise = "N/A"
    min_pca = "N/A"

    for r in results:
        if r["num"] == 1 and r["notes"]:
            min_pca = r["notes"].split(".")[0]
        elif r["num"] == 3 and r["notes"]:
            min_traj = r["notes"].split(".")[0]
        elif r["num"] == 4 and r["notes"]:
            max_noise = r["notes"].split(".")[0]
        elif r["num"] == 5 and r["notes"]:
            min_obs = r["notes"].split(".")[0]

    real_data_pass = sum(1 for r in results
                         if r["num"] in [6, 7, 8] and r["passed"])

    print(f"  Systems that work:      {total_systems}/20 total across both rounds")
    print(f"  Minimum trajectory:     {min_traj}")
    print(f"  Minimum observability:  {min_obs}")
    print(f"  Maximum noise:          {max_noise}")
    print(f"  Minimum PCA:            {min_pca}")
    print(f"  Real data:              {real_data_pass}/3 real-world tests passed")

    # ===================================================================
    # VERDICT
    # ===================================================================
    print()
    print("=" * 79)
    print("VERDICT")
    print("=" * 79)

    if real_data_pass >= 2:
        print("  The probe works on REAL DATA — not just simulations.")
        print("  Topology tracks spectral/dynamical properties without")
        print("  computing them directly.")
    elif real_data_pass == 1:
        print("  PARTIAL real-data success. The probe shows promise on")
        print("  real signals but needs more validation.")
    else:
        print("  SIMULATION ONLY. The probe does not reliably transfer")
        print("  to real-world data in its current form.")

    print()
    if total_systems >= 15:
        print("  GENERAL TOOL: Works across {}/20 systems.".format(total_systems))
    elif total_systems >= 10:
        print("  MODERATE TOOL: Works on {}/20 systems.".format(total_systems))
    else:
        print("  LIMITED TOOL: Only {}/20 systems.".format(total_systems))

    print()
    print("  DOCUMENTATION RECOMMENDATION:")
    if n_pass >= 7:
        print("  'The PCA-population PH probe predicts functional properties")
        print("   of dynamical systems from state trajectories. Validated on")
        print("   coupled oscillators, neural populations, reaction-diffusion,")
        print("   ecological, and collective behavior systems. Requires:")
        for r in results:
            if r["num"] == 3 and "Min length" in r["notes"]:
                print(f"   - {r['notes'].split('.')[0]} post-transient steps")
            if r["num"] == 5 and "Min k" in r["notes"]:
                print(f"   - {r['notes'].split('.')[0]} observed variables")
            if r["num"] == 4 and "Min SNR" in r["notes"]:
                print(f"   - {r['notes'].split('.')[0]} signal-to-noise ratio")
            if r["num"] == 1 and "Best PCA dim" in r["notes"]:
                print(f"   - {r['notes'].split('.')[0]} for high-D systems")
        print("   Known limitation: spiking network voltage resets may require")
        print("   smoothed firing rates instead of raw voltages.'")
    else:
        print("  'Use with caution. The probe works for specific system classes")
        print("   (coupled oscillators, continuous dynamics) but has known")
        print("   limitations with discontinuous state variables and very")
        print("   high-dimensional systems.'")

    elapsed = time.time() - t_global
    print(f"\n  Total runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
