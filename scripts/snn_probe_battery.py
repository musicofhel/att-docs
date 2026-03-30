#!/usr/bin/env python3
"""SNN probe battery: 20 tests across input, size, connectivity, noise, PCA, task.

Tests the operating envelope of the PCA-population PH probe for predicting
reservoir computing performance metrics from topological features.

Each test varies one axis while holding others at default:
  200 neurons, SR=0.95, white noise input, 5000 steps, 1000 washout,
  PCA 3 components, subsample=400.

3 seeds per condition.  SR sweep per test: [0.5, 0.7, 0.9, 0.95, 1.0, 1.1, 1.3].
Pass gate: |rho| > 0.6.

Usage:
    python scripts/snn_probe_battery.py
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
from att.synthetic import lorenz_system

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants / Defaults
# ---------------------------------------------------------------------------

BASE_SEED = 42
N_SEEDS = 3
N_NEURONS_DEFAULT = 200
SPARSITY_DEFAULT = 0.9
INPUT_SCALING = 0.1
WASHOUT = 1000
N_STEPS = 5000
MAX_LAG = 50
SUBSAMPLE = 400
PCA_COMPONENTS_DEFAULT = 3
RHO_GATE = 0.6

SR_SWEEP = [0.5, 0.7, 0.9, 0.95, 1.0, 1.1, 1.3]

TOPO_FEATURES = [
    "persistence_entropy",
    "total_H1_pers",
    "max_H1_pers",
    "betti_0",
    "betti_1",
]

ZERO_FEATURES = {k: 0.0 for k in TOPO_FEATURES}


# ---------------------------------------------------------------------------
# Minimal Echo State Network (from existing branch code)
# ---------------------------------------------------------------------------

class MinimalESN:
    def __init__(self, n_neurons=N_NEURONS_DEFAULT, spectral_radius=0.9,
                 input_scaling=INPUT_SCALING, sparsity=SPARSITY_DEFAULT, seed=42,
                 W_prebuilt=None, W_in=None):
        """If W_prebuilt is given, use that weight matrix directly (for custom topologies)."""
        self.n_neurons = n_neurons
        if W_prebuilt is not None:
            self.W = W_prebuilt.copy()
            self.W_base = None  # no rescaling base
            if W_in is not None:
                self.W_in = W_in.copy()
            else:
                rng = np.random.default_rng(seed)
                self.W_in = rng.standard_normal((n_neurons, 1)) * input_scaling
        else:
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

    def set_sr(self, spectral_radius):
        if self.W_base is not None:
            self.W = self.W_base * spectral_radius
        else:
            sr = np.max(np.abs(eigvals(self.W)))
            if sr > 1e-10:
                self.W = self.W / sr * spectral_radius
        self.reset()

    def run(self, inputs):
        states = np.zeros((len(inputs), self.n_neurons))
        for i, u in enumerate(inputs):
            self.state = np.tanh(self.W @ self.state + self.W_in.ravel() * u)
            states[i] = self.state
        return states

    def reset(self):
        self.state = np.zeros(self.n_neurons)


# ---------------------------------------------------------------------------
# PCA, MC, topology helpers
# ---------------------------------------------------------------------------

def pca_reduce(states, n_components=3, washout=WASHOUT):
    X = states[washout:]
    X_c = X - X.mean(axis=0)
    if np.std(X_c) < 1e-15:
        return X_c[:, :min(n_components, X_c.shape[1])]
    _, _, Vt = svd(X_c, full_matrices=False)
    return X_c @ Vt[:min(n_components, len(Vt))].T


def compute_mc(states, inputs, washout=WASHOUT, max_lag=MAX_LAG):
    states = states[washout:]
    inputs_trimmed = inputs[washout:]
    mc = 0.0
    for k in range(1, max_lag + 1):
        target = inputs_trimmed[:-k]
        X = states[k:]
        if len(X) < 10:
            continue
        try:
            W_out, _, _, _ = lstsq(X, target, rcond=None)
            pred = X @ W_out
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
            mc += r2
        except Exception:
            pass
    return mc


def _features_from_pa(pa):
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


def extract_topo_cloud(cloud, subsample=SUBSAMPLE, seed=BASE_SEED):
    if np.std(cloud) < 1e-15:
        return dict(ZERO_FEATURES)
    pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa.fit_transform(cloud, subsample=min(subsample, len(cloud)), seed=seed)
    return _features_from_pa(pa)


# ---------------------------------------------------------------------------
# NARMA-10
# ---------------------------------------------------------------------------

def generate_narma10(n_steps, seed=None):
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
    states_w = states[washout:]
    y_w = y_target[washout:]
    split = int(0.8 * len(states_w))
    X_train, X_test = states_w[:split], states_w[split:]
    y_train, y_test = y_w[:split], y_w[split:]
    W_out, _, _, _ = lstsq(X_train, y_train, rcond=None)
    pred = X_test @ W_out
    std_test = np.std(y_test)
    if std_test < 1e-10:
        return float("inf")
    return np.sqrt(np.mean((y_test - pred) ** 2)) / std_test


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------

def gen_white_noise(n_steps, seed=42):
    return np.random.default_rng(seed).uniform(0, 0.5, n_steps)


def gen_sine(n_steps, freq=0.1, seed=42):
    t = np.arange(n_steps, dtype=float)
    return 0.25 * np.sin(2 * np.pi * freq * t) + 0.25  # range [0, 0.5]


def gen_chirp(n_steps, f0=0.01, f1=0.5, seed=42):
    t = np.linspace(0, 1, n_steps)
    freq_t = f0 + (f1 - f0) * t
    phase = 2 * np.pi * np.cumsum(freq_t) / n_steps
    return 0.25 * np.sin(phase) + 0.25


def gen_lorenz(n_steps, seed=42):
    traj = lorenz_system(n_steps=n_steps + 1000, dt=0.01, seed=seed)
    x = traj[1000:1000 + n_steps, 0]  # discard transient, take x-component
    x = (x - x.min()) / (x.max() - x.min() + 1e-15) * 0.5
    return x


def gen_sparse_binary(n_steps, p=0.05, seed=42):
    rng = np.random.default_rng(seed)
    return rng.binomial(1, p, n_steps).astype(float) * 0.5


# ---------------------------------------------------------------------------
# Custom weight matrix builders
# ---------------------------------------------------------------------------

def build_dense_esn(n_neurons, spectral_radius, seed):
    """Fully connected (sparsity=0)."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_neurons, n_neurons))
    sr = np.max(np.abs(eigvals(W)))
    if sr < 1e-10:
        sr = 1.0
    W_base = W / sr
    W_in = rng.standard_normal((n_neurons, 1)) * INPUT_SCALING
    esn = MinimalESN(n_neurons=n_neurons, W_prebuilt=W_base * spectral_radius,
                     W_in=W_in, seed=seed)
    esn.W_base = W_base
    return esn


def build_very_sparse_esn(n_neurons, spectral_radius, seed, sparsity=0.98):
    """Very sparse: only 2% connectivity."""
    return MinimalESN(n_neurons=n_neurons, spectral_radius=spectral_radius,
                      sparsity=sparsity, seed=seed)


def build_small_world_esn(n_neurons, spectral_radius, seed, k=20, p_rewire=0.1):
    """Small-world ring lattice with rewiring."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_neurons, n_neurons))
    # Ring lattice: each neuron connects to k nearest neighbors
    half_k = k // 2
    for i in range(n_neurons):
        for j in range(1, half_k + 1):
            W[i, (i + j) % n_neurons] = rng.standard_normal()
            W[i, (i - j) % n_neurons] = rng.standard_normal()
    # Rewire
    for i in range(n_neurons):
        for j in range(1, half_k + 1):
            if rng.random() < p_rewire:
                target = rng.integers(0, n_neurons)
                while target == i:
                    target = rng.integers(0, n_neurons)
                W[i, (i + j) % n_neurons] = 0
                W[i, target] = rng.standard_normal()
    sr = np.max(np.abs(eigvals(W)))
    if sr < 1e-10:
        sr = 1.0
    W_base = W / sr
    W_in = rng.standard_normal((n_neurons, 1)) * INPUT_SCALING
    esn = MinimalESN(n_neurons=n_neurons, W_prebuilt=W_base * spectral_radius,
                     W_in=W_in, seed=seed)
    esn.W_base = W_base
    return esn


# ---------------------------------------------------------------------------
# Kernel quality
# ---------------------------------------------------------------------------

def compute_kernel_quality(states, washout=WASHOUT, threshold=1e-5):
    """KQ = rank(state_matrix) / min(n_steps_post_washout, n_neurons)."""
    X = states[washout:]
    s = svd(X, compute_uv=False)
    rank = int(np.sum(s > threshold))
    return rank / min(X.shape[0], X.shape[1])


# ---------------------------------------------------------------------------
# Generalization gap (NARMA)
# ---------------------------------------------------------------------------

def compute_gen_gap(states, u, y_target, washout=WASHOUT):
    """NRMSE(test) - NRMSE(train) on 50/50 split. High = poor generalization."""
    states_w = states[washout:]
    y_w = y_target[washout:]
    n = len(states_w)
    half = n // 2
    X_train, X_test = states_w[:half], states_w[half:]
    y_train, y_test = y_w[:half], y_w[half:]
    try:
        W_out, _, _, _ = lstsq(X_train, y_train, rcond=None)
    except Exception:
        return 0.0
    pred_train = X_train @ W_out
    pred_test = X_test @ W_out
    std_train = np.std(y_train)
    std_test = np.std(y_test)
    if std_train < 1e-10 or std_test < 1e-10:
        return 0.0
    nrmse_train = np.sqrt(np.mean((y_train - pred_train) ** 2)) / std_train
    nrmse_test = np.sqrt(np.mean((y_test - pred_test) ** 2)) / std_test
    return nrmse_test - nrmse_train


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------

def run_sr_sweep(
    sr_values=SR_SWEEP,
    n_seeds=N_SEEDS,
    n_neurons=N_NEURONS_DEFAULT,
    sparsity=SPARSITY_DEFAULT,
    pca_components=PCA_COMPONENTS_DEFAULT,
    input_gen=gen_white_noise,
    observation_noise_snr=None,
    esn_factory=None,
    metric="mc",
    narma_seed_offset=10000,
    subsample_neurons=None,
):
    """Run SR sweep and return (best_feature, best_abs_rho, best_p).

    metric: "mc" | "kq" | "gen_gap"
    esn_factory: if not None, callable(n_neurons, sr, seed) -> MinimalESN
    """
    all_mc = []
    all_features = []

    for si in range(n_seeds):
        seed = BASE_SEED + si

        # Build ESN once per seed (eigvals is O(N^3) — don't repeat)
        if esn_factory is not None:
            esn_template = esn_factory(n_neurons, sr_values[0], seed)
        else:
            esn_template = MinimalESN(n_neurons=n_neurons, spectral_radius=sr_values[0],
                                      sparsity=sparsity, seed=seed)

        # Build NARMA ESN template once if needed
        esn_narma_template = None
        if metric == "gen_gap":
            if esn_factory is not None:
                esn_narma_template = esn_factory(n_neurons, sr_values[0], seed)
            else:
                esn_narma_template = MinimalESN(n_neurons=n_neurons,
                                                 spectral_radius=sr_values[0],
                                                 sparsity=sparsity, seed=seed)

        # Generate input once per seed
        u = input_gen(N_STEPS, seed=seed)

        for sr in sr_values:
            # Rescale to target SR
            esn_template.set_sr(sr)

            # Run
            states = esn_template.run(u)

            # Subsample neurons if requested (for large networks)
            if subsample_neurons is not None and states.shape[1] > subsample_neurons:
                rng_sub = np.random.default_rng(seed)
                idx = rng_sub.choice(states.shape[1], subsample_neurons, replace=False)
                states_for_pca = states[:, idx]
            else:
                states_for_pca = states

            # Add observation noise if requested
            if observation_noise_snr is not None:
                rng_noise = np.random.default_rng(seed + 9999)
                signal_power = np.var(states_for_pca[WASHOUT:])
                if signal_power > 1e-15:
                    noise_power = signal_power / observation_noise_snr
                    noise = rng_noise.normal(0, np.sqrt(noise_power),
                                             states_for_pca.shape)
                    states_for_pca = states_for_pca + noise

            # Compute metric
            if metric == "mc":
                val = compute_mc(states, u)
            elif metric == "kq":
                val = compute_kernel_quality(states)
            elif metric == "gen_gap":
                u_narma, y_narma = generate_narma10(N_STEPS, seed=seed + narma_seed_offset)
                esn_narma_template.set_sr(sr)
                states_narma = esn_narma_template.run(u_narma)
                val = compute_gen_gap(states_narma, u_narma, y_narma)
            else:
                val = compute_mc(states, u)

            all_mc.append(val)

            # Topology
            cloud = pca_reduce(states_for_pca, n_components=pca_components)
            sub = min(SUBSAMPLE, len(cloud))
            feats = extract_topo_cloud(cloud, subsample=sub, seed=seed)
            all_features.append(feats)

    # Correlations
    metric_arr = np.array(all_mc)
    best_feat, best_rho, best_p = None, 0.0, 1.0
    for fname in TOPO_FEATURES:
        feat_arr = np.array([f[fname] for f in all_features])
        if np.std(feat_arr) < 1e-15 or np.std(metric_arr) < 1e-15:
            continue
        rho, p = spearmanr(feat_arr, metric_arr)
        if abs(rho) > abs(best_rho):
            best_feat, best_rho, best_p = fname, float(rho), float(p)

    return best_feat, abs(best_rho), best_p


# ---------------------------------------------------------------------------
# The 20 tests
# ---------------------------------------------------------------------------

def define_tests():
    """Return list of (test_number, name, category, kwargs_for_run_sr_sweep)."""
    tests = []

    # --- Input type (1-5) ---
    tests.append((1, "White noise (baseline)", "Input type",
                  dict(input_gen=gen_white_noise)))
    tests.append((2, "Sine input", "Input type",
                  dict(input_gen=gen_sine)))
    tests.append((3, "Chirp input", "Input type",
                  dict(input_gen=gen_chirp)))
    tests.append((4, "Chaotic (Lorenz) input", "Input type",
                  dict(input_gen=gen_lorenz)))
    tests.append((5, "Sparse binary pulses", "Input type",
                  dict(input_gen=gen_sparse_binary)))

    # --- Network size (6-9) ---
    tests.append((6, "N=30 neurons", "Network size",
                  dict(n_neurons=30)))
    tests.append((7, "N=50 neurons", "Network size",
                  dict(n_neurons=50)))
    tests.append((8, "N=100 neurons", "Network size",
                  dict(n_neurons=100)))
    tests.append((9, "N=1000 neurons", "Network size",
                  dict(n_neurons=1000, subsample_neurons=500)))

    # --- Connectivity (10-12) ---
    tests.append((10, "Dense (sparsity=0)", "Connectivity",
                  dict(esn_factory=build_dense_esn)))
    tests.append((11, "Very sparse (sparsity=0.98)", "Connectivity",
                  dict(esn_factory=lambda n, sr, s: build_very_sparse_esn(n, sr, s, 0.98))))
    tests.append((12, "Small-world topology", "Connectivity",
                  dict(esn_factory=build_small_world_esn)))

    # --- PCA components (13-15) ---
    tests.append((13, "PCA 1 component", "PCA components",
                  dict(pca_components=1)))
    tests.append((14, "PCA 5 components", "PCA components",
                  dict(pca_components=5)))
    tests.append((15, "PCA 10 components", "PCA components",
                  dict(pca_components=10)))

    # --- Noise robustness (16-18) ---
    tests.append((16, "Obs. noise SNR=20", "Noise robustness",
                  dict(observation_noise_snr=20)))
    tests.append((17, "Obs. noise SNR=10", "Noise robustness",
                  dict(observation_noise_snr=10)))
    tests.append((18, "Obs. noise SNR=3", "Noise robustness",
                  dict(observation_noise_snr=3)))

    # --- Task target (19-20) ---
    tests.append((19, "Kernel quality target", "Task target",
                  dict(metric="kq")))
    tests.append((20, "Generalization gap target", "Task target",
                  dict(metric="gen_gap")))

    return tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_global = time.time()

    print("=" * 79)
    print("SNN PROBE BATTERY: 20 TESTS")
    print("=" * 79)
    print(f"  Defaults: {N_NEURONS_DEFAULT} neurons, sparsity={SPARSITY_DEFAULT}, "
          f"PCA={PCA_COMPONENTS_DEFAULT}, subsample={SUBSAMPLE}")
    print(f"  SR sweep: {SR_SWEEP}")
    print(f"  Seeds: {N_SEEDS} per condition  →  "
          f"{len(SR_SWEEP)} SR x {N_SEEDS} seeds = {len(SR_SWEEP) * N_SEEDS} points/test")
    print(f"  Pass gate: |rho| > {RHO_GATE}")
    print()

    tests = define_tests()
    results = []

    for num, name, category, kwargs in tests:
        t0 = time.time()
        print(f"  [{num:2d}/20] {name:<30s} ...", end="", flush=True)
        best_feat, best_rho, best_p = run_sr_sweep(**kwargs)
        elapsed = time.time() - t0
        passed = best_rho > RHO_GATE
        results.append((num, name, category, best_feat, best_rho, best_p, passed))
        tag = "PASS" if passed else "FAIL"
        print(f"  |rho|={best_rho:.3f}  {tag}  ({elapsed:.1f}s)")

        # Time check — if > 20 min already and we're less than halfway, reduce
        total_so_far = time.time() - t_global
        if num <= 10 and total_so_far > 20 * 60:
            print(f"\n  WARNING: {total_so_far/60:.1f} min elapsed at test {num}. "
                  "Remaining tests may exceed 30 min budget.")

    # ===================================================================
    # SUMMARY TABLE
    # ===================================================================
    print()
    print("=" * 79)
    print("SNN PROBE BATTERY: 20 TESTS — RESULTS")
    print("=" * 79)
    print(f"{'#':>3}   {'Test':<32} {'Best Feature':<22} {'|rho|':>8} "
          f"{'p-value':>10}   {'Pass?':>5}")
    print("─" * 79)

    for num, name, cat, feat, rho, p, passed in results:
        feat_str = feat if feat else "none"
        p_str = f"{p:.4f}" if p > 0.0001 else "<0.0001"
        tag = "PASS" if passed else "FAIL"
        print(f"{num:>3}   {name:<32} {feat_str:<22} {rho:>8.3f} "
              f"{p_str:>10}   {tag:>5}")

    n_pass = sum(1 for r in results if r[6])
    n_fail = len(results) - n_pass
    print("─" * 79)
    print(f"PASSED: {n_pass}/20    FAILED: {n_fail}/20")

    # ===================================================================
    # PASS/FAIL BY CATEGORY
    # ===================================================================
    print()
    print("=" * 79)
    print("PASS/FAIL BY CATEGORY")
    print("=" * 79)

    categories = [
        ("Input type", [r for r in results if r[2] == "Input type"]),
        ("Network size", [r for r in results if r[2] == "Network size"]),
        ("Connectivity", [r for r in results if r[2] == "Connectivity"]),
        ("PCA components", [r for r in results if r[2] == "PCA components"]),
        ("Noise robustness", [r for r in results if r[2] == "Noise robustness"]),
        ("Task target", [r for r in results if r[2] == "Task target"]),
    ]

    for cat_name, cat_results in categories:
        n_cat_pass = sum(1 for r in cat_results if r[6])
        n_cat_total = len(cat_results)
        detail = ""
        if cat_name == "Network size":
            sizes = [r[1] for r in cat_results]
            detail = f"  (range: {sizes[0]} to {sizes[-1]})"
        elif cat_name == "Noise robustness":
            passing_snr = [r for r in cat_results if r[6]]
            if passing_snr:
                # Last passing test has lowest SNR
                min_snr = passing_snr[-1][1]  # name contains SNR
                detail = f"  (max noise that passes: {min_snr})"
            else:
                detail = "  (none pass)"
        print(f"  {cat_name:<20} {n_cat_pass}/{n_cat_total}{detail}")

    # ===================================================================
    # FEATURE ROBUSTNESS
    # ===================================================================
    print()
    print("=" * 79)
    print("FEATURE FREQUENCY (best feature across all 20 tests)")
    print("=" * 79)

    feat_counts = {}
    feat_rhos = {}
    for _, _, _, feat, rho, _, _ in results:
        if feat:
            feat_counts[feat] = feat_counts.get(feat, 0) + 1
            feat_rhos.setdefault(feat, []).append(rho)

    for feat in sorted(feat_counts, key=feat_counts.get, reverse=True):
        mean_rho = np.mean(feat_rhos[feat])
        print(f"  {feat:<25} best in {feat_counts[feat]:>2}/20 tests  "
              f"(mean |rho| when best: {mean_rho:.3f})")

    # ===================================================================
    # VERDICT
    # ===================================================================
    print()
    print("=" * 79)
    print("VERDICT")
    print("=" * 79)

    # Operating envelope
    print("\n  OPERATING ENVELOPE:")
    for cat_name, cat_results in categories:
        pass_names = [r[1] for r in cat_results if r[6]]
        fail_names = [r[1] for r in cat_results if not r[6]]
        if fail_names:
            print(f"    {cat_name}: PASS={pass_names or 'none'}")
            print(f"    {'':20s} FAIL={fail_names}")
        else:
            print(f"    {cat_name}: ALL PASS")

    # Minimum network size
    size_results = [r for r in results if r[2] == "Network size"]
    passing_sizes = [r for r in size_results if r[6]]
    if passing_sizes:
        print(f"\n  MINIMUM NETWORK SIZE: {passing_sizes[0][1]}")
    else:
        print(f"\n  MINIMUM NETWORK SIZE: > 1000 (none passed)")

    # Maximum noise
    noise_results = [r for r in results if r[2] == "Noise robustness"]
    passing_noise = [r for r in noise_results if r[6]]
    if passing_noise:
        print(f"  MAX OBSERVATION NOISE: {passing_noise[-1][1]}")
    else:
        print(f"  MAX OBSERVATION NOISE: probe fails at SNR=20 (too fragile)")

    # Input type
    input_results = [r for r in results if r[2] == "Input type"]
    n_input_pass = sum(1 for r in input_results if r[6])
    if n_input_pass == len(input_results):
        print(f"  INPUT TYPE: does NOT matter — probe works on all tested inputs")
    else:
        failing = [r[1] for r in input_results if not r[6]]
        print(f"  INPUT TYPE: matters — fails on: {', '.join(failing)}")

    # PCA sweet spot
    pca_results = [r for r in results if r[2] == "PCA components"]
    best_pca = max(pca_results, key=lambda r: r[4])
    baseline_rho = results[0][4]  # test 1 = PCA 3 baseline
    print(f"  PCA SWEET SPOT: {best_pca[1]} (|rho|={best_pca[4]:.3f}) "
          f"vs baseline PCA=3 (|rho|={baseline_rho:.3f})")

    # Most robust feature
    if feat_counts:
        most_robust = max(feat_counts, key=feat_counts.get)
        print(f"  MOST ROBUST FEATURE: {most_robust} "
              f"(best in {feat_counts[most_robust]}/20 tests, "
              f"mean |rho|={np.mean(feat_rhos[most_robust]):.3f})")

    elapsed = time.time() - t_global
    print(f"\n  Total runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
