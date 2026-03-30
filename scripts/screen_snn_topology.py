#!/usr/bin/env python3
"""Screen SNN topology: reservoir quality prediction from persistence features.

Tests whether topological features (persistence entropy, total/max H1
persistence, Betti counts) computed on Takens-embedded neuron trajectories
predict reservoir computing quality metrics:
  Part 1: Memory capacity (MC) vs topology across spectral radius sweep
  Part 2: NARMA-10 NRMSE vs topology across spectral radius sweep
  Part 3: Spiking (LIF) network MC vs topology (gated on rho > 0.7)

The core hypothesis: persistence features capture attractor complexity
that determines computational capacity, and this generalises from
rate-based to spiking neural networks.

Usage:
    python scripts/screen_snn_topology.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
from numpy.linalg import lstsq
from scipy.stats import spearmanr

from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N_SEEDS = 3
N_NEURONS = 100
SPARSITY = 0.9        # 90% zeros = 10% connectivity
INPUT_SCALING = 0.1
WASHOUT = 1000
N_STEPS = 5000
MAX_LAG = 50
SUBSAMPLE = 400       # for PH computation

SR_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]

# LIF parameters — higher CS values needed because LIF voltage integration
# is slower than ESN tanh (dt/tau = 0.05 per step), so connection_strength
# must be O(1) to produce spiking activity.
LIF_CS_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]

# Topology feature names
TOPO_FEATURES = [
    "persistence_entropy",
    "total_H1_pers",
    "max_H1_pers",
    "betti_0",
    "betti_1",
]

# Correlation threshold for gating Part 3
RHO_GATE = 0.7


# ---------------------------------------------------------------------------
# Minimal Echo State Network (from experiment/neuromorphic-reservoir)
# ---------------------------------------------------------------------------

class MinimalESN:
    """Echo state network with adjustable spectral radius.

    100 neurons, 10% connectivity, tanh activation.
    W_base = W / max_eigenvalue, then W = W_base * target_sr.
    """

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
        self.n_neurons = n_neurons

    def run(self, inputs):
        """inputs: (n_steps,) -> states: (n_steps, n_neurons)"""
        states = []
        for u in inputs:
            self.state = np.tanh(self.W @ self.state + self.W_in.ravel() * u)
            states.append(self.state.copy())
        return np.array(states)

    def reset(self):
        """Reset internal state to zeros."""
        self.state = np.zeros(self.n_neurons)


# ---------------------------------------------------------------------------
# Minimal LIF Spiking Network
# ---------------------------------------------------------------------------

class MinimalLIF:
    """Leaky integrate-and-fire population.

    Sparse random connectivity, exponential rate filter on spikes.
    connection_strength is analogous to spectral radius.
    """

    def __init__(self, n_neurons=100, sparsity=0.9, tau=20.0, v_thresh=1.0,
                 v_reset=0.0, dt=1.0, tau_filter=5.0, input_scaling=0.5,
                 bias=0.8, seed=None):
        rng = np.random.default_rng(seed)
        self.n = n_neurons
        self.tau = tau
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.dt = dt
        self.tau_filter = tau_filter
        self.input_scaling = input_scaling
        self.bias = bias  # sub-threshold DC bias to keep neurons near threshold

        # Sparse random weights (excitatory + inhibitory)
        W = rng.standard_normal((n_neurons, n_neurons))
        mask = rng.random((n_neurons, n_neurons)) > sparsity
        W *= mask
        # Normalise by sqrt(n_connections) for stable dynamics
        n_conn = mask.sum()
        if n_conn > 0:
            W *= np.sqrt(n_neurons) / np.sqrt(n_conn)
        self.W = W  # Will be scaled by connection_strength
        self.W_in = rng.standard_normal(n_neurons) * input_scaling

    def run(self, inputs, connection_strength=1.0):
        """Run LIF with given connection strength (analogous to spectral radius).

        Returns membrane voltages (continuous, more informative for topology
        than sparse spike-filtered rates): (n_steps, n_neurons).
        """
        n_steps = len(inputs)
        W_scaled = self.W * connection_strength

        v = np.full(self.n, self.bias * 0.5)  # start near resting
        all_v = np.zeros((n_steps, self.n))

        for t in range(n_steps):
            # Leak + recurrent + input + bias
            dv = self.dt / self.tau * (
                -v + self.bias + W_scaled @ np.clip(v, 0, self.v_thresh)
                + self.W_in * inputs[t]
            )
            v = v + dv
            spikes = v > self.v_thresh
            v[spikes] = self.v_reset
            all_v[t] = v

        return all_v


# ---------------------------------------------------------------------------
# Memory Capacity
# ---------------------------------------------------------------------------

def compute_memory_capacity(states, inputs, washout=WASHOUT,
                            max_lag=MAX_LAG):
    """Compute memory capacity via linear readout at each lag.

    MC = sum of R^2 across lags k=1..max_lag.
    Standard metric from Jaeger (2001).

    Parameters
    ----------
    states : (n_steps, n_neurons) reservoir states
    inputs : (n_steps,) driving input signal
    washout : int, initial transient to discard
    max_lag : int, maximum lag to evaluate

    Returns
    -------
    mc : float, total memory capacity
    mc_per_lag : list of float, R^2 at each lag
    """
    states = states[washout:]
    inputs_trimmed = inputs[washout:]

    mc = 0.0
    mc_per_lag = []
    for k in range(1, max_lag + 1):
        # Target: input shifted by k steps
        target = inputs_trimmed[:-k]
        X = states[k:]  # align states with targets

        # Linear regression (ridge via lstsq)
        W_out, _, _, _ = lstsq(X, target, rcond=None)
        pred = X @ W_out

        # R^2
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
        mc += r2
        mc_per_lag.append(r2)

    return mc, mc_per_lag


# ---------------------------------------------------------------------------
# NARMA-10 generation and evaluation
# ---------------------------------------------------------------------------

def generate_narma10(n_steps, seed=None):
    """Generate NARMA-10 time series.

    y[t] = 0.3*y[t-1] + 0.05*y[t-1]*sum(y[t-10:t]) + 1.5*u[t-1]*u[t-10] + 0.1

    Parameters
    ----------
    n_steps : int
    seed : int or None

    Returns
    -------
    u : (n_steps,) uniform input
    y : (n_steps,) NARMA-10 target
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n_steps)
    y = np.zeros(n_steps)
    for t in range(10, n_steps):
        y[t] = (0.3 * y[t - 1]
                + 0.05 * y[t - 1] * np.sum(y[t - 10:t])
                + 1.5 * u[t - 1] * u[t - 10]
                + 0.1)
        # Clip to prevent divergence
        y[t] = np.clip(y[t], -10, 10)
    return u, y


def evaluate_narma10(states, u, y_target, washout=WASHOUT):
    """Train linear readout on NARMA-10, return NRMSE.

    Parameters
    ----------
    states : (n_steps, n_neurons) reservoir states
    u : (n_steps,) input (unused, for API consistency)
    y_target : (n_steps,) NARMA-10 target
    washout : int, initial transient to discard

    Returns
    -------
    nrmse : float, normalised root mean square error
    """
    states = states[washout:]
    y_target = y_target[washout:]

    # Train/test split (80/20)
    split = int(0.8 * len(states))
    X_train, X_test = states[:split], states[split:]
    y_train, y_test = y_target[:split], y_target[split:]

    W_out, _, _, _ = lstsq(X_train, y_train, rcond=None)
    pred = X_test @ W_out

    std_test = np.std(y_test)
    if std_test < 1e-10:
        return float("inf")

    nrmse = np.sqrt(np.mean((y_test - pred) ** 2)) / std_test
    return nrmse


# ---------------------------------------------------------------------------
# Topological Feature Extraction
# ---------------------------------------------------------------------------

def extract_topo_features(signal, subsample=SUBSAMPLE, seed=SEED):
    """Compute topological features on a scalar trajectory.

    Takens embedding -> PersistenceAnalyzer -> extract features.

    Parameters
    ----------
    signal : (n_samples,) scalar time series (e.g., neuron 0 trajectory)
    subsample : int, number of points for PH computation
    seed : int

    Returns
    -------
    dict with keys: persistence_entropy, total_H1_pers, max_H1_pers,
                    betti_0, betti_1
    """
    # Takens embedding
    try:
        embedder = TakensEmbedder("auto", "auto")
        embedder.fit(signal[:min(len(signal), 20000)])
    except Exception:
        embedder = TakensEmbedder(delay=10, dimension=3)
        embedder.fit(signal)
    cloud = embedder.transform(signal)

    # Persistent homology
    pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa.fit_transform(cloud, subsample=subsample, seed=seed)

    features = {}

    # H0 features
    dgm0 = pa.diagrams_[0] if len(pa.diagrams_) > 0 else np.array([])
    if len(dgm0) > 0:
        lifetimes0 = dgm0[:, 1] - dgm0[:, 0]
        lifetimes0 = lifetimes0[lifetimes0 > 1e-10]
        features["betti_0"] = len(lifetimes0)
    else:
        lifetimes0 = np.array([])
        features["betti_0"] = 0

    # H1 features
    dgm1 = pa.diagrams_[1] if len(pa.diagrams_) > 1 else np.array([])
    if len(dgm1) > 0:
        lifetimes1 = dgm1[:, 1] - dgm1[:, 0]
        lifetimes1 = lifetimes1[lifetimes1 > 1e-10]
        features["betti_1"] = len(lifetimes1)
        if len(lifetimes1) > 0:
            total = float(lifetimes1.sum())
            features["total_H1_pers"] = total
            features["max_H1_pers"] = float(lifetimes1.max())
            # Persistence entropy (Shannon entropy of normalised lifetimes)
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


# ---------------------------------------------------------------------------
# Correlation analysis helper
# ---------------------------------------------------------------------------

def compute_correlations(param_values, metric_values, feature_matrix,
                         metric_name="metric"):
    """Compute Spearman correlations between each feature and metric.

    Parameters
    ----------
    param_values : list, parameter values per trial (for display only)
    metric_values : list, metric value per trial
    feature_matrix : list of dicts, topological features per trial
    metric_name : str, name for output

    Returns
    -------
    correlations : dict of (rho, p_value) per feature
    """
    correlations = {}
    metric_arr = np.array(metric_values)

    for feat_name in TOPO_FEATURES:
        feat_arr = np.array([f[feat_name] for f in feature_matrix])
        # Skip if no variance
        if np.std(feat_arr) < 1e-15 or np.std(metric_arr) < 1e-15:
            correlations[feat_name] = (0.0, 1.0)
            continue
        rho, p = spearmanr(feat_arr, metric_arr)
        correlations[feat_name] = (float(rho), float(p))

    return correlations


def print_correlation_table(correlations, metric_name):
    """Print formatted correlation table."""
    print(f"\n  {'Feature':<25} {metric_name + ' Spearman rho':>20} "
          f"{'p-value':>12} {'Sig':>5}")
    print(f"  {'-' * 65}")
    for feat_name in TOPO_FEATURES:
        rho, p = correlations[feat_name]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat_name:<25} {rho:>20.4f} {p:>12.4f} {sig:>5}")


def any_strong_correlation(correlations, threshold=RHO_GATE):
    """Check if any feature has |rho| > threshold."""
    for feat_name, (rho, p) in correlations.items():
        if abs(rho) > threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Part 1: Topology predicts memory capacity
# ---------------------------------------------------------------------------

def part1_mc():
    """Topology vs memory capacity across spectral radius sweep."""
    print(f"\n{'=' * 72}")
    print("PART 1: TOPOLOGY PREDICTS MEMORY CAPACITY")
    print(f"{'=' * 72}")
    print(f"ESN: {N_NEURONS} neurons, {int((1-SPARSITY)*100)}% connectivity")
    print(f"SR values: {SR_VALUES}")
    print(f"{N_SEEDS} seeds per SR, {N_STEPS} steps, washout={WASHOUT}, "
          f"max_lag={MAX_LAG}\n")

    # Collect results per SR
    sr_mc = {sr: [] for sr in SR_VALUES}
    sr_features = {sr: [] for sr in SR_VALUES}

    # Flat lists for correlation
    all_mc = []
    all_features = []
    all_sr = []

    t0 = time.time()

    for sr in SR_VALUES:
        for seed_offset in range(N_SEEDS):
            seed = SEED + seed_offset
            rng = np.random.default_rng(seed)

            esn = MinimalESN(spectral_radius=sr, seed=seed)
            inputs = rng.standard_normal(N_STEPS)
            states = esn.run(inputs)

            # Memory capacity
            mc, mc_per_lag = compute_memory_capacity(states, inputs)

            # Topological features on neuron 0
            signal = states[WASHOUT:, 0]
            feats = extract_topo_features(signal, seed=seed)

            sr_mc[sr].append(mc)
            sr_features[sr].append(feats)
            all_mc.append(mc)
            all_features.append(feats)
            all_sr.append(sr)

        mc_vals = sr_mc[sr]
        print(f"  SR={sr:<5.2f}: MC={np.mean(mc_vals):6.2f} +/- {np.std(mc_vals):5.2f}"
              f"  (seeds: {[f'{v:.1f}' for v in mc_vals]})", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Part 1 ESN+MC computation: {elapsed:.1f}s")

    # Per-SR summary with topological features
    print(f"\n  Per-SR summary (mean +/- std across {N_SEEDS} seeds):")
    header = (f"  {'SR':>5} | {'MC':>12} | {'H1_ent':>12} | {'H1_total':>12} | "
              f"{'H1_max':>12} | {'Betti-0':>12} | {'Betti-1':>12}")
    print(header)
    print(f"  {'-' * 85}")

    for sr in SR_VALUES:
        mc_vals = sr_mc[sr]
        feats_list = sr_features[sr]

        def fmt(key):
            vals = [f[key] for f in feats_list]
            return f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}"

        mc_s = f"{np.mean(mc_vals):.2f}+/-{np.std(mc_vals):.2f}"
        print(f"  {sr:>5.2f} | {mc_s:>12} | {fmt('persistence_entropy'):>12} | "
              f"{fmt('total_H1_pers'):>12} | {fmt('max_H1_pers'):>12} | "
              f"{fmt('betti_0'):>12} | {fmt('betti_1'):>12}")

    # Correlations
    correlations = compute_correlations(all_sr, all_mc, all_features, "MC")
    print_correlation_table(correlations, "MC")

    has_strong = any_strong_correlation(correlations)
    if has_strong:
        print(f"\n  -> At least one feature with |rho| > {RHO_GATE}: "
              "topology predicts memory capacity!")
    else:
        print(f"\n  -> No feature with |rho| > {RHO_GATE}: "
              "weak topology-MC relationship.")

    print(f"\n  Part 1 total: {time.time() - t0:.1f}s")
    return correlations, all_mc, all_features, all_sr


# ---------------------------------------------------------------------------
# Part 2: Topology predicts NARMA-10 performance
# ---------------------------------------------------------------------------

def part2_narma():
    """Topology vs NARMA-10 NRMSE across spectral radius sweep."""
    print(f"\n{'=' * 72}")
    print("PART 2: TOPOLOGY PREDICTS NARMA-10")
    print(f"{'=' * 72}")
    print(f"ESN: {N_NEURONS} neurons, {int((1-SPARSITY)*100)}% connectivity")
    print(f"SR values: {SR_VALUES}")
    print(f"{N_SEEDS} seeds per SR, {N_STEPS} steps, washout={WASHOUT}\n")

    sr_nrmse = {sr: [] for sr in SR_VALUES}
    sr_features = {sr: [] for sr in SR_VALUES}

    all_nrmse = []
    all_features = []
    all_sr = []

    t0 = time.time()

    for sr in SR_VALUES:
        for seed_offset in range(N_SEEDS):
            seed = SEED + seed_offset
            esn = MinimalESN(spectral_radius=sr, seed=seed)

            u, y = generate_narma10(N_STEPS, seed=seed)
            states = esn.run(u)

            nrmse = evaluate_narma10(states, u, y)

            # Topological features on neuron 0 (driven by NARMA input)
            signal = states[WASHOUT:, 0]
            feats = extract_topo_features(signal, seed=seed)

            sr_nrmse[sr].append(nrmse)
            sr_features[sr].append(feats)
            all_nrmse.append(nrmse)
            all_features.append(feats)
            all_sr.append(sr)

        nrmse_vals = sr_nrmse[sr]
        print(f"  SR={sr:<5.2f}: NRMSE={np.mean(nrmse_vals):6.3f} +/- "
              f"{np.std(nrmse_vals):5.3f}"
              f"  (seeds: {[f'{v:.3f}' for v in nrmse_vals]})", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Part 2 ESN+NARMA computation: {elapsed:.1f}s")

    # Per-SR summary
    print(f"\n  Per-SR summary (mean +/- std across {N_SEEDS} seeds):")
    header = (f"  {'SR':>5} | {'NRMSE':>12} | {'H1_ent':>12} | {'H1_total':>12} | "
              f"{'H1_max':>12} | {'Betti-0':>12} | {'Betti-1':>12}")
    print(header)
    print(f"  {'-' * 85}")

    for sr in SR_VALUES:
        nrmse_vals = sr_nrmse[sr]
        feats_list = sr_features[sr]

        def fmt(key):
            vals = [f[key] for f in feats_list]
            return f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}"

        nrmse_s = f"{np.mean(nrmse_vals):.3f}+/-{np.std(nrmse_vals):.3f}"
        print(f"  {sr:>5.2f} | {nrmse_s:>12} | {fmt('persistence_entropy'):>12} | "
              f"{fmt('total_H1_pers'):>12} | {fmt('max_H1_pers'):>12} | "
              f"{fmt('betti_0'):>12} | {fmt('betti_1'):>12}")

    # Correlations
    correlations = compute_correlations(all_sr, all_nrmse, all_features, "NARMA NRMSE")
    print_correlation_table(correlations, "NARMA NRMSE")

    has_strong = any_strong_correlation(correlations)
    if has_strong:
        print(f"\n  -> At least one feature with |rho| > {RHO_GATE}: "
              "topology predicts NARMA-10 performance!")
    else:
        print(f"\n  -> No feature with |rho| > {RHO_GATE}: "
              "weak topology-NARMA relationship.")

    # Note about optimal SR for NARMA
    best_sr = min(sr_nrmse, key=lambda s: np.mean(sr_nrmse[s]))
    print(f"\n  Best NARMA SR: {best_sr} (NRMSE={np.mean(sr_nrmse[best_sr]):.3f})")

    print(f"\n  Part 2 total: {time.time() - t0:.1f}s")
    return correlations, all_nrmse, all_features, all_sr


# ---------------------------------------------------------------------------
# Part 3: Spiking network (LIF)
# ---------------------------------------------------------------------------

def part3_spiking():
    """LIF spiking network: MC vs topology across connection strength."""
    print(f"\n{'=' * 72}")
    print("PART 3: SPIKING NETWORK (LIF)")
    print(f"{'=' * 72}")
    print(f"LIF: {N_NEURONS} neurons, sparsity={SPARSITY}, tau=20, "
          f"v_thresh=1.0, bias=0.8, membrane voltage readout")
    print(f"Connection strength values: {LIF_CS_VALUES}")
    print(f"{N_SEEDS} seeds per CS, {N_STEPS} steps, washout={WASHOUT}\n")

    cs_mc = {cs: [] for cs in LIF_CS_VALUES}
    cs_features = {cs: [] for cs in LIF_CS_VALUES}

    all_mc = []
    all_features = []
    all_cs = []

    t0 = time.time()

    for cs in LIF_CS_VALUES:
        for seed_offset in range(N_SEEDS):
            seed = SEED + seed_offset
            rng = np.random.default_rng(seed)

            lif = MinimalLIF(n_neurons=N_NEURONS, sparsity=SPARSITY, seed=seed)
            inputs = rng.standard_normal(N_STEPS)
            states = lif.run(inputs, connection_strength=cs)

            # Check for dead neurons (no variance after washout)
            active = np.std(states[WASHOUT:], axis=0) > 1e-10
            n_active = int(active.sum())

            # Memory capacity
            mc, mc_per_lag = compute_memory_capacity(states, inputs)

            # Topological features on neuron 0
            signal = states[WASHOUT:, 0]
            # If neuron 0 is dead, try first active neuron
            if np.std(signal) < 1e-15 and n_active > 0:
                first_active = int(np.where(active)[0][0])
                signal = states[WASHOUT:, first_active]

            if np.std(signal) < 1e-15:
                # All dead — zero features
                feats = {k: 0.0 for k in TOPO_FEATURES}
            else:
                feats = extract_topo_features(signal, seed=seed)

            cs_mc[cs].append(mc)
            cs_features[cs].append(feats)
            all_mc.append(mc)
            all_features.append(feats)
            all_cs.append(cs)

        mc_vals = cs_mc[cs]
        print(f"  CS={cs:<5.2f}: MC={np.mean(mc_vals):6.2f} +/- "
              f"{np.std(mc_vals):5.2f}  (active={n_active}/{N_NEURONS})",
              flush=True)

    elapsed = time.time() - t0
    print(f"\n  Part 3 LIF+MC computation: {elapsed:.1f}s")

    # Per-CS summary
    print(f"\n  Per-CS summary (mean +/- std across {N_SEEDS} seeds):")
    header = (f"  {'CS':>5} | {'MC':>12} | {'H1_ent':>12} | {'H1_total':>12} | "
              f"{'H1_max':>12} | {'Betti-0':>12} | {'Betti-1':>12}")
    print(header)
    print(f"  {'-' * 85}")

    for cs in LIF_CS_VALUES:
        mc_vals = cs_mc[cs]
        feats_list = cs_features[cs]

        def fmt(key):
            vals = [f[key] for f in feats_list]
            return f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}"

        mc_s = f"{np.mean(mc_vals):.2f}+/-{np.std(mc_vals):.2f}"
        print(f"  {cs:>5.2f} | {mc_s:>12} | {fmt('persistence_entropy'):>12} | "
              f"{fmt('total_H1_pers'):>12} | {fmt('max_H1_pers'):>12} | "
              f"{fmt('betti_0'):>12} | {fmt('betti_1'):>12}")

    # Correlations
    correlations = compute_correlations(all_cs, all_mc, all_features, "LIF MC")
    print_correlation_table(correlations, "LIF MC")

    has_strong = any_strong_correlation(correlations)
    if has_strong:
        print(f"\n  -> At least one feature with |rho| > {RHO_GATE}: "
              "topology-MC correlation survives rate->spiking transition!")
    else:
        print(f"\n  -> No feature with |rho| > {RHO_GATE}: "
              "topology-MC correlation does NOT generalise to spiking.")

    print(f"\n  Part 3 total: {time.time() - t0:.1f}s")
    return correlations


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def print_verdict(corr_mc, corr_narma, corr_lif=None):
    """Print final verdict summarising which features predict reservoir quality."""
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")

    # Identify best features for MC
    print("\n  Memory Capacity (Part 1):")
    mc_best = max(corr_mc, key=lambda k: abs(corr_mc[k][0]))
    mc_rho, mc_p = corr_mc[mc_best]
    print(f"    Best predictor: {mc_best} (rho={mc_rho:+.4f}, p={mc_p:.4f})")
    mc_strong = [k for k, (r, p) in corr_mc.items() if abs(r) > RHO_GATE]
    if mc_strong:
        print(f"    Strong predictors (|rho|>{RHO_GATE}): {mc_strong}")
    else:
        print(f"    No strong predictors (|rho|>{RHO_GATE})")

    # Identify best features for NARMA
    print("\n  NARMA-10 (Part 2):")
    narma_best = max(corr_narma, key=lambda k: abs(corr_narma[k][0]))
    narma_rho, narma_p = corr_narma[narma_best]
    print(f"    Best predictor: {narma_best} (rho={narma_rho:+.4f}, p={narma_p:.4f})")
    narma_strong = [k for k, (r, p) in corr_narma.items() if abs(r) > RHO_GATE]
    if narma_strong:
        print(f"    Strong predictors (|rho|>{RHO_GATE}): {narma_strong}")
    else:
        print(f"    No strong predictors (|rho|>{RHO_GATE})")

    # Shared strong predictors
    mc_strong_set = set(mc_strong)
    narma_strong_set = set(narma_strong)
    shared = mc_strong_set & narma_strong_set
    if shared:
        print(f"\n  Shared strong predictors across MC + NARMA: {list(shared)}")
    else:
        print(f"\n  No shared strong predictors across MC + NARMA")

    # LIF results
    if corr_lif is not None:
        print("\n  LIF Spiking Network (Part 3):")
        lif_best = max(corr_lif, key=lambda k: abs(corr_lif[k][0]))
        lif_rho, lif_p = corr_lif[lif_best]
        print(f"    Best predictor: {lif_best} (rho={lif_rho:+.4f}, p={lif_p:.4f})")
        lif_strong = [k for k, (r, p) in corr_lif.items() if abs(r) > RHO_GATE]
        if lif_strong:
            print(f"    Strong predictors (|rho|>{RHO_GATE}): {lif_strong}")
            generalises = mc_strong_set & set(lif_strong)
            if generalises:
                print(f"\n    GENERALISATION: {list(generalises)} predict reservoir "
                      f"quality in BOTH rate and spiking networks!")
            else:
                print(f"\n    Different features are predictive in rate vs spiking.")
        else:
            print(f"    No strong predictors (|rho|>{RHO_GATE})")
            print(f"\n    Topology-MC link does NOT survive rate->spiking transition.")
    else:
        print("\n  Part 3 (LIF) was skipped (no feature passed rho > "
              f"{RHO_GATE} gate in Parts 1-2).")

    # Overall summary
    print(f"\n  {'=' * 60}")
    all_strong = mc_strong_set | narma_strong_set
    if all_strong:
        print(f"  POSITIVE: Persistence features {list(all_strong)} correlate with")
        print(f"  reservoir computing quality metrics.")
        if corr_lif is not None and any(
            abs(corr_lif[k][0]) > RHO_GATE for k in TOPO_FEATURES
        ):
            print(f"  BONUS: Correlation generalises to spiking networks.")
        elif corr_lif is not None:
            print(f"  CAVEAT: Correlation does not generalise to spiking networks.")
    else:
        print(f"  NEGATIVE: No persistence feature reliably predicts reservoir")
        print(f"  computing quality. Topology may not capture the relevant")
        print(f"  dynamical structure at these parameter settings.")

    print(f"  {'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("Screen SNN Topology: Reservoir Quality Prediction from "
          "Persistence Features")
    print(f"{'=' * 72}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {SEED}-{SEED + N_SEEDS - 1}, steps={N_STEPS}, "
          f"washout={WASHOUT}")
    print(f"Subsample for PH: {SUBSAMPLE}")
    print()

    # Part 1: MC
    corr_mc, all_mc, all_features_mc, all_sr_mc = part1_mc()

    # Part 2: NARMA-10
    corr_narma, all_nrmse, all_features_narma, all_sr_narma = part2_narma()

    # Part 3: LIF (gated on rho > 0.7 in Parts 1 OR 2)
    gate_mc = any_strong_correlation(corr_mc)
    gate_narma = any_strong_correlation(corr_narma)

    corr_lif = None
    if gate_mc or gate_narma:
        print(f"\n  Gate passed: MC={gate_mc}, NARMA={gate_narma} "
              f"-> running Part 3 (LIF)")
        corr_lif = part3_spiking()
    else:
        print(f"\n  Gate FAILED: no feature with |rho| > {RHO_GATE} "
              f"in Parts 1 or 2 -> skipping Part 3 (LIF)")

    # Verdict
    print_verdict(corr_mc, corr_narma, corr_lif)

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
