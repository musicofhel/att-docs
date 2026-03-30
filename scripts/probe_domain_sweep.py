#!/usr/bin/env python3
"""Domain sweep: topology probe on 10 dynamical systems beyond reservoirs.

Tests whether PCA-population PH predicts a functional property of a dynamical
system without measuring that property directly. Proved on ESN memory capacity;
now sweep 10 different systems to find where else this works.

Protocol per domain:
  1. Generate system, sweep a control parameter (8 values, 3 seeds).
  2. Compute ground-truth metric at each parameter value.
  3. PCA(3) on state trajectory, PH via PersistenceAnalyzer(max_dim=1, subsample=400).
  4. Spearman rho between best topology feature and ground truth.
  5. Pass gate: |rho| > 0.6.

Usage:
    python scripts/probe_domain_sweep.py
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
from scipy.integrate import solve_ivp
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
TRANSIENT_FRAC = 0.2  # discard first 20%

TOPO_FEATURES = [
    "persistence_entropy",
    "total_H1_pers",
    "max_H1_pers",
    "betti_0",
    "betti_1",
]

ZERO_FEATURES = {k: 0.0 for k in TOPO_FEATURES}


# ---------------------------------------------------------------------------
# PCA + topology helpers (reused from probe battery)
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


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN 1: Kuramoto oscillator synchronization
# ═══════════════════════════════════════════════════════════════════════════

def kuramoto(N: int, K: float, n_steps: int, dt: float, seed: int):
    """Kuramoto model. Returns (n_steps, 2*N) sin/cos trajectory and order param r."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(N)  # natural frequencies
    theta = rng.uniform(0, 2 * np.pi, N)
    # Use sin/cos embedding — raw phases grow unbounded and kill PCA
    trajectory = np.zeros((n_steps, 2 * N))

    for t in range(n_steps):
        trajectory[t, :N] = np.cos(theta)
        trajectory[t, N:] = np.sin(theta)
        # Coupling term
        sin_diff = np.sin(theta[None, :] - theta[:, None])  # (N, N)
        coupling = (K / N) * sin_diff.sum(axis=1)
        theta = theta + dt * (omega + coupling)

    # Order parameter: r = |mean(exp(i*theta))| from the cos/sin columns
    cos_part = trajectory[:, :N]
    sin_part = trajectory[:, N:]
    r = np.sqrt(np.mean(cos_part, axis=1)**2 + np.mean(sin_part, axis=1)**2)
    r_mean = float(np.mean(r[int(len(r) * TRANSIENT_FRAC):]))
    return trajectory, r_mean


def run_domain_kuramoto(param: float, seed: int):
    """K = param. Returns (states, ground_truth)."""
    traj, r = kuramoto(N=50, K=param, n_steps=5000, dt=0.01, seed=seed)
    states = discard_transient(traj)
    return states, r


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN 2: Lotka-Volterra ecosystem stability
# ═══════════════════════════════════════════════════════════════════════════

def lotka_volterra(n_species: int, interaction_scale: float, n_steps: int,
                   dt: float, seed: int):
    """Generalized Lotka-Volterra. Returns trajectory and max real eigenvalue."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.5, 1.5, n_species)  # growth rates
    A = rng.standard_normal((n_species, n_species)) * interaction_scale / n_species
    np.fill_diagonal(A, -1.0)  # self-regulation

    x = rng.uniform(0.5, 2.0, n_species)
    trajectory = np.zeros((n_steps, n_species))

    for t in range(n_steps):
        trajectory[t] = x
        dx = x * (r + A @ x) * dt
        x = x + dx
        x = np.clip(x, 1e-6, 1e6)
        if np.any(~np.isfinite(x)):
            trajectory = trajectory[:t + 1]
            break

    # Stability: Jacobian at final state
    x_eq = trajectory[-1]
    J = np.diag(r + A @ x_eq) + np.diag(x_eq) @ A
    max_real_eig = float(np.max(np.real(np.linalg.eigvals(J))))
    return trajectory, max_real_eig


def run_domain_lotka_volterra(param: float, seed: int):
    """interaction_scale = param."""
    try:
        traj, stability = lotka_volterra(
            n_species=5, interaction_scale=param, n_steps=5000, dt=0.005, seed=seed
        )
        states = discard_transient(traj)
        if len(states) < 100:
            return None, None
        return states, stability
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN 3: Coupled map lattice (CML)
# ═══════════════════════════════════════════════════════════════════════════

def coupled_map_lattice(N: int, epsilon: float, n_steps: int, seed: int):
    """1D CML with logistic maps. Returns trajectory and spatial correlation length."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9, N)
    trajectory = np.zeros((n_steps, N))

    for t in range(n_steps):
        trajectory[t] = x
        f_x = 4.0 * x * (1.0 - x)  # logistic at r=4
        x_new = np.zeros(N)
        for i in range(N):
            left = f_x[(i - 1) % N]
            right = f_x[(i + 1) % N]
            x_new[i] = (1 - epsilon) * f_x[i] + epsilon / 2 * (left + right)
        x = np.clip(x_new, 0, 1)

    # Spatial correlation length
    post = trajectory[int(n_steps * TRANSIENT_FRAC):]
    corrs = []
    for lag in range(1, N // 2):
        c = np.mean([np.corrcoef(post[:, i], post[:, (i + lag) % N])[0, 1]
                      for i in range(N)])
        corrs.append(abs(c) if np.isfinite(c) else 0.0)
    # Correlation length = where correlation drops below 1/e
    corrs = np.array(corrs)
    below = np.where(corrs < 1.0 / np.e)[0]
    corr_length = float(below[0] + 1) if len(below) > 0 else float(N // 2)
    return trajectory, corr_length


def run_domain_cml(param: float, seed: int):
    """epsilon = param."""
    traj, corr_len = coupled_map_lattice(N=50, epsilon=param, n_steps=5000, seed=seed)
    states = discard_transient(traj)
    return states, corr_len


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN 4: Hopfield network memory retrieval
# ═══════════════════════════════════════════════════════════════════════════

def hopfield_retrieval(N: int, P: int, n_update_steps: int, seed: int):
    """Hopfield network. Returns continuous activation trajectory and retrieval accuracy."""
    rng = np.random.default_rng(seed)
    # Store P random patterns
    patterns = rng.choice([-1, 1], size=(P, N))
    W = (patterns.T @ patterns).astype(float) / N
    np.fill_diagonal(W, 0)

    beta = 5.0  # inverse temperature for continuous activations

    # Test retrieval from noisy cues (10 trials)
    overlaps = []
    all_trajectories = []

    n_trials = max(5, P)
    for trial in range(n_trials):
        pat_idx = trial % P
        cue = patterns[pat_idx].copy()
        # Flip 20% of bits
        flip_mask = rng.random(N) < 0.2
        cue[flip_mask] *= -1

        s = cue.astype(float)
        traj = np.zeros((n_update_steps, N))
        for t in range(n_update_steps):
            h = W @ s
            s = np.tanh(beta * h)  # continuous activations
            traj[t] = s

        all_trajectories.append(traj)
        # Retrieval overlap with original pattern
        final_binary = np.sign(s)
        overlap = abs(float(np.mean(final_binary * patterns[pat_idx])))
        overlaps.append(overlap)

    # Concatenate trajectories
    trajectory = np.concatenate(all_trajectories, axis=0)
    accuracy = float(np.mean(overlaps))
    return trajectory, accuracy


def run_domain_hopfield(param: float, seed: int):
    """P/N ratio = param, N=100."""
    N = 100
    P = max(1, int(param * N))
    traj, acc = hopfield_retrieval(N=N, P=P, n_update_steps=500, seed=seed)
    states = discard_transient(traj)
    return states, acc


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN 5: FitzHugh-Nagumo neural population
# ═══════════════════════════════════════════════════════════════════════════

def fitzhugh_nagumo_population(N: int, coupling: float, n_steps: int,
                                dt: float, seed: int):
    """Coupled FHN oscillators. Returns membrane voltage trajectory and
    information transfer (fraction of neurons that respond to driven neuron)."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(-1, 1, N)
    w = rng.uniform(-0.5, 0.5, N)

    # Random coupling matrix
    C = rng.standard_normal((N, N)) * coupling / N
    np.fill_diagonal(C, 0)

    I_ext = np.zeros(N)
    I_ext[0] = 0.5  # drive neuron 0

    trajectory = np.zeros((n_steps, N))

    for t in range(n_steps):
        trajectory[t] = v
        # FHN dynamics
        dv = v - v**3 / 3.0 - w + I_ext + C @ v
        dw = 0.08 * (v + 0.7 - 0.8 * w)
        v = v + dv * dt
        w = w + dw * dt
        v = np.clip(v, -5, 5)
        w = np.clip(w, -5, 5)

    # Transfer: correlation between driven neuron and each other
    post = trajectory[int(n_steps * TRANSIENT_FRAC):]
    driver = post[:, 0]
    n_responding = 0
    for i in range(1, N):
        c = abs(np.corrcoef(driver, post[:, i])[0, 1])
        if np.isfinite(c) and c > 0.3:
            n_responding += 1
    transfer = n_responding / (N - 1)
    return trajectory, transfer


def run_domain_fhn(param: float, seed: int):
    """coupling = param."""
    traj, transfer = fitzhugh_nagumo_population(
        N=20, coupling=param, n_steps=10000, dt=0.02, seed=seed
    )
    states = discard_transient(traj)
    return states, transfer


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN 6: Boolean network (Kauffman NK model)
# ═══════════════════════════════════════════════════════════════════════════

def boolean_network(N: int, K: int, n_steps: int, seed: int):
    """Random Boolean network. Returns continuous trajectory and attractor count."""
    rng = np.random.default_rng(seed)

    # Build random Boolean functions
    connections = np.zeros((N, K), dtype=int)
    lookup_tables = []
    for i in range(N):
        connections[i] = rng.choice(N, K, replace=False)
        # Random truth table for K inputs
        table = rng.choice([0, 1], size=2**K)
        lookup_tables.append(table)

    def step(state):
        new_state = np.zeros(N, dtype=int)
        for i in range(N):
            inputs = state[connections[i]]
            idx = sum(inputs[j] * (2**j) for j in range(K))
            new_state[i] = lookup_tables[i][idx]
        return new_state

    # Count attractors by running from 100 random initial states
    seen_attractors = set()
    for _ in range(100):
        s = rng.choice([0, 1], N)
        # Run to attractor (max 200 steps)
        visited = []
        for _ in range(200):
            s_tuple = tuple(s)
            if s_tuple in [tuple(v) for v in visited[-50:]]:
                break
            visited.append(s)
            s = step(s)
        attractor_state = tuple(s)
        seen_attractors.add(attractor_state)

    n_attractors = len(seen_attractors)

    # Continuous trajectory: sliding window density
    s = rng.choice([0, 1], N)
    raw = np.zeros((n_steps, N), dtype=int)
    for t in range(n_steps):
        raw[t] = s
        s = step(s)

    # Convert to continuous: sliding window mean (window=20)
    window = 20
    trajectory = np.zeros((n_steps - window, N))
    for t in range(n_steps - window):
        trajectory[t] = raw[t:t + window].mean(axis=0)

    return trajectory, float(n_attractors)


def run_domain_boolean(param: float, seed: int):
    """K = param (integer)."""
    K = max(1, int(round(param)))
    traj, n_att = boolean_network(N=50, K=K, n_steps=2000, seed=seed)
    states = discard_transient(traj)
    return states, n_att


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN 7: Lorenz-96 weather model
# ═══════════════════════════════════════════════════════════════════════════

def lorenz96(N: int, F: float, n_steps: int, dt: float, seed: int):
    """Lorenz-96 model. Returns trajectory and largest Lyapunov exponent estimate."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N) * 0.01
    x[0] += F

    trajectory = np.zeros((n_steps, N))

    for t in range(n_steps):
        trajectory[t] = x
        # dx_i/dt = (x_{i+1} - x_{i-2})*x_{i-1} - x_i + F
        dx = np.zeros(N)
        for i in range(N):
            dx[i] = ((x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N]
                     - x[i] + F)
        x = x + dx * dt
        if np.any(~np.isfinite(x)):
            trajectory = trajectory[:t + 1]
            break

    # Lyapunov exponent via trajectory divergence using stored trajectory
    n_trans = int(len(trajectory) * TRANSIENT_FRAC)
    post = trajectory[n_trans:]
    if len(post) < 200:
        return trajectory, 0.0

    def l96_rhs(x):
        """Vectorized Lorenz-96 RHS."""
        xp1 = np.roll(x, -1)  # x_{i+1}
        xm1 = np.roll(x, 1)   # x_{i-1}
        xm2 = np.roll(x, 2)   # x_{i-2}
        return (xp1 - xm2) * xm1 - x + F

    # Run reference + perturbed in parallel, renormalize periodically
    x_ref = post[0].copy()
    x_pert = x_ref + rng.standard_normal(N) * 1e-8
    d0 = 1e-8
    lyap_sum = 0.0
    n_lyap = 0
    renorm_interval = 5

    for t in range(min(3000, len(post) - 1)):
        # RK2 integration for better accuracy
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


def run_domain_lorenz96(param: float, seed: int):
    """F = param."""
    traj, lyap = lorenz96(N=40, F=param, n_steps=5000, dt=0.01, seed=seed)
    states = discard_transient(traj)
    if len(states) < 100:
        return None, None
    return states, lyap


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN 8: Spiking network with STDP
# ═══════════════════════════════════════════════════════════════════════════

def spiking_stdp(N: int, lr: float, n_steps: int, dt: float, seed: int):
    """LIF network with STDP. Returns membrane voltage trajectory and weight entropy."""
    rng = np.random.default_rng(seed)

    # LIF parameters
    tau_m = 10.0
    v_thresh = 1.0
    v_reset = 0.0
    v_rest = 0.0

    # Initial excitatory weights
    W = rng.uniform(0.1, 0.8, (N, N))
    np.fill_diagonal(W, 0)

    # STDP parameters
    tau_plus = 20.0
    tau_minus = 20.0
    A_plus = lr
    A_minus = lr * 1.05

    v = rng.uniform(0, 0.8, N)
    last_spike = np.full(N, -1000.0)
    prev_spikes = np.zeros(N)

    trajectory = np.zeros((n_steps, N))

    # STDP update interval (every 5 steps) to reduce O(spikes*N) cost
    stdp_interval = 5

    for t in range(n_steps):
        time_ms = t * dt
        I_ext = rng.poisson(3.0, N).astype(float) * 0.5

        recurrent = W @ prev_spikes
        dv = (-v + v_rest + recurrent + I_ext) / tau_m * dt
        v = v + dv
        trajectory[t] = v

        spikes = v >= v_thresh
        prev_spikes = spikes.astype(float)
        v[spikes] = v_reset

        # Vectorized STDP (batch update, reduced frequency)
        if np.any(spikes) and t % stdp_interval == 0:
            spike_idx = np.where(spikes)[0]
            last_spike[spike_idx] = time_ms
            # All-pairs timing differences for spiking neurons
            dt_all = time_ms - last_spike  # (N,)
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

    return trajectory, entropy


def run_domain_spiking(param: float, seed: int):
    """lr = param (STDP learning rate)."""
    traj, entropy = spiking_stdp(N=100, lr=param, n_steps=10000, dt=1.0, seed=seed)
    states = discard_transient(traj)
    return states, entropy


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN 9: Power grid frequency stability
# ═══════════════════════════════════════════════════════════════════════════

def power_grid(N: int, imbalance: float, n_steps: int, dt: float, seed: int):
    """Swing equation on a ring. Returns freq trajectory and max freq deviation."""
    rng = np.random.default_rng(seed)

    M = np.ones(N) * 2.0   # inertia
    D = np.ones(N) * 1.0   # damping
    K = 5.0                  # coupling strength

    # Power injection: generators (+) and loads (-), with imbalance perturbation
    P = np.zeros(N)
    P[:N // 2] = 1.0        # generators
    P[N // 2:] = -1.0       # loads
    # Add random imbalance
    P += rng.standard_normal(N) * imbalance

    theta = rng.uniform(-0.1, 0.1, N)
    omega = np.zeros(N)  # frequency deviation from 50Hz
    trajectory = np.zeros((n_steps, N))

    for t in range(n_steps):
        trajectory[t] = omega
        # Coupling on ring
        coupling = np.zeros(N)
        for i in range(N):
            left = (i - 1) % N
            right = (i + 1) % N
            coupling[i] = K * (np.sin(theta[left] - theta[i]) +
                               np.sin(theta[right] - theta[i]))

        d_omega = (P - D * omega + coupling) / M * dt
        omega = omega + d_omega
        theta = theta + omega * dt

        omega = np.clip(omega, -10, 10)

    # Ground truth: max absolute frequency deviation (post-transient)
    post = trajectory[int(n_steps * TRANSIENT_FRAC):]
    max_dev = float(np.max(np.abs(post)))
    return trajectory, max_dev


def run_domain_power_grid(param: float, seed: int):
    """imbalance = param."""
    traj, max_dev = power_grid(N=10, imbalance=param, n_steps=5000, dt=0.01, seed=seed)
    states = discard_transient(traj)
    return states, max_dev


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN 10: Genetic regulatory network (repressilator variant)
# ═══════════════════════════════════════════════════════════════════════════

def gene_regulatory_network(degradation: float, n_steps: int,
                             dt: float, seed: int):
    """6-gene regulatory network: 3-gene repressilator core + 3 activated
    downstream genes. Returns expression trajectory (12 vars: 6 mRNA + 6 protein)
    and oscillation CV.

    Even-length repression rings converge to fixed points, so we use a 3-gene
    core (odd cycle) which oscillates, plus 3 downstream genes activated by the
    core to get 6 total with mixed interactions."""
    rng = np.random.default_rng(seed)

    N = 6
    alpha = 100.0   # max production
    alpha0 = 0.5    # basal
    K_m = 1.0
    n_hill = 3

    m = rng.uniform(0.5, 5.0, N)
    p = rng.uniform(0.5, 5.0, N)
    trajectory = np.zeros((n_steps, 2 * N))

    for t in range(n_steps):
        trajectory[t, :N] = m
        trajectory[t, N:] = p

        dm = np.zeros(N)
        # Core repressilator: 0 -| 1 -| 2 -| 0
        dm[0] = alpha / (1.0 + p[2]**n_hill) + alpha0 - degradation * m[0]
        dm[1] = alpha / (1.0 + p[0]**n_hill) + alpha0 - degradation * m[1]
        dm[2] = alpha / (1.0 + p[1]**n_hill) + alpha0 - degradation * m[2]
        # Downstream activation: gene 3 activated by 0, gene 4 by 1, gene 5 by 2
        dm[3] = alpha * p[0]**n_hill / (K_m**n_hill + p[0]**n_hill) + alpha0 - degradation * m[3]
        dm[4] = alpha * p[1]**n_hill / (K_m**n_hill + p[1]**n_hill) + alpha0 - degradation * m[4]
        dm[5] = alpha * p[2]**n_hill / (K_m**n_hill + p[2]**n_hill) + alpha0 - degradation * m[5]

        dp = m - degradation * p

        m = m + dm * dt
        p = p + dp * dt
        m = np.clip(m, 1e-6, 500)
        p = np.clip(p, 1e-6, 500)

    # Oscillation regularity: CV of peak-to-peak intervals for protein 0
    post = trajectory[int(n_steps * TRANSIENT_FRAC):, N]  # protein 0
    # Smooth slightly to avoid spurious peaks
    if len(post) > 10:
        kernel = np.ones(5) / 5
        post_smooth = np.convolve(post, kernel, mode='valid')
    else:
        post_smooth = post
    peaks = []
    for t in range(1, len(post_smooth) - 1):
        if post_smooth[t] > post_smooth[t - 1] and post_smooth[t] > post_smooth[t + 1]:
            if post_smooth[t] > np.median(post_smooth):
                peaks.append(t)
    if len(peaks) > 2:
        intervals = np.diff(peaks)
        cv = float(np.std(intervals) / (np.mean(intervals) + 1e-10))
    else:
        cv = 1.0  # no oscillation = maximal irregularity
    return trajectory, cv


def run_domain_gene_reg(param: float, seed: int):
    """degradation = param."""
    traj, cv = gene_regulatory_network(
        degradation=param, n_steps=10000, dt=0.01, seed=seed
    )
    states = discard_transient(traj)
    return states, cv


# ═══════════════════════════════════════════════════════════════════════════
# Domain registry
# ═══════════════════════════════════════════════════════════════════════════

DOMAINS = [
    {
        "num": 1,
        "name": "Kuramoto oscillators",
        "ground_truth_name": "Order parameter",
        "param_sweep": np.linspace(0, 5, 8),
        "run_fn": run_domain_kuramoto,
    },
    {
        "num": 2,
        "name": "Lotka-Volterra ecosystem",
        "ground_truth_name": "Lyapunov stability",
        "param_sweep": np.linspace(0.1, 2.0, 8),
        "run_fn": run_domain_lotka_volterra,
    },
    {
        "num": 3,
        "name": "Coupled map lattice",
        "ground_truth_name": "Spatial corr. length",
        "param_sweep": np.linspace(0, 0.5, 8),
        "run_fn": run_domain_cml,
    },
    {
        "num": 4,
        "name": "Hopfield network",
        "ground_truth_name": "Retrieval accuracy",
        "param_sweep": np.linspace(0.05, 0.25, 8),
        "run_fn": run_domain_hopfield,
    },
    {
        "num": 5,
        "name": "FitzHugh-Nagumo population",
        "ground_truth_name": "Info transfer",
        "param_sweep": np.linspace(0.1, 3.0, 8),
        "run_fn": run_domain_fhn,
    },
    {
        "num": 6,
        "name": "Boolean network (NK)",
        "ground_truth_name": "Attractor count",
        "param_sweep": np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 5]),
        "run_fn": run_domain_boolean,
    },
    {
        "num": 7,
        "name": "Lorenz-96 weather",
        "ground_truth_name": "Lyapunov exponent",
        "param_sweep": np.linspace(2, 12, 8),
        "run_fn": run_domain_lorenz96,
    },
    {
        "num": 8,
        "name": "Spiking STDP network",
        "ground_truth_name": "Weight entropy",
        "param_sweep": np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]),
        "run_fn": run_domain_spiking,
    },
    {
        "num": 9,
        "name": "Power grid stability",
        "ground_truth_name": "Max freq deviation",
        "param_sweep": np.linspace(0.1, 2.0, 8),
        "run_fn": run_domain_power_grid,
    },
    {
        "num": 10,
        "name": "Gene regulatory network",
        "ground_truth_name": "Oscillation CV",
        "param_sweep": np.linspace(0.5, 5.0, 8),
        "run_fn": run_domain_gene_reg,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Main sweep engine
# ═══════════════════════════════════════════════════════════════════════════

def run_domain(domain: dict, n_seeds: int = N_SEEDS):
    """Run one domain. Returns (best_feature, |rho|, p_value)."""
    all_gt = []
    all_feats = []

    for si in range(n_seeds):
        seed = BASE_SEED + si
        for param in domain["param_sweep"]:
            try:
                states, gt = domain["run_fn"](param, seed)
            except Exception:
                continue
            if states is None or gt is None:
                continue
            if len(states) < 50 or not np.isfinite(gt):
                continue

            all_gt.append(gt)

            # PCA(3) + PH
            cloud = pca_reduce(states, PCA_DIM)
            feats = extract_topo(cloud, seed=seed)
            all_feats.append(feats)

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


def main():
    t_global = time.time()

    print("=" * 79)
    print("DOMAIN SWEEP: 10 SYSTEMS")
    print("Topology probe on dynamical systems beyond reservoirs")
    print("=" * 79)
    print(f"  Protocol: sweep control param (8 values), {N_SEEDS} seeds")
    print(f"  PCA({PCA_DIM}) + PH(max_dim=1, subsample={SUBSAMPLE})")
    print(f"  Discard first {int(TRANSIENT_FRAC*100)}% as transient")
    print(f"  Pass gate: |rho| > {RHO_GATE}")
    print()

    results = []

    for domain in DOMAINS:
        t0 = time.time()
        num = domain["num"]
        name = domain["name"]
        print(f"  [{num:2d}/10] {name:<30s} ...", end="", flush=True)

        best_feat, best_rho, best_p = run_domain(domain)
        elapsed = time.time() - t0
        passed = best_rho > RHO_GATE
        results.append({
            "num": num,
            "name": name,
            "gt_name": domain["ground_truth_name"],
            "best_feat": best_feat,
            "rho": best_rho,
            "p": best_p,
            "passed": passed,
        })
        tag = "PASS" if passed else "FAIL"
        print(f"  |rho|={best_rho:.3f}  {tag}  ({elapsed:.1f}s)")

        # Time budget check
        total_so_far = time.time() - t_global
        remaining = 10 - num
        if remaining > 0 and total_so_far > 15 * 60:
            avg_per = total_so_far / num
            est_total = avg_per * 10
            if est_total > 25 * 60:
                print(f"\n  WARNING: Projected {est_total/60:.0f} min total. "
                      "Continuing but may exceed 25 min budget.")

    # ===================================================================
    # RESULTS TABLE
    # ===================================================================
    print()
    print("=" * 79)
    print("DOMAIN SWEEP: 10 SYSTEMS")
    print("=" * 79)
    hdr = (f"{'#':>3}   {'Domain':<28} {'Ground Truth':<22} "
           f"{'Best Feature':<22} {'|rho|':>7}   {'Pass?':>5}")
    print(hdr)
    print("─" * 79)

    for r in results:
        feat_str = r["best_feat"] if r["best_feat"] else "none"
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"{r['num']:>3}   {r['name']:<28} {r['gt_name']:<22} "
              f"{feat_str:<22} {r['rho']:>7.3f}   {tag:>5}")

    n_pass = sum(1 for r in results if r["passed"])
    print("─" * 79)
    print(f"PASSED: {n_pass}/10")

    # ===================================================================
    # WHERE TOPOLOGY HAS AN EDGE vs WHERE IT FAILS
    # ===================================================================
    print()
    print("=" * 79)
    passing = [r for r in results if r["passed"]]
    failing = [r for r in results if not r["passed"]]

    print("WHERE TOPOLOGY HAS AN EDGE:")
    if passing:
        print(f"  Passing domains: {', '.join(r['name'] for r in passing)}")
        # Analyze common threads
        feat_counts = {}
        for r in passing:
            if r["best_feat"]:
                feat_counts[r["best_feat"]] = feat_counts.get(r["best_feat"], 0) + 1
        if feat_counts:
            dominant = max(feat_counts, key=feat_counts.get)
            print(f"  Dominant feature: {dominant} "
                  f"(best in {feat_counts[dominant]}/{len(passing)} passing domains)")

        # Check if continuous vs discrete state matters
        continuous_nums = {1, 2, 3, 5, 7, 9, 10}
        discrete_nums = {4, 6, 8}
        n_pass_continuous = sum(1 for r in passing if r["num"] in continuous_nums)
        n_pass_discrete = sum(1 for r in passing if r["num"] in discrete_nums)
        print(f"  Continuous-state systems: {n_pass_continuous}/{len(continuous_nums)} pass")
        print(f"  Discrete/spiking systems: {n_pass_discrete}/{len(discrete_nums)} pass")

        # Rho distribution
        rhos = [r["rho"] for r in passing]
        print(f"  |rho| range: [{min(rhos):.3f}, {max(rhos):.3f}], "
              f"mean={np.mean(rhos):.3f}")
    else:
        print("  NONE — topology probe has no predictive power outside reservoirs")

    print()
    print("WHERE IT FAILS:")
    if failing:
        print(f"  Failing domains: {', '.join(r['name'] for r in failing)}")
        rhos = [r["rho"] for r in failing]
        print(f"  |rho| range: [{min(rhos):.3f}, {max(rhos):.3f}], "
              f"mean={np.mean(rhos):.3f}")
    else:
        print("  NONE — topology probe works everywhere!")

    # ===================================================================
    # VERDICT
    # ===================================================================
    print()
    print("=" * 79)
    print("VERDICT")
    print("=" * 79)

    if n_pass >= 7:
        print("  The PCA-population PH probe is a GENERAL tool for dynamical")
        print("  systems diagnostics. It predicts functional properties across")
        print("  diverse system types without measuring those properties directly.")
    elif n_pass >= 4:
        print("  The probe has MODERATE generality. It works beyond reservoirs")
        print("  but not universally — system properties matter.")
    elif n_pass >= 1:
        print("  The probe has LIMITED generality. It works for some dynamical")
        print("  systems but is not a general diagnostic tool.")
    else:
        print("  The probe is SPECIFIC to reservoir-like networks.")
        print("  PCA-population PH does not generalize beyond ESN-type systems.")

    # System properties that predict success/failure
    print()
    print("  PREDICTORS OF SUCCESS:")
    if passing:
        if any(r["num"] in {1, 5, 9} for r in passing):
            print("  - Coupled oscillator / population dynamics with phase structure")
        if any(r["num"] in {2, 3} for r in passing):
            print("  - Systems with parameter-dependent spatial/temporal complexity")
        if any(r["num"] in {4, 6} for r in passing):
            print("  - Discrete-state systems IF continuized state has enough variance")
        if any(r["num"] in {10} for r in passing):
            print("  - Oscillatory biochemical networks (limit cycle topology)")
        # Feature diversity
        pass_feats = set(r["best_feat"] for r in passing if r["best_feat"])
        if len(pass_feats) > 2:
            print(f"  - Multiple features contribute: {', '.join(sorted(pass_feats))}")

    print()
    print("  PREDICTORS OF FAILURE:")
    if failing:
        if any(r["num"] == 8 for r in failing):
            print("  - Spiking networks: membrane voltage resets create discontinuities")
            print("    that PCA collapses; rate coding loses timing information")
        if any(r["num"] == 7 for r in failing):
            print("  - High-dimensional chaotic systems: Lorenz-96 Lyapunov exponent")
            print("    varies smoothly but PCA(3) may miss the relevant manifold")
            print("    structure in 40-dimensional state space")
        near_miss = [r for r in failing if r["rho"] > 0.35]
        if near_miss:
            rho_strs = ", ".join(f"{r['rho']:.3f}" for r in near_miss)
            print(f"  Note: {len(near_miss)} failure(s) are near-misses "
                  f"(|rho| = {rho_strs})")
            print("  — may pass with more seeds, higher PCA dim, or longer runs")

    elapsed = time.time() - t_global
    print(f"\n  Total runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
