#!/usr/bin/env python3
"""Branch 10: Multi-Agent Coordination — Topological Binding in Cooperative Systems.

Tests whether ATT's binding detector captures emergent coordination topology
in multi-agent dynamical systems (flocking, synchronization, coupled chaos).

Four experiments:
1. Vicsek flocking: binding in coordinated vs disordered regimes + noise sweep
2. Kuramoto oscillators: binding sweep vs known synchronization transition
3. 3-node coupled Lorenz: direct vs indirect coupling binding
4. Population-level N-body binding: joint 5-agent topology vs pairwise marginals
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from att.binding.detector import BindingDetector
from att.embedding.joint import JointEmbedder
from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer


# ============================================================================
# HELPERS
# ============================================================================


def pprint(msg: str) -> None:
    print(msg)
    sys.stdout.flush()


def to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def quick_binding(sig_a, sig_b, subsample=200, seed=42):
    """Compute binding score without surrogate testing (fast)."""
    det = BindingDetector(max_dim=1, method="persistence_image", baseline="max")
    det.fit(sig_a, sig_b, subsample=subsample, seed=seed)
    return det.binding_score()


def full_binding(sig_a, sig_b, subsample=200, n_surrogates=50, seed=42):
    """Compute binding score with surrogate testing."""
    det = BindingDetector(max_dim=1, method="persistence_image", baseline="max")
    det.fit(sig_a, sig_b, subsample=subsample, seed=seed)
    score = det.binding_score()
    sig = det.test_significance(
        n_surrogates=n_surrogates, method="phase_randomize", seed=seed,
        subsample=subsample,
    )
    return score, sig


# ============================================================================
# SYNTHETIC SYSTEMS
# ============================================================================


def vicsek_flock(
    n_agents=30,
    n_steps=3000,
    noise=0.1,
    interaction_radius=1.0,
    speed=0.5,
    box_size=10.0,
    seed=42,
):
    """2D Vicsek flocking model.

    Agents align heading with neighbors within interaction_radius, plus noise.
    Produces phase transition from disordered to flocked at critical noise.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, box_size, (n_agents, 2))
    theta = rng.uniform(0, 2 * np.pi, n_agents)

    positions = np.zeros((n_steps, n_agents, 2))
    headings = np.zeros((n_steps, n_agents))

    for t in range(n_steps):
        positions[t] = pos.copy()
        headings[t] = theta.copy()

        new_theta = np.zeros(n_agents)
        for i in range(n_agents):
            # Periodic distance
            delta = pos - pos[i]
            delta = delta - box_size * np.round(delta / box_size)
            dists = np.linalg.norm(delta, axis=1)
            neighbors = dists < interaction_radius
            new_theta[i] = np.arctan2(
                np.mean(np.sin(theta[neighbors])),
                np.mean(np.cos(theta[neighbors])),
            ) + noise * rng.uniform(-np.pi, np.pi)

        theta = new_theta
        pos += speed * np.column_stack([np.cos(theta), np.sin(theta)])
        pos %= box_size

    return positions, headings


def order_parameter_vicsek(headings):
    """Compute Vicsek order parameter: |mean(e^{i*theta})|."""
    return float(np.abs(np.mean(np.exp(1j * headings), axis=1)).mean())


def coupled_lorenz_3node(
    n_steps=10000,
    dt=0.01,
    coupling_12=0.5,
    coupling_23=0.5,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    seed=42,
):
    """Three diffusively coupled Lorenz systems.

    Coupling structure: 1 <-> 2 <-> 3 (chain topology).
    Node 1 and 3 are NOT directly coupled.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros(9)
    state[0:3] = [1.0, 1.0, 1.0] + rng.normal(0, 0.01, 3)
    state[3:6] = [-1.0, -1.0, 1.0] + rng.normal(0, 0.01, 3)
    state[6:9] = [0.5, 0.5, 0.5] + rng.normal(0, 0.01, 3)

    trajectory = np.zeros((n_steps, 3, 3))  # (time, node, xyz)

    for t in range(n_steps):
        x1, y1, z1 = state[0:3]
        x2, y2, z2 = state[3:6]
        x3, y3, z3 = state[6:9]

        trajectory[t, 0] = [x1, y1, z1]
        trajectory[t, 1] = [x2, y2, z2]
        trajectory[t, 2] = [x3, y3, z3]

        dx1 = sigma * (y1 - x1) + coupling_12 * (x2 - x1)
        dy1 = x1 * (rho - z1) - y1
        dz1 = x1 * y1 - beta * z1

        dx2 = sigma * (y2 - x2) + coupling_12 * (x1 - x2) + coupling_23 * (x3 - x2)
        dy2 = x2 * (rho - z2) - y2
        dz2 = x2 * y2 - beta * z2

        dx3 = sigma * (y3 - x3) + coupling_23 * (x2 - x3)
        dy3 = x3 * (rho - z3) - y3
        dz3 = x3 * y3 - beta * z3

        state[0:3] += dt * np.array([dx1, dy1, dz1])
        state[3:6] += dt * np.array([dx2, dy2, dz2])
        state[6:9] += dt * np.array([dx3, dy3, dz3])

    return trajectory


def kuramoto_model(
    n_agents=20,
    n_steps=5000,
    coupling=0.5,
    omega_spread=1.0,
    dt=0.01,
    seed=42,
):
    """Kuramoto phase oscillators with mean-field coupling.

    Returns phases (n_steps, n_agents) and natural frequencies omega.
    """
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(n_agents) * omega_spread
    theta = rng.uniform(0, 2 * np.pi, n_agents)

    phases = np.zeros((n_steps, n_agents))
    for t in range(n_steps):
        phases[t] = theta.copy()
        sin_diff = np.sin(theta[None, :] - theta[:, None])
        mean_field = np.mean(sin_diff, axis=1)
        theta += (omega + coupling * mean_field) * dt

    return phases, omega


def kuramoto_order_param(phases):
    """r(t) = |mean(e^{i*theta})|, averaged over time."""
    r_t = np.abs(np.mean(np.exp(1j * phases), axis=1))
    return float(r_t.mean())


# ============================================================================
# EXPERIMENT 1: Vicsek Flocking Binding
# ============================================================================


def run_exp1(args, fig_dir):
    """Binding in Vicsek flocking: coordinated vs disordered + noise sweep."""
    pprint("\n=== Experiment 1: Vicsek Flocking Binding ===")

    subsample = args.subsample
    seed = args.seed

    # --- Flocked regime (low noise) ---
    pprint("  Generating flocked flock (noise=0.1)...")
    pos_lo, head_lo = vicsek_flock(n_agents=30, n_steps=3000, noise=0.1, seed=seed)
    op_lo = order_parameter_vicsek(head_lo)
    pprint(f"  Flocked order parameter: {op_lo:.4f}")

    ts_x_lo = pos_lo[:, 0, 0]
    ts_y_lo = pos_lo[:, 1, 0]

    pprint("  Computing flocked binding (with surrogates)...")
    score_lo, sig_lo = full_binding(
        ts_x_lo, ts_y_lo, subsample=subsample,
        n_surrogates=args.n_surrogates, seed=seed,
    )
    pprint(f"  Flocked binding: {score_lo:.4f}, p={sig_lo['p_value']:.4f}, z={sig_lo['z_score']:.2f}")

    # --- Disordered regime (high noise) ---
    pprint("  Generating disordered flock (noise=2.0)...")
    pos_hi, head_hi = vicsek_flock(n_agents=30, n_steps=3000, noise=2.0, seed=seed)
    op_hi = order_parameter_vicsek(head_hi)
    pprint(f"  Disordered order parameter: {op_hi:.4f}")

    ts_x_hi = pos_hi[:, 0, 0]
    ts_y_hi = pos_hi[:, 1, 0]

    pprint("  Computing disordered binding (with surrogates)...")
    score_hi, sig_hi = full_binding(
        ts_x_hi, ts_y_hi, subsample=subsample,
        n_surrogates=args.n_surrogates, seed=seed,
    )
    pprint(f"  Disordered binding: {score_hi:.4f}, p={sig_hi['p_value']:.4f}, z={sig_hi['z_score']:.2f}")

    # --- Noise sweep (binding score only, no surrogates for speed) ---
    noise_values = np.linspace(0.05, 3.0, 10)
    sweep_scores = []
    sweep_order = []
    pprint(f"  Running noise sweep ({len(noise_values)} values, no surrogates)...")
    for i, eta in enumerate(noise_values):
        pos_s, head_s = vicsek_flock(n_agents=30, n_steps=3000, noise=eta, seed=seed)
        op_s = order_parameter_vicsek(head_s)
        sweep_order.append(op_s)

        sc = quick_binding(pos_s[:, 0, 0], pos_s[:, 1, 0], subsample=subsample, seed=seed)
        sweep_scores.append(sc)
        pprint(f"    noise={eta:.2f}: binding={sc:.4f}, order={op_s:.4f}")

    # --- Figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(pos_lo[:, 0, 0], pos_lo[:, 0, 1], "b-", alpha=0.3, lw=0.5, label="Agent 0")
        ax.plot(pos_lo[:, 1, 0], pos_lo[:, 1, 1], "r-", alpha=0.3, lw=0.5, label="Agent 1")
        ax.set_title(f"Flocked (η=0.1, OP={op_lo:.3f})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.plot(pos_hi[:, 0, 0], pos_hi[:, 0, 1], "b-", alpha=0.3, lw=0.5, label="Agent 0")
        ax.plot(pos_hi[:, 1, 0], pos_hi[:, 1, 1], "r-", alpha=0.3, lw=0.5, label="Agent 1")
        ax.set_title(f"Disordered (η=2.0, OP={op_hi:.3f})")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        ax.plot(noise_values, sweep_scores, "ko-", lw=2)
        ax.set_xlabel("Noise (η)"); ax.set_ylabel("Binding Score")
        ax.set_title("Binding vs Noise Level")

        ax = axes[1, 1]
        ax2 = ax.twinx()
        l1 = ax.plot(noise_values, sweep_scores, "ko-", lw=2, label="Binding")
        l2 = ax2.plot(noise_values, sweep_order, "rs--", lw=2, label="Order Param")
        ax.set_xlabel("Noise (η)"); ax.set_ylabel("Binding Score")
        ax2.set_ylabel("Order Parameter")
        ax.set_title("Binding vs Order Parameter")
        lines = l1 + l2
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

        fig.suptitle("Experiment 1: Vicsek Flocking Binding", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "exp1_vicsek_flocking.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'exp1_vicsek_flocking.png'}")
    except Exception as e:
        pprint(f"  Warning: figure failed: {e}")

    return {
        "flocked_binding": float(score_lo),
        "flocked_p": float(sig_lo["p_value"]),
        "flocked_z": float(sig_lo["z_score"]),
        "flocked_order_param": float(op_lo),
        "disordered_binding": float(score_hi),
        "disordered_p": float(sig_hi["p_value"]),
        "disordered_z": float(sig_hi["z_score"]),
        "disordered_order_param": float(op_hi),
        "flocked_greater": bool(score_lo > score_hi),
        "noise_sweep": {
            "noise_values": noise_values.tolist(),
            "binding_scores": [float(s) for s in sweep_scores],
            "order_params": [float(o) for o in sweep_order],
        },
    }


# ============================================================================
# EXPERIMENT 2: Kuramoto Synchronization Binding Sweep
# ============================================================================


def run_exp2(args, fig_dir):
    """Kuramoto binding sweep: compare binding transition to known K_c."""
    pprint("\n=== Experiment 2: Kuramoto Synchronization Binding ===")

    subsample = args.subsample
    seed = args.seed

    coupling_values = np.linspace(0.0, 3.0, 10)
    binding_scores = []
    order_params = []

    # Sweep: binding score only (no surrogates) for speed
    for i, K in enumerate(coupling_values):
        pprint(f"  K={K:.2f} ({i+1}/{len(coupling_values)})...")
        phases, omega = kuramoto_model(
            n_agents=20, n_steps=5000, coupling=K, seed=seed,
        )
        op = kuramoto_order_param(phases)
        order_params.append(op)

        sig_0 = np.sin(phases[:, 0])
        sig_1 = np.sin(phases[:, 1])

        sc = quick_binding(sig_0, sig_1, subsample=subsample, seed=seed)
        binding_scores.append(sc)
        pprint(f"    binding={sc:.4f}, order={op:.4f}")

    # Surrogate test at K=0 (baseline) and K=3 (strong coupling)
    pprint("  Running surrogate test at K=0 (baseline)...")
    phases_k0, _ = kuramoto_model(n_agents=20, n_steps=5000, coupling=0.0, seed=seed)
    _, sig_k0 = full_binding(
        np.sin(phases_k0[:, 0]), np.sin(phases_k0[:, 1]),
        subsample=subsample, n_surrogates=args.n_surrogates, seed=seed,
    )
    pprint(f"    K=0: p={sig_k0['p_value']:.4f}, z={sig_k0['z_score']:.2f}")

    pprint("  Running surrogate test at K=3 (strong coupling)...")
    phases_k3, _ = kuramoto_model(n_agents=20, n_steps=5000, coupling=3.0, seed=seed)
    _, sig_k3 = full_binding(
        np.sin(phases_k3[:, 0]), np.sin(phases_k3[:, 1]),
        subsample=subsample, n_surrogates=args.n_surrogates, seed=seed,
    )
    pprint(f"    K=3: p={sig_k3['p_value']:.4f}, z={sig_k3['z_score']:.2f}")

    known_kc = 2.0 / np.pi  # ≈ 0.637

    # Detect binding transition: first coupling where binding exceeds
    # mean + 2*std of first 3 values
    baseline_mean = np.mean(binding_scores[:3])
    baseline_std = max(np.std(binding_scores[:3]), 0.01)
    threshold = baseline_mean + 2 * baseline_std
    detected_kc = None
    for j, sc in enumerate(binding_scores):
        if sc > threshold and j >= 3:
            detected_kc = float(coupling_values[j])
            break

    pprint(f"  Known K_c: {known_kc:.3f}")
    pprint(f"  Detected K_c (binding threshold): {detected_kc}")

    # --- Figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax2 = ax.twinx()
        l1 = ax.plot(coupling_values, binding_scores, "ko-", lw=2, label="Binding Score")
        l2 = ax2.plot(coupling_values, order_params, "rs--", lw=2, label="Order Param r(K)")
        ax.axvline(known_kc, color="green", ls=":", lw=2, label=f"Known K_c={known_kc:.3f}")
        if detected_kc is not None:
            ax.axvline(detected_kc, color="purple", ls="--", lw=2, label=f"Detected K_c={detected_kc:.2f}")
        ax.set_xlabel("Coupling Strength K")
        ax.set_ylabel("Binding Score")
        ax2.set_ylabel("Order Parameter r(K)")
        ax.set_title("Kuramoto: Binding vs Synchronization")
        ax.legend(fontsize=8, loc="upper left")

        ax = axes[1]
        ax.bar(["K=0 (uncoupled)", "K=3 (synced)"],
               [sig_k0["z_score"], sig_k3["z_score"]],
               color=["lightblue", "steelblue"], edgecolor="black")
        ax.axhline(2, color="red", ls="--", label="z=2 (p≈0.05)")
        ax.set_ylabel("z-score")
        ax.set_title("Surrogate Test: Uncoupled vs Synchronized")
        ax.legend(fontsize=8)

        fig.suptitle("Experiment 2: Kuramoto Synchronization Binding", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "exp2_kuramoto_binding.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'exp2_kuramoto_binding.png'}")
    except Exception as e:
        pprint(f"  Warning: figure failed: {e}")

    return {
        "coupling_values": [float(c) for c in coupling_values],
        "binding_scores": [float(s) for s in binding_scores],
        "order_params": [float(o) for o in order_params],
        "sig_k0": {"p_value": float(sig_k0["p_value"]), "z_score": float(sig_k0["z_score"])},
        "sig_k3": {"p_value": float(sig_k3["p_value"]), "z_score": float(sig_k3["z_score"])},
        "detected_kc": detected_kc,
        "known_kc": float(known_kc),
    }


# ============================================================================
# EXPERIMENT 3: 3-Node Coupled Lorenz Binding
# ============================================================================


def run_exp3(args, fig_dir):
    """Pairwise binding in 3-node Lorenz chain: direct vs indirect coupling."""
    pprint("\n=== Experiment 3: 3-Node Lorenz Binding ===")

    subsample = args.subsample
    seed = args.seed

    pprint("  Generating 3-node coupled Lorenz (10000 steps)...")
    traj = coupled_lorenz_3node(n_steps=10000, dt=0.01, coupling_12=0.5, coupling_23=0.5, seed=seed)
    traj = traj[2000:]  # discard transient

    x1 = traj[:, 0, 0]
    x2 = traj[:, 1, 0]
    x3 = traj[:, 2, 0]

    pairs = {
        "1-2 (direct)": (x1, x2),
        "2-3 (direct)": (x2, x3),
        "1-3 (indirect)": (x1, x3),
    }

    results = {}
    for name, (sig_a, sig_b) in pairs.items():
        pprint(f"  Computing binding for pair {name} (with surrogates)...")
        sc, sig = full_binding(
            sig_a, sig_b, subsample=subsample,
            n_surrogates=args.n_surrogates, seed=seed,
        )
        results[name] = {
            "binding_score": float(sc),
            "p_value": float(sig["p_value"]),
            "z_score": float(sig["z_score"]),
        }
        pprint(f"    {name}: binding={sc:.4f}, p={sig['p_value']:.4f}, z={sig['z_score']:.2f}")

    direct_scores = [results["1-2 (direct)"]["binding_score"], results["2-3 (direct)"]["binding_score"]]
    indirect_score = results["1-3 (indirect)"]["binding_score"]
    mean_direct = float(np.mean(direct_scores))

    pprint(f"  Mean direct binding: {mean_direct:.4f}")
    pprint(f"  Indirect binding: {indirect_score:.4f}")
    pprint(f"  Direct > Indirect: {mean_direct > indirect_score}")

    # --- Figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax = axes[0]
        t = np.arange(min(500, len(x1))) * 0.01
        ax.plot(t, x1[:len(t)], "b-", lw=0.8, label="Node 1", alpha=0.8)
        ax.plot(t, x2[:len(t)], "r-", lw=0.8, label="Node 2", alpha=0.8)
        ax.plot(t, x3[:len(t)], "g-", lw=0.8, label="Node 3", alpha=0.8)
        ax.set_xlabel("Time"); ax.set_ylabel("x")
        ax.set_title("3-Node Lorenz Chain (first 5s)")
        ax.legend(fontsize=8)

        ax = axes[1]
        names = list(results.keys())
        scores = [results[n]["binding_score"] for n in names]
        colors = ["steelblue", "steelblue", "coral"]
        bars = ax.bar(range(len(names)), scores, color=colors, edgecolor="black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Binding Score")
        ax.set_title("Pairwise Binding Scores")
        for bar, s in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{s:.4f}", ha="center", va="bottom", fontsize=9)

        ax = axes[2]
        zscores = [results[n]["z_score"] for n in names]
        bars = ax.bar(range(len(names)), zscores, color=colors, edgecolor="black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.axhline(2, color="red", ls="--", label="z=2 (p≈0.05)")
        ax.set_ylabel("z-score"); ax.set_title("Surrogate Test z-scores")
        ax.legend(fontsize=8)
        for bar, z in zip(bars, zscores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{z:.2f}", ha="center", va="bottom", fontsize=9)

        fig.suptitle("Experiment 3: 3-Node Lorenz Chain Binding", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "exp3_lorenz_3node.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'exp3_lorenz_3node.png'}")
    except Exception as e:
        pprint(f"  Warning: figure failed: {e}")

    return {
        "pairs": results,
        "direct_binding_12": results["1-2 (direct)"]["binding_score"],
        "direct_binding_23": results["2-3 (direct)"]["binding_score"],
        "indirect_binding_13": results["1-3 (indirect)"]["binding_score"],
        "mean_direct": mean_direct,
        "direct_greater_than_indirect": bool(mean_direct > indirect_score),
    }


# ============================================================================
# EXPERIMENT 4: Population-Level N-Body Binding
# ============================================================================


def run_exp4(args, fig_dir):
    """N-body binding: joint 5-agent topology vs all pairwise marginals."""
    pprint("\n=== Experiment 4: Population-Level 5-Agent Binding ===")

    subsample = args.subsample
    seed = args.seed
    n_agents_exp4 = 5

    # Flocked state
    pprint(f"  Generating flocked 5-agent system (noise=0.1)...")
    pos, head = vicsek_flock(
        n_agents=n_agents_exp4, n_steps=3000, noise=0.1,
        interaction_radius=1.5, seed=seed,
    )
    op = order_parameter_vicsek(head)
    pprint(f"  Order parameter: {op:.4f}")

    channels = [pos[:, i, 0] for i in range(n_agents_exp4)]

    # Joint embedding of all 5 agents
    pprint("  Computing joint 5-agent embedding...")
    joint_emb = JointEmbedder(delays="auto", dimensions="auto")
    joint_cloud = joint_emb.fit_transform(channels)
    pprint(f"  Joint cloud shape: {joint_cloud.shape}")

    pa_joint = PersistenceAnalyzer(max_dim=1, backend="ripser")
    result_joint = pa_joint.fit_transform(joint_cloud, subsample=subsample, seed=seed)
    joint_h1_count = len(pa_joint.diagrams_[1]) if len(pa_joint.diagrams_) > 1 else 0
    joint_h1_entropy = result_joint["persistence_entropy"][1] if len(result_joint["persistence_entropy"]) > 1 else 0.0
    pprint(f"  Joint: H1 features={joint_h1_count}, H1 entropy={joint_h1_entropy:.4f}")

    # All 10 pairwise marginals
    pprint("  Computing all 10 pairwise marginal topologies...")
    pairwise_h1_counts = []
    pairwise_h1_entropies = []
    pair_names = []

    for i, j in combinations(range(n_agents_exp4), 2):
        pair_emb = JointEmbedder(delays="auto", dimensions="auto")
        pair_cloud = pair_emb.fit_transform([channels[i], channels[j]])

        pa_pair = PersistenceAnalyzer(max_dim=1, backend="ripser")
        result_pair = pa_pair.fit_transform(pair_cloud, subsample=subsample, seed=seed)
        h1_count = len(pa_pair.diagrams_[1]) if len(pa_pair.diagrams_) > 1 else 0
        h1_ent = result_pair["persistence_entropy"][1] if len(result_pair["persistence_entropy"]) > 1 else 0.0
        pairwise_h1_counts.append(h1_count)
        pairwise_h1_entropies.append(h1_ent)
        pair_names.append(f"{i}-{j}")
        pprint(f"    Pair {i}-{j}: H1={h1_count}, entropy={h1_ent:.4f}")

    max_pairwise_h1 = max(pairwise_h1_counts)
    emergent_features = max(0, joint_h1_count - max_pairwise_h1)

    pprint(f"  Joint H1 features: {joint_h1_count}")
    pprint(f"  Max pairwise H1 features: {max_pairwise_h1}")
    pprint(f"  Emergent (excess) features: {emergent_features}")

    # Disordered comparison
    pprint("  Computing disordered 5-agent comparison...")
    pos_dis, head_dis = vicsek_flock(
        n_agents=n_agents_exp4, n_steps=3000, noise=2.0,
        interaction_radius=1.5, seed=seed,
    )
    channels_dis = [pos_dis[:, i, 0] for i in range(n_agents_exp4)]
    joint_emb_dis = JointEmbedder(delays="auto", dimensions="auto")
    joint_cloud_dis = joint_emb_dis.fit_transform(channels_dis)
    pa_joint_dis = PersistenceAnalyzer(max_dim=1, backend="ripser")
    result_joint_dis = pa_joint_dis.fit_transform(joint_cloud_dis, subsample=subsample, seed=seed)
    dis_h1_count = len(pa_joint_dis.diagrams_[1]) if len(pa_joint_dis.diagrams_) > 1 else 0
    dis_h1_entropy = result_joint_dis["persistence_entropy"][1] if len(result_joint_dis["persistence_entropy"]) > 1 else 0.0
    pprint(f"  Disordered joint: H1 features={dis_h1_count}, entropy={dis_h1_entropy:.4f}")

    # --- Figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax = axes[0]
        x_pos = np.arange(len(pair_names))
        ax.bar(x_pos, pairwise_h1_counts, color="lightblue", edgecolor="black", label="Pairwise")
        ax.axhline(joint_h1_count, color="red", lw=2, ls="--", label=f"Joint={joint_h1_count}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pair_names, rotation=45, fontsize=8)
        ax.set_ylabel("H1 Feature Count")
        ax.set_title("Joint vs Pairwise H1 (Flocked)")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.bar(x_pos, pairwise_h1_entropies, color="lightyellow", edgecolor="black", label="Pairwise")
        ax.axhline(joint_h1_entropy, color="red", lw=2, ls="--", label=f"Joint={joint_h1_entropy:.3f}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pair_names, rotation=45, fontsize=8)
        ax.set_ylabel("H1 Persistence Entropy")
        ax.set_title("Joint vs Pairwise Entropy (Flocked)")
        ax.legend(fontsize=8)

        ax = axes[2]
        bars = ax.bar(["Flocked", "Disordered"], [joint_h1_count, dis_h1_count],
                       color=["steelblue", "coral"], edgecolor="black")
        ax.set_ylabel("Joint H1 Feature Count")
        ax.set_title("Flocked vs Disordered Joint Topology")
        for bar, val in zip(bars, [joint_h1_count, dis_h1_count]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=11)

        fig.suptitle("Experiment 4: Population-Level 5-Agent Binding", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "exp4_population_binding.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'exp4_population_binding.png'}")
    except Exception as e:
        pprint(f"  Warning: figure failed: {e}")

    return {
        "flocked_order_param": float(op),
        "joint_5agent_h1": int(joint_h1_count),
        "joint_5agent_h1_entropy": float(joint_h1_entropy),
        "max_pairwise_h1": int(max_pairwise_h1),
        "emergent_features": int(emergent_features),
        "pairwise_h1_counts": {n: int(c) for n, c in zip(pair_names, pairwise_h1_counts)},
        "pairwise_h1_entropies": {n: float(e) for n, e in zip(pair_names, pairwise_h1_entropies)},
        "disordered_joint_h1": int(dis_h1_count),
        "disordered_joint_h1_entropy": float(dis_h1_entropy),
    }


# ============================================================================
# OVERVIEW FIGURE
# ============================================================================


def make_overview_figure(exp1, exp2, exp3, exp4, fig_dir):
    """Summary overview combining key results from all 4 experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Exp1 noise sweep
        ax = axes[0, 0]
        nv = exp1["noise_sweep"]["noise_values"]
        bs = exp1["noise_sweep"]["binding_scores"]
        op = exp1["noise_sweep"]["order_params"]
        ax.plot(nv, bs, "ko-", lw=2, label="Binding")
        ax2 = ax.twinx()
        ax2.plot(nv, op, "rs--", lw=2, label="Order Param")
        ax.set_xlabel("Noise (η)"); ax.set_ylabel("Binding Score")
        ax2.set_ylabel("Order Parameter")
        ax.set_title("Exp 1: Vicsek Flocking")

        # Panel 2: Exp2 Kuramoto sweep
        ax = axes[0, 1]
        cv = exp2["coupling_values"]
        bs2 = exp2["binding_scores"]
        op2 = exp2["order_params"]
        ax.plot(cv, bs2, "ko-", lw=2, label="Binding")
        ax2 = ax.twinx()
        ax2.plot(cv, op2, "rs--", lw=2, label="Order Param")
        ax.axvline(exp2["known_kc"], color="green", ls=":", lw=2)
        ax.set_xlabel("Coupling K"); ax.set_ylabel("Binding Score")
        ax2.set_ylabel("Order Parameter r(K)")
        ax.set_title("Exp 2: Kuramoto Synchronization")

        # Panel 3: Exp3 Lorenz
        ax = axes[1, 0]
        pair_names = ["1↔2\n(direct)", "2↔3\n(direct)", "1↔3\n(indirect)"]
        pair_scores = [exp3["direct_binding_12"], exp3["direct_binding_23"], exp3["indirect_binding_13"]]
        colors = ["steelblue", "steelblue", "coral"]
        bars = ax.bar(pair_names, pair_scores, color=colors, edgecolor="black")
        ax.set_ylabel("Binding Score")
        ax.set_title("Exp 3: 3-Node Lorenz Chain")
        for bar, s in zip(bars, pair_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{s:.4f}", ha="center", va="bottom", fontsize=9)

        # Panel 4: Exp4 joint vs pairwise
        ax = axes[1, 1]
        bars = ax.bar(
            ["Joint (5-agent)", "Max Pairwise", "Disordered\nJoint"],
            [exp4["joint_5agent_h1"], exp4["max_pairwise_h1"], exp4["disordered_joint_h1"]],
            color=["steelblue", "lightblue", "coral"], edgecolor="black",
        )
        ax.set_ylabel("H1 Feature Count")
        ax.set_title("Exp 4: Population-Level Binding")
        for bar, val in zip(bars, [exp4["joint_5agent_h1"], exp4["max_pairwise_h1"], exp4["disordered_joint_h1"]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=11)

        fig.suptitle("Branch 10: Multi-Agent Coordination — Topological Binding",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "overview.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'overview.png'}")
    except Exception as e:
        pprint(f"  Warning: overview figure failed: {e}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Branch 10: Multi-Agent Coordination Topological Analysis"
    )
    parser.add_argument("--subsample", type=int, default=200, help="PH subsample size")
    parser.add_argument("--n_surrogates", type=int, default=50, help="Surrogates for significance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    pprint("=" * 70)
    pprint("Branch 10: Multi-Agent Coordination — Topological Binding")
    pprint("=" * 70)

    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data" / "multiagent"
    fig_dir = base_dir / "figures" / "multiagent"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    exp1 = run_exp1(args, fig_dir)
    exp2 = run_exp2(args, fig_dir)
    exp3 = run_exp3(args, fig_dir)
    exp4 = run_exp4(args, fig_dir)

    pprint("\n=== Generating Overview Figure ===")
    make_overview_figure(exp1, exp2, exp3, exp4, fig_dir)

    pprint(f"\n{'=' * 70}")
    pprint("RESULTS SUMMARY")
    pprint(f"{'=' * 70}")
    pprint(f"  Exp 1 — Flocked binding: {exp1['flocked_binding']:.4f} (p={exp1['flocked_p']:.4f}, z={exp1['flocked_z']:.2f})")
    pprint(f"          Disordered binding: {exp1['disordered_binding']:.4f} (p={exp1['disordered_p']:.4f}, z={exp1['disordered_z']:.2f})")
    pprint(f"          Flocked > Disordered: {exp1['flocked_greater']}")
    pprint(f"  Exp 2 — Known K_c: {exp2['known_kc']:.3f}, Detected: {exp2['detected_kc']}")
    pprint(f"          K=0 z={exp2['sig_k0']['z_score']:.2f}, K=3 z={exp2['sig_k3']['z_score']:.2f}")
    pprint(f"  Exp 3 — Direct binding (mean): {exp3['mean_direct']:.4f}")
    pprint(f"          Indirect binding: {exp3['indirect_binding_13']:.4f}")
    pprint(f"          Direct > Indirect: {exp3['direct_greater_than_indirect']}")
    pprint(f"  Exp 4 — Joint 5-agent H1: {exp4['joint_5agent_h1']}")
    pprint(f"          Max pairwise H1: {exp4['max_pairwise_h1']}")
    pprint(f"          Emergent features: {exp4['emergent_features']}")

    total_time = time.time() - t_start
    pprint(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    results = to_serializable({
        "branch": "experiment/tda-multiagent",
        "config": {
            "subsample": args.subsample,
            "n_surrogates": args.n_surrogates,
            "seed": args.seed,
        },
        "exp1_flocked_binding": exp1["flocked_binding"],
        "exp1_flocked_p": exp1["flocked_p"],
        "exp1_disordered_binding": exp1["disordered_binding"],
        "exp1_disordered_p": exp1["disordered_p"],
        "exp1_noise_sweep": exp1["noise_sweep"],
        "exp2_kuramoto_sweep": {
            "coupling_values": exp2["coupling_values"],
            "binding_scores": exp2["binding_scores"],
            "order_params": exp2["order_params"],
        },
        "exp2_detected_kc": exp2["detected_kc"],
        "exp2_known_kc": exp2["known_kc"],
        "exp3_direct_binding": exp3["mean_direct"],
        "exp3_indirect_binding": exp3["indirect_binding_13"],
        "exp3_direct_greater": exp3["direct_greater_than_indirect"],
        "exp4_joint_5agent_h1": exp4["joint_5agent_h1"],
        "exp4_max_pairwise_h1": exp4["max_pairwise_h1"],
        "exp4_emergent_features": exp4["emergent_features"],
        "exp1_details": exp1,
        "exp2_details": exp2,
        "exp3_details": exp3,
        "exp4_details": exp4,
        "runtime_seconds": round(total_time, 1),
    })

    results_path = data_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    pprint(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
