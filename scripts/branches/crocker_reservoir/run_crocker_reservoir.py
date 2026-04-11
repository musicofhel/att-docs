#!/usr/bin/env python3
"""Branch 3 tracer bullet — Crocker stack process control on reservoir SR drift.

Architecture #10 from the hybrid topological monitoring document. Build a
Crocker matrix (Betti curve per window stacked into a 2D heatmap) over a
sliding window of an ESN under spectral-radius drift, then apply three
detection arms:

  A. Column-wise L1 to baseline   — drift detection in feature-vector space.
  B. Image CUSUM                  — pixel-wise CUSUM, the document's framing.
  C. Total Betti mass per column  — dumb scalar reduction (sanity baseline).

Baseline to beat: F1 = 0.06 (per-window PH binding score, existing reservoir
monitoring experiment). Tracer bullet success criterion: best arm F1 >= 0.20.

This script does NOT depend on att.llm.crocker (which only exists on the
TDA-LLM branches). It builds the Crocker stack inline using
PersistenceAnalyzer + a shared-grid Betti curve computation.

Usage:
    python scripts/branches/crocker_reservoir/run_crocker_reservoir.py
    python scripts/branches/crocker_reservoir/run_crocker_reservoir.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer

import reservoir_degradation_monitor as rdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW = 2000
STRIDE = 200
N_HEALTHY = rdm.N_HEALTHY  # 5000
N_TOTAL = rdm.N_TOTAL  # 15000
N_FILTRATION_STEPS = 100
SUBSAMPLE = 300
MAX_DIM = 1
SIGMA_ALARM = 2.0  # n-sigma vs first-25-window baseline
N_BASELINE_WINDOWS = 25  # ground-truth healthy region in window-index space

DATA_DIR = os.path.join(REPO_ROOT, "data", "crocker_reservoir")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "crocker_reservoir")
HANDOFF_DIR = os.path.join(REPO_ROOT, ".claude", "handoff")


# ---------------------------------------------------------------------------
# PCA -> 1D
# ---------------------------------------------------------------------------

def pc1(states: np.ndarray) -> np.ndarray:
    """First PCA component of (n_steps, n_neurons) reservoir states."""
    Xc = states - states.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[0]


# ---------------------------------------------------------------------------
# Per-window persistence diagrams
# ---------------------------------------------------------------------------

def window_diagrams(signal_window: np.ndarray, seed: int = 42) -> list[np.ndarray]:
    """Takens-embed a 1D window, run ripser, return [H0_dgm, H1_dgm]."""
    if np.std(signal_window) < 1e-12:
        return [np.empty((0, 2)) for _ in range(MAX_DIM + 1)]
    try:
        emb = TakensEmbedder(delay="auto", dimension="auto")
        emb.fit(signal_window)
    except Exception:
        emb = TakensEmbedder(delay=10, dimension=3)
        emb.fit(signal_window)
    cloud = emb.transform(signal_window)
    pa = PersistenceAnalyzer(max_dim=MAX_DIM, backend="ripser")
    pa.fit_transform(cloud, subsample=SUBSAMPLE, seed=seed)
    return [pa.diagrams_[d] if d < len(pa.diagrams_) else np.empty((0, 2))
            for d in range(MAX_DIM + 1)]


# ---------------------------------------------------------------------------
# Build Crocker matrix on shared filtration grid
# ---------------------------------------------------------------------------

def build_crocker(diagrams_list: list[list[np.ndarray]],
                  n_steps: int = N_FILTRATION_STEPS) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Stack diagrams into per-dimension Betti matrices on a shared grid.

    Parameters
    ----------
    diagrams_list : list of list of np.ndarray
        Outer index = window, inner index = homology dim, value = (n_features, 2) (b, d) array.
    n_steps : int
        Filtration grid resolution.

    Returns
    -------
    matrices : list of (n_steps, n_windows) arrays, one per homology dim.
    grids    : list of (n_steps,) filtration parameter grids, one per homology dim.
    """
    n_windows = len(diagrams_list)
    n_dims = MAX_DIM + 1
    matrices = []
    grids = []
    for d in range(n_dims):
        all_b, all_dth = [], []
        for ws in diagrams_list:
            dgm = ws[d]
            if len(dgm) > 0:
                all_b.extend(dgm[:, 0].tolist())
                all_dth.extend(dgm[:, 1].tolist())
        if not all_b:
            grids.append(np.linspace(0.0, 1.0, n_steps))
            matrices.append(np.zeros((n_steps, n_windows)))
            continue
        eps_min = float(min(all_b))
        eps_max = float(max(all_dth))
        if eps_max <= eps_min:
            eps_max = eps_min + 1.0
        grid = np.linspace(eps_min, eps_max, n_steps)
        mat = np.zeros((n_steps, n_windows))
        for w_idx, ws in enumerate(diagrams_list):
            dgm = ws[d]
            if len(dgm) == 0:
                continue
            for b, dth in dgm:
                mat[:, w_idx] += (grid >= b) & (grid < dth)
        grids.append(grid)
        matrices.append(mat)
    return matrices, grids


# ---------------------------------------------------------------------------
# Detection arms
# ---------------------------------------------------------------------------

def arm_column_l1(crocker: np.ndarray, n_baseline: int) -> np.ndarray:
    baseline_col = crocker[:, :n_baseline].mean(axis=1)
    return np.linalg.norm(crocker - baseline_col[:, None], ord=1, axis=0)


def arm_image_cusum(crocker: np.ndarray, n_baseline: int) -> np.ndarray:
    baseline_col = crocker[:, :n_baseline].mean(axis=1)
    diff = np.abs(crocker - baseline_col[:, None]).sum(axis=0)
    # Reset accumulator at start of streaming (i.e. cumulative sum of per-step deviation)
    return np.cumsum(diff)


def arm_total_betti(crocker: np.ndarray, n_baseline: int) -> np.ndarray:
    return crocker.sum(axis=0)


# ---------------------------------------------------------------------------
# Threshold + F1
# ---------------------------------------------------------------------------

def alarms_from_signal(signal: np.ndarray, n_baseline: int, k: float = SIGMA_ALARM) -> np.ndarray:
    base = signal[:n_baseline]
    mu = float(base.mean())
    sd = float(base.std())
    if sd < 1e-15:
        sd = 1e-15
    return signal > mu + k * sd


def prf1(alarms: np.ndarray, ground_truth: np.ndarray) -> dict:
    tp = int(((alarms == 1) & (ground_truth == 1)).sum())
    fp = int(((alarms == 1) & (ground_truth == 0)).sum())
    fn = int(((alarms == 0) & (ground_truth == 1)).sum())
    tn = int(((alarms == 0) & (ground_truth == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def first_alarm_index(alarms: np.ndarray) -> int | None:
    idx = np.where(alarms)[0]
    return int(idx[0]) if len(idx) > 0 else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(crocker_h1: np.ndarray, grid_h1: np.ndarray,
                 arm_signals: dict[str, np.ndarray],
                 alarm_indices: dict[str, int | None],
                 ground_truth: np.ndarray,
                 out_path: str) -> None:
    n_windows = crocker_h1.shape[1]
    n_arms = len(arm_signals)

    fig = plt.figure(figsize=(13, 4 + 2 * n_arms))
    gs = fig.add_gridspec(n_arms + 1, 1, height_ratios=[2.5] + [1] * n_arms, hspace=0.45)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(crocker_h1, aspect="auto", origin="lower",
                    extent=[0, n_windows - 1, grid_h1[0], grid_h1[-1]],
                    cmap="viridis")
    ax0.axvline(N_BASELINE_WINDOWS - 0.5, color="white", linestyle=":", linewidth=1.5,
                label=f"end of baseline (w={N_BASELINE_WINDOWS - 1})")
    onset_window = N_HEALTHY // STRIDE
    ax0.axvline(onset_window - 0.5, color="red", linestyle="--", linewidth=1.5,
                label=f"drift onset (w={onset_window})")
    ax0.set_ylabel("filtration ε")
    ax0.set_title("H1 Crocker stack — ESN spectral-radius drift")
    ax0.legend(loc="upper left", fontsize=8)
    plt.colorbar(im, ax=ax0, fraction=0.025, label="β₁(ε)")

    for i, (name, sig) in enumerate(arm_signals.items()):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.plot(sig, lw=1.5, color=f"C{i}", label=name)
        ax.axvline(onset_window - 0.5, color="red", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(N_BASELINE_WINDOWS - 0.5, color="black", linestyle=":", linewidth=1, alpha=0.4)
        ai = alarm_indices.get(name)
        if ai is not None:
            ax.axvline(ai, color="orange", linestyle="-", linewidth=1.2,
                       label=f"first alarm w={ai}")
        ax.set_xlim(0, n_windows - 1)
        ax.set_ylabel(name)
        ax.legend(loc="upper left", fontsize=8)
        if i == n_arms - 1:
            ax.set_xlabel("window index")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Branch 3 — Crocker stack reservoir tracer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="sr_drift",
                        choices=["sr_drift", "noise", "connection_death"])
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(HANDOFF_DIR, exist_ok=True)

    t0 = time.time()
    print(f"[branch 3] Generating reservoir trajectory: mode={args.mode} seed={args.seed}")
    states, inputs, _ = rdm.simulate_degradation(args.mode, args.seed)
    print(f"  states shape: {states.shape}, dtype: {states.dtype}")
    signal = pc1(states)
    print(f"  PC1 shape: {signal.shape}")

    starts = list(range(0, N_TOTAL - WINDOW + 1, STRIDE))
    n_windows = len(starts)
    print(f"  windows: {n_windows} (window={WINDOW}, stride={STRIDE})")

    # Per-window persistence diagrams
    diagrams_list: list[list[np.ndarray]] = []
    t_per_window_acc = 0.0
    for w_idx, s in enumerate(starts):
        t_w = time.time()
        dgms = window_diagrams(signal[s:s + WINDOW], seed=args.seed + w_idx)
        diagrams_list.append(dgms)
        dt = time.time() - t_w
        t_per_window_acc += dt
        if w_idx == 0 or w_idx % 10 == 9 or w_idx == n_windows - 1:
            print(f"    window {w_idx + 1}/{n_windows} (start={s}) "
                  f"H0={len(dgms[0])} H1={len(dgms[1])} {dt:.2f}s")

    # Build Crocker stack on shared grid
    matrices, grids = build_crocker(diagrams_list, n_steps=N_FILTRATION_STEPS)
    crocker_h0, crocker_h1 = matrices
    grid_h0, grid_h1 = grids
    print(f"  crocker_h0 shape: {crocker_h0.shape}, h1 shape: {crocker_h1.shape}")
    print(f"  H0 grid range: [{grid_h0[0]:.4f}, {grid_h0[-1]:.4f}]")
    print(f"  H1 grid range: [{grid_h1[0]:.4f}, {grid_h1[-1]:.4f}]")

    # Ground truth: window i is "drift" iff start[i] >= N_HEALTHY
    ground_truth = np.array([1 if s >= N_HEALTHY else 0 for s in starts], dtype=int)
    print(f"  ground truth: {int(ground_truth.sum())} drift / {int((1 - ground_truth).sum())} healthy")

    # Three arms on H1 (where the action is)
    arm_signals = {
        "A_column_L1": arm_column_l1(crocker_h1, N_BASELINE_WINDOWS),
        "B_image_CUSUM": arm_image_cusum(crocker_h1, N_BASELINE_WINDOWS),
        "C_total_betti": arm_total_betti(crocker_h1, N_BASELINE_WINDOWS),
    }

    arm_results = {}
    alarm_indices = {}
    for name, sig in arm_signals.items():
        alarms = alarms_from_signal(sig, N_BASELINE_WINDOWS, SIGMA_ALARM)
        first_alarm = first_alarm_index(alarms)
        scores = prf1(alarms.astype(int), ground_truth)
        scores["first_alarm_window"] = first_alarm
        scores["onset_window"] = int(N_HEALTHY // STRIDE)
        scores["lag_windows"] = (first_alarm - scores["onset_window"]) if first_alarm is not None else None
        arm_results[name] = scores
        alarm_indices[name] = first_alarm
        print(f"  arm {name}: F1={scores['f1']:.3f} P={scores['precision']:.3f} R={scores['recall']:.3f} "
              f"first_alarm=w{first_alarm} (onset=w{scores['onset_window']})")

    # Verdict
    best_arm = max(arm_results.items(), key=lambda kv: kv[1]["f1"])
    best_name, best_scores = best_arm
    target_f1 = 0.20
    baseline_f1 = 0.06
    passed = best_scores["f1"] >= target_f1
    verdict_str = "PASS" if passed else "FAIL"
    print(f"\n[branch 3] VERDICT: {verdict_str} — "
          f"best arm {best_name} F1={best_scores['f1']:.3f} "
          f"(target≥{target_f1}, baseline={baseline_f1})")

    # Persist
    runtime_s = time.time() - t0
    output = {
        "branch": "experiment/crocker-stack-reservoir",
        "mode": args.mode,
        "seed": args.seed,
        "config": {
            "window": WINDOW,
            "stride": STRIDE,
            "n_filtration_steps": N_FILTRATION_STEPS,
            "subsample": SUBSAMPLE,
            "max_dim": MAX_DIM,
            "n_baseline_windows": N_BASELINE_WINDOWS,
            "sigma_alarm": SIGMA_ALARM,
        },
        "n_windows": n_windows,
        "n_drift_windows": int(ground_truth.sum()),
        "n_healthy_windows": int((1 - ground_truth).sum()),
        "onset_window": int(N_HEALTHY // STRIDE),
        "arms": arm_results,
        "best_arm": best_name,
        "best_f1": best_scores["f1"],
        "baseline_f1": baseline_f1,
        "target_f1": target_f1,
        "verdict": verdict_str,
        "runtime_s": runtime_s,
        "per_window_avg_s": t_per_window_acc / max(n_windows, 1),
    }
    json_path = os.path.join(DATA_DIR, f"results_{args.mode}_seed{args.seed}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  wrote {json_path}")

    npz_path = os.path.join(DATA_DIR, f"crocker_h0_h1_{args.mode}_seed{args.seed}.npz")
    np.savez_compressed(npz_path,
                        crocker_h0=crocker_h0, crocker_h1=crocker_h1,
                        grid_h0=grid_h0, grid_h1=grid_h1,
                        ground_truth=ground_truth,
                        starts=np.array(starts))
    print(f"  wrote {npz_path}")

    fig_path = os.path.join(FIG_DIR, f"crocker_stack_{args.mode}_seed{args.seed}.png")
    plot_results(crocker_h1, grid_h1, arm_signals, alarm_indices, ground_truth, fig_path)
    print(f"  wrote {fig_path}")

    print(f"\n[branch 3] runtime: {runtime_s:.1f}s ({t_per_window_acc / max(n_windows, 1):.2f}s/window avg)")
    print(f"[branch 3] one-liner: best={best_name} F1={best_scores['f1']:.3f} verdict={verdict_str}")


if __name__ == "__main__":
    main()
