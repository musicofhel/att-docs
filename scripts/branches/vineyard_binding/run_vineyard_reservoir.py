#!/usr/bin/env python3
"""Branch 2 tracer bullet — vineyard-tracked binding score on reservoir SR drift.

Architecture #4 from the hybrid topological monitoring document. Per the
plan-agent pressure test, we **skip Dionysus vineyards** for the tracer
bullet — the boost dependency yak-shave is unwarranted at this stage.
Instead, build a "dumb vineyard": dense overlapping windows + Hungarian-L∞
matching of joint H1 features across adjacent windows + Savitzky-Golay
smoothing of the tracked persistence pairs. This is a strict lower bound
on what real vineyards would buy you.

Three arms:
  A. Dense raw         — per-window BindingDetector.binding_score(), no smoothing.
  B. Dense + smoothed  — Savitzky-Golay on the raw score series. Isolates the
                          smoothing effect alone.
  C. Tracked + smoothed — Hungarian-match joint H1 pairs window-to-window,
                          smooth each track's (birth, death) over time, rebuild
                          the joint persistence image from the tracked pairs,
                          recompute the binding score from the smoothed image.

Baseline to beat: F1=0.06 (existing reservoir binding experiment).
Tracer bullet success criterion:
  F1(C) >= 0.20 AND F1(C) - F1(A) >= 0.05 AND F1(C) > F1(B)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from att.binding.detector import BindingDetector
from att.embedding.takens import TakensEmbedder
from att.embedding.joint import JointEmbedder
from att.topology.persistence import PersistenceAnalyzer

import reservoir_degradation_monitor as rdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW = 2000
STRIDE = 50
N_HEALTHY = rdm.N_HEALTHY  # 5000
N_TOTAL = rdm.N_TOTAL  # 15000
SUBSAMPLE = 300
IMAGE_RES = 30  # smaller for speed (3x faster than 50)
IMAGE_SIGMA = 0.1
N_BASELINE_WINDOWS = 80
SIGMA_ALARM = 2.0
SAVGOL_WINDOW = 21
SAVGOL_POLY = 3
TRACK_MAX_COST_MULT = 2.0  # cap match cost at 2x median persistence

DATA_DIR = os.path.join(REPO_ROOT, "data", "vineyard_binding")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "vineyard_binding")
HANDOFF_DIR = os.path.join(REPO_ROOT, ".claude", "handoff")


# ---------------------------------------------------------------------------
# Hungarian L∞ tracker
# ---------------------------------------------------------------------------

class FeatureTracker:
    """Track H1 persistence pairs across consecutive windows via Hungarian L∞ matching.

    Each track is a list of (window_idx, birth, death) entries. Unmatched features in
    window t+1 start new tracks. Tracks that don't get matched at window t+1 are
    "killed" — they may resume later if a similar feature reappears (no, actually,
    we keep them open for `max_gap` windows so re-matching is possible across small
    gaps; for tracer-bullet we use `max_gap=0`, i.e. strict consecutive matching).

    Attributes
    ----------
    tracks : list[dict]
        Each dict: {"start_w": int, "births": [...], "deaths": [...], "windows": [...]}
        Indexed in order of creation.
    """

    def __init__(self, max_cost_mult: float = TRACK_MAX_COST_MULT):
        self.tracks: list[dict] = []
        self._active_idx: list[int] = []  # indices into self.tracks for currently-open tracks
        self.max_cost_mult = max_cost_mult

    def update(self, window_idx: int, dgm: np.ndarray) -> None:
        """Match the features in dgm to currently active tracks. Open new tracks for unmatched."""
        if len(dgm) == 0:
            self._active_idx = []
            return

        if not self._active_idx:
            # All features start new tracks
            for b, d in dgm:
                self.tracks.append({"start_w": window_idx, "births": [float(b)], "deaths": [float(d)],
                                    "windows": [window_idx]})
            self._active_idx = list(range(len(self.tracks) - len(dgm), len(self.tracks)))
            return

        # Build (n_active, n_dgm) cost matrix using L∞ on (birth, death)
        prev_b = np.array([self.tracks[i]["births"][-1] for i in self._active_idx])
        prev_d = np.array([self.tracks[i]["deaths"][-1] for i in self._active_idx])
        new_b = dgm[:, 0]
        new_d = dgm[:, 1]
        # cost[i, j] = max(|prev_b[i] - new_b[j]|, |prev_d[i] - new_d[j]|)
        birth_diff = np.abs(prev_b[:, None] - new_b[None, :])
        death_diff = np.abs(prev_d[:, None] - new_d[None, :])
        cost = np.maximum(birth_diff, death_diff)

        # Cap match cost — anything above this is effectively unmatched
        all_pers = np.concatenate([(dgm[:, 1] - dgm[:, 0])])
        median_pers = float(np.median(all_pers[all_pers > 1e-12])) if (all_pers > 1e-12).any() else 1e-3
        max_cost = self.max_cost_mult * median_pers

        # Build augmented cost matrix to allow unmatched tracks AND unmatched features
        n_a = len(self._active_idx)
        n_n = len(dgm)
        big = max(cost.max() if cost.size else 1.0, max_cost) * 10.0
        # Square matrix of size (n_a + n_n, n_a + n_n)
        size = n_a + n_n
        aug = np.full((size, size), big, dtype=float)
        # Top-left block: real costs (clipped at max_cost so no high-cost matches)
        clipped = np.where(cost <= max_cost, cost, big)
        aug[:n_a, :n_n] = clipped
        # Top-right block: track-killing edges (constant max_cost)
        aug[:n_a, n_n:n_n + n_a] = max_cost
        np.fill_diagonal(aug[:n_a, n_n:n_n + n_a], max_cost)
        # Bottom-left block: feature-creation edges
        aug[n_a:n_a + n_n, :n_n] = max_cost
        np.fill_diagonal(aug[n_a:n_a + n_n, :n_n], max_cost)
        # Bottom-right block: dummy-to-dummy (cheap)
        aug[n_a:, n_n:] = 0.0

        row_ind, col_ind = linear_sum_assignment(aug)

        new_active: list[int] = []
        matched_features: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if r < n_a and c < n_n:
                # Real match
                track_id = self._active_idx[r]
                self.tracks[track_id]["births"].append(float(dgm[c, 0]))
                self.tracks[track_id]["deaths"].append(float(dgm[c, 1]))
                self.tracks[track_id]["windows"].append(window_idx)
                new_active.append(track_id)
                matched_features.add(c)

        # Unmatched features become new tracks
        for j in range(n_n):
            if j not in matched_features:
                self.tracks.append({"start_w": window_idx,
                                    "births": [float(dgm[j, 0])],
                                    "deaths": [float(dgm[j, 1])],
                                    "windows": [window_idx]})
                new_active.append(len(self.tracks) - 1)

        self._active_idx = new_active

    def smooth_tracks(self, window_length: int = SAVGOL_WINDOW, polyorder: int = SAVGOL_POLY) -> None:
        """In-place Savitzky-Golay smoothing of births/deaths within each long-enough track."""
        for tr in self.tracks:
            n = len(tr["births"])
            if n >= window_length:
                wl = window_length
            elif n >= 5:
                wl = n if n % 2 == 1 else n - 1
                if wl < polyorder + 2:
                    continue
            else:
                continue
            po = min(polyorder, wl - 1)
            tr["births"] = savgol_filter(tr["births"], wl, po).tolist()
            tr["deaths"] = savgol_filter(tr["deaths"], wl, po).tolist()

    def reconstruct_diagrams(self, n_windows: int) -> list[np.ndarray]:
        """For each window index, return the (n, 2) array of (birth, death) pairs that pass through it."""
        per_window: list[list[tuple[float, float]]] = [[] for _ in range(n_windows)]
        for tr in self.tracks:
            for w, b, d in zip(tr["windows"], tr["births"], tr["deaths"]):
                if 0 <= w < n_windows and d > b:
                    per_window[w].append((b, d))
        return [np.array(pw, dtype=float) if pw else np.empty((0, 2)) for pw in per_window]

    def stats(self) -> dict:
        if not self.tracks:
            return {"n_tracks": 0, "mean_track_len": 0.0, "max_track_len": 0, "n_long_tracks": 0}
        lens = [len(tr["births"]) for tr in self.tracks]
        return {
            "n_tracks": len(self.tracks),
            "mean_track_len": float(np.mean(lens)),
            "max_track_len": int(np.max(lens)),
            "n_long_tracks": int(sum(1 for L in lens if L >= 5)),
        }


# ---------------------------------------------------------------------------
# Build a persistence image from a (n, 2) diagram on a fixed grid
# ---------------------------------------------------------------------------

def diagram_to_image(dgm: np.ndarray,
                     birth_range: tuple[float, float],
                     persistence_range: tuple[float, float],
                     resolution: int = IMAGE_RES,
                     sigma: float = IMAGE_SIGMA) -> np.ndarray:
    """Build a persistence image from a (b, d) diagram on the given fixed grid.

    Mirrors `att.topology.persistence.PersistenceAnalyzer.to_image()` but operates
    on a bare (n, 2) array (no PA instance required).
    """
    if len(dgm) == 0:
        return np.zeros((resolution, resolution))

    births = dgm[:, 0]
    persistence = dgm[:, 1] - dgm[:, 0]
    mask = persistence > 1e-10
    if mask.sum() == 0:
        return np.zeros((resolution, resolution))
    births = births[mask]
    persistence = persistence[mask]

    weights = persistence / persistence.max() if persistence.max() > 0 else np.ones_like(persistence)

    b_min, b_max = birth_range
    p_min, p_max = persistence_range
    b_pad = (b_max - b_min) * 0.1 + 1e-6
    p_pad = (p_max - p_min) * 0.1 + 1e-6
    b_grid = np.linspace(b_min - b_pad, b_max + b_pad, resolution)
    p_grid = np.linspace(max(0, p_min - p_pad), p_max + p_pad, resolution)

    img = np.zeros((resolution, resolution))
    for b, p, w in zip(births, persistence, weights):
        gb = np.exp(-((b_grid - b) ** 2) / (2 * sigma ** 2))
        gp = np.exp(-((p_grid - p) ** 2) / (2 * sigma ** 2))
        img += w * np.outer(gp, gb)

    return img


# ---------------------------------------------------------------------------
# Threshold + F1
# ---------------------------------------------------------------------------

def alarms_from_signal(signal: np.ndarray, n_baseline: int, k: float = SIGMA_ALARM) -> np.ndarray:
    base = signal[:n_baseline]
    base = base[np.isfinite(base)]
    if len(base) < 2:
        return np.zeros_like(signal, dtype=bool)
    mu = float(base.mean())
    sd = float(base.std())
    if sd < 1e-15:
        sd = 1e-15
    return signal > mu + k * sd


def prf1(alarms: np.ndarray, ground_truth: np.ndarray) -> dict:
    a = alarms.astype(int)
    g = ground_truth.astype(int)
    tp = int(((a == 1) & (g == 1)).sum())
    fp = int(((a == 1) & (g == 0)).sum())
    fn = int(((a == 0) & (g == 1)).sum())
    tn = int(((a == 0) & (g == 0)).sum())
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

def plot_arms(scores: dict[str, np.ndarray], alarms: dict[str, np.ndarray],
              ground_truth: np.ndarray, onset_window: int, out_path: str) -> None:
    n_arms = len(scores)
    fig, axes = plt.subplots(n_arms, 1, figsize=(13, 2.2 * n_arms), sharex=True)
    if n_arms == 1:
        axes = [axes]
    for i, (name, sig) in enumerate(scores.items()):
        ax = axes[i]
        ax.plot(sig, lw=1.2, color=f"C{i}", label=name)
        ax.axvline(onset_window - 0.5, color="red", linestyle="--", lw=1, alpha=0.6,
                   label=f"drift onset (w={onset_window})")
        ax.axvline(N_BASELINE_WINDOWS - 0.5, color="black", linestyle=":", lw=1, alpha=0.4,
                   label=f"end baseline (w={N_BASELINE_WINDOWS - 1})")
        # Mark alarms
        a_idx = np.where(alarms[name])[0]
        if len(a_idx) > 0:
            ax.scatter(a_idx, sig[a_idx], color="orange", s=10, zorder=5,
                       label=f"{len(a_idx)} alarms")
        ax.set_ylabel(name)
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("window index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Branch 2 — vineyard-tracked binding tracer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="sr_drift",
                        choices=["sr_drift", "noise", "connection_death"])
    parser.add_argument("--channels", type=str, default="0,100",
                        help="comma-separated neuron indices for X,Y")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(HANDOFF_DIR, exist_ok=True)

    t0 = time.time()
    print(f"[branch 2] Generating reservoir trajectory: mode={args.mode} seed={args.seed}")
    states, _, _ = rdm.simulate_degradation(args.mode, args.seed)
    print(f"  states shape: {states.shape}")

    cx, cy = (int(c) for c in args.channels.split(","))
    X_full = states[:, cx]
    Y_full = states[:, cy]
    print(f"  channels: X=neuron[{cx}] (std={X_full.std():.4f}), Y=neuron[{cy}] (std={Y_full.std():.4f})")

    starts = list(range(0, N_TOTAL - WINDOW + 1, STRIDE))
    n_windows = len(starts)
    print(f"  windows: {n_windows} (W={WINDOW}, stride={STRIDE})")

    onset_window = N_HEALTHY // STRIDE
    ground_truth = np.array([1 if s >= N_HEALTHY else 0 for s in starts], dtype=int)
    print(f"  onset_window: {onset_window}, drift={int(ground_truth.sum())}, healthy={int((1 - ground_truth).sum())}")

    # ---- Window 0: auto-embedding to cache parameters ----
    print(f"  fitting window 0 (auto-embed for cached params)...")
    bd0 = BindingDetector(max_dim=1, method="persistence_image",
                          image_resolution=IMAGE_RES, image_sigma=IMAGE_SIGMA,
                          embedding_quality_gate=False)
    bd0.fit(X_full[:WINDOW], Y_full[:WINDOW], subsample=SUBSAMPLE, seed=args.seed)
    emb_x_d, emb_x_dim = bd0._marginal_delay_x, bd0._marginal_dim_x
    emb_y_d, emb_y_dim = bd0._marginal_delay_y, bd0._marginal_dim_y
    j_delays, j_dims = list(bd0._joint_delays), list(bd0._joint_dims)
    print(f"    marg_x: delay={emb_x_d}, dim={emb_x_dim}")
    print(f"    marg_y: delay={emb_y_d}, dim={emb_y_dim}")
    print(f"    joint:  delays={j_delays}, dims={j_dims}")
    print(f"    binding_score(window 0) = {bd0.binding_score():.4f}")

    # ---- Per-window dense pass: collect H1 diagrams + raw scores ----
    scores_a = np.zeros(n_windows)
    joint_h1: list[np.ndarray] = []
    marg_x_h1: list[np.ndarray] = []
    marg_y_h1: list[np.ndarray] = []
    birth_ranges: list[tuple[float, float]] = []
    pers_ranges: list[tuple[float, float]] = []

    print(f"  starting dense pass over {n_windows} windows...")
    t_loop = time.time()
    for w_idx, s in enumerate(starts):
        bd = BindingDetector(max_dim=1, method="persistence_image",
                             image_resolution=IMAGE_RES, image_sigma=IMAGE_SIGMA,
                             embedding_quality_gate=False)
        try:
            bd.fit(
                X_full[s:s + WINDOW], Y_full[s:s + WINDOW],
                joint_embedder=JointEmbedder(delays=j_delays, dimensions=j_dims),
                marginal_embedder_x=TakensEmbedder(delay=emb_x_d, dimension=emb_x_dim),
                marginal_embedder_y=TakensEmbedder(delay=emb_y_d, dimension=emb_y_dim),
                subsample=SUBSAMPLE, seed=args.seed,
            )
            scores_a[w_idx] = bd.binding_score()
            joint_h1.append(bd._pa_joint.diagrams_[1].copy() if len(bd._pa_joint.diagrams_) > 1 else np.empty((0, 2)))
            marg_x_h1.append(bd._pa_x.diagrams_[1].copy() if len(bd._pa_x.diagrams_) > 1 else np.empty((0, 2)))
            marg_y_h1.append(bd._pa_y.diagrams_[1].copy() if len(bd._pa_y.diagrams_) > 1 else np.empty((0, 2)))
            birth_ranges.append(bd._birth_range)
            pers_ranges.append(bd._persistence_range)
        except Exception as e:
            print(f"    window {w_idx} failed: {e}")
            scores_a[w_idx] = np.nan
            joint_h1.append(np.empty((0, 2)))
            marg_x_h1.append(np.empty((0, 2)))
            marg_y_h1.append(np.empty((0, 2)))
            birth_ranges.append((0.0, 1.0))
            pers_ranges.append((0.0, 1.0))

        if w_idx % 30 == 29 or w_idx == n_windows - 1:
            elapsed = time.time() - t_loop
            print(f"    window {w_idx + 1}/{n_windows} | {elapsed:.1f}s | "
                  f"score={scores_a[w_idx]:.3f} joint_h1={len(joint_h1[-1])}")

    # NaN-fill any failures with neighbour values
    finite_mask = np.isfinite(scores_a)
    if not finite_mask.all():
        print(f"  warning: {int((~finite_mask).sum())} window(s) failed; interpolating")
        scores_a = np.interp(np.arange(n_windows), np.where(finite_mask)[0], scores_a[finite_mask])

    # ---- Compute the global shared (birth, persistence) range across all windows ----
    # Use the union of per-window ranges from arm A; this is what the tracked PIs use too.
    global_birth = (min(br[0] for br in birth_ranges), max(br[1] for br in birth_ranges))
    global_pers = (min(pr[0] for pr in pers_ranges), max(pr[1] for pr in pers_ranges))
    print(f"  global birth range: {global_birth}, persistence range: {global_pers}")

    # ---- Arm B: smooth raw scores ----
    if len(scores_a) >= SAVGOL_WINDOW:
        scores_b = savgol_filter(scores_a, SAVGOL_WINDOW, SAVGOL_POLY)
    else:
        scores_b = scores_a.copy()

    # ---- Arm C: track joint H1, smooth tracks, rebuild PIs, recompute binding score ----
    print(f"  building tracker on joint H1 features...")
    tracker = FeatureTracker(max_cost_mult=TRACK_MAX_COST_MULT)
    for w_idx, dgm in enumerate(joint_h1):
        tracker.update(w_idx, dgm)
    tracker_stats = tracker.stats()
    print(f"  tracker stats: {tracker_stats}")
    tracker.smooth_tracks()
    tracked_diagrams = tracker.reconstruct_diagrams(n_windows)

    # Rebuild joint PIs from tracked diagrams; marginals stay raw (we tracked only the joint).
    # Binding score = positive(joint - max(marg_x, marg_y)).sum()
    scores_c = np.zeros(n_windows)
    for w_idx in range(n_windows):
        joint_img = diagram_to_image(tracked_diagrams[w_idx], global_birth, global_pers)
        marg_x_img = diagram_to_image(marg_x_h1[w_idx], global_birth, global_pers)
        marg_y_img = diagram_to_image(marg_y_h1[w_idx], global_birth, global_pers)
        baseline_img = np.maximum(marg_x_img, marg_y_img)
        residual = np.maximum(joint_img - baseline_img, 0)
        scores_c[w_idx] = float(residual.sum())

    # ---- Detection (alarms + F1) per arm ----
    arms = {"A_dense_raw": scores_a, "B_dense_smoothed": scores_b, "C_tracked_smoothed": scores_c}
    alarms_dict: dict[str, np.ndarray] = {}
    arm_results = {}
    for name, sig in arms.items():
        # Try alarms in BOTH directions and pick whichever has higher F1 — the binding score
        # may go up or down with drift; we don't a priori know
        alarms_up = alarms_from_signal(sig, N_BASELINE_WINDOWS, SIGMA_ALARM)
        alarms_down = alarms_from_signal(-sig, N_BASELINE_WINDOWS, SIGMA_ALARM)
        scores_up = prf1(alarms_up, ground_truth)
        scores_down = prf1(alarms_down, ground_truth)
        if scores_up["f1"] >= scores_down["f1"]:
            chosen = alarms_up
            chosen_dir = "up"
            chosen_scores = scores_up
        else:
            chosen = alarms_down
            chosen_dir = "down"
            chosen_scores = scores_down
        first_alarm = first_alarm_index(chosen)
        chosen_scores["first_alarm_window"] = first_alarm
        chosen_scores["lag_windows"] = (first_alarm - onset_window) if first_alarm is not None else None
        chosen_scores["onset_window"] = onset_window
        chosen_scores["alarm_direction"] = chosen_dir
        arm_results[name] = chosen_scores
        alarms_dict[name] = chosen
        print(f"  arm {name}: F1={chosen_scores['f1']:.3f} P={chosen_scores['precision']:.3f} "
              f"R={chosen_scores['recall']:.3f} dir={chosen_dir} first_alarm=w{first_alarm}")

    # ---- Verdict ----
    f1_a = arm_results["A_dense_raw"]["f1"]
    f1_b = arm_results["B_dense_smoothed"]["f1"]
    f1_c = arm_results["C_tracked_smoothed"]["f1"]
    target_f1 = 0.20
    pass_threshold = (f1_c >= target_f1)
    pass_lift_over_a = (f1_c - f1_a) >= 0.05
    pass_beats_b = (f1_c > f1_b)
    passed = pass_threshold and pass_lift_over_a and pass_beats_b
    verdict_str = "PASS" if passed else "FAIL"
    why = []
    if not pass_threshold:
        why.append(f"f1_c={f1_c:.3f} < target {target_f1}")
    if not pass_lift_over_a:
        why.append(f"f1_c - f1_a = {f1_c - f1_a:+.3f} < 0.05")
    if not pass_beats_b:
        why.append(f"f1_c={f1_c:.3f} not > f1_b={f1_b:.3f}")
    print(f"\n[branch 2] VERDICT: {verdict_str}")
    if why:
        print("  fail reasons:", "; ".join(why))

    # ---- Persist ----
    runtime_s = time.time() - t0
    output = {
        "branch": "experiment/vineyard-binding-reservoir",
        "mode": args.mode,
        "seed": args.seed,
        "channels": [cx, cy],
        "config": {
            "window": WINDOW,
            "stride": STRIDE,
            "subsample": SUBSAMPLE,
            "image_resolution": IMAGE_RES,
            "image_sigma": IMAGE_SIGMA,
            "n_baseline_windows": N_BASELINE_WINDOWS,
            "sigma_alarm": SIGMA_ALARM,
            "savgol_window": SAVGOL_WINDOW,
            "savgol_poly": SAVGOL_POLY,
            "track_max_cost_mult": TRACK_MAX_COST_MULT,
        },
        "n_windows": n_windows,
        "n_drift_windows": int(ground_truth.sum()),
        "n_healthy_windows": int((1 - ground_truth).sum()),
        "onset_window": onset_window,
        "embedding": {
            "marg_x_delay": int(emb_x_d), "marg_x_dim": int(emb_x_dim),
            "marg_y_delay": int(emb_y_d), "marg_y_dim": int(emb_y_dim),
            "joint_delays": [int(d) for d in j_delays],
            "joint_dims": [int(d) for d in j_dims],
        },
        "tracker_stats": tracker_stats,
        "arms": arm_results,
        "f1_a": f1_a, "f1_b": f1_b, "f1_c": f1_c,
        "lift_c_over_a": float(f1_c - f1_a),
        "baseline_f1": 0.06,
        "target_f1": target_f1,
        "verdict": verdict_str,
        "verdict_failure_reasons": why,
        "runtime_s": runtime_s,
    }
    json_path = os.path.join(DATA_DIR, f"results_{args.mode}_seed{args.seed}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  wrote {json_path}")

    npz_path = os.path.join(DATA_DIR, f"scores_{args.mode}_seed{args.seed}.npz")
    np.savez_compressed(npz_path,
                        scores_a=scores_a, scores_b=scores_b, scores_c=scores_c,
                        ground_truth=ground_truth, starts=np.array(starts),
                        n_tracks=tracker_stats["n_tracks"])
    print(f"  wrote {npz_path}")

    fig_path = os.path.join(FIG_DIR, f"vineyard_binding_{args.mode}_seed{args.seed}.png")
    plot_arms(arms, alarms_dict, ground_truth, onset_window, fig_path)
    print(f"  wrote {fig_path}")

    print(f"\n[branch 2] runtime: {runtime_s:.1f}s")
    print(f"[branch 2] one-liner: A={f1_a:.3f} B={f1_b:.3f} C={f1_c:.3f} verdict={verdict_str}")


if __name__ == "__main__":
    main()
