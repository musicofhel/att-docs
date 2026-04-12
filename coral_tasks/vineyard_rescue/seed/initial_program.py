"""Vineyard rescue — seed program.

Implements two binding detection arms on sr_drift (seed=42):
  - Arm A (dense raw binding):      baseline F1 = 0.929
  - Arm C (Hungarian-tracked + smooth): F1 = 0.898

The seed reports arm C's F1 as the score. YOUR JOB: beat arm A's F1 = 0.929
using a tracking-based approach. See task.yaml for suggested strategies.

Approach history:
  - Hungarian L-inf tracker with augmented cost matrix (arm C) — F1=0.898
    → tracking HURTS vs dense raw (arm A=0.929)
  - Try: greedy matching, vineyard approximation, modified Hungarian, hybrid

References (read-only, in reference/):
  - reference/run_vineyard_reservoir.py — full 3-arm tracer
  - reference/run_crocker_reservoir.py — Crocker stack (F1=0.988 on sr_drift)
"""

from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from att.binding.detector import BindingDetector
from att.embedding.takens import TakensEmbedder
from att.embedding.joint import JointEmbedder

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
IMAGE_RES = 30
IMAGE_SIGMA = 0.1
N_BASELINE_WINDOWS = 80
SIGMA_ALARM = 2.0
SAVGOL_WINDOW = 21
SAVGOL_POLY = 3
TRACK_MAX_COST_MULT = 2.0

SEED = 42
MODE = "sr_drift"
CHANNELS = (0, 100)


# ---------------------------------------------------------------------------
# Hungarian L-inf tracker (from Branch 2 tracer)
# ---------------------------------------------------------------------------

class FeatureTracker:
    """Track H1 persistence pairs across consecutive windows via Hungarian L-inf."""

    def __init__(self, max_cost_mult: float = TRACK_MAX_COST_MULT):
        self.tracks: list[dict] = []
        self._active_idx: list[int] = []
        self.max_cost_mult = max_cost_mult

    def update(self, window_idx: int, dgm: np.ndarray) -> None:
        if len(dgm) == 0:
            self._active_idx = []
            return

        if not self._active_idx:
            for b, d in dgm:
                self.tracks.append({
                    "start_w": window_idx,
                    "births": [float(b)],
                    "deaths": [float(d)],
                    "windows": [window_idx],
                })
            self._active_idx = list(range(len(self.tracks) - len(dgm), len(self.tracks)))
            return

        prev_b = np.array([self.tracks[i]["births"][-1] for i in self._active_idx])
        prev_d = np.array([self.tracks[i]["deaths"][-1] for i in self._active_idx])
        new_b = dgm[:, 0]
        new_d = dgm[:, 1]
        birth_diff = np.abs(prev_b[:, None] - new_b[None, :])
        death_diff = np.abs(prev_d[:, None] - new_d[None, :])
        cost = np.maximum(birth_diff, death_diff)

        all_pers = dgm[:, 1] - dgm[:, 0]
        median_pers = float(np.median(all_pers[all_pers > 1e-12])) if (all_pers > 1e-12).any() else 1e-3
        max_cost = self.max_cost_mult * median_pers

        n_a = len(self._active_idx)
        n_n = len(dgm)
        big = max(cost.max() if cost.size else 1.0, max_cost) * 10.0
        size = n_a + n_n
        aug = np.full((size, size), big, dtype=float)
        clipped = np.where(cost <= max_cost, cost, big)
        aug[:n_a, :n_n] = clipped
        aug[:n_a, n_n:n_n + n_a] = max_cost
        np.fill_diagonal(aug[:n_a, n_n:n_n + n_a], max_cost)
        aug[n_a:n_a + n_n, :n_n] = max_cost
        np.fill_diagonal(aug[n_a:n_a + n_n, :n_n], max_cost)
        aug[n_a:, n_n:] = 0.0

        row_ind, col_ind = linear_sum_assignment(aug)

        new_active: list[int] = []
        matched_features: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if r < n_a and c < n_n:
                track_id = self._active_idx[r]
                self.tracks[track_id]["births"].append(float(dgm[c, 0]))
                self.tracks[track_id]["deaths"].append(float(dgm[c, 1]))
                self.tracks[track_id]["windows"].append(window_idx)
                new_active.append(track_id)
                matched_features.add(c)

        for j in range(n_n):
            if j not in matched_features:
                self.tracks.append({
                    "start_w": window_idx,
                    "births": [float(dgm[j, 0])],
                    "deaths": [float(dgm[j, 1])],
                    "windows": [window_idx],
                })
                new_active.append(len(self.tracks) - 1)

        self._active_idx = new_active

    def smooth_tracks(self, window_length: int = SAVGOL_WINDOW, polyorder: int = SAVGOL_POLY) -> None:
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
        per_window: list[list[tuple[float, float]]] = [[] for _ in range(n_windows)]
        for tr in self.tracks:
            for w, b, d in zip(tr["windows"], tr["births"], tr["deaths"]):
                if 0 <= w < n_windows and d > b:
                    per_window[w].append((b, d))
        return [np.array(pw, dtype=float) if pw else np.empty((0, 2)) for pw in per_window]


# ---------------------------------------------------------------------------
# Persistence image from bare diagram
# ---------------------------------------------------------------------------

def diagram_to_image(dgm: np.ndarray,
                     birth_range: tuple[float, float],
                     persistence_range: tuple[float, float],
                     resolution: int = IMAGE_RES,
                     sigma: float = IMAGE_SIGMA) -> np.ndarray:
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
# Alarm + F1
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
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    # --- Generate reservoir trajectory ---
    print(f"Generating reservoir: mode={MODE} seed={SEED}")
    states, _, _ = rdm.simulate_degradation(MODE, SEED)
    cx, cy = CHANNELS
    X_full = states[:, cx]
    Y_full = states[:, cy]

    starts = list(range(0, N_TOTAL - WINDOW + 1, STRIDE))
    n_windows = len(starts)
    onset_window = N_HEALTHY // STRIDE
    ground_truth = np.array([1 if s >= N_HEALTHY else 0 for s in starts], dtype=int)
    print(f"  {n_windows} windows, onset at w={onset_window}")

    # --- Cache embedding params from window 0 ---
    bd0 = BindingDetector(max_dim=1, method="persistence_image",
                          image_resolution=IMAGE_RES, image_sigma=IMAGE_SIGMA,
                          embedding_quality_gate=False)
    bd0.fit(X_full[:WINDOW], Y_full[:WINDOW], subsample=SUBSAMPLE, seed=SEED)
    emb_x_d, emb_x_dim = bd0._marginal_delay_x, bd0._marginal_dim_x
    emb_y_d, emb_y_dim = bd0._marginal_delay_y, bd0._marginal_dim_y
    j_delays, j_dims = list(bd0._joint_delays), list(bd0._joint_dims)

    # --- Dense pass: collect per-window scores + H1 diagrams ---
    scores_a = np.zeros(n_windows)
    joint_h1: list[np.ndarray] = []
    marg_x_h1: list[np.ndarray] = []
    marg_y_h1: list[np.ndarray] = []
    birth_ranges: list[tuple[float, float]] = []
    pers_ranges: list[tuple[float, float]] = []

    print(f"  Dense pass over {n_windows} windows...")
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
                subsample=SUBSAMPLE, seed=SEED,
            )
            scores_a[w_idx] = bd.binding_score()
            joint_h1.append(bd._pa_joint.diagrams_[1].copy() if len(bd._pa_joint.diagrams_) > 1 else np.empty((0, 2)))
            marg_x_h1.append(bd._pa_x.diagrams_[1].copy() if len(bd._pa_x.diagrams_) > 1 else np.empty((0, 2)))
            marg_y_h1.append(bd._pa_y.diagrams_[1].copy() if len(bd._pa_y.diagrams_) > 1 else np.empty((0, 2)))
            birth_ranges.append(bd._birth_range)
            pers_ranges.append(bd._persistence_range)
        except Exception as e:
            print(f"    window {w_idx} failed: {e}", file=sys.stderr)
            scores_a[w_idx] = np.nan
            joint_h1.append(np.empty((0, 2)))
            marg_x_h1.append(np.empty((0, 2)))
            marg_y_h1.append(np.empty((0, 2)))
            birth_ranges.append((0.0, 1.0))
            pers_ranges.append((0.0, 1.0))

        if w_idx % 50 == 49 or w_idx == n_windows - 1:
            print(f"    w {w_idx + 1}/{n_windows}")

    # Interpolate NaN failures
    finite_mask = np.isfinite(scores_a)
    if not finite_mask.all():
        scores_a = np.interp(np.arange(n_windows), np.where(finite_mask)[0], scores_a[finite_mask])

    # Global ranges for tracked PIs
    global_birth = (min(br[0] for br in birth_ranges), max(br[1] for br in birth_ranges))
    global_pers = (min(pr[0] for pr in pers_ranges), max(pr[1] for pr in pers_ranges))

    # --- Arm A: dense raw binding (baseline) ---
    alarms_a_up = alarms_from_signal(scores_a, N_BASELINE_WINDOWS, SIGMA_ALARM)
    alarms_a_down = alarms_from_signal(-scores_a, N_BASELINE_WINDOWS, SIGMA_ALARM)
    res_a_up = prf1(alarms_a_up, ground_truth)
    res_a_down = prf1(alarms_a_down, ground_truth)
    if res_a_up["f1"] >= res_a_down["f1"]:
        res_a = res_a_up
    else:
        res_a = res_a_down
    arm_a_f1 = res_a["f1"]
    print(f"  Arm A (dense raw): F1={arm_a_f1:.4f}")

    # --- Arm C: Hungarian-tracked + smoothed ---
    tracker = FeatureTracker(max_cost_mult=TRACK_MAX_COST_MULT)
    for w_idx, dgm in enumerate(joint_h1):
        tracker.update(w_idx, dgm)
    tracker.smooth_tracks()
    tracked_diagrams = tracker.reconstruct_diagrams(n_windows)

    scores_c = np.zeros(n_windows)
    for w_idx in range(n_windows):
        joint_img = diagram_to_image(tracked_diagrams[w_idx], global_birth, global_pers)
        marg_x_img = diagram_to_image(marg_x_h1[w_idx], global_birth, global_pers)
        marg_y_img = diagram_to_image(marg_y_h1[w_idx], global_birth, global_pers)
        baseline_img = np.maximum(marg_x_img, marg_y_img)
        residual = np.maximum(joint_img - baseline_img, 0)
        scores_c[w_idx] = float(residual.sum())

    alarms_c_up = alarms_from_signal(scores_c, N_BASELINE_WINDOWS, SIGMA_ALARM)
    alarms_c_down = alarms_from_signal(-scores_c, N_BASELINE_WINDOWS, SIGMA_ALARM)
    res_c_up = prf1(alarms_c_up, ground_truth)
    res_c_down = prf1(alarms_c_down, ground_truth)
    if res_c_up["f1"] >= res_c_down["f1"]:
        res_c = res_c_up
        alarms_c = alarms_c_up
    else:
        res_c = res_c_down
        alarms_c = alarms_c_down
    tracked_f1 = res_c["f1"]
    print(f"  Arm C (Hungarian tracked): F1={tracked_f1:.4f}")

    first_alarm = first_alarm_index(alarms_c)
    runtime_s = time.time() - t0
    lift = tracked_f1 - arm_a_f1

    print(f"  Tracking lift: {lift:+.4f}")
    print(f"  Runtime: {runtime_s:.1f}s")

    # --- Output ---
    payload = {
        "score": round(tracked_f1, 6),
        "f1": round(tracked_f1, 6),
        "precision": round(res_c["precision"], 4),
        "recall": round(res_c["recall"], 4),
        "strategy": "hungarian_tracked_smoothed",
        "uses_tracking": True,
        "n_windows": n_windows,
        "onset_window": onset_window,
        "first_alarm_window": first_alarm,
        "arm_a_f1": round(arm_a_f1, 4),
        "tracking_lift": round(lift, 4),
        "runtime_s": round(runtime_s, 1),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
