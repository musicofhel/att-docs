#!/usr/bin/env python3
"""Branch 1 tracer bullet — tiered sentinel architecture on MIT-BIH cardiac arrhythmia.

Architecture #1 from the hybrid topological monitoring document. The existing
ATT cardiac experiment (`scripts/branches/cardiac_arrhythmia_topology.py` on
`experiment/tda-cardiac`) hits ~20% precision with NEGATIVE mean detection lag —
variance spikes pre-empt the topology-based alarm. Tiered sentinel turns this
bug into a feature: variance + ACF1 become the *gate* (Layer 1) for cheap
topology (Layer 2 — persistence entropy), and only entropy-confirmed windows
get the full TransitionDetector treatment (Layer 3).

Three arms (all on the same windowing for an apples-to-apples comparison):
  A. Topology-only — per-window persistence entropy + 2σ alarm. The existing
                     reservoir-style "PH on every window" baseline.
  B. Variance-only — Layer 1 alone (CUSUM on rolling variance). Establishes
                     the lower bound the tiered sentinel must beat.
  C. Tiered sentinel — Layer 1 → if triggered, Layer 2 (entropy) → if confirmed,
                       Layer 3 (TransitionDetector). Alarm only at Layer 3.

Success criterion (tracer bullet):
  precision(C) >= 0.40 (~2x existing baseline) AND f1(C) > max(f1(A), f1(B))
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from math import gcd

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, resample_poly

import wfdb

from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants — match the existing cardiac script's preprocessing exactly
# ---------------------------------------------------------------------------

ORIGINAL_FS = 360
TARGET_FS = 128
WINDOW = 1280  # 10 seconds @ 128 Hz
STRIDE = 640   # 5 seconds (50% overlap)
ARRHYTHMIA_THRESHOLD = 0.50  # window is "arrhythmia" iff >50% non-N beats
NORMAL_SYMBOLS = {"N", "·", ".", "/"}
BEAT_SYMBOLS = set("NLRBAaJSVFejnE/fQ·.")

# Layer 1 / detection params
SUBSAMPLE = 200  # for PH on Takens-embedded windows
SIGMA_ALARM = 2.0
CUSUM_K = 0.5  # CUSUM slack (in z-score units)
CUSUM_H = 5.0  # CUSUM threshold (in z-score units)
ENTROPY_GATE_K = 2.0  # confirm window iff entropy z-score > k

# Healthy region: first 5 minutes — used to build z-score baselines
BASELINE_SECONDS = 5 * 60
BASELINE_SAMPLES = BASELINE_SECONDS * TARGET_FS

DATA_DIR = os.path.join(REPO_ROOT, "data", "cardiac_tiered")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "cardiac_tiered")
HANDOFF_DIR = os.path.join(REPO_ROOT, ".claude", "handoff")


# ---------------------------------------------------------------------------
# Signal preprocessing (verbatim from cardiac_arrhythmia_topology.py)
# ---------------------------------------------------------------------------

def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    nyq = fs / 2.0
    sos = butter(4, [low / nyq, min(high / nyq, 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def downsample(signal: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    g = gcd(fs_orig, fs_target)
    return resample_poly(signal, fs_target // g, fs_orig // g)


def load_record(record_num: int) -> dict:
    rec_str = str(record_num)
    record = wfdb.rdrecord(rec_str, pn_dir="mitdb")
    annotation = wfdb.rdann(rec_str, "atr", pn_dir="mitdb")
    signal = record.p_signal[:, 0].astype(np.float64)
    fs = record.fs
    signal = bandpass_filter(signal, fs)
    signal_ds = downsample(signal, fs, TARGET_FS)
    scale = TARGET_FS / fs
    ann_samples = (annotation.sample * scale).astype(int)
    ann_symbols = annotation.symbol
    return {
        "record": record_num,
        "signal": signal_ds,
        "fs": TARGET_FS,
        "ann_samples": ann_samples,
        "ann_symbols": ann_symbols,
        "duration_s": len(signal_ds) / TARGET_FS,
    }


def label_window(ann_samples: np.ndarray, ann_symbols: list, start: int, end: int) -> int:
    """Returns 1 if window is arrhythmic (>50% non-N beats), else 0."""
    mask = (ann_samples >= start) & (ann_samples < end)
    syms = [ann_symbols[i] for i in range(len(ann_symbols)) if mask[i]]
    beats = [s for s in syms if s in BEAT_SYMBOLS]
    if len(beats) < 3:
        return 0  # too few beats to label — treat as not-arrhythmic
    n_normal = sum(1 for s in beats if s in NORMAL_SYMBOLS)
    frac_normal = n_normal / len(beats)
    return 1 if frac_normal < (1.0 - ARRHYTHMIA_THRESHOLD) else 0


# ---------------------------------------------------------------------------
# Layer 1 — rolling variance + ACF1 + CUSUM
# ---------------------------------------------------------------------------

def rolling_variance(signal: np.ndarray, win: int, stride: int) -> np.ndarray:
    n_windows = (len(signal) - win) // stride + 1
    out = np.zeros(n_windows)
    for i in range(n_windows):
        s = i * stride
        out[i] = float(np.var(signal[s:s + win]))
    return out


def rolling_acf1(signal: np.ndarray, win: int, stride: int) -> np.ndarray:
    n_windows = (len(signal) - win) // stride + 1
    out = np.zeros(n_windows)
    for i in range(n_windows):
        s = i * stride
        x = signal[s:s + win]
        x_c = x - x.mean()
        denom = float(np.dot(x_c, x_c))
        if denom < 1e-15:
            out[i] = 0.0
        else:
            out[i] = float(np.dot(x_c[:-1], x_c[1:])) / denom
    return out


def cusum_alarm(z: np.ndarray, k: float = CUSUM_K, h: float = CUSUM_H) -> np.ndarray:
    """Two-sided CUSUM. Returns boolean alarm array."""
    pos = np.zeros_like(z)
    neg = np.zeros_like(z)
    alarms = np.zeros_like(z, dtype=bool)
    for i in range(len(z)):
        pos[i] = max(0.0, (pos[i - 1] if i > 0 else 0.0) + (z[i] - k))
        neg[i] = max(0.0, (neg[i - 1] if i > 0 else 0.0) + (-z[i] - k))
        if pos[i] > h or neg[i] > h:
            alarms[i] = True
            pos[i] = 0.0
            neg[i] = 0.0
    return alarms


def zscore(x: np.ndarray, baseline_n: int) -> np.ndarray:
    base = x[:baseline_n]
    base = base[np.isfinite(base)]
    if len(base) < 2:
        return np.zeros_like(x)
    mu = float(base.mean())
    sd = float(base.std())
    if sd < 1e-15:
        sd = 1e-15
    return (x - mu) / sd


# ---------------------------------------------------------------------------
# Layer 2 — persistence entropy from a Takens-embedded window
# ---------------------------------------------------------------------------

def window_h1_entropy(signal_window: np.ndarray, seed: int) -> float:
    if np.std(signal_window) < 1e-12:
        return 0.0
    try:
        emb = TakensEmbedder(delay="auto", dimension="auto")
        emb.fit(signal_window)
    except Exception:
        emb = TakensEmbedder(delay=10, dimension=3)
        emb.fit(signal_window)
    cloud = emb.transform(signal_window)
    if cloud.shape[0] < 5:
        return 0.0
    pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
    result = pa.fit_transform(cloud, subsample=SUBSAMPLE, seed=seed)
    pe = result.get("persistence_entropy", [0.0, 0.0])
    return float(pe[1] if len(pe) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Layer 3 — TransitionDetector-style score from a window
# ---------------------------------------------------------------------------

def window_transition_score(signal_window: np.ndarray, seed: int) -> float:
    """A simple per-window topological-change score: max H1 persistence + total persistence.

    Avoids the full TransitionDetector windowing (which is for streaming over a long
    sequence) — at this layer we already have a single window and just need a scalar
    indicating "how unusual is this window's H1 structure".
    """
    if np.std(signal_window) < 1e-12:
        return 0.0
    try:
        emb = TakensEmbedder(delay="auto", dimension="auto")
        emb.fit(signal_window)
    except Exception:
        emb = TakensEmbedder(delay=10, dimension=3)
        emb.fit(signal_window)
    cloud = emb.transform(signal_window)
    if cloud.shape[0] < 5:
        return 0.0
    pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa.fit_transform(cloud, subsample=SUBSAMPLE, seed=seed)
    if len(pa.diagrams_) < 2 or len(pa.diagrams_[1]) == 0:
        return 0.0
    d = pa.diagrams_[1]
    life = d[:, 1] - d[:, 0]
    life = life[life > 1e-12]
    if len(life) == 0:
        return 0.0
    return float(life.sum() + life.max())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def prf1(pred: np.ndarray, truth: np.ndarray) -> dict:
    p = pred.astype(int)
    t = truth.astype(int)
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(prec), "recall": float(rec), "f1": float(f1)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(signal: np.ndarray, fs: int, window_centers_s: np.ndarray,
                 truth: np.ndarray,
                 var_z: np.ndarray, l1_alarm: np.ndarray,
                 entropy_z: np.ndarray, l2_confirm: np.ndarray,
                 l3_score: np.ndarray, l3_alarm: np.ndarray,
                 record: int, out_path: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 1. Raw ECG with truth
    ax = axes[0]
    t = np.arange(len(signal)) / fs
    ax.plot(t, signal, lw=0.4, color="black")
    for i, ws in enumerate(window_centers_s):
        if truth[i] == 1:
            ax.axvspan(ws - WINDOW / (2 * fs), ws + WINDOW / (2 * fs),
                       color="red", alpha=0.10, lw=0)
    ax.set_ylabel("ECG")
    ax.set_title(f"Record {record} — red shading = arrhythmic windows (ground truth)")
    ax.set_xlim(0, t[-1])

    # 2. Layer 1 — variance z-score with CUSUM alarms
    ax = axes[1]
    ax.plot(window_centers_s, var_z, lw=1, color="C0", label="variance z-score")
    a = np.where(l1_alarm)[0]
    if len(a) > 0:
        ax.scatter(window_centers_s[a], var_z[a], color="orange", s=12, zorder=5,
                   label=f"L1 trigger ({len(a)})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("L1: variance z")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Layer 2 — entropy z on triggered windows
    ax = axes[2]
    ax.plot(window_centers_s, entropy_z, lw=1, color="C1", label="H1 entropy z (triggered only)")
    c = np.where(l2_confirm)[0]
    if len(c) > 0:
        ax.scatter(window_centers_s[c], entropy_z[c], color="purple", s=14, zorder=5,
                   label=f"L2 confirm ({len(c)})")
    ax.axhline(ENTROPY_GATE_K, color="purple", linestyle=":", lw=1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("L2: entropy z")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Layer 3 — transition score on confirmed windows + final alarm
    ax = axes[3]
    ax.plot(window_centers_s, l3_score, lw=1, color="C2", label="L3 transition score")
    a3 = np.where(l3_alarm)[0]
    if len(a3) > 0:
        ax.scatter(window_centers_s[a3], l3_score[a3], color="red", s=18, zorder=5,
                   label=f"L3 alarm ({len(a3)})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("L3: PH score")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Branch 1 — tiered sentinel cardiac tracer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record", type=int, default=207)
    parser.add_argument("--fallback-record", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(HANDOFF_DIR, exist_ok=True)

    t0 = time.time()
    print(f"[branch 1] Loading MIT-BIH record {args.record}...")
    try:
        data = load_record(args.record)
    except Exception as e:
        print(f"  primary record {args.record} failed: {e}")
        print(f"  falling back to record {args.fallback_record}...")
        data = load_record(args.fallback_record)
    record_used = data["record"]
    signal = data["signal"]
    fs = data["fs"]
    print(f"  record {record_used}: {len(signal)} samples ({data['duration_s']:.1f}s @ {fs} Hz)")
    print(f"  annotations: {len(data['ann_samples'])}")

    # ---- Window the signal ----
    starts = list(range(0, len(signal) - WINDOW + 1, STRIDE))
    n_windows = len(starts)
    centers_s = np.array([(s + WINDOW / 2) / fs for s in starts])
    print(f"  windows: {n_windows} (W={WINDOW}, stride={STRIDE}, ~{WINDOW/fs:.1f}s each)")

    # Ground truth labels per window
    truth = np.array([
        label_window(data["ann_samples"], data["ann_symbols"], s, s + WINDOW)
        for s in starts
    ])
    print(f"  ground truth: {int(truth.sum())} arrhythmic / {int((1 - truth).sum())} normal")

    # Baseline window count (first 5 minutes)
    baseline_n = max(1, int(BASELINE_SAMPLES // STRIDE))
    baseline_n = min(baseline_n, n_windows // 2)
    print(f"  baseline window count for z-scoring: {baseline_n}")

    # ---- Layer 1 — rolling variance + ACF1 + CUSUM ----
    print(f"  Layer 1: rolling variance + ACF1 + CUSUM...")
    var_series = rolling_variance(signal, WINDOW, STRIDE)
    acf_series = rolling_acf1(signal, WINDOW, STRIDE)
    var_z = zscore(var_series, baseline_n)
    acf_z = zscore(acf_series, baseline_n)
    var_alarm = cusum_alarm(var_z, k=CUSUM_K, h=CUSUM_H)
    acf_alarm = cusum_alarm(acf_z, k=CUSUM_K, h=CUSUM_H)
    l1_trigger = var_alarm | acf_alarm
    print(f"    var triggers: {int(var_alarm.sum())}, acf triggers: {int(acf_alarm.sum())}, "
          f"any: {int(l1_trigger.sum())}")

    # ---- Layer 2 — persistence entropy on TRIGGERED windows only ----
    print(f"  Layer 2: persistence entropy on {int(l1_trigger.sum())} triggered window(s)...")
    entropy_series = np.full(n_windows, np.nan)
    t_l2 = time.time()
    for i in range(n_windows):
        if l1_trigger[i]:
            s = starts[i]
            entropy_series[i] = window_h1_entropy(signal[s:s + WINDOW], seed=args.seed + i)
    print(f"    Layer 2 runtime: {time.time() - t_l2:.1f}s")

    # We need a baseline for entropy z-scoring — but Layer 2 only fires on triggered windows.
    # Use the FULL entropy of the first baseline_n windows (compute once) as the baseline.
    print(f"  Layer 2 baseline: computing entropy on first {baseline_n} windows for z-scoring...")
    baseline_entropy = np.zeros(baseline_n)
    for i in range(baseline_n):
        s = starts[i]
        baseline_entropy[i] = window_h1_entropy(signal[s:s + WINDOW], seed=args.seed + i)
    base_mu = float(baseline_entropy.mean())
    base_sd = float(baseline_entropy.std())
    if base_sd < 1e-15:
        base_sd = 1e-15
    print(f"    baseline entropy: mean={base_mu:.3f}, std={base_sd:.3f}")

    entropy_z = np.zeros(n_windows)
    for i in range(n_windows):
        if not np.isnan(entropy_series[i]):
            entropy_z[i] = (entropy_series[i] - base_mu) / base_sd
        else:
            entropy_z[i] = np.nan
    l2_confirm = (np.nan_to_num(entropy_z, nan=-np.inf) > ENTROPY_GATE_K) & l1_trigger
    print(f"    Layer 2 confirmed: {int(l2_confirm.sum())} window(s)")

    # ---- Layer 3 — TransitionDetector-style PH score on confirmed windows ----
    print(f"  Layer 3: PH transition score on {int(l2_confirm.sum())} confirmed window(s)...")
    l3_score = np.zeros(n_windows)
    t_l3 = time.time()
    for i in range(n_windows):
        if l2_confirm[i]:
            s = starts[i]
            l3_score[i] = window_transition_score(signal[s:s + WINDOW], seed=args.seed + i)
    print(f"    Layer 3 runtime: {time.time() - t_l3:.1f}s")

    # Layer 3 alarm: confirmed AND score above 75th percentile of confirmed-window scores
    confirmed_scores = l3_score[l2_confirm]
    if len(confirmed_scores) > 0:
        l3_threshold = float(np.percentile(confirmed_scores, 75))
    else:
        l3_threshold = float("inf")
    l3_alarm = l2_confirm & (l3_score >= l3_threshold)
    print(f"    Layer 3 alarm threshold (75th pctile of confirmed): {l3_threshold:.3f}")
    print(f"    Layer 3 alarms: {int(l3_alarm.sum())}")

    # ---- Arms ----
    # Arm A: topology-only baseline — per-window entropy + 2σ alarm using PROPER baseline.
    # This needs entropy on EVERY window (not just triggered). We already have it for the
    # baseline region; compute for the rest now.
    print(f"  Arm A baseline (topology-only): computing entropy on remaining windows...")
    full_entropy = np.zeros(n_windows)
    full_entropy[:baseline_n] = baseline_entropy
    t_a = time.time()
    for i in range(baseline_n, n_windows):
        s = starts[i]
        full_entropy[i] = window_h1_entropy(signal[s:s + WINDOW], seed=args.seed + i)
    print(f"    Arm A entropy runtime: {time.time() - t_a:.1f}s")
    arm_a_alarm = (full_entropy - base_mu) / base_sd > SIGMA_ALARM

    # Arm B: variance-only — Layer 1 alone
    arm_b_alarm = l1_trigger

    # Arm C: tiered sentinel — final L3 alarms
    arm_c_alarm = l3_alarm

    arms = {
        "A_topology_only": arm_a_alarm,
        "B_variance_only": arm_b_alarm,
        "C_tiered_sentinel": arm_c_alarm,
    }
    arm_results = {}
    for name, alarms in arms.items():
        scores = prf1(alarms, truth)
        scores["n_alarms"] = int(alarms.sum())
        arm_results[name] = scores
        print(f"  {name}: F1={scores['f1']:.3f} P={scores['precision']:.3f} "
              f"R={scores['recall']:.3f} alarms={int(alarms.sum())}")

    # ---- Verdict ----
    f1_a = arm_results["A_topology_only"]["f1"]
    f1_b = arm_results["B_variance_only"]["f1"]
    f1_c = arm_results["C_tiered_sentinel"]["f1"]
    p_c = arm_results["C_tiered_sentinel"]["precision"]
    target_p = 0.40
    target_f1_lift = max(f1_a, f1_b)
    pass_precision = p_c >= target_p
    pass_f1 = f1_c > target_f1_lift
    passed = pass_precision and pass_f1
    verdict = "PASS" if passed else "FAIL"
    why = []
    if not pass_precision:
        why.append(f"precision(C)={p_c:.3f} < target {target_p}")
    if not pass_f1:
        why.append(f"f1(C)={f1_c:.3f} not > max(f1_A={f1_a:.3f}, f1_B={f1_b:.3f})")
    print(f"\n[branch 1] VERDICT: {verdict}")
    if why:
        print("  fail reasons:", "; ".join(why))

    # ---- Persist ----
    runtime_s = time.time() - t0
    output = {
        "branch": "experiment/tiered-sentinel-cardiac",
        "record": record_used,
        "seed": args.seed,
        "config": {
            "fs": fs, "window": WINDOW, "stride": STRIDE,
            "subsample": SUBSAMPLE, "sigma_alarm": SIGMA_ALARM,
            "cusum_k": CUSUM_K, "cusum_h": CUSUM_H,
            "entropy_gate_k": ENTROPY_GATE_K,
            "baseline_seconds": BASELINE_SECONDS,
            "baseline_windows": baseline_n,
            "arrhythmia_threshold": ARRHYTHMIA_THRESHOLD,
        },
        "n_windows": n_windows,
        "n_arrhythmic_windows": int(truth.sum()),
        "n_normal_windows": int((1 - truth).sum()),
        "layer_counts": {
            "l1_var_alarms": int(var_alarm.sum()),
            "l1_acf_alarms": int(acf_alarm.sum()),
            "l1_any_triggers": int(l1_trigger.sum()),
            "l2_confirmed": int(l2_confirm.sum()),
            "l3_alarms": int(l3_alarm.sum()),
        },
        "arms": arm_results,
        "f1_a": f1_a, "f1_b": f1_b, "f1_c": f1_c,
        "precision_c": p_c,
        "target_precision": target_p,
        "verdict": verdict,
        "verdict_failure_reasons": why,
        "runtime_s": runtime_s,
    }
    json_path = os.path.join(DATA_DIR, f"results_record{record_used}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  wrote {json_path}")

    fig_path = os.path.join(FIG_DIR, f"tiered_sentinel_record{record_used}.png")
    plot_results(signal, fs, centers_s, truth,
                 var_z, l1_trigger, entropy_z, l2_confirm,
                 l3_score, l3_alarm, record_used, fig_path)
    print(f"  wrote {fig_path}")

    print(f"\n[branch 1] runtime: {runtime_s:.1f}s")
    print(f"[branch 1] one-liner: A={f1_a:.3f} B={f1_b:.3f} C={f1_c:.3f} P_C={p_c:.3f} verdict={verdict}")


if __name__ == "__main__":
    main()
