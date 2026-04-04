#!/usr/bin/env python3
"""Branch 4: Cardiac Arrhythmia — Attractor Topology of Heart Rhythm.

Applies Takens embedding + persistent homology to ECG recordings from
MIT-BIH Arrhythmia Database. Tests whether attractor topology discriminates
normal sinus rhythm from arrhythmias, and whether topology tracks transitions.

Three experiments:
  1. Normal vs arrhythmia attractor topology (H1 entropy, Wasserstein, permutation test)
  2. Transition detection at arrhythmia onset (TransitionDetector)
  3. RR-interval dynamics (HRV attractor topology)
"""

import argparse
import functools
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Force unbuffered output
print = functools.partial(print, flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.stats import wasserstein_distance

# ATT imports
from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer
from att.transitions.detector import TransitionDetector

# wfdb for MIT-BIH data
import wfdb

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")

# ── Constants ────────────────────────────────────────────────────────────────

ARRHYTHMIA_RECORDS = [200, 201, 207, 210, 217]  # frequent PVCs / sustained arrhythmia
ORIGINAL_FS = 360  # MIT-BIH sampling rate
TARGET_FS = 128    # downsample target
SEGMENT_SECONDS = 30
SEGMENT_SAMPLES = SEGMENT_SECONDS * TARGET_FS  # 3840 samples per segment
NORMAL_THRESHOLD = 0.90   # >90% N beats → normal segment
ARRHYTHMIA_THRESHOLD = 0.50  # >50% non-N beats → arrhythmia segment

# Beat annotation symbols
NORMAL_SYMBOLS = {"N", "·", ".", "/"}  # Normal beats in MIT-BIH
# Everything else is considered abnormal


# ── Signal Processing ────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Bandpass filter ECG signal."""
    nyq = fs / 2.0
    sos = butter(4, [low / nyq, min(high / nyq, 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def downsample(signal: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """Downsample signal from fs_orig to fs_target."""
    from math import gcd
    g = gcd(fs_orig, fs_target)
    return resample_poly(signal, fs_target // g, fs_orig // g)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_record(record_num: int) -> dict | None:
    """Load MIT-BIH record with signal and annotations."""
    rec_str = str(record_num)
    try:
        record = wfdb.rdrecord(rec_str, pn_dir="mitdb")
        annotation = wfdb.rdann(rec_str, "atr", pn_dir="mitdb")
    except Exception as e:
        print(f"  [WARN] Failed to load record {record_num}: {e}")
        return None

    # Lead MLII (column 0)
    signal = record.p_signal[:, 0].astype(np.float64)
    fs = record.fs  # should be 360

    # Bandpass filter
    signal = bandpass_filter(signal, fs)

    # Downsample
    signal_ds = downsample(signal, fs, TARGET_FS)

    # Rescale annotation sample indices to downsampled rate
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


def classify_segment(ann_samples: np.ndarray, ann_symbols: list, seg_start: int, seg_end: int) -> str:
    """Classify a segment as 'normal', 'arrhythmia', or 'transition'."""
    mask = (ann_samples >= seg_start) & (ann_samples < seg_end)
    symbols = [ann_symbols[i] for i in range(len(ann_symbols)) if mask[i]]

    if len(symbols) < 3:
        return "insufficient"

    n_normal = sum(1 for s in symbols if s in NORMAL_SYMBOLS)
    frac_normal = n_normal / len(symbols)

    if frac_normal >= NORMAL_THRESHOLD:
        return "normal"
    elif frac_normal <= (1.0 - ARRHYTHMIA_THRESHOLD):
        return "arrhythmia"
    else:
        return "transition"


def extract_segments(data: dict) -> dict:
    """Extract normal, arrhythmia, and transition segments from a record."""
    signal = data["signal"]
    n_samples = len(signal)
    segments = {"normal": [], "arrhythmia": [], "transition": []}

    # Non-overlapping 30s windows
    for start in range(0, n_samples - SEGMENT_SAMPLES, SEGMENT_SAMPLES):
        end = start + SEGMENT_SAMPLES
        label = classify_segment(data["ann_samples"], data["ann_symbols"], start, end)
        if label in segments:
            segments[label].append(signal[start:end])

    return segments


def extract_rr_intervals(data: dict) -> dict:
    """Extract RR intervals for normal and arrhythmia episodes."""
    ann_samples = data["ann_samples"]
    ann_symbols = data["ann_symbols"]
    n_total = len(data["signal"])

    # Get all beat annotations (filter out non-beat annotations)
    beat_symbols = set("NLRBAaJSVFejnE/fQ·.")
    beat_mask = [s in beat_symbols for s in ann_symbols]
    beat_samples = ann_samples[beat_mask]
    beat_syms = [ann_symbols[i] for i, m in enumerate(beat_mask) if m]

    if len(beat_samples) < 10:
        return {"normal_rr": np.array([]), "arrhythmia_rr": np.array([])}

    # RR intervals in seconds
    rr_all = np.diff(beat_samples) / TARGET_FS

    # Classify each RR interval by surrounding beats
    normal_rr = []
    arrhythmia_rr = []

    for i in range(len(rr_all)):
        s1, s2 = beat_syms[i], beat_syms[i + 1]
        if s1 in NORMAL_SYMBOLS and s2 in NORMAL_SYMBOLS:
            normal_rr.append(rr_all[i])
        elif s1 not in NORMAL_SYMBOLS or s2 not in NORMAL_SYMBOLS:
            arrhythmia_rr.append(rr_all[i])

    return {
        "normal_rr": np.array(normal_rr) if normal_rr else np.array([]),
        "arrhythmia_rr": np.array(arrhythmia_rr) if arrhythmia_rr else np.array([]),
    }


def find_transition_segment(data: dict, min_run: int = 5) -> dict | None:
    """Find a 5-minute segment spanning an arrhythmia onset.

    Looks for a run of ≥min_run consecutive abnormal beats preceded by normal beats.
    Returns a 5-minute window centered on the onset.
    """
    ann_samples = data["ann_samples"]
    ann_symbols = data["ann_symbols"]
    signal = data["signal"]

    five_min = 5 * 60 * TARGET_FS  # 5 minutes in samples

    # Find runs of abnormal beats
    is_abnormal = [s not in NORMAL_SYMBOLS for s in ann_symbols]

    onset_idx = None
    run_count = 0
    for i in range(len(is_abnormal)):
        if is_abnormal[i]:
            run_count += 1
            if run_count >= min_run and onset_idx is None:
                onset_idx = i - min_run + 1
                break
        else:
            run_count = 0

    if onset_idx is None:
        return None

    onset_sample = ann_samples[onset_idx]
    # Center the 5-min window on the onset
    start = max(0, onset_sample - five_min // 2)
    end = min(len(signal), start + five_min)
    start = max(0, end - five_min)

    return {
        "signal": signal[start:end],
        "onset_sample_relative": onset_sample - start,
        "onset_time_s": (onset_sample - start) / TARGET_FS,
        "start": start,
        "end": end,
        "record": data["record"],
    }


# ── Synthetic Fallback ───────────────────────────────────────────────────────

def synthetic_ecg_segment(segment_type: str, fs: int = 128, duration_s: int = 30, seed: int = 42) -> np.ndarray:
    """Generate synthetic ECG-like signal for fallback."""
    rng = np.random.default_rng(seed)
    t = np.arange(duration_s * fs) / fs

    if segment_type == "normal":
        # Regular sinus rhythm ~72 bpm
        hr = 72 / 60  # Hz
        ecg = np.sin(2 * np.pi * hr * t) * 0.5
        ecg += 0.3 * np.sin(2 * np.pi * 2 * hr * t)  # QRS complex harmonic
        ecg += rng.normal(0, 0.02, len(t))
    else:
        # Irregular rhythm with PVCs
        hr = 80 / 60
        ecg = np.sin(2 * np.pi * hr * t) * 0.5
        # Add irregular beats
        pvc_times = rng.choice(len(t), size=int(duration_s * 0.3), replace=False)
        for pt in pvc_times:
            if pt + 20 < len(t):
                ecg[pt:pt + 20] += rng.normal(0, 0.5, 20)
        ecg += rng.normal(0, 0.05, len(t))

    return ecg


def synthetic_rr_intervals(segment_type: str, n: int = 150, seed: int = 42) -> np.ndarray:
    """Generate synthetic RR intervals."""
    rng = np.random.default_rng(seed)
    if segment_type == "normal":
        # Normal HRV: mean ~0.83s (72 bpm), moderate variability
        rr = 0.83 + 0.05 * np.sin(2 * np.pi * np.arange(n) / 25)  # respiratory sinus arrhythmia
        rr += rng.normal(0, 0.02, n)
    else:
        # Arrhythmic: irregular intervals
        rr = 0.75 + rng.exponential(0.1, n)
        # Occasional very short intervals (PVCs)
        pvc_mask = rng.random(n) < 0.15
        rr[pvc_mask] *= 0.5
    return np.clip(rr, 0.2, 2.0)


# ── TDA Helpers ──────────────────────────────────────────────────────────────

def embed_and_ph(signal: np.ndarray, delay="auto", dimension="auto", max_dim: int = 1,
                 subsample: int | None = None, seed: int = 42,
                 fallback_delay: int = 8, fallback_dim: int = 5) -> tuple:
    """Takens embed + PH. Returns (cloud, ph_results, embedder, used_fallback)."""
    used_fallback = False
    try:
        embedder = TakensEmbedder(delay=delay, dimension=dimension)
        cloud = embedder.fit_transform(signal)
        d = embedder.delay_ if hasattr(embedder, "delay_") else fallback_delay
        dim = embedder.dimension_ if hasattr(embedder, "dimension_") else fallback_dim
    except Exception:
        used_fallback = True
        embedder = TakensEmbedder(delay=fallback_delay, dimension=fallback_dim)
        cloud = embedder.fit_transform(signal)
        d = fallback_delay
        dim = fallback_dim

    if cloud.shape[0] < 20:
        return cloud, None, embedder, used_fallback

    pa = PersistenceAnalyzer(max_dim=max_dim)
    ph = pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return cloud, ph, embedder, used_fallback


def count_features(diagrams: list, dim: int) -> int:
    """Count finite-lifetime features in dimension dim."""
    if dim >= len(diagrams):
        return 0
    dgm = diagrams[dim]
    if len(dgm) == 0:
        return 0
    finite = dgm[np.isfinite(dgm[:, 1])]
    return len(finite)


def get_lifetimes(diagrams: list, dim: int) -> np.ndarray:
    """Get finite lifetimes in dimension dim."""
    if dim >= len(diagrams):
        return np.array([])
    dgm = diagrams[dim]
    if len(dgm) == 0:
        return np.array([])
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return np.array([])
    return finite[:, 1] - finite[:, 0]


def wasserstein_1d(dgm1: list, dgm2: list, dim: int = 1) -> float:
    """Wasserstein-1 via lifetime distribution comparison."""
    l1 = get_lifetimes(dgm1, dim)
    l2 = get_lifetimes(dgm2, dim)
    if len(l1) == 0 or len(l2) == 0:
        return 0.0
    return float(wasserstein_distance(l1, l2))


# ── Experiment 1: Normal vs Arrhythmia Attractor Topology ────────────────────

def run_exp1(normal_segments: list, arrhythmia_segments: list,
             subsample: int = 500, seed: int = 42, n_perms: int = 200) -> dict:
    """Compare attractor topology of normal vs arrhythmia ECG segments."""
    print("\n═══ Experiment 1: Normal vs Arrhythmia Attractor Topology ═══")

    normal_h1_features = []
    normal_h1_entropy = []
    normal_diagrams = []
    arrhythmia_h1_features = []
    arrhythmia_h1_entropy = []
    arrhythmia_diagrams = []
    embedding_info = {"delay": None, "dimension": None, "method": "auto"}

    for i, seg in enumerate(normal_segments):
        print(f"  [Normal {i+1}/{len(normal_segments)}] len={len(seg)}...", end=" ")
        cloud, ph, emb, fb = embed_and_ph(seg, subsample=subsample, seed=seed)
        if ph is None:
            print("SKIP (cloud too small)")
            continue
        if embedding_info["delay"] is None and hasattr(emb, "delay_"):
            embedding_info["delay"] = int(emb.delay_)
            embedding_info["dimension"] = int(emb.dimension_)
            if fb:
                embedding_info["method"] = "fallback"
        n_h1 = count_features(ph["diagrams"], 1)
        ent = ph["persistence_entropy"][1] if len(ph["persistence_entropy"]) > 1 else 0.0
        normal_h1_features.append(n_h1)
        normal_h1_entropy.append(ent)
        normal_diagrams.append(ph["diagrams"])
        print(f"H1={n_h1}, ent={ent:.3f}")

    for i, seg in enumerate(arrhythmia_segments):
        print(f"  [Arrhythmia {i+1}/{len(arrhythmia_segments)}] len={len(seg)}...", end=" ")
        cloud, ph, emb, fb = embed_and_ph(seg, subsample=subsample, seed=seed)
        if ph is None:
            print("SKIP (cloud too small)")
            continue
        n_h1 = count_features(ph["diagrams"], 1)
        ent = ph["persistence_entropy"][1] if len(ph["persistence_entropy"]) > 1 else 0.0
        arrhythmia_h1_features.append(n_h1)
        arrhythmia_h1_entropy.append(ent)
        arrhythmia_diagrams.append(ph["diagrams"])
        print(f"H1={n_h1}, ent={ent:.3f}")

    n_normal = len(normal_h1_features)
    n_arr = len(arrhythmia_h1_features)
    print(f"  Normal segments analyzed: {n_normal}")
    print(f"  Arrhythmia segments analyzed: {n_arr}")

    if n_normal == 0 or n_arr == 0:
        print("  [WARN] Insufficient segments for comparison")
        return {
            "n_normal": n_normal, "n_arrhythmia": n_arr,
            "normal_h1_entropy_mean": 0.0, "arrhythmia_h1_entropy_mean": 0.0,
            "wasserstein_p": 1.0, "wasserstein_z": 0.0,
            "embedding_params": embedding_info,
            "normal_h1_features": [], "arrhythmia_h1_features": [],
            "normal_diagrams": normal_diagrams, "arrhythmia_diagrams": arrhythmia_diagrams,
        }

    mean_normal_ent = float(np.mean(normal_h1_entropy))
    mean_arr_ent = float(np.mean(arrhythmia_h1_entropy))
    mean_normal_h1 = float(np.mean(normal_h1_features))
    mean_arr_h1 = float(np.mean(arrhythmia_h1_features))

    print(f"  Normal:     H1 features={mean_normal_h1:.1f}±{np.std(normal_h1_features):.1f}, "
          f"entropy={mean_normal_ent:.3f}±{np.std(normal_h1_entropy):.3f}")
    print(f"  Arrhythmia: H1 features={mean_arr_h1:.1f}±{np.std(arrhythmia_h1_features):.1f}, "
          f"entropy={mean_arr_ent:.3f}±{np.std(arrhythmia_h1_entropy):.3f}")

    # Aggregate Wasserstein: pool all H1 lifetimes per class
    all_normal_lt = np.concatenate([get_lifetimes(d, 1) for d in normal_diagrams if len(get_lifetimes(d, 1)) > 0])
    all_arr_lt = np.concatenate([get_lifetimes(d, 1) for d in arrhythmia_diagrams if len(get_lifetimes(d, 1)) > 0])

    if len(all_normal_lt) > 0 and len(all_arr_lt) > 0:
        observed_w = float(wasserstein_distance(all_normal_lt, all_arr_lt))
    else:
        observed_w = 0.0

    print(f"  Observed Wasserstein (pooled H1): {observed_w:.4f}")

    # Permutation test: shuffle segment labels
    all_diagrams = normal_diagrams + arrhythmia_diagrams
    all_labels = [0] * len(normal_diagrams) + [1] * len(arrhythmia_diagrams)
    rng = np.random.default_rng(seed)
    null_w = []

    for _ in range(n_perms):
        perm = rng.permutation(len(all_labels))
        perm_labels = [all_labels[p] for p in perm]
        perm_normal_lt = []
        perm_arr_lt = []
        for j, lab in enumerate(perm_labels):
            lt = get_lifetimes(all_diagrams[j], 1)
            if len(lt) > 0:
                if lab == 0:
                    perm_normal_lt.append(lt)
                else:
                    perm_arr_lt.append(lt)

        if perm_normal_lt and perm_arr_lt:
            w = wasserstein_distance(np.concatenate(perm_normal_lt), np.concatenate(perm_arr_lt))
            null_w.append(w)

    null_w = np.array(null_w)
    if len(null_w) > 0:
        p_value = float(np.mean(null_w >= observed_w))
        z_score = float((observed_w - np.mean(null_w)) / max(np.std(null_w), 1e-10))
    else:
        p_value = 1.0
        z_score = 0.0

    print(f"  Permutation test: p={p_value:.4f}, z={z_score:.2f} ({n_perms} permutations)")
    print(f"  → {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'}")

    return {
        "n_normal": n_normal,
        "n_arrhythmia": n_arr,
        "normal_h1_features_mean": round(mean_normal_h1, 2),
        "normal_h1_features_std": round(float(np.std(normal_h1_features)), 2),
        "arrhythmia_h1_features_mean": round(mean_arr_h1, 2),
        "arrhythmia_h1_features_std": round(float(np.std(arrhythmia_h1_features)), 2),
        "normal_h1_entropy_mean": round(mean_normal_ent, 4),
        "normal_h1_entropy_std": round(float(np.std(normal_h1_entropy)), 4),
        "arrhythmia_h1_entropy_mean": round(mean_arr_ent, 4),
        "arrhythmia_h1_entropy_std": round(float(np.std(arrhythmia_h1_entropy)), 4),
        "wasserstein_observed": round(observed_w, 4),
        "wasserstein_p": round(p_value, 4),
        "wasserstein_z": round(z_score, 2),
        "n_permutations": n_perms,
        "embedding_params": embedding_info,
        "normal_h1_features": normal_h1_features,
        "arrhythmia_h1_features": arrhythmia_h1_features,
        "normal_h1_entropy_list": [round(e, 4) for e in normal_h1_entropy],
        "arrhythmia_h1_entropy_list": [round(e, 4) for e in arrhythmia_h1_entropy],
        "null_distribution": null_w.tolist() if len(null_w) > 0 else [],
        "normal_diagrams": normal_diagrams,
        "arrhythmia_diagrams": arrhythmia_diagrams,
    }


# ── Experiment 2: Transition Detection ───────────────────────────────────────

def run_exp2(transition_data: list, subsample: int = 500, seed: int = 42) -> dict:
    """Detect arrhythmia onset via sliding-window topology."""
    print("\n═══ Experiment 2: Transition Detection ═══")

    if not transition_data:
        print("  [WARN] No transition segments found")
        return {
            "n_transitions": 0,
            "detection_lag_seconds": float("nan"),
            "precision": 0.0,
            "recall": 0.0,
        }

    all_lags = []
    all_tp = 0
    all_fp = 0
    all_fn = 0
    per_record = []

    # Limit to 3 transition records for tractable computation
    transition_data = transition_data[:3]

    for td_info in transition_data:
        signal = td_info["signal"]
        onset_s = td_info["onset_time_s"]
        rec = td_info["record"]
        print(f"\n  Record {rec}: onset at {onset_s:.1f}s, signal length={len(signal)/TARGET_FS:.1f}s")

        # Estimate embedding params from the segment
        try:
            emb_probe = TakensEmbedder(delay="auto", dimension="auto")
            emb_probe.fit_transform(signal)
            e_delay = emb_probe.delay_ if hasattr(emb_probe, "delay_") else 8
            e_dim = emb_probe.dimension_ if hasattr(emb_probe, "dimension_") else 5
        except Exception:
            e_delay, e_dim = 8, 5

        # Window: 10s, step: 5s (larger step for tractable PH computation)
        window_size = 10 * TARGET_FS  # 1280
        step_size = 5 * TARGET_FS     # 640

        try:
            td = TransitionDetector(
                window_size=window_size,
                step_size=step_size,
                max_dim=1,
                subsample=min(subsample, 200),  # cap subsample for speed
            )
            result = td.fit_transform(signal, seed=seed, embedding_dim=e_dim, embedding_delay=e_delay)
        except Exception as e:
            print(f"  [WARN] TransitionDetector failed: {e}")
            per_record.append({"record": rec, "error": str(e)})
            continue

        scores = result["transition_scores"]
        centers = result["window_centers"]

        # transition_scores has N-1 elements (distances between consecutive windows)
        if len(scores) < len(centers):
            score_times = ((centers[:-1] + centers[1:]) / 2.0) / TARGET_FS
        else:
            score_times = centers / TARGET_FS

        if len(scores) == 0:
            print(f"  [WARN] No transition scores computed for record {rec}")
            per_record.append({"record": rec, "error": "no scores"})
            continue

        # Detect peaks: scores above 75th percentile
        threshold = np.percentile(scores, 75)
        peak_mask = scores > threshold
        peak_times = score_times[peak_mask]

        # Define onset window: ±15s around true onset
        onset_window = 15.0
        detected_onset = False
        lag = float("nan")

        for pt in peak_times:
            if abs(pt - onset_s) <= onset_window:
                detected_onset = True
                lag = pt - onset_s  # positive = detected after onset
                all_lags.append(lag)
                break

        # Count normal-region false positives
        # Normal region: first half before onset minus a buffer
        normal_end = max(0, onset_s - 30)
        fp_peaks = sum(1 for pt in peak_times if pt < normal_end)

        tp = 1 if detected_onset else 0
        fn = 0 if detected_onset else 1
        all_tp += tp
        all_fn += fn
        all_fp += fp_peaks

        print(f"  Embedding: delay={e_delay}, dim={e_dim}")
        print(f"  Scores: {len(scores)} values, range=[{scores.min():.2f}, {scores.max():.2f}]")
        print(f"  Threshold (75th): {threshold:.2f}")
        print(f"  Onset detected: {detected_onset}" + (f" (lag={lag:.1f}s)" if detected_onset else ""))
        print(f"  False positives in normal region: {fp_peaks}")

        per_record.append({
            "record": rec,
            "onset_s": round(onset_s, 1),
            "detected": detected_onset,
            "lag_s": round(lag, 2) if not np.isnan(lag) else None,
            "n_scores": len(scores),
            "score_range": [round(float(scores.min()), 2), round(float(scores.max()), 2)],
            "threshold": round(float(threshold), 2),
            "fp_in_normal": fp_peaks,
            "scores": scores.tolist(),
            "score_times": score_times.tolist(),
        })

    # Aggregate
    mean_lag = float(np.nanmean(all_lags)) if all_lags else float("nan")
    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)

    print(f"\n  Summary: TP={all_tp}, FP={all_fp}, FN={all_fn}")
    print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}")
    print(f"  Mean detection lag: {mean_lag:.1f}s" if not np.isnan(mean_lag) else "  No detections")

    return {
        "n_transitions": len(transition_data),
        "n_analyzed": len(per_record),
        "detection_lag_seconds": round(mean_lag, 2) if not np.isnan(mean_lag) else None,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "per_record": per_record,
    }


# ── Experiment 3: RR-Interval Dynamics ───────────────────────────────────────

def run_exp3(normal_rr_all: np.ndarray, arrhythmia_rr_all: np.ndarray,
             subsample: int = 500, seed: int = 42) -> dict:
    """Compare HRV attractor topology: normal vs arrhythmia RR intervals."""
    print("\n═══ Experiment 3: RR-Interval Dynamics ═══")
    print(f"  Normal RR intervals: {len(normal_rr_all)}")
    print(f"  Arrhythmia RR intervals: {len(arrhythmia_rr_all)}")

    if len(normal_rr_all) < 30 or len(arrhythmia_rr_all) < 30:
        print("  [WARN] Insufficient RR intervals")
        return {
            "normal_rr_n": len(normal_rr_all),
            "arrhythmia_rr_n": len(arrhythmia_rr_all),
            "normal_rr_h1_features": 0,
            "arrhythmia_rr_h1_features": 0,
        }

    # Use smaller delay/dim for RR (shorter series, different dynamics)
    # RR intervals are ~1 per heartbeat, so auto parameters may differ
    normal_cloud, normal_ph, normal_emb, _ = embed_and_ph(
        normal_rr_all, delay="auto", dimension="auto", max_dim=1,
        subsample=subsample, seed=seed, fallback_delay=3, fallback_dim=4
    )
    arr_cloud, arr_ph, arr_emb, _ = embed_and_ph(
        arrhythmia_rr_all, delay="auto", dimension="auto", max_dim=1,
        subsample=subsample, seed=seed, fallback_delay=3, fallback_dim=4
    )

    if normal_ph is None or arr_ph is None:
        print("  [WARN] PH computation failed")
        return {
            "normal_rr_n": len(normal_rr_all),
            "arrhythmia_rr_n": len(arrhythmia_rr_all),
            "normal_rr_h1_features": 0,
            "arrhythmia_rr_h1_features": 0,
        }

    n_h1_normal = count_features(normal_ph["diagrams"], 1)
    n_h1_arr = count_features(arr_ph["diagrams"], 1)
    ent_normal = normal_ph["persistence_entropy"][1] if len(normal_ph["persistence_entropy"]) > 1 else 0.0
    ent_arr = arr_ph["persistence_entropy"][1] if len(arr_ph["persistence_entropy"]) > 1 else 0.0

    normal_delay = normal_emb.delay_ if hasattr(normal_emb, "delay_") else "?"
    normal_dim = normal_emb.dimension_ if hasattr(normal_emb, "dimension_") else "?"
    arr_delay = arr_emb.delay_ if hasattr(arr_emb, "delay_") else "?"
    arr_dim = arr_emb.dimension_ if hasattr(arr_emb, "dimension_") else "?"

    print(f"  Normal RR:     cloud={normal_cloud.shape}, H1={n_h1_normal}, entropy={ent_normal:.3f} "
          f"(τ={normal_delay}, d={normal_dim})")
    print(f"  Arrhythmia RR: cloud={arr_cloud.shape}, H1={n_h1_arr}, entropy={ent_arr:.3f} "
          f"(τ={arr_delay}, d={arr_dim})")

    # Wasserstein on H1 lifetimes
    w = wasserstein_1d(normal_ph["diagrams"], arr_ph["diagrams"], dim=1)
    print(f"  Wasserstein (H1): {w:.4f}")

    return {
        "normal_rr_n": len(normal_rr_all),
        "arrhythmia_rr_n": len(arrhythmia_rr_all),
        "normal_rr_h1_features": int(n_h1_normal),
        "normal_rr_h1_entropy": round(float(ent_normal), 4),
        "arrhythmia_rr_h1_features": int(n_h1_arr),
        "arrhythmia_rr_h1_entropy": round(float(ent_arr), 4),
        "rr_wasserstein": round(float(w), 4),
        "normal_rr_cloud_shape": list(normal_cloud.shape),
        "arrhythmia_rr_cloud_shape": list(arr_cloud.shape),
        "normal_rr_embedding": {"delay": int(normal_delay) if isinstance(normal_delay, (int, np.integer)) else None,
                                 "dimension": int(normal_dim) if isinstance(normal_dim, (int, np.integer)) else None},
        "arrhythmia_rr_embedding": {"delay": int(arr_delay) if isinstance(arr_delay, (int, np.integer)) else None,
                                     "dimension": int(arr_dim) if isinstance(arr_dim, (int, np.integer)) else None},
        "normal_ph": normal_ph,
        "arrhythmia_ph": arr_ph,
    }


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_exp1(exp1: dict, fig_dir: Path):
    """Exp 1: Normal vs arrhythmia attractor comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: H1 feature counts
    ax = axes[0, 0]
    normal_f = exp1["normal_h1_features"]
    arr_f = exp1["arrhythmia_h1_features"]
    if normal_f and arr_f:
        ax.boxplot([normal_f, arr_f], tick_labels=["Normal", "Arrhythmia"])
        ax.set_ylabel("H1 Features")
        ax.set_title("H1 Feature Count by Rhythm Type")

    # Panel 2: H1 entropy
    ax = axes[0, 1]
    normal_e = exp1.get("normal_h1_entropy_list", [])
    arr_e = exp1.get("arrhythmia_h1_entropy_list", [])
    if normal_e and arr_e:
        ax.boxplot([normal_e, arr_e], tick_labels=["Normal", "Arrhythmia"])
        ax.set_ylabel("H1 Persistence Entropy")
        ax.set_title("H1 Entropy by Rhythm Type")

    # Panel 3: Permutation null distribution
    ax = axes[1, 0]
    null = exp1.get("null_distribution", [])
    obs = exp1.get("wasserstein_observed", 0)
    if null:
        ax.hist(null, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(obs, color="red", linewidth=2, label=f"Observed={obs:.4f}")
        ax.set_xlabel("Wasserstein Distance (H1)")
        ax.set_ylabel("Count")
        ax.set_title(f"Permutation Test (p={exp1['wasserstein_p']:.4f}, z={exp1['wasserstein_z']:.1f})")
        ax.legend()

    # Panel 4: Example persistence diagrams
    ax = axes[1, 1]
    normal_dgms = exp1.get("normal_diagrams", [])
    arr_dgms = exp1.get("arrhythmia_diagrams", [])
    if normal_dgms and arr_dgms:
        # Plot first normal and first arrhythmia H1 diagram
        dgm_n = normal_dgms[0][1] if len(normal_dgms[0]) > 1 else np.array([])
        dgm_a = arr_dgms[0][1] if len(arr_dgms[0]) > 1 else np.array([])
        if len(dgm_n) > 0:
            finite_n = dgm_n[np.isfinite(dgm_n[:, 1])]
            if len(finite_n) > 0:
                ax.scatter(finite_n[:, 0], finite_n[:, 1], alpha=0.5, s=20, label="Normal", color="blue")
        if len(dgm_a) > 0:
            finite_a = dgm_a[np.isfinite(dgm_a[:, 1])]
            if len(finite_a) > 0:
                ax.scatter(finite_a[:, 0], finite_a[:, 1], alpha=0.5, s=20, label="Arrhythmia", color="red")
        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title("H1 Persistence Diagrams (Example)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "exp1_normal_vs_arrhythmia.png", dpi=150)
    plt.close()
    print(f"  Saved {fig_dir / 'exp1_normal_vs_arrhythmia.png'}")


def plot_exp2(exp2: dict, fig_dir: Path):
    """Exp 2: Transition detection."""
    per_record = exp2.get("per_record", [])
    valid = [r for r in per_record if "scores" in r]

    if not valid:
        print("  [WARN] No transition data to plot")
        return

    n = len(valid)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)

    for i, rec_data in enumerate(valid):
        ax = axes[i, 0]
        times = rec_data["score_times"]
        scores = rec_data["scores"]
        onset = rec_data["onset_s"]
        threshold = rec_data["threshold"]

        ax.plot(times, scores, "b-", alpha=0.7, linewidth=1)
        ax.axhline(threshold, color="orange", linestyle="--", alpha=0.5, label=f"75th pctl={threshold:.1f}")
        ax.axvline(onset, color="red", linewidth=2, label=f"Arrhythmia onset ({onset:.0f}s)")

        # Highlight detected peaks
        peak_mask = np.array(scores) > threshold
        peak_t = np.array(times)[peak_mask]
        peak_s = np.array(scores)[peak_mask]
        ax.scatter(peak_t, peak_s, color="orange", s=15, zorder=5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Transition Score")
        ax.set_title(f"Record {rec_data['record']}: "
                      f"{'DETECTED' if rec_data['detected'] else 'MISSED'}"
                      + (f" (lag={rec_data['lag_s']:.1f}s)" if rec_data.get('lag_s') is not None else ""))
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fig_dir / "exp2_transition_detection.png", dpi=150)
    plt.close()
    print(f"  Saved {fig_dir / 'exp2_transition_detection.png'}")


def plot_exp3(exp3: dict, fig_dir: Path):
    """Exp 3: RR-interval attractor topology."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    normal_ph = exp3.get("normal_ph")
    arr_ph = exp3.get("arrhythmia_ph")

    # Panel 1: H1 feature comparison
    ax = axes[0]
    vals = [exp3.get("normal_rr_h1_features", 0), exp3.get("arrhythmia_rr_h1_features", 0)]
    bars = ax.bar(["Normal", "Arrhythmia"], vals, color=["steelblue", "coral"])
    ax.set_ylabel("H1 Features")
    ax.set_title("RR-Interval H1 Topology")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(v),
                ha="center", va="bottom", fontsize=11)

    # Panel 2: Persistence diagrams
    ax = axes[1]
    if normal_ph is not None and len(normal_ph["diagrams"]) > 1:
        dgm = normal_ph["diagrams"][1]
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.array([])
        if len(finite) > 0:
            ax.scatter(finite[:, 0], finite[:, 1], alpha=0.5, s=20, label="Normal RR", color="blue")
    if arr_ph is not None and len(arr_ph["diagrams"]) > 1:
        dgm = arr_ph["diagrams"][1]
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.array([])
        if len(finite) > 0:
            ax.scatter(finite[:, 0], finite[:, 1], alpha=0.5, s=20, label="Arrhythmia RR", color="red")
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("H1 Persistence: RR Intervals")
    ax.legend()

    # Panel 3: H1 entropy comparison
    ax = axes[2]
    ents = [exp3.get("normal_rr_h1_entropy", 0), exp3.get("arrhythmia_rr_h1_entropy", 0)]
    bars = ax.bar(["Normal", "Arrhythmia"], ents, color=["steelblue", "coral"])
    ax.set_ylabel("H1 Persistence Entropy")
    ax.set_title("RR-Interval H1 Entropy")
    for bar, v in zip(bars, ents):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(fig_dir / "exp3_rr_intervals.png", dpi=150)
    plt.close()
    print(f"  Saved {fig_dir / 'exp3_rr_intervals.png'}")


def plot_overview(exp1: dict, exp2: dict, exp3: dict, fig_dir: Path):
    """4-panel overview figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: H1 features — normal vs arrhythmia (ECG)
    ax = axes[0, 0]
    n_mean = exp1.get("normal_h1_features_mean", 0)
    a_mean = exp1.get("arrhythmia_h1_features_mean", 0)
    n_std = exp1.get("normal_h1_features_std", 0)
    a_std = exp1.get("arrhythmia_h1_features_std", 0)
    bars = ax.bar(["Normal", "Arrhythmia"], [n_mean, a_mean], yerr=[n_std, a_std],
                  color=["steelblue", "coral"], capsize=5)
    ax.set_ylabel("H1 Features (mean)")
    ax.set_title(f"ECG Attractor: p={exp1.get('wasserstein_p', 1.0):.4f}")

    # Panel 2: Transition detection summary
    ax = axes[0, 1]
    tp = exp2.get("tp", 0)
    fp = exp2.get("fp", 0)
    fn = exp2.get("fn", 0)
    ax.bar(["True Pos", "False Pos", "False Neg"], [tp, fp, fn],
           color=["green", "orange", "red"])
    ax.set_title(f"Transition Detection: P={exp2.get('precision', 0):.2f}, R={exp2.get('recall', 0):.2f}")

    # Panel 3: RR-interval H1
    ax = axes[1, 0]
    rr_vals = [exp3.get("normal_rr_h1_features", 0), exp3.get("arrhythmia_rr_h1_features", 0)]
    ax.bar(["Normal RR", "Arrhythmia RR"], rr_vals, color=["steelblue", "coral"])
    ax.set_ylabel("H1 Features")
    ax.set_title(f"RR-Interval Topology (W={exp3.get('rr_wasserstein', 0):.4f})")

    # Panel 4: Entropy comparison (ECG + RR)
    ax = axes[1, 1]
    labels = ["ECG Normal", "ECG Arrhythmia", "RR Normal", "RR Arrhythmia"]
    ents = [
        exp1.get("normal_h1_entropy_mean", 0),
        exp1.get("arrhythmia_h1_entropy_mean", 0),
        exp3.get("normal_rr_h1_entropy", 0),
        exp3.get("arrhythmia_rr_h1_entropy", 0),
    ]
    colors = ["steelblue", "coral", "steelblue", "coral"]
    ax.bar(labels, ents, color=colors)
    ax.set_ylabel("H1 Persistence Entropy")
    ax.set_title("Entropy Summary")
    ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Branch 4: Cardiac Arrhythmia — Topological Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_dir / 'overview.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Branch 4: Cardiac Arrhythmia Topology")
    parser.add_argument("--subsample", type=int, default=500)
    parser.add_argument("--n-perms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-synthetic", action="store_true")
    parser.add_argument("--output-dir", type=str, default="data/cardiac")
    parser.add_argument("--fig-dir", type=str, default="figures/cardiac")
    args = parser.parse_args()

    t0 = time.time()

    out_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Data ────────────────────────────────────────────────────────
    print("Loading MIT-BIH Arrhythmia Database records...")
    all_normal = []
    all_arrhythmia = []
    all_transition = []
    all_normal_rr = []
    all_arrhythmia_rr = []
    records_loaded = 0
    use_synthetic = args.force_synthetic

    if not use_synthetic:
        for rec_num in ARRHYTHMIA_RECORDS:
            print(f"\n  Record {rec_num}...")
            data = load_record(rec_num)
            if data is None:
                print(f"  [WARN] Skipping record {rec_num}")
                continue

            records_loaded += 1
            segs = extract_segments(data)
            all_normal.extend(segs["normal"])
            all_arrhythmia.extend(segs["arrhythmia"])
            all_transition.extend(segs["transition"])

            print(f"    Duration: {data['duration_s']:.0f}s, "
                  f"Normal: {len(segs['normal'])}, Arrhythmia: {len(segs['arrhythmia'])}, "
                  f"Transition: {len(segs['transition'])}")

            # RR intervals
            rr = extract_rr_intervals(data)
            if len(rr["normal_rr"]) > 0:
                all_normal_rr.append(rr["normal_rr"])
            if len(rr["arrhythmia_rr"]) > 0:
                all_arrhythmia_rr.append(rr["arrhythmia_rr"])

            # Find transition onset for Exp 2
            trans = find_transition_segment(data, min_run=5)
            if trans is not None:
                all_transition.append(trans)

        if records_loaded == 0:
            print("\n  [WARN] No records loaded, falling back to synthetic data")
            use_synthetic = True

    if use_synthetic:
        print("  Using synthetic ECG data (fallback)")
        rng = np.random.default_rng(args.seed)
        for i in range(10):
            all_normal.append(synthetic_ecg_segment("normal", seed=args.seed + i))
        for i in range(10):
            all_arrhythmia.append(synthetic_ecg_segment("arrhythmia", seed=args.seed + 100 + i))
        all_normal_rr.append(synthetic_rr_intervals("normal", n=500, seed=args.seed))
        all_arrhythmia_rr.append(synthetic_rr_intervals("arrhythmia", n=500, seed=args.seed + 1))
        records_loaded = 0

    # Concatenate RR intervals
    normal_rr_cat = np.concatenate(all_normal_rr) if all_normal_rr else np.array([])
    arrhythmia_rr_cat = np.concatenate(all_arrhythmia_rr) if all_arrhythmia_rr else np.array([])

    # Cap segments at 15 per type for tractable PH computation
    MAX_SEGMENTS = 15
    if len(all_normal) > MAX_SEGMENTS:
        rng_cap = np.random.default_rng(args.seed)
        idx = rng_cap.choice(len(all_normal), MAX_SEGMENTS, replace=False)
        all_normal = [all_normal[i] for i in sorted(idx)]
    if len(all_arrhythmia) > MAX_SEGMENTS:
        rng_cap = np.random.default_rng(args.seed + 1)
        idx = rng_cap.choice(len(all_arrhythmia), MAX_SEGMENTS, replace=False)
        all_arrhythmia = [all_arrhythmia[i] for i in sorted(idx)]

    print(f"\n  Total: {len(all_normal)} normal segments, {len(all_arrhythmia)} arrhythmia segments")
    print(f"  RR intervals: {len(normal_rr_cat)} normal, {len(arrhythmia_rr_cat)} arrhythmia")

    # ── Run Experiments ──────────────────────────────────────────────────

    exp1 = run_exp1(all_normal, all_arrhythmia,
                    subsample=args.subsample, seed=args.seed, n_perms=args.n_perms)

    # For Exp 2, find transition segments from loaded records
    transition_segments = [t for t in all_transition if isinstance(t, dict) and "signal" in t]
    exp2 = run_exp2(transition_segments, subsample=args.subsample, seed=args.seed)

    exp3 = run_exp3(normal_rr_cat, arrhythmia_rr_cat,
                    subsample=args.subsample, seed=args.seed)

    # ── Save Results ─────────────────────────────────────────────────────
    runtime = time.time() - t0

    results = {
        "branch": "experiment/tda-cardiac",
        "data_source": "synthetic" if use_synthetic else "MIT-BIH",
        "n_records": records_loaded,
        "n_normal_segments": len(all_normal),
        "n_arrhythmia_segments": len(all_arrhythmia),
        "exp1_normal_h1_features_mean": exp1.get("normal_h1_features_mean", 0),
        "exp1_arrhythmia_h1_features_mean": exp1.get("arrhythmia_h1_features_mean", 0),
        "exp1_normal_h1_entropy": exp1.get("normal_h1_entropy_mean", 0),
        "exp1_arrhythmia_h1_entropy": exp1.get("arrhythmia_h1_entropy_mean", 0),
        "exp1_wasserstein_observed": exp1.get("wasserstein_observed", 0),
        "exp1_wasserstein_p": exp1.get("wasserstein_p", 1.0),
        "exp1_wasserstein_z": exp1.get("wasserstein_z", 0.0),
        "exp2_n_transitions": exp2.get("n_transitions", 0),
        "exp2_detection_lag_seconds": exp2.get("detection_lag_seconds"),
        "exp2_precision": exp2.get("precision", 0.0),
        "exp2_recall": exp2.get("recall", 0.0),
        "exp3_normal_rr_h1_features": exp3.get("normal_rr_h1_features", 0),
        "exp3_arrhythmia_rr_h1_features": exp3.get("arrhythmia_rr_h1_features", 0),
        "exp3_normal_rr_h1_entropy": exp3.get("normal_rr_h1_entropy", 0),
        "exp3_arrhythmia_rr_h1_entropy": exp3.get("arrhythmia_rr_h1_entropy", 0),
        "exp3_rr_wasserstein": exp3.get("rr_wasserstein", 0),
        "embedding_params": exp1.get("embedding_params", {}),
        "runtime_seconds": round(runtime, 1),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_dir / 'results.json'}")

    # ── Figures ──────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_exp1(exp1, fig_dir)
    plot_exp2(exp2, fig_dir)
    plot_exp3(exp3, fig_dir)
    plot_overview(exp1, exp2, exp3, fig_dir)

    print(f"\n{'='*60}")
    print(f"Branch 4: Cardiac Arrhythmia Topology — COMPLETE")
    print(f"Runtime: {runtime:.1f}s")
    print(f"Data source: {'synthetic' if use_synthetic else 'MIT-BIH'}")
    print(f"Records: {records_loaded}")
    print(f"Segments: {len(all_normal)} normal, {len(all_arrhythmia)} arrhythmia")
    print(f"Exp1: Wasserstein p={exp1.get('wasserstein_p', '?')}, z={exp1.get('wasserstein_z', '?')}")
    print(f"Exp2: Precision={exp2.get('precision', '?')}, Recall={exp2.get('recall', '?')}")
    print(f"Exp3: RR H1 normal={exp3.get('normal_rr_h1_features', '?')}, "
          f"arrhythmia={exp3.get('arrhythmia_rr_h1_features', '?')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
