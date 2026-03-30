#!/usr/bin/env python3
"""Screen: Does the topological transition detector generalize to sleep stages?

Tests whether ATT's sliding-window PH — validated on binocular rivalry at 94%
precision / 41% recall — can detect sleep stage transitions in PhysioNet
Sleep-EDF data. Uses ground-truth hypnogram annotations as reference.

Protocol:
  1. Download PhysioNet Sleep-EDF (SC) via MNE: subject 0, recording 1
  2. Extract Fpz-Cz channel, bandpass 0.5-30 Hz
  3. Takens embed, parallel sliding-window PH + CUSUM changepoint detection
  4. Compare detected changepoints against hypnogram transitions (±30s tolerance)
  5. Compute precision and recall
  6. Null control: phase-randomize signal, run same pipeline
  7. Fisher exact test: real vs null hit rate

Pass criterion: precision > 50% AND recall > 15% AND Fisher p < 0.05.

Usage:
    python scripts/screen_sleep_transitions.py
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import fisher_exact
from tqdm import tqdm

from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer
from att.surrogates import phase_randomize

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TD_WINDOW_SIZE = 500
TD_STEP_SIZE = 200
TD_MAX_DIM = 1
TD_SUBSAMPLE = 400
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 30.0
BANDPASS_ORDER = 4
TOLERANCE_S = 30.0
MAX_DURATION_S = 1800    # 30 minutes
SEED = 42
CHANNEL_PRIORITY = ["EEG Fpz-Cz", "EEG Pz-Oz"]
N_JOBS = min(16, mp.cpu_count())


# ---------------------------------------------------------------------------
# Data loading via MNE
# ---------------------------------------------------------------------------

def load_sleep_data(subject: int = 0, recording: int = 1):
    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError:
        print("ERROR: MNE not installed. pip install mne")
        sys.exit(1)

    print(f"Fetching Sleep-EDF data: subject={subject}, recording={recording}")
    try:
        paths = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=[subject], recording=[recording],
        )
    except Exception as e:
        print(f"ERROR: Failed to download sleep data: {e}")
        sys.exit(1)

    raw_fname, annot_fname = paths[0]
    raw = mne.io.read_raw_edf(raw_fname, preload=True)
    annots = mne.read_annotations(annot_fname)
    raw.set_annotations(annots)
    return raw, annots


def extract_channel(raw, channel_priority: list[str]) -> tuple[np.ndarray, str, float]:
    available = raw.ch_names
    for ch in channel_priority:
        if ch in available:
            signal = raw.get_data(picks=[ch])[0]
            sfreq = raw.info["sfreq"]
            return signal, ch, sfreq
    raise ValueError(f"None of {channel_priority} found in {available}")


def extract_transitions(annots, sfreq: float, max_sample: int | None = None) -> np.ndarray:
    stage_prefixes = ("Sleep stage",)
    stages = []
    for desc, onset in zip(annots.description, annots.onset):
        if any(desc.startswith(p) for p in stage_prefixes):
            sample = int(onset * sfreq)
            if max_sample is not None and sample >= max_sample:
                break
            stages.append((sample, desc))

    transitions = []
    for i in range(1, len(stages)):
        if stages[i][1] != stages[i - 1][1]:
            transitions.append(stages[i][0])

    return np.array(transitions, dtype=int)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_alignment(
    true_samples: np.ndarray,
    detected_samples: np.ndarray,
    tolerance: int,
) -> dict:
    if len(detected_samples) == 0:
        return {
            "true_positives": 0, "false_positives": 0, "switches_detected": 0,
            "precision": 0.0, "recall": 0.0,
            "n_true": len(true_samples), "n_detected": 0,
        }
    if len(true_samples) == 0:
        return {
            "true_positives": 0, "false_positives": len(detected_samples),
            "switches_detected": 0, "precision": 0.0, "recall": 0.0,
            "n_true": 0, "n_detected": len(detected_samples),
        }

    tp = sum(1 for det in detected_samples
             if any(abs(det - t) <= tolerance for t in true_samples))
    fp = len(detected_samples) - tp

    detected_count = sum(1 for t in true_samples
                         if any(abs(det - t) <= tolerance for det in detected_samples))

    n_det = len(detected_samples)
    n_true = len(true_samples)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "switches_detected": detected_count,
        "precision": tp / n_det if n_det > 0 else 0.0,
        "recall": detected_count / n_true if n_true > 0 else 0.0,
        "n_true": n_true,
        "n_detected": n_det,
    }


# ---------------------------------------------------------------------------
# Parallel sliding-window PH
# ---------------------------------------------------------------------------

def _ph_worker(args):
    """Worker: compute PH on one window cloud."""
    cloud, max_dim, subsample, seed = args
    pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser")
    pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return pa.diagrams_


def _cusum_changepoints(scores: np.ndarray, threshold: float | None = None) -> list[int]:
    """Forward CUSUM changepoint detection."""
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    if threshold is None:
        threshold = mean + 2 * std
    cusum = 0.0
    changepoints = []
    for i, s in enumerate(scores):
        cusum = max(0.0, cusum + (s - mean))
        if cusum > threshold:
            changepoints.append(i)
            cusum = 0.0
    return changepoints


def run_detection(signal: np.ndarray, sfreq: float, label: str = "real") -> tuple[np.ndarray, dict]:
    """Embed + parallel sliding-window PH + CUSUM changepoints."""
    # Bandpass
    sos = butter(BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH],
                 btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfilt(sos, signal)

    # Takens embed
    embedder = TakensEmbedder("auto", "auto")
    embedder.fit(filtered)
    cloud = embedder.transform(filtered)
    delay = embedder.delay_
    dim = embedder.dimension_

    # Window the cloud
    n_points = len(cloud)
    window_starts = list(range(0, n_points - TD_WINDOW_SIZE + 1, TD_STEP_SIZE))
    window_centers = np.array([s + TD_WINDOW_SIZE // 2 for s in window_starts])
    windows = [cloud[s : s + TD_WINDOW_SIZE] for s in window_starts]

    # Parallel PH
    args = [(w, TD_MAX_DIM, TD_SUBSAMPLE, SEED) for w in windows]
    with mp.Pool(N_JOBS) as pool:
        all_diagrams = list(tqdm(
            pool.imap(_ph_worker, args),
            total=len(args),
            desc=f"  PH ({label})",
        ))

    # Shared ranges
    all_births, all_persistences = [], []
    for dgms in all_diagrams:
        for dgm in dgms:
            if len(dgm) > 0:
                all_births.extend(dgm[:, 0].tolist())
                pers = dgm[:, 1] - dgm[:, 0]
                all_persistences.extend(pers[pers > 1e-10].tolist())

    birth_range = (min(all_births), max(all_births)) if all_births else (0, 1)
    persistence_range = (0.0, max(all_persistences)) if all_persistences else (0, 1)

    # Shared images
    shared_images = []
    for dgms in all_diagrams:
        pa = PersistenceAnalyzer(max_dim=TD_MAX_DIM)
        pa.diagrams_ = dgms
        imgs = pa.to_image(birth_range=birth_range, persistence_range=persistence_range)
        shared_images.append(imgs)

    # Consecutive L2 distances
    image_distances = []
    for i in range(len(shared_images) - 1):
        dist = sum(
            float(np.sqrt(np.sum((shared_images[i][d] - shared_images[i + 1][d]) ** 2)))
            for d in range(TD_MAX_DIM + 1)
        )
        image_distances.append(dist)
    image_distances = np.array(image_distances)

    # CUSUM changepoints
    changepoints = _cusum_changepoints(image_distances)

    # Convert to sample positions
    dist_x = (window_centers[:-1] + window_centers[1:]) / 2.0
    detected_samples = np.array(
        [int(dist_x[cp]) for cp in changepoints if cp < len(dist_x)]
    )

    meta = {
        "delay": delay, "dim": dim,
        "n_windows": len(window_centers),
        "n_changepoints": len(changepoints),
    }
    return detected_samples, meta


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results_table(rows: list[dict]) -> None:
    hdr = (f"{'Condition':<14} {'n_true':>6} {'n_det':>6} {'prec':>7} {'recall':>7} "
           f"{'TP':>4} {'FP':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['condition']:<14} {r['n_true']:>6} {r['n_detected']:>6} "
              f"{r['precision']:>7.1%} {r['recall']:>7.1%} "
              f"{r['true_positives']:>4} {r['false_positives']:>4}")


def print_verdict(real: dict, null: dict, fisher_p: float) -> None:
    prec_pass = real["precision"] > 0.50
    recall_pass = real["recall"] > 0.15
    fisher_pass = fisher_p < 0.05
    overall = "PASS" if (prec_pass and recall_pass and fisher_pass) else "FAIL"

    print()
    print("=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"Sleep transition detection: {overall}")
    print(f"  Precision: {real['precision']:.1%} (threshold: >50%) — {'PASS' if prec_pass else 'FAIL'}")
    print(f"  Recall:    {real['recall']:.1%} (threshold: >15%) — {'PASS' if recall_pass else 'FAIL'}")
    print(f"  Fisher p:  {fisher_p:.4f} (threshold: <0.05) — {'PASS' if fisher_pass else 'FAIL'}")
    print(f"  Null precision: {null['precision']:.1%}, null recall: {null['recall']:.1%}")
    print(f"  Reference (rivalry): 94% precision, 41% recall")
    if overall == "PASS":
        print(f"  TransitionDetector generalizes to sleep stage transitions")
    else:
        print(f"  Insufficient evidence for generalization to sleep data")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Screen: sleep transitions")
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--recording", type=int, default=1)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Parallel workers: {N_JOBS}")

    # Load sleep data
    raw, annots = load_sleep_data(args.subject, args.recording)
    signal, ch_name, sfreq = extract_channel(raw, CHANNEL_PRIORITY)
    print(f"Channel: {ch_name}, {len(signal)} samples @ {sfreq} Hz "
          f"({len(signal) / sfreq / 60:.0f} min)")

    # Find segment with transitions (first 30 min may be all-wake)
    all_transitions = extract_transitions(annots, sfreq, max_sample=len(signal))
    print(f"Total stage transitions in recording: {len(all_transitions)}")

    if len(all_transitions) == 0:
        print("ERROR: No sleep stage transitions found.")
        sys.exit(1)

    max_samples = int(MAX_DURATION_S * sfreq)
    first_trans = all_transitions[0]
    start_sample = max(0, first_trans - int(2 * 60 * sfreq))
    end_sample = min(len(signal), start_sample + max_samples)
    signal = signal[start_sample:end_sample]
    transitions = all_transitions[
        (all_transitions >= start_sample) & (all_transitions < end_sample)
    ] - start_sample
    print(f"Analysis window: samples {start_sample}-{end_sample} "
          f"({(end_sample - start_sample) / sfreq / 60:.0f} min)")
    print(f"Stage transitions in window: {len(transitions)}")

    if len(transitions) == 0:
        print("ERROR: No transitions in window.")
        sys.exit(1)

    tolerance_samples = int(TOLERANCE_S * sfreq)

    # Real detection
    print("\nRunning parallel PH on real signal...")
    t1 = time.time()
    detected_real, meta = run_detection(signal, sfreq, label="real")
    print(f"  embed: delay={meta['delay']}, dim={meta['dim']}")
    print(f"  windows: {meta['n_windows']}, changepoints: {meta['n_changepoints']}")
    print(f"  Time: {time.time() - t1:.1f}s")

    real_eval = evaluate_alignment(transitions, detected_real, tolerance_samples)
    real_eval["condition"] = "real"

    # Null control: phase-randomized signal
    print("\nRunning parallel PH on phase-randomized null...")
    t2 = time.time()
    surr = phase_randomize(signal, n_surrogates=1, seed=SEED)[0]
    detected_null, meta_null = run_detection(surr, sfreq, label="null")
    print(f"  windows: {meta_null['n_windows']}, changepoints: {meta_null['n_changepoints']}")
    print(f"  Time: {time.time() - t2:.1f}s")

    null_eval = evaluate_alignment(transitions, detected_null, tolerance_samples)
    null_eval["condition"] = "null (phase)"

    # Fisher exact test
    table = [
        [real_eval["true_positives"], real_eval["false_positives"]],
        [null_eval["true_positives"], null_eval["false_positives"]],
    ]
    if sum(table[0]) == 0 and sum(table[1]) == 0:
        fisher_p = 1.0
    else:
        _, fisher_p = fisher_exact(table, alternative="greater")

    # Output
    print("\n")
    print_results_table([real_eval, null_eval])
    print_verdict(real_eval, null_eval, fisher_p)

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    passed = (real_eval["precision"] > 0.50
              and real_eval["recall"] > 0.15
              and fisher_p < 0.05)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
