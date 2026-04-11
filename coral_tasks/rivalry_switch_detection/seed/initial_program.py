"""Rivalry switch detection — seed program (PI-L2 + CUSUM baseline).

Loads 3 binocular rivalry EEG subjects from the Nie/Katyal/Engel (2023)
dataset, runs the existing TransitionDetector pipeline, evaluates against
behavioral button-press switches at ±5s tolerance, and reports mean F1.

Background
==========
The published baseline (`scripts/batch_eeg.py` on `experiment/tda-sleep`)
achieves precision 94.1%, recall 40.6% (mean across N=80 subjects, ±5s).
Per-subject F1 ranges from ~0.30 to ~0.75. Mean F1 ≈ 0.57.

The recall gap is the dominant problem: precision is already publication-grade
but the detector misses most behavioral switches. Improving recall while
keeping precision high would lift this from "interesting" to "deployable".

YOUR JOB
========
Edit `initial_program.py` to maximize mean F1 across the 3 subjects
{3629, 3630, 3632}. The grader requires:
  - All 3 subjects present in per_subject
  - tolerance_s = 5.0
  - F1, precision, recall all in [0, 1]

Strategies to consider:
  - Different topological features: persistence entropy, max H1 lifetime,
    Betti curve shape distance, persistence landscape L1
  - Different window/step sizes (current: 500/200 samples = 1.4s/0.55s @ 360Hz)
  - Different change-detection statistics: BOCPD, EWMA, ADWIN, page hinkley
  - Lower CUSUM threshold (current: mean+2σ from baseline) → trade precision
    for recall
  - Band-specific topology: filter to alpha-only or theta-only or
    gamma-only before Takens embedding
  - Multi-channel: use Pz, P3, P4, or O1/O2 in addition to Oz
  - **Hybrid spectral-topological**: compute alpha-band power as a
    fast pre-filter, only run PH on flagged windows. The Künneth-aware
    interpretation is that perceptual switches modulate occipital α power
    (well-known) — topology may help distinguish *which* switches matter.
  - Cross-channel binding: use BindingDetector on Oz/Pz pairs as a
    secondary signal that the existing baseline doesn't use
  - Adaptive thresholds per subject

Output format (LAST stdout line MUST be a single JSON object):
  {
    "score": 0.62,
    "per_subject": [
        {"subject": "3629", "f1": 0.71, "precision": 0.95, "recall": 0.57,
         "n_changepoints": 12, "n_switches": 30},
        {"subject": "3630", "f1": ..., ...},
        {"subject": "3632", "f1": ..., ...}
    ],
    "strategy": "ph_pi_l2_cusum",
    "tolerance_s": 5.0
  }

Constraints
===========
  - Don't touch tolerance_s. The 5s value matches the published baseline
    and the grader enforces it.
  - The data is large (~1.4 GB across the 3 subjects) and lives at
    /home/musicofhel/att-docs/data/eeg/rivalry_ssvep/. The path is hardcoded
    below — don't relocate the data.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import scipy.io
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")

from att.embedding.takens import TakensEmbedder  # noqa: E402
from att.neuro.embedding import embed_channel  # noqa: E402
from att.transitions import TransitionDetector  # noqa: E402

DATA_ROOT = "/home/musicofhel/att-docs/data/eeg/rivalry_ssvep"

# (subject_id, directory_name) — these dirs hold preprocessed Epochs/ + Behavior/
SUBJECTS = [
    ("3629", "Sucharit - 012516_3629"),
    ("3630", "Sucharit - 012716_3630"),
    ("3632", "Sucharit - 012816_3632"),
]

SFREQ = 360
TRANSITION_CHANNEL = "Oz"
BANDPASS_LOW = 4
BANDPASS_HIGH = 13
EPOCH_FILE = "csd_rejevs_icacomprem_gaprem_filt_rivindiff_riv_12.mat"
PARAM_SET = 2
EPOCH_INDEX = 0

# Detector parameters (matched to published baseline)
TD_WINDOW_SIZE = 500
TD_STEP_SIZE = 200
TD_MAX_DIM = 1
TD_SUBSAMPLE = 300
TOLERANCE_S = 5.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rivalry_epoch(subject_dir: str) -> tuple[np.ndarray, list[str]]:
    """Load a single rivalry epoch. Returns (signal[n_ch, n_samp], ch_names)."""
    mat_path = os.path.join(DATA_ROOT, subject_dir, "Epochs", EPOCH_FILE)
    eeg = scipy.io.loadmat(mat_path, simplify_cells=True)
    ch_names = [ch["labels"] for ch in eeg["chanlocs"]]
    epochs = eeg["epochs"]
    if isinstance(epochs, np.ndarray) and epochs.ndim == 2:
        data = epochs
    else:
        data = epochs[EPOCH_INDEX]
    return data.astype(np.float64), ch_names


def load_behavioral_switches(subject_id: str, subject_dir: str) -> list[float]:
    """Return behavioral switch times in seconds for the rivalry session.

    Excludes practice files (BR_Rivalry_*_PRACT_*.mat).
    """
    bdir = os.path.join(DATA_ROOT, subject_dir, "Behavior")
    candidates = [
        f
        for f in os.listdir(bdir)
        if f.startswith("BR_Rivalry_") and "PRACT" not in f and f.endswith(".mat")
    ]
    if not candidates:
        return []
    beh = scipy.io.loadmat(os.path.join(bdir, candidates[0]), simplify_cells=True)
    results_beh = beh["results"]
    if not isinstance(results_beh, (list, np.ndarray)):
        return []

    matching = []
    for r in results_beh:
        try:
            if r["params"]["paramSet"] == PARAM_SET:
                matching.append(r)
        except (KeyError, TypeError):
            continue
    if not matching:
        return []

    r = matching[0]
    psycho = r["psycho"]
    t_key = psycho["tKeyPress"]
    keys = psycho["responseKey"]
    out = []
    for i in range(1, len(keys)):
        if keys[i] != keys[i - 1]:
            out.append(float(t_key[i]))
    return out


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def bandpass(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfilt(sos, signal)


def detect_changepoints(channel_signal: np.ndarray, seed: int = 42) -> np.ndarray:
    """Returns array of detected changepoint times in seconds."""
    cloud, _ = embed_channel(
        channel_signal, band="theta_alpha", sfreq=float(SFREQ),
        condition_threshold=1e4,
    )
    det = TransitionDetector(
        window_size=TD_WINDOW_SIZE,
        step_size=TD_STEP_SIZE,
        max_dim=TD_MAX_DIM,
        backend="ripser",
        subsample=TD_SUBSAMPLE,
    )
    result = det.fit_transform(cloud, seed=seed)
    cps = det.detect_changepoints(method="cusum")
    centers = result["window_centers"]
    dist_x = (centers[:-1] + centers[1:]) / 2.0
    cp_samples = np.array([dist_x[c] for c in cps if c < len(dist_x)])
    return cp_samples / SFREQ


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_alignment(
    cp_times: np.ndarray, switch_times: np.ndarray, tolerance_s: float,
) -> dict:
    if len(cp_times) == 0:
        return {
            "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "n_changepoints": 0, "n_switches": int(len(switch_times)),
        }
    if len(switch_times) == 0:
        return {
            "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "n_changepoints": int(len(cp_times)), "n_switches": 0,
        }
    # Recall: switches with a nearby changepoint
    sw_detected = 0
    for sw in switch_times:
        if np.any(np.abs(cp_times - sw) < tolerance_s):
            sw_detected += 1
    # Precision: changepoints with a nearby switch
    tp = 0
    for cp in cp_times:
        if np.any(np.abs(switch_times - cp) < tolerance_s):
            tp += 1
    precision = tp / len(cp_times) if len(cp_times) > 0 else 0.0
    recall = sw_detected / len(switch_times) if len(switch_times) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "n_changepoints": int(len(cp_times)),
        "n_switches": int(len(switch_times)),
    }


def main():
    per_subject = []
    for subj_id, subj_dir in SUBJECTS:
        print(f"loading subject={subj_id}", file=sys.stderr)
        try:
            signal, ch_names = load_rivalry_epoch(subj_dir)
        except Exception as e:
            print(f"FAIL load epoch {subj_id}: {e}", file=sys.stderr)
            return

        if TRANSITION_CHANNEL not in ch_names:
            print(f"FAIL channel {TRANSITION_CHANNEL} not in {subj_id}", file=sys.stderr)
            return
        ch_idx = ch_names.index(TRANSITION_CHANNEL)
        sig = signal[ch_idx]
        sig = bandpass(sig, BANDPASS_LOW, BANDPASS_HIGH, SFREQ)
        print(f"  signal shape={sig.shape}", file=sys.stderr)

        switches = load_behavioral_switches(subj_id, subj_dir)
        print(f"  n_behavioral_switches={len(switches)}", file=sys.stderr)

        cps = detect_changepoints(sig, seed=42)
        print(f"  n_changepoints={len(cps)}", file=sys.stderr)

        scores = evaluate_alignment(cps, np.array(switches), TOLERANCE_S)
        scores["subject"] = subj_id
        per_subject.append(scores)
        print(
            f"  subject={subj_id} F1={scores['f1']:.3f} "
            f"P={scores['precision']:.3f} R={scores['recall']:.3f}",
            file=sys.stderr,
        )

    f1s = [s["f1"] for s in per_subject]
    mean_f1 = float(sum(f1s) / len(f1s))

    payload = {
        "score": mean_f1,
        "per_subject": per_subject,
        "strategy": "ph_pi_l2_cusum",
        "tolerance_s": TOLERANCE_S,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
