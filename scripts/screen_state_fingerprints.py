#!/usr/bin/env python3
"""Screen: Do perceptual states produce different attractor topologies?

Tests whether stable-percept windows and transition windows have measurably
different persistence images (PI). Uses sliding-window PH on Takens-embedded
EEG from the Oz channel, labels windows by proximity to behavioral switch
events, then runs a permutation test and leave-one-out classification.

Protocol:
  1. Load rivalry EEG (Oz channel) + behavioral switch events
  2. Bandpass 4-13 Hz, Takens embed (auto params)
  3. Sliding-window PH (500 pts, step 250, max_dim=1, subsample=400)
     — parallelized across 28 cores
  4. Label: "stable" (>2s from switch) vs "transition" (within 1s of switch)
  5. Mean PI per class, L2 distance, permutation test (200 shuffles)
  6. LOO logistic regression on flattened PI vectors
  7. Null control: split stable windows into halves — their L2 should be smaller

Pass criterion: permutation p < 0.05 AND classification accuracy > 60%
for at least 2/3 subjects.

Usage:
    python scripts/screen_state_fingerprints.py
    python scripts/screen_state_fingerprints.py --data-dir data/eeg/rivalry_ssvep
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import butter, sosfilt
from tqdm import tqdm

from att.embedding.takens import TakensEmbedder
from att.neuro.embedding import embed_channel
from att.topology.persistence import PersistenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 500
STEP_SIZE = 250
SUBSAMPLE = 400
MAX_DIM = 1
BANDPASS_LOW = 4
BANDPASS_HIGH = 13
BANDPASS_ORDER = 4
STABLE_MARGIN_S = 2.0
TRANS_MARGIN_S = 1.0
N_PERMUTATIONS = 200
SEED = 42
MAX_SUBJECTS = 3
CONDITION_SUFFIX = "riv_12"
PARAM_SET = 2
EPOCH_INDEX = 0
DEFAULT_DATA_DIR = Path("data/eeg/rivalry_ssvep")
N_JOBS = min(16, mp.cpu_count())  # cap at 16 to avoid memory pressure


# ---------------------------------------------------------------------------
# Data loading (from batch_eeg.py patterns)
# ---------------------------------------------------------------------------

def discover_subjects(data_dir: Path) -> list[dict]:
    subjects = []
    if not data_dir.exists():
        return subjects
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir():
            continue
        epochs_dir = p / "Epochs"
        behavior_dir = p / "Behavior"
        if epochs_dir.exists():
            subjects.append({
                "name": p.name,
                "path": p,
                "epochs_dir": epochs_dir,
                "behavior_dir": behavior_dir,
            })
    return subjects


def load_rivalry_epoch(
    epochs_dir: Path, condition_suffix: str, epoch_index: int = 0,
) -> tuple[np.ndarray, list[str], int] | None:
    prefix = "csd_rejevs_icacomprem_gaprem_filt_rivindiff_"
    mat_path = epochs_dir / f"{prefix}{condition_suffix}.mat"
    if not mat_path.exists():
        return None
    eeg = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    ch_names = [ch["labels"] for ch in eeg["chanlocs"]]
    sfreq = int(eeg["fs"])
    epochs = eeg["epochs"]
    if isinstance(epochs, np.ndarray) and epochs.ndim == 2:
        epoch_data = epochs
    elif isinstance(epochs, (list, np.ndarray)):
        if epoch_index >= len(epochs):
            return None
        epoch_data = epochs[epoch_index]
    else:
        return None
    return epoch_data.astype(np.float64), ch_names, sfreq


def load_behavioral_switches(
    behavior_dir: Path, param_set: int, sfreq: int,
) -> list[dict] | None:
    if not behavior_dir.exists():
        return None
    beh_files = [f for f in behavior_dir.glob("BR_Rivalry_*.mat")
                 if "PRACT" not in f.name]
    if not beh_files:
        return None
    try:
        beh = scipy.io.loadmat(str(beh_files[0]), simplify_cells=True)
        results_beh = beh["results"]
    except Exception:
        return None
    matching = []
    for i, r in enumerate(results_beh):
        try:
            if r["params"]["paramSet"] == param_set:
                matching.append(i)
        except (KeyError, TypeError):
            continue
    if not matching:
        return None
    r = results_beh[matching[0]]
    psycho = r["psycho"]
    t_key = psycho["tKeyPress"]
    resp_key = psycho["responseKey"]
    switches = []
    for i in range(1, len(resp_key)):
        if resp_key[i] != resp_key[i - 1]:
            switches.append({
                "time": float(t_key[i]),
                "sample": int(t_key[i] * sfreq),
            })
    return switches


def bandpass_filter(signal, low, high, sfreq, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, signal)


def find_channel(ch_names, target="Oz"):
    for i, name in enumerate(ch_names):
        if name == target:
            return i
    for i, name in enumerate(ch_names):
        if name.startswith(target):
            return i
    return None


# ---------------------------------------------------------------------------
# Parallel sliding-window PH (replaces sequential TransitionDetector)
# ---------------------------------------------------------------------------

def _ph_worker(args):
    """Worker: embed one window + compute PH. Called in pool."""
    cloud, max_dim, subsample, seed = args
    pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser")
    pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return pa.diagrams_


def parallel_windowed_ph(
    signal: np.ndarray,
    window_size: int,
    step_size: int,
    max_dim: int,
    subsample: int,
    seed: int,
    embedding_delay: int,
    embedding_dim: int,
    n_jobs: int = N_JOBS,
) -> dict:
    """Sliding-window PH with parallel ripser calls.

    Returns dict with shared_images, image_distances, window_centers.
    """
    n_samples = len(signal)
    window_starts = list(range(0, n_samples - window_size + 1, step_size))
    window_centers = np.array([s + window_size // 2 for s in window_starts])

    # Embed each window (fast, sequential — just array slicing)
    embedder = TakensEmbedder(delay=embedding_delay, dimension=embedding_dim)
    clouds = []
    for s in window_starts:
        embedder.fit(signal[s : s + window_size])
        clouds.append(embedder.transform(signal[s : s + window_size]))

    # Parallel PH computation
    args = [(c, max_dim, subsample, seed) for c in clouds]
    with mp.Pool(n_jobs) as pool:
        all_diagrams = list(tqdm(
            pool.imap(_ph_worker, args),
            total=len(args),
            desc="  PH windows",
        ))

    # Shared birth/persistence ranges across all windows
    all_births = []
    all_persistences = []
    for dgms in all_diagrams:
        for dgm in dgms:
            if len(dgm) > 0:
                all_births.extend(dgm[:, 0].tolist())
                pers = dgm[:, 1] - dgm[:, 0]
                all_persistences.extend(pers[pers > 1e-10].tolist())

    if all_births and all_persistences:
        birth_range = (min(all_births), max(all_births))
        persistence_range = (0.0, max(all_persistences))
    else:
        birth_range = (0.0, 1.0)
        persistence_range = (0.0, 1.0)

    # Recompute images on shared grid at reduced resolution
    # 20×20 not 50×50 — keeps LOO logistic regression tractable (800 vs 5000 features)
    PI_RES = 20
    shared_images = []
    for dgms in all_diagrams:
        pa = PersistenceAnalyzer(max_dim=max_dim)
        pa.diagrams_ = dgms
        imgs = pa.to_image(
            resolution=PI_RES,
            birth_range=birth_range,
            persistence_range=persistence_range,
        )
        shared_images.append(imgs)

    # Consecutive L2 image distances
    image_distances = []
    for i in range(len(shared_images) - 1):
        dist = sum(
            float(np.sqrt(np.sum((shared_images[i][d] - shared_images[i + 1][d]) ** 2)))
            for d in range(max_dim + 1)
        )
        image_distances.append(dist)

    return {
        "shared_images": shared_images,
        "image_distances": np.array(image_distances),
        "window_centers": window_centers,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_fingerprint(signal: np.ndarray, switch_samples: list[int], sfreq: int) -> dict:
    """Run state fingerprint analysis on one subject's channel signal."""
    # Bandpass
    filtered = bandpass_filter(signal, BANDPASS_LOW, BANDPASS_HIGH, sfreq, BANDPASS_ORDER)

    # Estimate embedding params from full signal
    cloud, meta = embed_channel(filtered, band="theta_alpha", sfreq=float(sfreq))
    delay = meta["delay"]
    dim = meta["dimension"]
    print(f"  embed: delay={delay}, dim={dim}, method={meta['method']}")

    # Parallel sliding-window PH
    result = parallel_windowed_ph(
        filtered, WINDOW_SIZE, STEP_SIZE, MAX_DIM, SUBSAMPLE, SEED,
        embedding_delay=delay, embedding_dim=dim,
    )

    shared_images = result["shared_images"]
    window_centers = result["window_centers"]

    # Flatten per-window PIs to vectors
    pi_vectors = np.array([
        np.concatenate([img.ravel() for img in imgs])
        for imgs in shared_images
    ])

    # Label windows by proximity to switch events
    stable_margin = int(STABLE_MARGIN_S * sfreq)
    trans_margin = int(TRANS_MARGIN_S * sfreq)

    stable_idxs = []
    trans_idxs = []
    for i, wc in enumerate(window_centers):
        if len(switch_samples) == 0:
            stable_idxs.append(i)
            continue
        min_dist = min(abs(int(wc) - s) for s in switch_samples)
        if min_dist <= trans_margin:
            trans_idxs.append(i)
        elif min_dist >= stable_margin:
            stable_idxs.append(i)

    n_stable = len(stable_idxs)
    n_trans = len(trans_idxs)

    if n_trans < 3:
        return {"error": f"too few transition windows ({n_trans})", "n_stable": n_stable, "n_trans": n_trans}
    if n_stable < 3:
        return {"error": f"too few stable windows ({n_stable})", "n_stable": n_stable, "n_trans": n_trans}

    stable_pis = pi_vectors[stable_idxs]
    trans_pis = pi_vectors[trans_idxs]

    # Mean PI per class + L2 distance
    mean_stable = stable_pis.mean(axis=0)
    mean_trans = trans_pis.mean(axis=0)
    observed_l2 = float(np.linalg.norm(mean_stable - mean_trans))

    # Null control: split stable windows into halves
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_stable)
    half = n_stable // 2
    null_a = stable_pis[perm[:half]].mean(axis=0)
    null_b = stable_pis[perm[half : 2 * half]].mean(axis=0)
    null_l2 = float(np.linalg.norm(null_a - null_b))

    # Permutation test: shuffle labels 200 times
    all_idxs = stable_idxs + trans_idxs
    all_pis = pi_vectors[all_idxs]
    all_labels = np.array([0] * n_stable + [1] * n_trans)

    n_exceed = 0
    for _ in range(N_PERMUTATIONS):
        perm_labels = rng.permutation(all_labels)
        m0 = all_pis[perm_labels == 0].mean(axis=0)
        m1 = all_pis[perm_labels == 1].mean(axis=0)
        if np.linalg.norm(m0 - m1) >= observed_l2:
            n_exceed += 1
    perm_p = (n_exceed + 1) / (N_PERMUTATIONS + 1)

    # LOO classification
    accuracy = _loo_classify(all_pis, all_labels)
    chance = max(n_stable, n_trans) / (n_stable + n_trans)

    return {
        "n_stable": n_stable,
        "n_trans": n_trans,
        "n_windows": len(window_centers),
        "l2_distance": observed_l2,
        "null_l2": null_l2,
        "perm_p": perm_p,
        "accuracy": accuracy,
        "chance": chance,
        "embedding_delay": delay,
        "embedding_dim": dim,
    }


def _loo_classify(pis: np.ndarray, labels: np.ndarray) -> float:
    """LOO classification: logistic regression via cross_val_score, fallback to centroid."""
    n = len(pis)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut, cross_val_score

        clf = LogisticRegression(max_iter=500, C=0.1, solver="lbfgs")
        scores = cross_val_score(clf, pis, labels, cv=LeaveOneOut(), n_jobs=-1)
        return float(scores.mean())
    except ImportError:
        pass

    # Fallback: nearest centroid (no sklearn needed)
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        train_pis = pis[mask]
        train_labels = labels[mask]
        c0 = train_pis[train_labels == 0].mean(axis=0)
        c1 = train_pis[train_labels == 1].mean(axis=0)
        d0 = np.linalg.norm(pis[i] - c0)
        d1 = np.linalg.norm(pis[i] - c1)
        pred = 0 if d0 < d1 else 1
        if pred == labels[i]:
            correct += 1
    return correct / n


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results_table(rows: list[dict]) -> None:
    hdr = (f"{'Subject':<12} {'n_stbl':>6} {'n_trns':>6} {'L2_dist':>9} "
           f"{'null_L2':>9} {'perm_p':>8} {'acc':>6} {'chance':>6} {'verdict':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['subject']:<12} {r.get('n_stable', 0):>6} {r.get('n_trans', 0):>6} "
                  f"{'---':>9} {'---':>9} {'---':>8} {'---':>6} {'---':>6} {'SKIP':>7}")
            continue
        passed = r["perm_p"] < 0.05 and r["accuracy"] > 0.60
        v = "PASS" if passed else "FAIL"
        print(f"{r['subject']:<12} {r['n_stable']:>6} {r['n_trans']:>6} "
              f"{r['l2_distance']:>9.4f} {r['null_l2']:>9.4f} {r['perm_p']:>8.3f} "
              f"{r['accuracy']:>6.1%} {r['chance']:>6.1%} {v:>7}")


def print_verdict(rows: list[dict]) -> None:
    valid = [r for r in rows if "error" not in r]
    n_pass = sum(1 for r in valid if r["perm_p"] < 0.05 and r["accuracy"] > 0.60)
    n_valid = len(valid)
    threshold = max(1, int(np.ceil(n_valid * 2 / 3)))
    overall = "PASS" if n_pass >= threshold else "FAIL"

    print()
    print("=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"State fingerprints: {overall}")
    print(f"  Subjects passing: {n_pass}/{n_valid}")
    print(f"  Criterion: perm p<0.05 AND acc>60% for >={threshold}/{n_valid} subjects")
    if valid:
        mean_l2 = np.mean([r["l2_distance"] for r in valid])
        mean_null = np.mean([r["null_l2"] for r in valid])
        mean_acc = np.mean([r["accuracy"] for r in valid])
        print(f"  Mean L2 (stable vs trans): {mean_l2:.4f}")
        print(f"  Mean L2 (null control):    {mean_null:.4f}")
        print(f"  Mean accuracy:             {mean_acc:.1%}")
        if mean_l2 <= mean_null:
            print("  WARNING: stable-vs-transition L2 <= null control L2")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Screen: state fingerprints")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Data dir: {data_dir}")
    print(f"Parallel workers: {N_JOBS}")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Cannot run state fingerprint screen without EEG rivalry data.")
        sys.exit(1)

    subjects = discover_subjects(data_dir)
    if not subjects:
        print("ERROR: No subjects found.")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects, using first {MAX_SUBJECTS}")
    subjects = subjects[:MAX_SUBJECTS]

    rows = []
    for si, subj in enumerate(subjects):
        t0 = time.time()
        name = subj["name"][:12]
        print(f"\n--- Subject {si + 1}/{len(subjects)}: {name} ---")

        # Load EEG
        result = load_rivalry_epoch(subj["epochs_dir"], CONDITION_SUFFIX, EPOCH_INDEX)
        if result is None:
            print(f"  Skipping: epoch not found")
            rows.append({"subject": name, "error": "no epoch"})
            continue
        epoch_data, ch_names, sfreq = result

        ch_idx = find_channel(ch_names, "Oz")
        if ch_idx is None:
            print(f"  Skipping: Oz channel not found")
            rows.append({"subject": name, "error": "no Oz"})
            continue
        signal = epoch_data[ch_idx]
        print(f"  Channel: {ch_names[ch_idx]}, {len(signal)} samples @ {sfreq} Hz")

        # Load behavioral switches
        switches = load_behavioral_switches(subj["behavior_dir"], PARAM_SET, sfreq)
        if switches is None:
            print(f"  Skipping: no behavioral data")
            rows.append({"subject": name, "error": "no behavior"})
            continue
        switch_samples = [sw["sample"] for sw in switches]
        print(f"  Switches: {len(switches)}")

        # Run analysis
        r = run_fingerprint(signal, switch_samples, sfreq)
        r["subject"] = name
        if "error" in r:
            print(f"  Skipping: {r['error']}")
        else:
            print(f"  L2={r['l2_distance']:.4f}, null_L2={r['null_l2']:.4f}, "
                  f"perm_p={r['perm_p']:.3f}, acc={r['accuracy']:.1%}")
        rows.append(r)
        print(f"  Time: {time.time() - t0:.1f}s")

    print("\n")
    print_results_table(rows)
    print_verdict(rows)

    valid = [r for r in rows if "error" not in r]
    n_pass = sum(1 for r in valid if r["perm_p"] < 0.05 and r["accuracy"] > 0.60)
    threshold = max(1, int(np.ceil(len(valid) * 2 / 3)))
    sys.exit(0 if n_pass >= threshold else 1)


if __name__ == "__main__":
    main()
