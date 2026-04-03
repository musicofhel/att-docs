#!/usr/bin/env python3
"""Branch 2: Sleep Stage Topology — Attractor Switching During Sleep.

Applies ATT's sliding-window PH and transition detector to sleep EEG from
the Sleep-EDF (Expanded) dataset (PhysioNet). Tests whether topological
transitions align with expert-annotated sleep stage transitions and whether
persistent homology discriminates sleep stages.

Three experiments:
  1. Per-stage point cloud PH: H1 entropy per stage, Wasserstein distance
     matrix, permutation test on distance matrix.
  2. Sliding-window transition detection: precision/recall of detected
     changepoints vs annotated stage transitions.
  3. Cross-region binding (Fpz-Cz vs Pz-Oz): REM vs N3 coupling with
     surrogate significance testing.

Usage:
    python scripts/branches/sleep_stage_topology.py [--subjects 0 1 2]
        [--n_perms 200] [--seed 42]

Output:
    data/sleep/results.json
    figures/sleep/*.png
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Suppress MNE info messages
import mne
mne.set_log_level("WARNING")

# Suppress Ripser warnings about more columns than rows
warnings.filterwarnings("ignore", message=".*more columns than rows.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ATT toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from att.topology.persistence import PersistenceAnalyzer
from att.transitions.detector import TransitionDetector
from att.binding.detector import BindingDetector
from att.embedding.takens import TakensEmbedder
from att.neuro.embedding import embed_channel
from att.neuro.eeg_params import get_fallback_params


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # Merge N3+N4 into N3
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#3498db", 3: "#2c3e50", 4: "#9b59b6"}


def parse_args():
    p = argparse.ArgumentParser(description="Sleep stage topological analysis")
    p.add_argument("--subjects", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--n_perms", type=int, default=200)
    p.add_argument("--n_surrogates", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample", type=int, default=500)
    p.add_argument("--window_sec", type=float, default=30.0,
                   help="Sliding window size in seconds")
    p.add_argument("--step_sec", type=float, default=15.0,
                   help="Sliding window step in seconds")
    p.add_argument("--target_sfreq", type=float, default=100.0,
                   help="Target sampling frequency after resampling")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sleep_data(subjects, target_sfreq=100.0):
    """Download and preprocess Sleep-EDF data.

    Returns list of dicts, each with:
        raw: preprocessed MNE Raw
        epochs: list of (stage, data_fpz, data_pz) tuples per 30s epoch
        annotations: list of (onset_sec, duration_sec, stage_int)
        subject_id: int
    """
    from mne.datasets import sleep_physionet

    all_data = []
    for subj in subjects:
        print(f"  Loading subject {subj}...")
        try:
            paths = sleep_physionet.age.fetch_data(subjects=[subj], recording=[1])
        except Exception as e:
            print(f"  WARNING: Could not fetch subject {subj}: {e}")
            continue

        psg_path, hyp_path = paths[0]
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        annot = mne.read_annotations(hyp_path)
        raw.set_annotations(annot)

        # Pick EEG channels: Fpz-Cz (primary) and Pz-Oz (secondary for binding)
        available = raw.ch_names
        picks = []
        fpz_ch = None
        pz_ch = None
        for ch in available:
            if "fpz" in ch.lower() or "fpz-cz" in ch.lower():
                fpz_ch = ch
            if "pz" in ch.lower() or "pz-oz" in ch.lower():
                pz_ch = ch

        if fpz_ch is None:
            print(f"  WARNING: No Fpz-Cz channel found for subject {subj}, channels: {available}")
            continue

        pick_chs = [fpz_ch]
        if pz_ch is not None:
            pick_chs.append(pz_ch)

        raw.pick(pick_chs)

        # Bandpass filter 0.5-30 Hz
        raw.filter(0.5, 30.0, verbose=False)

        # Resample to target
        if raw.info["sfreq"] != target_sfreq:
            raw.resample(target_sfreq, verbose=False)

        sfreq = raw.info["sfreq"]
        epoch_samples = int(30.0 * sfreq)  # 30-second epochs

        # Extract epochs aligned to hypnogram
        epochs = []
        stage_annotations = []

        for ann in annot:
            stage = STAGE_MAP.get(ann["description"], -1)
            if stage < 0:
                continue

            onset_sec = ann["onset"]
            duration_sec = ann["duration"]
            n_epochs_in_ann = int(duration_sec / 30.0)

            for k in range(n_epochs_in_ann):
                epoch_start = onset_sec + k * 30.0
                start_samp = int(epoch_start * sfreq)
                end_samp = start_samp + epoch_samples

                if end_samp > raw.n_times:
                    break

                data = raw.get_data(start=start_samp, stop=end_samp)
                # Scale from Volts to microvolts (MNE stores SI units)
                data = data * 1e6
                fpz_data = data[0]  # Fpz-Cz
                pz_data = data[1] if data.shape[0] > 1 else None

                epochs.append({
                    "stage": stage,
                    "fpz": fpz_data,
                    "pz": pz_data,
                    "onset_sec": epoch_start,
                })
                stage_annotations.append((epoch_start, 30.0, stage))

        # Build continuous data for transition detector (Fpz-Cz only)
        # Scale to microvolts
        continuous_fpz = raw.get_data(picks=[0])[0] * 1e6

        all_data.append({
            "raw": raw,
            "epochs": epochs,
            "annotations": stage_annotations,
            "subject_id": subj,
            "sfreq": sfreq,
            "continuous_fpz": continuous_fpz,
            "fpz_ch": fpz_ch,
            "pz_ch": pz_ch,
        })
        print(f"    {len(epochs)} epochs, sfreq={sfreq} Hz, "
              f"channels: {pick_chs}, duration={raw.times[-1]/3600:.1f}h")

    return all_data


def get_stage_transition_times(annotations):
    """Extract times where sleep stage changes.

    Returns list of (time_sec, from_stage, to_stage) tuples.
    """
    transitions = []
    prev_stage = None
    for onset, dur, stage in annotations:
        if prev_stage is not None and stage != prev_stage:
            transitions.append((onset, prev_stage, stage))
        prev_stage = stage
    return transitions


# ---------------------------------------------------------------------------
# Experiment 1: Per-stage point cloud topology
# ---------------------------------------------------------------------------
def experiment1_per_stage_ph(all_data, args):
    """Compute PH per sleep stage, compare via Wasserstein + permutation test."""
    print("\n=== Experiment 1: Per-stage point cloud topology ===")
    rng = np.random.default_rng(args.seed)

    # Collect all epochs across subjects by stage
    stage_epochs = {s: [] for s in range(5)}
    for subj_data in all_data:
        for ep in subj_data["epochs"]:
            stage_epochs[ep["stage"]].append(ep["fpz"])

    n_per_stage = {s: len(eps) for s, eps in stage_epochs.items()}
    print(f"  Epochs per stage: {', '.join(f'{STAGE_NAMES[s]}={n}' for s, n in sorted(n_per_stage.items()))}")

    # Takens-embed each stage's concatenated epochs into a point cloud
    # Use literature fallback params for sleep EEG (broadband 0.5-30 Hz)
    params = get_fallback_params("broadband", sfreq=100.0)
    delay = params["delay"]
    dimension = params["dimension"]
    print(f"  Embedding: delay={delay}, dimension={dimension}")

    stage_clouds = {}
    stage_results = {}
    stage_analyzers = {}

    for stage in range(5):
        if len(stage_epochs[stage]) == 0:
            print(f"  WARNING: No epochs for stage {STAGE_NAMES[stage]}")
            continue

        # Concatenate all epochs for this stage
        all_epoch_data = np.concatenate(stage_epochs[stage])

        # Takens embedding
        embedder = TakensEmbedder(delay=delay, dimension=dimension)
        embedder.fit(all_epoch_data)
        cloud = embedder.transform(all_epoch_data)

        # Subsample for PH computation
        n_sub = min(args.subsample, cloud.shape[0])
        idx = rng.choice(cloud.shape[0], n_sub, replace=False)
        cloud_sub = cloud[idx]

        stage_clouds[stage] = cloud_sub

        pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
        result = pa.fit_transform(cloud_sub, seed=args.seed)
        stage_results[stage] = result
        stage_analyzers[stage] = pa

        h0_ent = result["persistence_entropy"][0]
        h1_ent = result["persistence_entropy"][1]
        n_h0 = len(result["diagrams"][0])
        n_h1 = len(result["diagrams"][1])
        print(f"  {STAGE_NAMES[stage]}: H0_ent={h0_ent:.3f} ({n_h0} feat), "
              f"H1_ent={h1_ent:.3f} ({n_h1} feat)")

    # Wasserstein distance matrix (5x5)
    stages_present = sorted(stage_analyzers.keys())
    n_stages = len(stages_present)
    wass_matrix = np.zeros((n_stages, n_stages))

    for i, s1 in enumerate(stages_present):
        for j, s2 in enumerate(stages_present):
            if i < j:
                d = stage_analyzers[s1].distance(stage_analyzers[s2], metric="wasserstein_1")
                wass_matrix[i, j] = d
                wass_matrix[j, i] = d

    print(f"\n  Wasserstein-1 distance matrix:")
    header = "       " + "  ".join(f"{STAGE_NAMES[s]:>8}" for s in stages_present)
    print(f"  {header}")
    for i, s1 in enumerate(stages_present):
        row = f"  {STAGE_NAMES[s1]:>5}  " + "  ".join(f"{wass_matrix[i,j]:8.3f}" for j in range(n_stages))
        print(row)

    # Permutation test: is the observed total pairwise distance significant?
    observed_total = np.sum(wass_matrix[np.triu_indices(n_stages, k=1)])

    # Build label array for permutation
    all_clouds_list = []
    labels = []
    for stage in stages_present:
        all_clouds_list.append(stage_clouds[stage])
        labels.extend([stage] * stage_clouds[stage].shape[0])
    all_points = np.vstack(all_clouds_list)
    labels = np.array(labels)

    # For permutation test, use smaller subsample for speed (200 points)
    # Recompute observed total at same subsample for fair comparison
    perm_sub = min(200, args.subsample)

    obs_analyzers = {}
    for stage in stages_present:
        n_sub = min(perm_sub, stage_clouds[stage].shape[0])
        idx = rng.choice(stage_clouds[stage].shape[0], n_sub, replace=False)
        pa_obs = PersistenceAnalyzer(max_dim=1, backend="ripser")
        pa_obs.fit_transform(stage_clouds[stage][idx], seed=args.seed)
        obs_analyzers[stage] = pa_obs

    observed_total = 0.0
    for i, s1 in enumerate(stages_present):
        for j, s2 in enumerate(stages_present):
            if i < j:
                d = obs_analyzers[s1].distance(obs_analyzers[s2], metric="wasserstein_1")
                observed_total += d

    print(f"\n  Permutation test ({args.n_perms} permutations, subsample={perm_sub})...")
    print(f"  Observed total (sub={perm_sub}): {observed_total:.1f}")
    null_totals = []
    for perm_i in range(args.n_perms):
        shuffled = rng.permutation(labels)
        perm_analyzers = {}

        for stage in stages_present:
            mask = shuffled == stage
            cloud_perm = all_points[mask]
            n_sub = min(perm_sub, cloud_perm.shape[0])
            idx = rng.choice(cloud_perm.shape[0], n_sub, replace=False)

            pa_perm = PersistenceAnalyzer(max_dim=1, backend="ripser")
            pa_perm.fit_transform(cloud_perm[idx], seed=args.seed + perm_i + 1)
            perm_analyzers[stage] = pa_perm

        perm_total = 0.0
        for i, s1 in enumerate(stages_present):
            for j, s2 in enumerate(stages_present):
                if i < j:
                    d = perm_analyzers[s1].distance(perm_analyzers[s2], metric="wasserstein_1")
                    perm_total += d
        null_totals.append(perm_total)

        if (perm_i + 1) % 25 == 0:
            print(f"    {perm_i + 1}/{args.n_perms}")

    null_totals = np.array(null_totals)
    p_value = (np.sum(null_totals >= observed_total) + 1) / (args.n_perms + 1)
    z_score = (observed_total - np.mean(null_totals)) / (np.std(null_totals) + 1e-10)

    print(f"\n  Observed total Wasserstein: {observed_total:.3f}")
    print(f"  Null mean: {np.mean(null_totals):.3f} +/- {np.std(null_totals):.3f}")
    print(f"  z-score: {z_score:.3f}, p-value: {p_value:.4f}")

    # Extract results
    h0_entropy = {STAGE_NAMES[s]: stage_results[s]["persistence_entropy"][0]
                  for s in stages_present}
    h1_entropy = {STAGE_NAMES[s]: stage_results[s]["persistence_entropy"][1]
                  for s in stages_present}

    # Pairwise Wasserstein as named dict
    wass_pairs = {}
    for i, s1 in enumerate(stages_present):
        for j, s2 in enumerate(stages_present):
            if i < j:
                key = f"{STAGE_NAMES[s1]}_vs_{STAGE_NAMES[s2]}"
                wass_pairs[key] = round(wass_matrix[i, j], 4)

    return {
        "h0_entropy_per_stage": {k: round(v, 4) for k, v in h0_entropy.items()},
        "h1_entropy_per_stage": {k: round(v, 4) for k, v in h1_entropy.items()},
        "wasserstein_matrix": wass_matrix.tolist(),
        "wasserstein_pairs": wass_pairs,
        "wasserstein_z": round(z_score, 4),
        "wasserstein_p": round(p_value, 4),
        "n_per_stage": {STAGE_NAMES[s]: n_per_stage[s] for s in stages_present},
        "stage_results": stage_results,
        "stages_present": stages_present,
        "null_totals": null_totals,
        "observed_total": observed_total,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Sliding-window transition detection
# ---------------------------------------------------------------------------
def experiment2_transition_detection(all_data, args):
    """Run TransitionDetector on continuous EEG, compare to annotations."""
    print("\n=== Experiment 2: Sliding-window transition detection ===")

    sfreq = all_data[0]["sfreq"]
    window_samples = int(args.window_sec * sfreq)
    step_samples = int(args.step_sec * sfreq)

    # Use fallback embedding params
    params = get_fallback_params("broadband", sfreq=sfreq)
    emb_delay = params["delay"]
    emb_dim = params["dimension"]

    all_results = []

    for subj_data in all_data:
        subj_id = subj_data["subject_id"]
        continuous = subj_data["continuous_fpz"]
        annotations = subj_data["annotations"]

        # Get ground truth transition times
        gt_transitions = get_stage_transition_times(annotations)
        gt_times = np.array([t[0] for t in gt_transitions])

        print(f"\n  Subject {subj_id}: {len(continuous)/sfreq/3600:.1f}h, "
              f"{len(gt_transitions)} annotated transitions")

        # Find sleep onset (first non-Wake annotation) and extract ~2h around it
        sleep_onset = None
        for onset, dur, stage in annotations:
            if stage != 0:  # not Wake
                sleep_onset = onset
                break

        if sleep_onset is None:
            print(f"    No sleep onset found, skipping subject {subj_id}")
            continue

        # Start 5 min before sleep onset, take 1 hour
        start_time = max(0, sleep_onset - 300)
        duration = 1.0 * 3600  # 1 hour
        start_samp = int(start_time * sfreq)
        end_samp = min(int((start_time + duration) * sfreq), len(continuous))
        continuous = continuous[start_samp:end_samp]

        # Adjust ground truth times relative to the extracted segment
        gt_times = gt_times - start_time
        gt_times = gt_times[(gt_times >= 0) & (gt_times < duration)]

        print(f"    Sleep onset at {sleep_onset/3600:.2f}h, "
              f"using {start_time/3600:.2f}h-{(start_time+duration)/3600:.2f}h "
              f"({len(continuous)/sfreq/60:.0f} min, {len(gt_times)} transitions)")

        # Run transition detector
        print(f"    Running TransitionDetector (window={args.window_sec}s, "
              f"step={args.step_sec}s, emb_dim={emb_dim}, delay={emb_delay})...")

        det = TransitionDetector(
            window_size=window_samples,
            step_size=step_samples,
            max_dim=1,
            backend="ripser",
            subsample=200,  # subsample within each window for speed
        )

        result = det.fit_transform(
            continuous,
            seed=args.seed,
            embedding_dim=emb_dim,
            embedding_delay=emb_delay,
        )

        # Detect changepoints
        changepoints = det.detect_changepoints(method="cusum")
        window_centers = result["window_centers"]
        scores = result["transition_scores"]

        # Convert changepoint indices to seconds
        detected_times = np.array([window_centers[cp] / sfreq for cp in changepoints])
        print(f"    Detected {len(detected_times)} changepoints")

        # Precision/recall against ground truth (tolerance = 30s)
        tolerance = 30.0
        n_detected = len(detected_times)
        n_gt = len(gt_times)

        if n_detected == 0 or n_gt == 0:
            precision = 0.0
            recall = 0.0
            median_lag = float("nan")
        else:
            # For each detected changepoint, find nearest GT transition
            tp_detected = 0
            lags = []
            for dt in detected_times:
                dists = np.abs(gt_times - dt)
                min_dist = np.min(dists)
                if min_dist <= tolerance:
                    tp_detected += 1
                    nearest_gt = gt_times[np.argmin(dists)]
                    lags.append(dt - nearest_gt)

            # For each GT transition, check if any detected point is within tolerance
            tp_gt = 0
            for gt in gt_times:
                if len(detected_times) > 0 and np.min(np.abs(detected_times - gt)) <= tolerance:
                    tp_gt += 1

            precision = tp_detected / n_detected if n_detected > 0 else 0.0
            recall = tp_gt / n_gt if n_gt > 0 else 0.0
            median_lag = float(np.median(lags)) if lags else float("nan")

        print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, "
              f"Median lag: {median_lag:.1f}s")

        all_results.append({
            "subject_id": subj_id,
            "n_detected": n_detected,
            "n_annotated": n_gt,
            "precision": precision,
            "recall": recall,
            "median_lag_seconds": median_lag,
            "scores": scores,
            "window_centers": window_centers,
            "detected_times": detected_times,
            "gt_times": gt_times,
            "sfreq": sfreq,
        })

    # Aggregate across subjects
    precisions = [r["precision"] for r in all_results]
    recalls = [r["recall"] for r in all_results]
    lags = [r["median_lag_seconds"] for r in all_results if not np.isnan(r["median_lag_seconds"])]

    return {
        "per_subject": all_results,
        "mean_precision": round(float(np.mean(precisions)), 4),
        "mean_recall": round(float(np.mean(recalls)), 4),
        "mean_median_lag": round(float(np.mean(lags)), 2) if lags else float("nan"),
        "total_detected": sum(r["n_detected"] for r in all_results),
        "total_annotated": sum(r["n_annotated"] for r in all_results),
    }


# ---------------------------------------------------------------------------
# Experiment 3: Cross-region binding (REM vs N3)
# ---------------------------------------------------------------------------
def experiment3_cross_region_binding(all_data, args):
    """Compare Fpz-Cz ↔ Pz-Oz topological binding in REM vs N3."""
    print("\n=== Experiment 3: Cross-region binding (REM vs N3) ===")

    # Check if we have multi-channel data
    has_pz = any(d["pz_ch"] is not None for d in all_data)
    if not has_pz:
        print("  No Pz-Oz channel available. Skipping Experiment 3.")
        return {
            "rem_binding": None,
            "rem_p": None,
            "n3_binding": None,
            "n3_p": None,
            "skipped": True,
            "reason": "No Pz-Oz channel available",
        }

    # Collect REM and N3 epochs with both channels
    rem_fpz, rem_pz = [], []
    n3_fpz, n3_pz = [], []

    for subj_data in all_data:
        if subj_data["pz_ch"] is None:
            continue
        for ep in subj_data["epochs"]:
            if ep["pz"] is None:
                continue
            if ep["stage"] == 4:  # REM
                rem_fpz.append(ep["fpz"])
                rem_pz.append(ep["pz"])
            elif ep["stage"] == 3:  # N3
                n3_fpz.append(ep["fpz"])
                n3_pz.append(ep["pz"])

    print(f"  REM epochs (with both channels): {len(rem_fpz)}")
    print(f"  N3 epochs (with both channels): {len(n3_fpz)}")

    if len(rem_fpz) < 5 or len(n3_fpz) < 5:
        print("  Not enough epochs for binding analysis.")
        return {
            "rem_binding": None,
            "rem_p": None,
            "n3_binding": None,
            "n3_p": None,
            "skipped": True,
            "reason": f"Too few epochs: REM={len(rem_fpz)}, N3={len(n3_fpz)}",
        }

    # Concatenate epochs for each condition
    # Use up to 20 epochs (10 min) per condition for manageable computation
    max_epochs = 20
    rng = np.random.default_rng(args.seed)

    def select_epochs(fpz_list, pz_list, max_n):
        if len(fpz_list) > max_n:
            idx = rng.choice(len(fpz_list), max_n, replace=False)
            return (np.concatenate([fpz_list[i] for i in idx]),
                    np.concatenate([pz_list[i] for i in idx]))
        return np.concatenate(fpz_list), np.concatenate(pz_list)

    rem_fpz_cat, rem_pz_cat = select_epochs(rem_fpz, rem_pz, max_epochs)
    n3_fpz_cat, n3_pz_cat = select_epochs(n3_fpz, n3_pz, max_epochs)

    print(f"  REM: {len(rem_fpz_cat)} samples ({len(rem_fpz_cat)/100:.0f}s)")
    print(f"  N3:  {len(n3_fpz_cat)} samples ({len(n3_fpz_cat)/100:.0f}s)")

    # Use pre-configured embedders with literature params (skip auto AMI/FNN)
    sfreq = all_data[0]["sfreq"]
    params = get_fallback_params("broadband", sfreq=sfreq)
    emb_delay = params["delay"]
    emb_dim = params["dimension"]
    print(f"  Embedding: delay={emb_delay}, dimension={emb_dim}")

    from att.embedding.joint import JointEmbedder

    def make_embedders():
        """Create pre-configured embedders (no auto estimation)."""
        emb_x = TakensEmbedder(delay=emb_delay, dimension=emb_dim)
        emb_y = TakensEmbedder(delay=emb_delay, dimension=emb_dim)
        je = JointEmbedder(delays=[emb_delay, emb_delay],
                           dimensions=[emb_dim, emb_dim])
        return emb_x, emb_y, je

    # Binding: REM
    print("  Computing REM binding...")
    emb_x, emb_y, je = make_embedders()
    bd_rem = BindingDetector(max_dim=1, method="persistence_image", baseline="max",
                             embedding_quality_gate=False)
    bd_rem.fit(rem_fpz_cat, rem_pz_cat,
               marginal_embedder_x=emb_x, marginal_embedder_y=emb_y,
               joint_embedder=je,
               subsample=args.subsample, seed=args.seed)
    rem_score = bd_rem.binding_score()
    print(f"    REM binding score: {rem_score:.4f}")

    print(f"  REM significance test ({args.n_surrogates} surrogates)...")
    rem_sig = bd_rem.test_significance(
        n_surrogates=args.n_surrogates,
        method="phase_randomize",
        seed=args.seed,
        subsample=args.subsample,
    )
    print(f"    REM p={rem_sig['p_value']:.4f}, z={rem_sig['z_score']:.3f}")

    # Binding: N3
    print("  Computing N3 binding...")
    emb_x, emb_y, je = make_embedders()
    bd_n3 = BindingDetector(max_dim=1, method="persistence_image", baseline="max",
                            embedding_quality_gate=False)
    bd_n3.fit(n3_fpz_cat, n3_pz_cat,
              marginal_embedder_x=emb_x, marginal_embedder_y=emb_y,
              joint_embedder=je,
              subsample=args.subsample, seed=args.seed)
    n3_score = bd_n3.binding_score()
    print(f"    N3 binding score: {n3_score:.4f}")

    print(f"  N3 significance test ({args.n_surrogates} surrogates)...")
    n3_sig = bd_n3.test_significance(
        n_surrogates=args.n_surrogates,
        method="phase_randomize",
        seed=args.seed,
        subsample=args.subsample,
    )
    print(f"    N3 p={n3_sig['p_value']:.4f}, z={n3_sig['z_score']:.3f}")

    return {
        "rem_binding": round(rem_score, 4),
        "rem_p": round(rem_sig["p_value"], 4),
        "rem_z": round(rem_sig["z_score"], 4),
        "rem_significant": bool(rem_sig["significant"]),
        "n3_binding": round(n3_score, 4),
        "n3_p": round(n3_sig["p_value"], 4),
        "n3_z": round(n3_sig["z_score"], 4),
        "n3_significant": bool(n3_sig["significant"]),
        "rem_surrogate_mean": round(rem_sig["surrogate_mean"], 4),
        "n3_surrogate_mean": round(n3_sig["surrogate_mean"], 4),
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_entropy_by_stage(exp1_results, outdir):
    """Bar chart of H0 and H1 persistence entropy per sleep stage."""
    stages_present = exp1_results["stages_present"]
    stage_names = [STAGE_NAMES[s] for s in stages_present]

    h0 = [exp1_results["h0_entropy_per_stage"][STAGE_NAMES[s]] for s in stages_present]
    h1 = [exp1_results["h1_entropy_per_stage"][STAGE_NAMES[s]] for s in stages_present]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = [STAGE_COLORS[s] for s in stages_present]

    axes[0].bar(stage_names, h0, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("H0 Persistence Entropy")
    axes[0].set_xlabel("Sleep Stage")
    axes[0].set_title("H0 Entropy by Sleep Stage")

    axes[1].bar(stage_names, h1, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("H1 Persistence Entropy")
    axes[1].set_xlabel("Sleep Stage")
    axes[1].set_title("H1 Entropy by Sleep Stage")

    plt.tight_layout()
    path = os.path.join(outdir, "entropy_by_stage.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_wasserstein_matrix(exp1_results, outdir):
    """Heatmap of pairwise Wasserstein distances between stages."""
    stages_present = exp1_results["stages_present"]
    stage_names = [STAGE_NAMES[s] for s in stages_present]
    matrix = np.array(exp1_results["wasserstein_matrix"])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names)
    ax.set_yticks(range(len(stage_names)))
    ax.set_yticklabels(stage_names)

    # Annotate
    for i in range(len(stage_names)):
        for j in range(len(stage_names)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=10)

    ax.set_title(f"Wasserstein-1 Distance (z={exp1_results['wasserstein_z']:.2f}, "
                 f"p={exp1_results['wasserstein_p']:.3f})")
    plt.colorbar(im, ax=ax, label="Wasserstein-1")
    plt.tight_layout()
    path = os.path.join(outdir, "wasserstein_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_permutation_test(exp1_results, outdir):
    """Null distribution histogram with observed value."""
    null_totals = exp1_results["null_totals"]
    observed = exp1_results["observed_total"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_totals, bins=30, alpha=0.7, color="#3498db", edgecolor="black",
            linewidth=0.5, label="Null distribution")
    ax.axvline(observed, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"Observed ({observed:.1f})")
    ax.set_xlabel("Total pairwise Wasserstein-1 distance")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test (z={exp1_results['wasserstein_z']:.2f}, "
                 f"p={exp1_results['wasserstein_p']:.3f})")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "permutation_test.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_transition_timeline(exp2_results, outdir):
    """Transition score time series with detected + annotated transitions."""
    # Plot first subject
    for subj_result in exp2_results["per_subject"]:
        subj_id = subj_result["subject_id"]
        scores = subj_result["scores"]
        wc = subj_result["window_centers"]
        sfreq = subj_result["sfreq"]
        detected = subj_result["detected_times"]
        gt_times = subj_result["gt_times"]

        # Time axis in minutes
        time_min = wc[:-1] / sfreq / 60.0  # scores has len(wc)-1 entries

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(time_min, scores, color="#3498db", linewidth=0.5, alpha=0.8,
                label="Transition score")

        # Plot GT transitions
        for gt in gt_times:
            gt_min = gt / 60.0
            if gt_min <= time_min[-1]:
                ax.axvline(gt_min, color="#2ecc71", alpha=0.4, linewidth=1)
        ax.axvline(-999, color="#2ecc71", alpha=0.4, linewidth=1,
                   label="Annotated transitions")

        # Plot detected changepoints
        for dt in detected:
            dt_min = dt / 60.0
            if dt_min <= time_min[-1]:
                ax.axvline(dt_min, color="#e74c3c", alpha=0.6, linewidth=1.5,
                           linestyle="--")
        ax.axvline(-999, color="#e74c3c", alpha=0.6, linewidth=1.5, linestyle="--",
                   label="Detected changepoints")

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Topological transition score")
        ax.set_title(f"Subject {subj_id}: Transition Detection "
                     f"(P={subj_result['precision']:.2f}, R={subj_result['recall']:.2f})")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(time_min[0], time_min[-1])
        plt.tight_layout()
        path = os.path.join(outdir, f"transition_timeline_subj{subj_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")


def plot_binding_comparison(exp3_results, outdir):
    """Bar chart comparing REM vs N3 binding scores."""
    if exp3_results.get("skipped"):
        print("  Skipping binding plot (no multi-channel data)")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["REM", "N3"]
    scores = [exp3_results["rem_binding"], exp3_results["n3_binding"]]
    surr_means = [exp3_results["rem_surrogate_mean"], exp3_results["n3_surrogate_mean"]]
    colors = [STAGE_COLORS[4], STAGE_COLORS[3]]
    p_vals = [exp3_results["rem_p"], exp3_results["n3_p"]]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, scores, width, color=colors, alpha=0.8,
                   edgecolor="black", linewidth=0.5, label="Observed")
    bars2 = ax.bar(x + width / 2, surr_means, width, color="gray", alpha=0.5,
                   edgecolor="black", linewidth=0.5, label="Surrogate mean")

    # Significance stars
    for i, p in enumerate(p_vals):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(x[i], max(scores[i], surr_means[i]) * 1.05, star,
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Binding Score")
    ax.set_title("Cross-Region Topological Binding: REM vs N3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "binding_rem_vs_n3.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_epoch_distribution(all_data, outdir):
    """Pie/bar chart of epoch counts per stage across all subjects."""
    stage_counts = {s: 0 for s in range(5)}
    for subj_data in all_data:
        for ep in subj_data["epochs"]:
            stage_counts[ep["stage"]] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [STAGE_NAMES[s] for s in range(5)]
    counts = [stage_counts[s] for s in range(5)]
    colors = [STAGE_COLORS[s] for s in range(5)]

    bars = ax.bar(names, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Number of 30s Epochs")
    ax.set_xlabel("Sleep Stage")
    ax.set_title(f"Epoch Distribution ({sum(counts)} total, {len(all_data)} subjects)")
    plt.tight_layout()
    path = os.path.join(outdir, "epoch_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(base_dir, "..", "..")
    data_dir = os.path.join(repo_root, "data", "sleep")
    fig_dir = os.path.join(repo_root, "figures", "sleep")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("Branch 2: Sleep Stage Topology — Attractor Switching")
    print("=" * 60)

    # Load data
    print(f"\nLoading Sleep-EDF data (subjects: {args.subjects})...")
    all_data = load_sleep_data(args.subjects, target_sfreq=args.target_sfreq)
    if not all_data:
        print("ERROR: No data loaded. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(all_data)} subjects successfully.")

    # Epoch distribution figure
    print("\nPlotting epoch distribution...")
    plot_epoch_distribution(all_data, fig_dir)

    # Experiment 1
    exp1 = experiment1_per_stage_ph(all_data, args)

    # Experiment 2
    exp2 = experiment2_transition_detection(all_data, args)

    # Experiment 3
    exp3 = experiment3_cross_region_binding(all_data, args)

    # Generate figures
    print("\n=== Generating Figures ===")
    plot_entropy_by_stage(exp1, fig_dir)
    plot_wasserstein_matrix(exp1, fig_dir)
    plot_permutation_test(exp1, fig_dir)
    plot_transition_timeline(exp2, fig_dir)
    plot_binding_comparison(exp3, fig_dir)

    # Save results
    results = {
        "branch": "experiment/tda-sleep",
        "n_subjects": len(all_data),
        "subjects": [d["subject_id"] for d in all_data],
        "n_epochs_per_stage": exp1["n_per_stage"],
        "exp1_h0_entropy_per_stage": exp1["h0_entropy_per_stage"],
        "exp1_h1_entropy_per_stage": exp1["h1_entropy_per_stage"],
        "exp1_wasserstein_pairs": exp1["wasserstein_pairs"],
        "exp1_wasserstein_z": exp1["wasserstein_z"],
        "exp1_wasserstein_p": exp1["wasserstein_p"],
        "exp1_n_perms": args.n_perms,
        "exp2_mean_precision": exp2["mean_precision"],
        "exp2_mean_recall": exp2["mean_recall"],
        "exp2_mean_median_lag_seconds": exp2["mean_median_lag"],
        "exp2_total_detected": exp2["total_detected"],
        "exp2_total_annotated": exp2["total_annotated"],
        "exp2_per_subject": [
            {
                "subject_id": r["subject_id"],
                "n_detected": r["n_detected"],
                "n_annotated": r["n_annotated"],
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "median_lag_seconds": round(r["median_lag_seconds"], 2)
                    if not np.isnan(r["median_lag_seconds"]) else None,
            }
            for r in exp2["per_subject"]
        ],
        "exp3_rem_binding": exp3.get("rem_binding"),
        "exp3_rem_p": exp3.get("rem_p"),
        "exp3_rem_z": exp3.get("rem_z"),
        "exp3_n3_binding": exp3.get("n3_binding"),
        "exp3_n3_p": exp3.get("n3_p"),
        "exp3_n3_z": exp3.get("n3_z"),
        "exp3_skipped": exp3.get("skipped", False),
        "exp3_reason": exp3.get("reason"),
        "config": {
            "seed": args.seed,
            "n_perms": args.n_perms,
            "n_surrogates": args.n_surrogates,
            "subsample": args.subsample,
            "window_sec": args.window_sec,
            "step_sec": args.step_sec,
            "target_sfreq": args.target_sfreq,
        },
    }

    results_path = os.path.join(data_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nExp 1 — Per-stage PH:")
    print(f"  H1 entropy: {exp1['h1_entropy_per_stage']}")
    print(f"  Wasserstein permutation: z={exp1['wasserstein_z']}, p={exp1['wasserstein_p']}")

    print(f"\nExp 2 — Transition detection:")
    print(f"  Mean precision: {exp2['mean_precision']:.3f}")
    print(f"  Mean recall: {exp2['mean_recall']:.3f}")
    print(f"  Mean median lag: {exp2['mean_median_lag']:.1f}s")

    print(f"\nExp 3 — Cross-region binding:")
    if exp3.get("skipped"):
        print(f"  Skipped: {exp3['reason']}")
    else:
        print(f"  REM binding: {exp3['rem_binding']:.4f} (p={exp3['rem_p']:.4f})")
        print(f"  N3 binding:  {exp3['n3_binding']:.4f} (p={exp3['n3_p']:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
