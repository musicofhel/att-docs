#!/usr/bin/env python3
"""Batch EEG analysis pipeline for multi-subject binocular rivalry data.

Runs the transition detection (fig7) and binding analysis (fig8) pipelines
across all subjects in the Nie/Katyal/Engel (2023) dataset. Produces per-subject
JSON results and an aggregated summary CSV.

Dataset: 84 subjects, 34-channel EEG at 360 Hz, binocular rivalry SSVEP.
DOI: 10.13020/9sy5-a716

Usage:
    # Process all subjects (transition detection only, ~5 min/subject)
    python scripts/batch_eeg.py \\
        --data-dir data/eeg/rivalry_ssvep \\
        --output-dir results/batch_eeg \\
        --skip-binding

    # Process first 5 subjects with binding (~10 min/subject)
    python scripts/batch_eeg.py \\
        --data-dir data/eeg/rivalry_ssvep \\
        --output-dir results/batch_eeg \\
        --n-subjects 5

    # Full pipeline, all subjects, custom config
    python scripts/batch_eeg.py \\
        --data-dir data/eeg/rivalry_ssvep \\
        --output-dir results/batch_eeg \\
        --config configs/batch_eeg.yaml

Estimated runtime:
    - Transition detection only: ~5 minutes per subject (~7 hours for 84)
    - With binding: ~10 minutes per subject (~14 hours for 84)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import butter, sosfilt
from scipy.stats import spearmanr

# ATT imports
from att.config import set_seed
from att.neuro.embedding import embed_channel
from att.embedding.takens import TakensEmbedder
from att.embedding.joint import JointEmbedder
from att.transitions import TransitionDetector
from att.binding import BindingDetector

log = logging.getLogger("batch_eeg")

# ---------------------------------------------------------------------------
# Configuration defaults (overridden by YAML config if provided)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "seed": 42,
    "sampling_rate": 360,
    "bandpass_low": 4,
    "bandpass_high": 13,
    "bandpass_order": 4,
    "transition_channel": "Oz",
    "binding_pair": ["Oz", "Pz"],
    "embedding_band": "theta_alpha",
    "condition_threshold": 1e4,
    "td_window_size": 500,
    "td_step_size": 200,
    "td_max_dim": 1,
    "td_backend": "ripser",
    "td_subsample": 300,
    "td_changepoint_method": "cusum",
    "td_changepoint_threshold": None,
    "eval_tolerances": [3.0, 5.0],
    "binding_max_dim": 1,
    "binding_baseline": "max",
    "binding_image_resolution": 50,
    "binding_image_sigma": 0.1,
    "binding_subsample": 400,
    "binding_shared_dim": 5,
    "binding_window_seconds": 10,
    "binding_step_seconds": 5,
    "epoch_condition_suffix": "riv_12",
    "epoch_param_set": 2,
    "epoch_index": 0,
}


def load_config(config_path: Path | None) -> dict:
    """Load configuration from YAML file, merged with defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if config_path is None:
        return cfg

    try:
        import yaml
    except ImportError:
        log.warning("PyYAML not installed; ignoring config file. pip install pyyaml")
        return cfg

    with open(config_path) as f:
        user_cfg = yaml.safe_load(f)

    if user_cfg is None:
        return cfg

    # Flatten nested YAML into flat dict matching DEFAULT_CONFIG keys
    _flat_map = {
        ("seed",): "seed",
        ("dataset", "sampling_rate"): "sampling_rate",
        ("preprocessing", "bandpass", "low"): "bandpass_low",
        ("preprocessing", "bandpass", "high"): "bandpass_high",
        ("preprocessing", "bandpass", "order"): "bandpass_order",
        ("channels", "transition_channel"): "transition_channel",
        ("channels", "binding_pair"): "binding_pair",
        ("embedding", "fallback_band"): "embedding_band",
        ("embedding", "condition_threshold"): "condition_threshold",
        ("transition_detection", "window_size"): "td_window_size",
        ("transition_detection", "step_size"): "td_step_size",
        ("transition_detection", "max_dim"): "td_max_dim",
        ("transition_detection", "backend"): "td_backend",
        ("transition_detection", "subsample"): "td_subsample",
        ("transition_detection", "changepoint_method"): "td_changepoint_method",
        ("transition_detection", "changepoint_threshold"): "td_changepoint_threshold",
        ("evaluation", "tolerance_seconds"): "eval_tolerances",
        ("binding", "max_dim"): "binding_max_dim",
        ("binding", "baseline"): "binding_baseline",
        ("binding", "image_resolution"): "binding_image_resolution",
        ("binding", "image_sigma"): "binding_image_sigma",
        ("binding", "subsample"): "binding_subsample",
        ("binding", "shared_dimension"): "binding_shared_dim",
        ("binding", "window_seconds"): "binding_window_seconds",
        ("binding", "step_seconds"): "binding_step_seconds",
        ("epochs", "conditions", 0, "suffix"): "epoch_condition_suffix",
        ("epochs", "conditions", 0, "param_set"): "epoch_param_set",
        ("epochs", "epoch_index"): "epoch_index",
    }

    for key_path, flat_key in _flat_map.items():
        obj = user_cfg
        try:
            for k in key_path:
                if isinstance(obj, list):
                    obj = obj[k]
                else:
                    obj = obj[k]
            cfg[flat_key] = obj
        except (KeyError, IndexError, TypeError):
            pass

    return cfg


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def discover_subjects(data_dir: Path) -> list[dict]:
    """Discover subject directories under the data path.

    Each subject directory should contain Epochs/ and Behavior/ subdirectories.
    Returns a sorted list of dicts with 'name', 'path', 'epochs_dir', 'behavior_dir'.
    """
    subjects = []
    if not data_dir.exists():
        log.error("Data directory does not exist: %s", data_dir)
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


# ---------------------------------------------------------------------------
# Data loading (mirrors fig7/fig8 notebooks)
# ---------------------------------------------------------------------------

def load_rivalry_epoch(
    epochs_dir: Path,
    condition_suffix: str,
    epoch_index: int,
) -> tuple[np.ndarray, list[str], int] | None:
    """Load a rivalry epoch from .mat file.

    Returns (epoch_data, channel_names, sfreq) or None on failure.
    epoch_data shape: (n_channels, n_samples)
    """
    # Find the epoch file matching the condition suffix
    prefix = "csd_rejevs_icacomprem_gaprem_filt_rivindiff_"
    mat_name = f"{prefix}{condition_suffix}.mat"
    mat_path = epochs_dir / mat_name

    if not mat_path.exists():
        log.warning("Epoch file not found: %s", mat_path)
        return None

    eeg = scipy.io.loadmat(str(mat_path), simplify_cells=True)

    ch_names = [ch["labels"] for ch in eeg["chanlocs"]]
    sfreq = int(eeg["fs"])
    epochs = eeg["epochs"]

    # Handle both single-epoch and multi-epoch cases
    if isinstance(epochs, np.ndarray) and epochs.ndim == 2:
        # Single epoch: shape (n_channels, n_samples)
        if epoch_index != 0:
            log.warning("Only 1 epoch available, ignoring epoch_index=%d", epoch_index)
        epoch_data = epochs
    elif isinstance(epochs, (list, np.ndarray)):
        # Multiple epochs
        if epoch_index >= len(epochs):
            log.warning(
                "Requested epoch %d but only %d available",
                epoch_index, len(epochs),
            )
            return None
        epoch_data = epochs[epoch_index]
    else:
        log.warning("Unexpected epochs type: %s", type(epochs))
        return None

    return epoch_data.astype(np.float64), ch_names, sfreq


def load_behavioral_switches(
    behavior_dir: Path,
    param_set: int,
    session_index: int = 0,
    sfreq: int = 360,
) -> list[dict] | None:
    """Load behavioral switch times from button-press .mat files.

    Returns a list of switch dicts with keys: time, from_key, to_key, sample.
    Returns None if behavioral data is unavailable.
    """
    if not behavior_dir.exists():
        log.warning("Behavior directory not found: %s", behavior_dir)
        return None

    # Find the rivalry behavioral file: BR_Rivalry_*_*.mat
    beh_files = list(behavior_dir.glob("BR_Rivalry_*.mat"))
    # Exclude practice files
    beh_files = [f for f in beh_files if "PRACT" not in f.name]

    if not beh_files:
        log.warning("No BR_Rivalry behavioral file found in %s", behavior_dir)
        return None

    beh_file = beh_files[0]  # Should be exactly one

    try:
        beh = scipy.io.loadmat(str(beh_file), simplify_cells=True)
        results_beh = beh["results"]
    except Exception as e:
        log.warning("Failed to load behavioral file %s: %s", beh_file, e)
        return None

    # Find sessions matching the requested paramSet
    matching_indices = []
    for i, r in enumerate(results_beh):
        try:
            if r["params"]["paramSet"] == param_set:
                matching_indices.append(i)
        except (KeyError, TypeError):
            continue

    if not matching_indices:
        log.warning(
            "No sessions with paramSet=%d found in %s", param_set, beh_file.name,
        )
        return None

    if session_index >= len(matching_indices):
        log.warning(
            "Requested session %d but only %d paramSet=%d sessions found",
            session_index, len(matching_indices), param_set,
        )
        return None

    r = results_beh[matching_indices[session_index]]
    psycho = r["psycho"]
    t_key_press = psycho["tKeyPress"]
    response_key = psycho["responseKey"]

    # Extract perceptual switches (any key change)
    switches = []
    for i in range(1, len(response_key)):
        if response_key[i] != response_key[i - 1]:
            switches.append({
                "time": float(t_key_press[i]),
                "from_key": int(response_key[i - 1]),
                "to_key": int(response_key[i]),
                "sample": int(t_key_press[i] * sfreq),
            })

    return switches


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    low: float,
    high: float,
    sfreq: float,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfilt(sos, signal)


# ---------------------------------------------------------------------------
# Transition detection pipeline (fig7 equivalent)
# ---------------------------------------------------------------------------

def run_transition_detection(
    signal: np.ndarray,
    sfreq: int,
    cfg: dict,
) -> dict:
    """Run transition detection on a single channel signal.

    Returns a dict with:
        embedding_meta, n_windows, n_changepoints, changepoint_times,
        image_distance_stats, etc.
    """
    # Takens embedding with auto+fallback
    cloud, meta = embed_channel(
        signal,
        band=cfg["embedding_band"],
        sfreq=float(sfreq),
        condition_threshold=cfg["condition_threshold"],
    )

    # TransitionDetector
    det = TransitionDetector(
        window_size=cfg["td_window_size"],
        step_size=cfg["td_step_size"],
        max_dim=cfg["td_max_dim"],
        backend=cfg["td_backend"],
        subsample=cfg["td_subsample"],
    )

    result = det.fit_transform(cloud, seed=cfg["seed"])

    # Changepoint detection
    cp_kwargs = {"method": cfg["td_changepoint_method"]}
    if cfg["td_changepoint_threshold"] is not None:
        cp_kwargs["threshold"] = cfg["td_changepoint_threshold"]
    changepoints = det.detect_changepoints(**cp_kwargs)

    # Convert changepoint indices to times
    window_centers = result["window_centers"]
    dist_x = (window_centers[:-1] + window_centers[1:]) / 2.0
    detected_samples = np.array(
        [dist_x[cp] for cp in changepoints if cp < len(dist_x)]
    )
    detected_times = detected_samples / sfreq

    image_distances = result["image_distances"]

    return {
        "embedding_method": meta["method"],
        "embedding_delay": meta["delay"],
        "embedding_dimension": meta["dimension"],
        "embedding_condition": meta["condition_number"],
        "fallback_reason": meta["fallback_reason"],
        "cloud_shape": list(cloud.shape),
        "n_windows": len(window_centers),
        "n_changepoints": len(changepoints),
        "changepoint_times": detected_times.tolist(),
        "image_distance_mean": float(image_distances.mean()) if len(image_distances) else 0.0,
        "image_distance_std": float(image_distances.std()) if len(image_distances) else 0.0,
        "image_distance_max": float(image_distances.max()) if len(image_distances) else 0.0,
    }


# ---------------------------------------------------------------------------
# Precision / recall evaluation
# ---------------------------------------------------------------------------

def evaluate_alignment(
    switch_samples: np.ndarray,
    detected_samples: np.ndarray,
    tolerance_samples: int,
) -> dict:
    """Compute precision and recall of changepoints vs behavioral switches."""
    if len(detected_samples) == 0:
        return {
            "hits": 0,
            "false_alarms": 0,
            "precision": 0.0,
            "recall": 0.0,
            "n_switches": int(len(switch_samples)),
            "n_changepoints": 0,
        }

    if len(switch_samples) == 0:
        return {
            "hits": 0,
            "false_alarms": int(len(detected_samples)),
            "precision": 0.0,
            "recall": 0.0,
            "n_switches": 0,
            "n_changepoints": int(len(detected_samples)),
        }

    # Hits: switches with a nearby changepoint
    hits = 0
    for sw_s in switch_samples:
        for det_s in detected_samples:
            if abs(det_s - sw_s) < tolerance_samples:
                hits += 1
                break

    # False alarms: changepoints with no nearby switch
    false_alarms = 0
    for det_s in detected_samples:
        matched = any(abs(det_s - sw_s) < tolerance_samples for sw_s in switch_samples)
        if not matched:
            false_alarms += 1

    n_det = len(detected_samples)
    n_sw = len(switch_samples)
    precision = hits / n_det if n_det > 0 else 0.0
    recall = hits / n_sw if n_sw > 0 else 0.0

    return {
        "hits": hits,
        "false_alarms": false_alarms,
        "precision": precision,
        "recall": recall,
        "n_switches": int(n_sw),
        "n_changepoints": int(n_det),
    }


# ---------------------------------------------------------------------------
# Binding analysis pipeline (fig8 equivalent)
# ---------------------------------------------------------------------------

def run_binding_analysis(
    signal_x: np.ndarray,
    signal_y: np.ndarray,
    sfreq: int,
    switch_times: np.ndarray | None,
    cfg: dict,
) -> dict:
    """Run sliding-window binding analysis on a channel pair.

    Returns a dict with: mean_binding, binding_rho, binding_p, binding_scores, etc.
    """
    # Estimate shared embedding parameters from full signals
    cloud_x, meta_x = embed_channel(
        signal_x, band=cfg["embedding_band"], sfreq=float(sfreq),
        condition_threshold=cfg["condition_threshold"],
    )
    cloud_y, meta_y = embed_channel(
        signal_y, band=cfg["embedding_band"], sfreq=float(sfreq),
        condition_threshold=cfg["condition_threshold"],
    )

    # Shared parameters: average delay, capped dimension
    shared_delay = round((meta_x["delay"] + meta_y["delay"]) / 2)
    shared_dim = min(meta_x["dimension"], meta_y["dimension"])
    shared_dim = min(shared_dim, cfg["binding_shared_dim"])

    # Sliding window binding
    window_samples = int(cfg["binding_window_seconds"] * sfreq)
    step_samples = int(cfg["binding_step_seconds"] * sfreq)
    n_samples = len(signal_x)

    starts = list(range(0, n_samples - window_samples + 1, step_samples))
    if not starts:
        log.warning("Signal too short for binding windows")
        return {"mean_binding": float("nan"), "binding_rho": float("nan"),
                "binding_p": float("nan"), "n_binding_windows": 0}

    binding_scores = []
    window_centers_sec = []

    for start in starts:
        end = start + window_samples
        seg_x = signal_x[start:end]
        seg_y = signal_y[start:end]
        center_sec = (start + end) / 2.0 / sfreq
        window_centers_sec.append(center_sec)

        emb_x = TakensEmbedder(delay=shared_delay, dimension=shared_dim)
        emb_y = TakensEmbedder(delay=shared_delay, dimension=shared_dim)
        joint_emb = JointEmbedder(
            delays=[shared_delay, shared_delay],
            dimensions=[shared_dim, shared_dim],
        )

        bd = BindingDetector(
            max_dim=cfg["binding_max_dim"],
            baseline=cfg["binding_baseline"],
            image_resolution=cfg["binding_image_resolution"],
            image_sigma=cfg["binding_image_sigma"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bd.fit(
                seg_x, seg_y,
                marginal_embedder_x=emb_x,
                marginal_embedder_y=emb_y,
                joint_embedder=joint_emb,
                subsample=cfg["binding_subsample"],
                seed=cfg["seed"],
            )

        binding_scores.append(bd.binding_score())

    binding_scores = np.array(binding_scores)
    window_centers_sec = np.array(window_centers_sec)

    # Compute correlation with switch density (if behavioral data available)
    binding_rho = float("nan")
    binding_p = float("nan")

    if switch_times is not None and len(switch_times) > 0:
        n_switches_per_window = np.zeros(len(starts), dtype=int)
        for i, start in enumerate(starts):
            win_start_sec = start / sfreq
            win_end_sec = (start + window_samples) / sfreq
            n_sw = np.sum(
                (switch_times >= win_start_sec) & (switch_times <= win_end_sec)
            )
            n_switches_per_window[i] = n_sw

        # Only compute correlation if there is variance in switch counts
        if n_switches_per_window.std() > 0 and binding_scores.std() > 0:
            rho, p = spearmanr(binding_scores, n_switches_per_window)
            binding_rho = float(rho)
            binding_p = float(p)

    return {
        "mean_binding": float(binding_scores.mean()),
        "std_binding": float(binding_scores.std()),
        "min_binding": float(binding_scores.min()),
        "max_binding": float(binding_scores.max()),
        "binding_rho": binding_rho,
        "binding_p": binding_p,
        "n_binding_windows": len(starts),
        "shared_delay": shared_delay,
        "shared_dimension": shared_dim,
        "embedding_x_method": meta_x["method"],
        "embedding_y_method": meta_y["method"],
        "binding_scores": binding_scores.tolist(),
        "window_centers_sec": window_centers_sec.tolist(),
    }


# ---------------------------------------------------------------------------
# Per-subject pipeline
# ---------------------------------------------------------------------------

def process_subject(
    subject: dict,
    cfg: dict,
    skip_binding: bool = False,
) -> dict | None:
    """Run the full pipeline on a single subject.

    Returns a results dict or None on total failure.
    """
    name = subject["name"]
    log.info("Processing: %s", name)

    # 1. Load rivalry epoch
    loaded = load_rivalry_epoch(
        subject["epochs_dir"],
        cfg["epoch_condition_suffix"],
        cfg["epoch_index"],
    )
    if loaded is None:
        log.error("  Failed to load epoch data for %s", name)
        return None

    epoch_data, ch_names, sfreq = loaded
    duration = epoch_data.shape[1] / sfreq
    log.info("  Epoch: %d channels x %d samples (%.0fs at %d Hz)",
             epoch_data.shape[0], epoch_data.shape[1], duration, sfreq)

    # 2. Extract transition channel
    trans_ch = cfg["transition_channel"]
    if trans_ch not in ch_names:
        log.error("  Channel %s not found. Available: %s", trans_ch, ch_names)
        return None

    ch_idx = ch_names.index(trans_ch)
    signal_raw = epoch_data[ch_idx]

    # Bandpass filter
    signal = bandpass_filter(
        signal_raw,
        cfg["bandpass_low"],
        cfg["bandpass_high"],
        float(sfreq),
        cfg["bandpass_order"],
    )

    # 3. Load behavioral data
    switches = load_behavioral_switches(
        subject["behavior_dir"],
        cfg["epoch_param_set"],
        session_index=cfg["epoch_index"],
        sfreq=sfreq,
    )
    n_switches = len(switches) if switches else 0
    switch_times = np.array([s["time"] for s in switches]) if switches else np.array([])
    switch_samples = np.array([s["sample"] for s in switches]) if switches else np.array([])

    log.info("  Behavioral switches: %d", n_switches)

    # 4. Run transition detection
    log.info("  Running transition detection (%s channel) ...", trans_ch)
    t0 = time.time()
    td_result = run_transition_detection(signal, sfreq, cfg)
    td_elapsed = time.time() - t0
    log.info("  Transition detection: %d changepoints in %.1fs",
             td_result["n_changepoints"], td_elapsed)

    # 5. Evaluate precision/recall
    detected_times = np.array(td_result["changepoint_times"])
    detected_samples = (detected_times * sfreq).astype(int)

    eval_results = {}
    for tol_sec in cfg["eval_tolerances"]:
        tol_samp = int(tol_sec * sfreq)
        ev = evaluate_alignment(switch_samples, detected_samples, tol_samp)
        eval_results[f"{tol_sec:.0f}s"] = ev
        log.info("  [%ss tol] Precision=%.1f%%, Recall=%.1f%%, FA=%d",
                 f"{tol_sec:.0f}", ev["precision"] * 100, ev["recall"] * 100,
                 ev["false_alarms"])

    # 6. Binding analysis (optional)
    binding_result = None
    if not skip_binding:
        bind_ch_x, bind_ch_y = cfg["binding_pair"]
        if bind_ch_x in ch_names and bind_ch_y in ch_names:
            idx_x = ch_names.index(bind_ch_x)
            idx_y = ch_names.index(bind_ch_y)
            sig_x = bandpass_filter(
                epoch_data[idx_x].astype(np.float64),
                cfg["bandpass_low"], cfg["bandpass_high"],
                float(sfreq), cfg["bandpass_order"],
            )
            sig_y = bandpass_filter(
                epoch_data[idx_y].astype(np.float64),
                cfg["bandpass_low"], cfg["bandpass_high"],
                float(sfreq), cfg["bandpass_order"],
            )

            log.info("  Running binding analysis (%s-%s) ...", bind_ch_x, bind_ch_y)
            t0 = time.time()
            binding_result = run_binding_analysis(
                sig_x, sig_y, sfreq, switch_times, cfg,
            )
            b_elapsed = time.time() - t0
            log.info("  Binding: mean=%.2f, rho=%.3f, p=%.4f in %.1fs",
                     binding_result["mean_binding"],
                     binding_result["binding_rho"],
                     binding_result["binding_p"],
                     b_elapsed)
        else:
            missing = [c for c in [bind_ch_x, bind_ch_y] if c not in ch_names]
            log.warning("  Binding channels not found: %s", missing)

    # 7. Assemble result
    result = {
        "subject_name": name,
        "subject_path": str(subject["path"]),
        "sampling_rate": sfreq,
        "epoch_duration_s": duration,
        "n_channels": epoch_data.shape[0],
        "channel_names": ch_names,
        "transition_channel": trans_ch,
        "n_switches": n_switches,
        "transition_detection": td_result,
        "evaluation": eval_results,
        "binding": binding_result,
        "timestamp": datetime.now().isoformat(),
    }

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    results: list[dict],
    tolerances: list[float],
) -> list[dict]:
    """Aggregate per-subject results into summary rows for CSV.

    Each row has: subject_id, n_changepoints, n_switches,
    precision_3s, recall_3s, precision_5s, recall_5s,
    mean_binding, binding_rho, binding_p
    """
    rows = []
    for r in results:
        row = {
            "subject_id": r["subject_name"],
            "n_changepoints": r["transition_detection"]["n_changepoints"],
            "n_switches": r["n_switches"],
        }

        # Precision/recall at each tolerance
        for tol in tolerances:
            key = f"{tol:.0f}s"
            ev = r["evaluation"].get(key, {})
            row[f"precision_{key}"] = ev.get("precision", float("nan"))
            row[f"recall_{key}"] = ev.get("recall", float("nan"))

        # Binding metrics
        bd = r.get("binding")
        if bd is not None:
            row["mean_binding"] = bd.get("mean_binding", float("nan"))
            row["binding_rho"] = bd.get("binding_rho", float("nan"))
            row["binding_p"] = bd.get("binding_p", float("nan"))
        else:
            row["mean_binding"] = float("nan")
            row["binding_rho"] = float("nan")
            row["binding_p"] = float("nan")

        rows.append(row)

    return rows


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    """Write summary CSV from aggregated rows."""
    if not rows:
        log.warning("No results to write.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Summary CSV written to: %s", output_path)


def save_subject_json(result: dict, output_dir: Path) -> Path:
    """Save per-subject result to JSON."""
    # Sanitize name for filename
    safe_name = result["subject_name"].replace(" ", "_").replace("/", "_")
    path = output_dir / f"{safe_name}.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=convert)

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure logging to both console and file."""
    level = logging.DEBUG if verbose else logging.INFO

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    ))

    # File handler
    log_path = output_dir / "batch_eeg.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))

    log.setLevel(logging.DEBUG)
    log.addHandler(console)
    log.addHandler(file_handler)

    log.info("Log file: %s", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch EEG analysis pipeline for Nie/Katyal/Engel (2023) "
            "binocular rivalry dataset. Runs transition detection and "
            "binding analysis across multiple subjects."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=Path("data/eeg/rivalry_ssvep"),
        help="Root directory containing subject folders (default: data/eeg/rivalry_ssvep)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("results/batch_eeg"),
        help="Output directory for results (default: results/batch_eeg)",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to YAML config file (default: use built-in defaults)",
    )
    parser.add_argument(
        "--n-subjects", "-n",
        type=int,
        default=None,
        help="Limit to first N subjects (for testing)",
    )
    parser.add_argument(
        "--skip-binding",
        action="store_true",
        help="Skip binding analysis (much faster, transition detection only)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help='Process specific subjects by name substring (comma-separated). '
             'E.g., "3629,3630" to match subject directories containing those IDs.',
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover subjects and print plan, but do not process",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir, args.verbose)

    # Load config
    cfg = load_config(args.config)
    log.info("Configuration loaded (seed=%d)", cfg["seed"])

    # Set global seed
    set_seed(cfg["seed"])

    # Discover subjects
    data_dir = args.data_dir.resolve()
    all_subjects = discover_subjects(data_dir)
    log.info("Discovered %d subject directories in %s", len(all_subjects), data_dir)

    if not all_subjects:
        log.error(
            "No subject directories found. Expected structure:\n"
            "  %s/<SubjectName>/Epochs/*.mat\n"
            "  %s/<SubjectName>/Behavior/*.mat\n"
            "Download data with: python scripts/download_eeg.py",
            data_dir, data_dir,
        )
        sys.exit(1)

    # Filter by name substring if requested
    if args.subjects:
        name_filters = [s.strip() for s in args.subjects.split(",")]
        filtered = [
            s for s in all_subjects
            if any(f in s["name"] for f in name_filters)
        ]
        log.info("Filtered to %d subjects matching: %s", len(filtered), name_filters)
        all_subjects = filtered

    # Limit number of subjects
    if args.n_subjects is not None:
        all_subjects = all_subjects[:args.n_subjects]
        log.info("Limited to first %d subjects", args.n_subjects)

    # Print plan
    log.info("")
    log.info("=" * 60)
    log.info("Batch EEG Analysis Pipeline")
    log.info("  Dataset: Nie/Katyal/Engel (2023)")
    log.info("  Subjects: %d", len(all_subjects))
    log.info("  Transition channel: %s", cfg["transition_channel"])
    log.info("  Binding: %s", "SKIP" if args.skip_binding else
             f"{cfg['binding_pair'][0]}-{cfg['binding_pair'][1]}")
    log.info("  Output: %s", output_dir)
    log.info("=" * 60)
    log.info("")

    for i, s in enumerate(all_subjects):
        log.info("  [%d] %s", i + 1, s["name"])

    if args.dry_run:
        log.info("")
        log.info("Dry run complete. Use without --dry-run to process.")
        return

    # Process subjects
    log.info("")
    results = []
    failures = []
    total_t0 = time.time()

    for i, subject in enumerate(all_subjects):
        log.info("")
        log.info("--- Subject %d/%d ---", i + 1, len(all_subjects))

        try:
            t0 = time.time()
            result = process_subject(subject, cfg, skip_binding=args.skip_binding)
            elapsed = time.time() - t0

            if result is not None:
                results.append(result)

                # Save per-subject JSON
                json_path = save_subject_json(result, output_dir)
                log.info("  Saved: %s (%.1fs)", json_path.name, elapsed)

                # ETA estimate
                avg_time = (time.time() - total_t0) / (i + 1)
                remaining = avg_time * (len(all_subjects) - i - 1)
                log.info("  ETA for remaining %d subjects: %.0f min",
                         len(all_subjects) - i - 1, remaining / 60)
            else:
                failures.append({"subject": subject["name"], "error": "returned None"})
                log.error("  FAILED: %s (returned None)", subject["name"])

        except Exception as e:
            elapsed = time.time() - t0
            failures.append({
                "subject": subject["name"],
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            log.error("  FAILED: %s in %.1fs", subject["name"], elapsed)
            log.error("  Error: %s", e)
            log.debug("  Traceback:\n%s", traceback.format_exc())

    total_elapsed = time.time() - total_t0

    # Aggregate and write summary CSV
    log.info("")
    log.info("=" * 60)
    log.info("BATCH COMPLETE")
    log.info("  Total time: %.1f min", total_elapsed / 60)
    log.info("  Subjects processed: %d/%d", len(results), len(all_subjects))
    log.info("  Failures: %d", len(failures))
    log.info("=" * 60)

    if results:
        rows = aggregate_results(results, cfg["eval_tolerances"])
        csv_path = output_dir / "batch_eeg_summary.csv"
        write_summary_csv(rows, csv_path)

        # Print summary table
        log.info("")
        log.info("Summary:")
        header = (
            f"{'Subject':<40s} {'CPs':>4s} {'SW':>4s} "
            f"{'P@3s':>6s} {'R@3s':>6s} {'P@5s':>6s} {'R@5s':>6s} "
            f"{'Bind':>6s} {'Rho':>6s} {'p':>6s}"
        )
        log.info(header)
        log.info("-" * len(header))
        for row in rows:
            p3 = row.get("precision_3s", float("nan"))
            r3 = row.get("recall_3s", float("nan"))
            p5 = row.get("precision_5s", float("nan"))
            r5 = row.get("recall_5s", float("nan"))
            mb = row.get("mean_binding", float("nan"))
            rho = row.get("binding_rho", float("nan"))
            bp = row.get("binding_p", float("nan"))

            def fmt(v: float) -> str:
                return f"{v:.3f}" if not np.isnan(v) else "  N/A"

            log.info(
                f"{row['subject_id']:<40s} "
                f"{row['n_changepoints']:4d} {row['n_switches']:4d} "
                f"{fmt(p3):>6s} {fmt(r3):>6s} {fmt(p5):>6s} {fmt(r5):>6s} "
                f"{fmt(mb):>6s} {fmt(rho):>6s} {fmt(bp):>6s}"
            )

        # Grand averages
        log.info("-" * len(header))
        avg_row = {}
        numeric_keys = [
            "n_changepoints", "n_switches",
            "precision_3s", "recall_3s", "precision_5s", "recall_5s",
            "mean_binding", "binding_rho",
        ]
        for k in numeric_keys:
            vals = [r[k] for r in rows if not np.isnan(r.get(k, float("nan")))]
            avg_row[k] = np.mean(vals) if vals else float("nan")

        def fmt(v: float) -> str:
            return f"{v:.3f}" if not np.isnan(v) else "  N/A"

        log.info(
            f"{'MEAN':<40s} "
            f"{avg_row['n_changepoints']:4.0f} {avg_row['n_switches']:4.0f} "
            f"{fmt(avg_row['precision_3s']):>6s} {fmt(avg_row['recall_3s']):>6s} "
            f"{fmt(avg_row['precision_5s']):>6s} {fmt(avg_row['recall_5s']):>6s} "
            f"{fmt(avg_row['mean_binding']):>6s} {fmt(avg_row['binding_rho']):>6s} "
            f"{'':>6s}"
        )

    # Save failure log
    if failures:
        failure_path = output_dir / "failures.json"
        with open(failure_path, "w") as f:
            json.dump(failures, f, indent=2)
        log.info("")
        log.info("Failure details written to: %s", failure_path)
        for fail in failures:
            log.info("  FAILED: %s -- %s", fail["subject"], fail["error"])

    log.info("")
    log.info("Results directory: %s", output_dir)


if __name__ == "__main__":
    main()
