#!/usr/bin/env python3
"""Branch 4 tracer bullet — zigzag temporal persistence on reservoir SR drift.

Architecture #7 from the hybrid topological monitoring document. Treat
overlapping reservoir windows as a sequence of point clouds (the temporal
analogue of `att.llm.zigzag.ZigzagLayerAnalyzer`'s "layer" axis), then run
zigzag persistent homology over the sequence to find topological features
that persist across multiple windows.

Detection: H1 features whose **birth** falls in the post-onset half AND whose
lifetime exceeds the median healthy-region lifetime are evidence of structural
change visible only to a multi-window primitive (snapshot PH on each window
independently can never see them).

Baseline to beat: per-window ripser PH on identical windows, scalar = sum of
H1 lifetimes, 2σ alarm.

This script does NOT depend on att.llm.zigzag (which only exists on the
TDA-LLM branches). The zigzag construction is inlined using dionysus
directly — same logic as `_build_and_compute_zigzag()` from
`att/llm/zigzag.py` on the LLM branches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dionysus

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
STRIDE = 500  # 26 windows total — appropriate for zigzag
N_HEALTHY = rdm.N_HEALTHY  # 5000
N_TOTAL = rdm.N_TOTAL  # 15000
SUBSAMPLE_PER_WINDOW = 60  # MUST be constant across all windows for zigzag vertex indexing.
# Triangle enumeration is O(n_pts^3); 60^3 = 216k is comfortable for a tracer.
N_PCA = 20
THRESHOLD_PERCENTILE = 70  # adaptive VR-complex threshold (drift clouds are 30x larger
# in scale than healthy clouds; p30 is too restrictive — see handoff)
SEED = 42

DATA_DIR = os.path.join(REPO_ROOT, "data", "zigzag_reservoir")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "zigzag_reservoir")
HANDOFF_DIR = os.path.join(REPO_ROOT, ".claude", "handoff")


# ---------------------------------------------------------------------------
# PC1
# ---------------------------------------------------------------------------

def pc1(states: np.ndarray) -> np.ndarray:
    Xc = states - states.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[0]


# ---------------------------------------------------------------------------
# Inlined zigzag (from att/llm/zigzag.py — loader-decoupled version)
# ---------------------------------------------------------------------------

def pca_reduce_clouds(clouds: list[np.ndarray], n_components: int) -> list[np.ndarray]:
    """PCA-reduce each cloud independently."""
    out = []
    for cloud in clouds:
        n_comp = min(n_components, cloud.shape[0] - 1, cloud.shape[1])
        if n_comp < cloud.shape[1]:
            pca = PCA(n_components=n_comp)
            out.append(pca.fit_transform(cloud))
        else:
            out.append(cloud)
    return out


def _presence_to_intervals(present: np.ndarray) -> list[float]:
    """Boolean presence array → flat [appear, disappear, appear, ...] list."""
    intervals: list[float] = []
    in_interval = False
    for t, p in enumerate(present):
        if p and not in_interval:
            intervals.append(float(t))
            in_interval = True
        elif not p and in_interval:
            intervals.append(float(t))
            in_interval = False
    if in_interval:
        intervals.append(float(len(present)))
    return intervals


def _edge_intervals(i: int, j: int, dist_matrices: list[np.ndarray],
                    threshold: float, total_times: int) -> list[float]:
    n_layers = len(dist_matrices)
    present = np.zeros(total_times, dtype=bool)
    for k in range(n_layers):
        t_layer = 2 * k
        if dist_matrices[k][i, j] <= threshold:
            present[t_layer] = True
        if k < n_layers - 1:
            t_union = 2 * k + 1
            min_dist = min(dist_matrices[k][i, j], dist_matrices[k + 1][i, j])
            if min_dist <= threshold:
                present[t_union] = True
    return _presence_to_intervals(present)


def _edge_presence_matrix(n_pts: int, dist_matrices: list[np.ndarray],
                          threshold: float, total_times: int) -> np.ndarray:
    n_layers = len(dist_matrices)
    present = np.zeros((n_pts, n_pts, total_times), dtype=bool)
    for k in range(n_layers):
        t_layer = 2 * k
        present[:, :, t_layer] = dist_matrices[k] <= threshold
        if k < n_layers - 1:
            t_union = 2 * k + 1
            min_dist = np.minimum(dist_matrices[k], dist_matrices[k + 1])
            present[:, :, t_union] = min_dist <= threshold
    return present


def _triangle_intervals(i: int, j: int, k: int,
                        edge_at_time: np.ndarray, total_times: int) -> list[float]:
    present = (edge_at_time[i, j, :total_times]
               & edge_at_time[i, k, :total_times]
               & edge_at_time[j, k, :total_times])
    return _presence_to_intervals(present)


def build_and_compute_zigzag(dist_matrices: list[np.ndarray],
                             n_pts: int,
                             n_layers: int,
                             threshold: float,
                             max_dim: int = 1) -> dict[int, np.ndarray]:
    """Build the zigzag filtration and compute persistence.

    Mirrors `att.llm.zigzag.ZigzagLayerAnalyzer._build_and_compute_zigzag` exactly.
    Time axis: t=2k = X_k, t=2k+1 = X_k ∪ X_{k+1}. So total_times = 2*n_layers - 1.
    Cloud index = zigzag_time / 2.
    """
    total_times = 2 * n_layers - 1

    simplex_list = []
    times_list: list[list[float]] = []

    # Vertices — present throughout
    for i in range(n_pts):
        simplex_list.append(dionysus.Simplex([i]))
        times_list.append([0, total_times])

    # Edges
    if max_dim >= 1:
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                intervals = _edge_intervals(i, j, dist_matrices, threshold, total_times)
                if intervals:
                    simplex_list.append(dionysus.Simplex([i, j]))
                    times_list.append(intervals)

    # Triangles (needed for H1)
    if max_dim >= 1:
        edge_at_time = _edge_presence_matrix(n_pts, dist_matrices, threshold, total_times)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                for k in range(j + 1, n_pts):
                    intervals = _triangle_intervals(i, j, k, edge_at_time, total_times)
                    if intervals:
                        simplex_list.append(dionysus.Simplex([i, j, k]))
                        times_list.append(intervals)

    f = dionysus.Filtration(simplex_list)
    zz, dgms, cells = dionysus.zigzag_homology_persistence(f, times_list)

    barcodes: dict[int, np.ndarray] = {}
    for dim in range(len(dgms)):
        bars: list[list[float]] = []
        for pt in dgms[dim]:
            b, d = pt.birth, pt.death
            if d == float("inf"):
                d = total_times
            # Zigzag time → cloud index
            bars.append([b / 2.0, d / 2.0])
        barcodes[dim] = np.array(bars) if bars else np.empty((0, 2))
    return barcodes


# ---------------------------------------------------------------------------
# Per-window snapshot baseline (ripser) — for comparison
# ---------------------------------------------------------------------------

def per_window_h1_total_persistence(clouds: list[np.ndarray], seed: int = SEED) -> np.ndarray:
    """Compute Σ (death - birth) over H1 features per window using ripser."""
    out = np.zeros(len(clouds))
    for i, c in enumerate(clouds):
        if c.shape[0] < 5:
            continue
        pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
        pa.fit_transform(c, subsample=None, seed=seed)
        if len(pa.diagrams_) > 1 and len(pa.diagrams_[1]) > 0:
            d = pa.diagrams_[1]
            life = d[:, 1] - d[:, 0]
            out[i] = float(life[life > 1e-12].sum())
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_zigzag_barcode(barcodes: dict[int, np.ndarray],
                        n_clouds: int,
                        onset_cloud_idx: float,
                        per_window_baseline: np.ndarray,
                        out_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})

    # H1 zigzag barcode
    ax = axes[0]
    bars_h1 = barcodes.get(1, np.empty((0, 2)))
    if len(bars_h1) > 0:
        order = np.argsort(bars_h1[:, 0])
        bars_h1 = bars_h1[order]
        lifetimes = bars_h1[:, 1] - bars_h1[:, 0]
        max_lt = max(lifetimes.max(), 1e-10)
        for i, (bar, life) in enumerate(zip(bars_h1, lifetimes)):
            color = plt.cm.viridis(life / max_lt)
            ax.barh(i, bar[1] - bar[0], left=bar[0], height=0.85,
                    color=color, alpha=0.85)
    ax.axvline(onset_cloud_idx, color="red", linestyle="--", linewidth=2,
               label=f"drift onset (cloud={onset_cloud_idx:.1f})")
    ax.set_xlim(-0.5, n_clouds + 0.5)
    ax.set_xlabel("zigzag time (cloud index)")
    ax.set_ylabel("H1 feature index")
    ax.set_title(f"Zigzag H1 Barcode — Reservoir SR Drift ({n_clouds} clouds)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2, axis="x")

    # Per-window baseline (snapshot PH)
    ax2 = axes[1]
    ax2.plot(np.arange(n_clouds), per_window_baseline, marker="o", lw=1.5, color="C1",
             label="snapshot H1 total persistence")
    ax2.axvline(onset_cloud_idx, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlim(-0.5, n_clouds + 0.5)
    ax2.set_xlabel("cloud index")
    ax2.set_ylabel("Σ (death - birth)")
    ax2.set_title("Per-window (snapshot PH) baseline at same stride")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Branch 4 — zigzag reservoir tracer")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mode", type=str, default="sr_drift",
                        choices=["sr_drift", "noise", "connection_death"])
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(HANDOFF_DIR, exist_ok=True)

    t0 = time.time()
    print(f"[branch 4] Generating reservoir trajectory: mode={args.mode} seed={args.seed}")
    states, _, _ = rdm.simulate_degradation(args.mode, args.seed)
    print(f"  states shape: {states.shape}")
    signal = pc1(states)

    starts = list(range(0, N_TOTAL - WINDOW + 1, args.stride))
    n_clouds = len(starts)
    onset_cloud = N_HEALTHY / args.stride  # fractional cloud index of onset
    print(f"  windows: {n_clouds} (W={WINDOW}, stride={args.stride})")
    print(f"  onset (cloud index, fractional): {onset_cloud:.2f}")

    # ---- Build constant-size point clouds via Takens embedding + subsample ----
    print(f"  building Takens-embedded point clouds (subsample={SUBSAMPLE_PER_WINDOW})...")
    clouds: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)
    for s in starts:
        sig = signal[s:s + WINDOW]
        try:
            emb = TakensEmbedder(delay="auto", dimension="auto")
            emb.fit(sig)
        except Exception:
            emb = TakensEmbedder(delay=10, dimension=3)
            emb.fit(sig)
        cloud = emb.transform(sig)
        # Constant subsample size required for zigzag (vertices are vertex-indexed across clouds)
        if cloud.shape[0] < SUBSAMPLE_PER_WINDOW:
            print(f"    cloud at start={s} too small ({cloud.shape[0]} < {SUBSAMPLE_PER_WINDOW}); padding")
            pad = SUBSAMPLE_PER_WINDOW - cloud.shape[0]
            cloud = np.vstack([cloud, cloud[rng.integers(0, cloud.shape[0], size=pad)]])
        else:
            idx = rng.choice(cloud.shape[0], size=SUBSAMPLE_PER_WINDOW, replace=False)
            idx.sort()
            cloud = cloud[idx]
        clouds.append(cloud)
    print(f"  built {len(clouds)} clouds, each {clouds[0].shape}")

    # ---- PCA-reduce each cloud independently (matches the LLM ZigzagLayerAnalyzer) ----
    print(f"  PCA-reducing clouds to n_components={N_PCA}")
    clouds_pca = pca_reduce_clouds(clouds, n_components=N_PCA)
    print(f"    cloud[0] after PCA: {clouds_pca[0].shape}")

    # ---- Distance matrices + adaptive threshold ----
    print(f"  computing distance matrices...")
    dist_matrices = [squareform(pdist(c)) for c in clouds_pca]
    n_pts = clouds_pca[0].shape[0]
    n_layers = len(clouds_pca)
    all_d = np.concatenate([dm[np.triu_indices(n_pts, k=1)] for dm in dist_matrices])
    threshold = float(np.percentile(all_d, THRESHOLD_PERCENTILE))
    print(f"    n_pts={n_pts}, n_layers={n_layers}, threshold (p{THRESHOLD_PERCENTILE})={threshold:.4f}")

    # ---- Run zigzag ----
    print(f"  running zigzag persistence (this is the expensive step)...")
    t_z = time.time()
    barcodes = build_and_compute_zigzag(dist_matrices, n_pts, n_layers, threshold, max_dim=1)
    print(f"    zigzag computation: {time.time() - t_z:.1f}s")
    n_h0 = len(barcodes.get(0, []))
    n_h1 = len(barcodes.get(1, []))
    print(f"    H0 features: {n_h0}, H1 features: {n_h1}")

    # ---- Lifetime stats ----
    h1 = barcodes.get(1, np.empty((0, 2)))
    h1_lifetimes = h1[:, 1] - h1[:, 0] if len(h1) > 0 else np.empty(0)
    h1_lifetimes = h1_lifetimes[h1_lifetimes > 0]
    if len(h1_lifetimes) > 0:
        print(f"    H1 lifetimes — mean={h1_lifetimes.mean():.2f}, "
              f"median={np.median(h1_lifetimes):.2f}, max={h1_lifetimes.max():.2f}")
    else:
        print(f"    no H1 features with positive lifetime")

    # ---- Detection logic: H1 features born in healthy region but with long lifetime
    #      that crosses the onset, vs features born in drift region ----
    healthy_births = h1[h1[:, 0] < onset_cloud] if len(h1) > 0 else np.empty((0, 2))
    drift_births = h1[h1[:, 0] >= onset_cloud] if len(h1) > 0 else np.empty((0, 2))
    healthy_life = healthy_births[:, 1] - healthy_births[:, 0] if len(healthy_births) > 0 else np.empty(0)
    drift_life = drift_births[:, 1] - drift_births[:, 0] if len(drift_births) > 0 else np.empty(0)

    healthy_median_life = float(np.median(healthy_life)) if len(healthy_life) > 0 else 0.0
    healthy_mean_life = float(np.mean(healthy_life)) if len(healthy_life) > 0 else 0.0
    drift_median_life = float(np.median(drift_life)) if len(drift_life) > 0 else 0.0
    drift_mean_life = float(np.mean(drift_life)) if len(drift_life) > 0 else 0.0

    # "Long-lived post-onset features": features born after onset with lifetime > healthy median
    long_post_onset = drift_births[(drift_births[:, 1] - drift_births[:, 0]) > max(healthy_median_life, 0.5)] \
        if len(drift_births) > 0 else np.empty((0, 2))
    n_long_post_onset = int(len(long_post_onset))

    # Earliest detection: birth time of the longest-lived post-onset feature
    if n_long_post_onset > 0:
        lifetimes = long_post_onset[:, 1] - long_post_onset[:, 0]
        earliest_long_birth = float(long_post_onset[:, 0].min())
        earliest_long_death = float(long_post_onset[:, 1].min())
        longest_life = float(lifetimes.max())
    else:
        earliest_long_birth = None
        earliest_long_death = None
        longest_life = None

    print(f"\n  H1 birth-region split:")
    print(f"    healthy region (birth<{onset_cloud:.1f}): n={len(healthy_births)}, "
          f"mean_life={healthy_mean_life:.2f}, median_life={healthy_median_life:.2f}")
    print(f"    drift region (birth>={onset_cloud:.1f}):  n={len(drift_births)}, "
          f"mean_life={drift_mean_life:.2f}, median_life={drift_median_life:.2f}")
    print(f"    n_long_post_onset (lifetime > healthy median): {n_long_post_onset}")
    if earliest_long_birth is not None:
        print(f"    earliest long post-onset feature: birth={earliest_long_birth:.2f}, "
              f"death={earliest_long_death:.2f}, longest={longest_life:.2f}")

    # ---- Per-window snapshot baseline ----
    print(f"\n  computing per-window snapshot PH baseline (ripser, same windows)...")
    baseline_signal = per_window_h1_total_persistence(clouds_pca, seed=args.seed)
    print(f"    baseline signal: {baseline_signal.tolist()}")
    healthy_base = baseline_signal[:int(np.floor(onset_cloud))]
    if len(healthy_base) >= 2 and healthy_base.std() > 1e-15:
        threshold_base = healthy_base.mean() + 2 * healthy_base.std()
    else:
        threshold_base = healthy_base.max() if len(healthy_base) > 0 else 0.0
    baseline_alarms = baseline_signal > threshold_base
    baseline_first_alarm = int(np.where(baseline_alarms)[0][0]) if baseline_alarms.any() else None
    print(f"    baseline 2σ threshold: {threshold_base:.4f}")
    print(f"    baseline first alarm cloud: {baseline_first_alarm}")

    # ---- Verdict ----
    # Zigzag wins if it finds a long-lived post-onset feature that the snapshot baseline misses
    # OR if it detects earlier (lower cloud index) than the baseline.
    zigzag_detects = n_long_post_onset > 0
    baseline_detects = baseline_alarms.any()

    # Compare detection points
    if zigzag_detects and baseline_first_alarm is None:
        verdict = "PASS"
        verdict_reason = f"zigzag finds {n_long_post_onset} long post-onset feature(s); snapshot baseline misses entirely"
    elif zigzag_detects and earliest_long_birth is not None and baseline_first_alarm is not None:
        if earliest_long_birth < baseline_first_alarm:
            verdict = "PASS"
            verdict_reason = (f"zigzag detects at cloud {earliest_long_birth:.2f} "
                              f"vs baseline at cloud {baseline_first_alarm} (earlier by "
                              f"{baseline_first_alarm - earliest_long_birth:.2f} clouds)")
        else:
            verdict = "PARTIAL"
            verdict_reason = (f"zigzag detects ({n_long_post_onset} long features) but no earlier than baseline "
                              f"(zigzag={earliest_long_birth:.2f}, baseline={baseline_first_alarm}); "
                              f"qualitative tracer purpose still satisfied — different feature set")
    elif zigzag_detects:
        verdict = "PASS"
        verdict_reason = f"zigzag finds {n_long_post_onset} long post-onset feature(s)"
    else:
        verdict = "FAIL"
        verdict_reason = "no long-lived post-onset H1 features in zigzag barcode"

    print(f"\n[branch 4] VERDICT: {verdict}")
    print(f"  reason: {verdict_reason}")

    # ---- Persist ----
    runtime_s = time.time() - t0
    output = {
        "branch": "experiment/zigzag-reservoir",
        "mode": args.mode,
        "seed": args.seed,
        "config": {
            "window": WINDOW,
            "stride": args.stride,
            "n_clouds": n_clouds,
            "subsample_per_window": SUBSAMPLE_PER_WINDOW,
            "n_pca": N_PCA,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "vr_threshold": threshold,
        },
        "n_clouds": n_clouds,
        "onset_cloud": onset_cloud,
        "barcode_h0_count": n_h0,
        "barcode_h1_count": n_h1,
        "h1_birth_region_split": {
            "healthy": {"n": int(len(healthy_births)),
                        "mean_lifetime": healthy_mean_life,
                        "median_lifetime": healthy_median_life},
            "drift": {"n": int(len(drift_births)),
                      "mean_lifetime": drift_mean_life,
                      "median_lifetime": drift_median_life},
        },
        "n_long_post_onset": n_long_post_onset,
        "earliest_long_birth": earliest_long_birth,
        "longest_post_onset_life": longest_life,
        "snapshot_baseline": {
            "signal": baseline_signal.tolist(),
            "threshold_2sigma": float(threshold_base),
            "first_alarm_cloud": baseline_first_alarm,
            "any_detect": bool(baseline_detects),
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "runtime_s": runtime_s,
    }
    json_path = os.path.join(DATA_DIR, f"results_{args.mode}_seed{args.seed}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  wrote {json_path}")

    # Save full barcodes for follow-up analysis
    npz_path = os.path.join(DATA_DIR, f"zigzag_barcodes_{args.mode}_seed{args.seed}.npz")
    np.savez_compressed(npz_path,
                        h0=barcodes.get(0, np.empty((0, 2))),
                        h1=barcodes.get(1, np.empty((0, 2))),
                        baseline_signal=baseline_signal)
    print(f"  wrote {npz_path}")

    fig_path = os.path.join(FIG_DIR, f"zigzag_barcode_{args.mode}_seed{args.seed}.png")
    plot_zigzag_barcode(barcodes, n_clouds, onset_cloud, baseline_signal, fig_path)
    print(f"  wrote {fig_path}")

    print(f"\n[branch 4] runtime: {runtime_s:.1f}s")
    print(f"[branch 4] one-liner: H1={n_h1} feats, "
          f"long_post_onset={n_long_post_onset}, "
          f"baseline_alarm=cloud{baseline_first_alarm}, verdict={verdict}")


if __name__ == "__main__":
    main()
