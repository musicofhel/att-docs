#!/usr/bin/env python3
"""Branch 8: Financial Regime Detection — Market State Topology.

Applies ATT's transition detection to financial time series (S&P 500 daily).
Detects market regime changes (bull->bear, low-vol->high-vol) from topological
signatures of price dynamics. Compares with VIX as ground truth.

Prior ATT results on financial data showed negative surrogate z-scores (-8 to -10),
so we approach with appropriate skepticism.

Experiments:
  1. Sliding-window topology on log returns — crisis detection
  2. Surrogate testing — phase-randomize + time-shuffle null models
  3. Topology vs VIX correlation — rolling H1 entropy vs VIX
  4. Bull vs Bear attractor topology — point cloud PH comparison

Usage:
    python -u scripts/branches/financial_regime_topology.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ATT imports
from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer
from att.transitions.detector import TransitionDetector
from att.surrogates import phase_randomize, time_shuffle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pprint(msg: str) -> None:
    """Print with flush for unbuffered output."""
    print(msg)
    sys.stdout.flush()


def rolling_stat(x: np.ndarray, window: int, stat: str = "variance") -> np.ndarray:
    """Compute rolling statistic over 1D array."""
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(window, len(x)):
        chunk = x[i - window:i]
        if stat == "variance":
            out[i] = np.var(chunk)
        elif stat == "autocorr":
            if np.std(chunk) < 1e-12:
                out[i] = 0.0
            else:
                c = chunk - np.mean(chunk)
                out[i] = np.corrcoef(c[:-1], c[1:])[0, 1]
        elif stat == "skewness":
            out[i] = stats.skew(chunk)
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def download_financial_data(start: str = "2005-01-01", end: str = "2025-12-31",
                            seed: int = 42) -> dict:
    """Download SPY and VIX data via yfinance."""
    pprint("=== Downloading financial data ===")
    try:
        import yfinance as yf
    except ImportError:
        pprint("ERROR: yfinance not installed. pip install yfinance")
        sys.exit(1)

    pprint(f"Downloading SPY {start} to {end}...")
    spy_df = yf.download("SPY", start=start, end=end, progress=False)
    pprint(f"Downloading VIX {start} to {end}...")
    vix_df = yf.download("^VIX", start=start, end=end, progress=False)

    # Align dates
    spy_close = spy_df["Close"].squeeze()
    vix_close = vix_df["Close"].squeeze()
    common_idx = spy_close.index.intersection(vix_close.index)
    spy_close = spy_close.loc[common_idx]
    vix_close = vix_close.loc[common_idx]

    spy = spy_close.values.astype(float)
    vix = vix_close.values.astype(float)
    dates = common_idx

    # Log returns
    returns = np.diff(np.log(spy))

    pprint(f"  SPY: {len(spy)} trading days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    pprint(f"  VIX: mean={np.mean(vix):.1f}, max={np.max(vix):.1f}")
    pprint(f"  Returns: mean={np.mean(returns)*252:.4f}/yr, std={np.std(returns)*np.sqrt(252):.4f}/yr")

    return {
        "spy": spy, "vix": vix, "returns": returns, "dates": dates,
        "date_start": dates[0].strftime("%Y-%m-%d"),
        "date_end": dates[-1].strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# Known crisis events
# ---------------------------------------------------------------------------

CRISIS_EVENTS = {
    "2008": {"label": "2008 Financial Crisis", "onset": "2008-09-15"},
    "2020": {"label": "2020 COVID Crash", "onset": "2020-03-09"},
    "2022": {"label": "2022 Rate Hike Selloff", "onset": "2022-01-03"},
}


def find_date_index(dates, target_str: str) -> int:
    """Find closest trading day index to a date string."""
    import pandas as pd
    target = pd.Timestamp(target_str)
    diffs = np.abs(dates - target)
    return int(np.argmin(diffs))


# ---------------------------------------------------------------------------
# Experiment 1: Sliding-window topology on returns
# ---------------------------------------------------------------------------

def run_exp1(returns: np.ndarray, dates, vix: np.ndarray,
             window_size: int, step_size: int, subsample: int,
             seed: int, fig_dir: Path) -> dict:
    """Sliding-window topology on log returns. Detect crises."""
    pprint("\n=== Experiment 1: Sliding-window topology on returns ===")

    # Run TransitionDetector
    pprint(f"  TransitionDetector: window={window_size}, step={step_size}, subsample={subsample}")
    detector = TransitionDetector(
        window_size=window_size, step_size=step_size,
        max_dim=1, backend="ripser", subsample=subsample,
    )

    t0 = time.time()
    result = detector.fit_transform(
        returns, seed=seed, embedding_dim=5, embedding_delay=1,
    )
    elapsed = time.time() - t0
    pprint(f"  TransitionDetector completed in {elapsed:.1f}s")

    centers = result["window_centers"]
    transition_scores = result["transition_scores"]
    distances = result["distances"]

    # Extract H1 entropy per window
    topo_ts = result["topology_timeseries"]
    h1_entropy = []
    for wt in topo_ts:
        if wt is not None and "persistence_entropy" in wt:
            pe = wt["persistence_entropy"]
            h1_entropy.append(pe[1] if len(pe) > 1 else 0.0)
        else:
            h1_entropy.append(0.0)
    h1_entropy = np.array(h1_entropy)

    # Detect changepoints
    changepoints = detector.detect_changepoints(method="cusum")
    cp_indices = [centers[cp] for cp in changepoints if cp < len(centers)]
    pprint(f"  Changepoints detected: {len(cp_indices)}")

    # Interpolate transition scores to full returns length
    score_centers = centers[:len(transition_scores)]
    topo_full = np.full(len(returns), np.nan)
    for i, c in enumerate(score_centers):
        if 0 <= c < len(returns):
            topo_full[c] = transition_scores[i]
    valid = ~np.isnan(topo_full)
    if np.sum(valid) > 2:
        interp_fn = interp1d(
            np.where(valid)[0], topo_full[valid],
            kind="linear", fill_value="extrapolate", bounds_error=False,
        )
        topo_interp = interp_fn(np.arange(len(returns)))
    else:
        topo_interp = topo_full

    # Check crisis detection
    # Use 2-sigma threshold from first 500 days as baseline
    baseline_end = min(500, len(topo_interp) // 4)
    baseline = topo_interp[:baseline_end]
    baseline = baseline[~np.isnan(baseline)]
    if len(baseline) > 10:
        threshold = np.mean(baseline) + 2 * np.std(baseline)
    else:
        threshold = np.nanmean(topo_interp) + 2 * np.nanstd(topo_interp)

    crisis_results = {}
    for key, event in CRISIS_EVENTS.items():
        onset_idx = find_date_index(dates[1:], event["onset"])  # dates[1:] for returns
        # Look in window [-252, +63] around onset (1yr before to 3mo after)
        search_start = max(0, onset_idx - 252)
        search_end = min(len(topo_interp), onset_idx + 63)
        region = topo_interp[search_start:search_end]

        # Find first exceedance of threshold in this region
        exceedances = np.where(region > threshold)[0]
        if len(exceedances) >= 3:
            # 3-consecutive-point criterion
            consecutive = False
            for j in range(len(exceedances) - 2):
                if exceedances[j+1] == exceedances[j] + 1 and exceedances[j+2] == exceedances[j] + 2:
                    detect_idx = search_start + exceedances[j]
                    lag = detect_idx - onset_idx  # negative = detected before onset
                    crisis_results[key] = {
                        "detected": True, "lag_days": int(lag),
                        "detect_date": str(dates[1:][detect_idx].strftime("%Y-%m-%d")) if detect_idx < len(dates) - 1 else "N/A",
                    }
                    consecutive = True
                    pprint(f"  {event['label']}: DETECTED, lag={lag} days")
                    break
            if not consecutive:
                # Relax: just need any exceedance
                detect_idx = search_start + exceedances[0]
                lag = detect_idx - onset_idx
                crisis_results[key] = {
                    "detected": True, "lag_days": int(lag),
                    "detect_date": str(dates[1:][detect_idx].strftime("%Y-%m-%d")) if detect_idx < len(dates) - 1 else "N/A",
                }
                pprint(f"  {event['label']}: DETECTED (relaxed), lag={lag} days")
        else:
            crisis_results[key] = {"detected": False, "lag_days": 0, "detect_date": "N/A"}
            pprint(f"  {event['label']}: NOT DETECTED")

    # --- Figure: time series + topology + VIX ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        ret_dates = dates[1:]

        # Panel 1: Log returns
        axes[0].plot(ret_dates, returns, color="steelblue", alpha=0.5, linewidth=0.3)
        axes[0].set_ylabel("Log Returns")
        axes[0].set_title("S&P 500 Daily Log Returns + Topological Transition Scores")
        for key, event in CRISIS_EVENTS.items():
            idx = find_date_index(ret_dates, event["onset"])
            axes[0].axvline(ret_dates[idx], color="red", linestyle="--", alpha=0.7, label=event["label"])
        axes[0].legend(fontsize=8)

        # Panel 2: Transition scores
        score_dates = [ret_dates[c] for c in score_centers if c < len(ret_dates)]
        axes[1].plot(score_dates[:len(transition_scores)], transition_scores[:len(score_dates)],
                     color="darkorange", linewidth=1.0)
        axes[1].axhline(threshold, color="gray", linestyle=":", alpha=0.7, label=f"2σ threshold={threshold:.0f}")
        axes[1].set_ylabel("Transition Score")
        for key, event in CRISIS_EVENTS.items():
            idx = find_date_index(ret_dates, event["onset"])
            axes[1].axvline(ret_dates[idx], color="red", linestyle="--", alpha=0.7)
        # Mark detections
        for key, cr in crisis_results.items():
            if cr["detected"] and cr["detect_date"] != "N/A":
                import pandas as pd
                det_ts = pd.Timestamp(cr["detect_date"])
                axes[1].axvline(det_ts, color="green", linestyle="-", alpha=0.7, linewidth=2)
        axes[1].legend(fontsize=8)

        # Panel 3: VIX
        axes[2].plot(dates, vix, color="purple", linewidth=0.8)
        axes[2].set_ylabel("VIX")
        axes[2].axhline(30, color="red", linestyle=":", alpha=0.5, label="VIX=30")
        axes[2].axhline(20, color="orange", linestyle=":", alpha=0.5, label="VIX=20")
        for key, event in CRISIS_EVENTS.items():
            idx = find_date_index(dates, event["onset"])
            axes[2].axvline(dates[idx], color="red", linestyle="--", alpha=0.7)
        axes[2].legend(fontsize=8)

        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
        plt.tight_layout()
        fig.savefig(fig_dir / "exp1_topology_crisis_detection.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'exp1_topology_crisis_detection.png'}")
    except Exception as e:
        pprint(f"  Warning: figure failed: {e}")

    return {
        "crisis_detection": crisis_results,
        "n_changepoints": len(cp_indices),
        "threshold": float(threshold),
        "mean_transition_score": float(np.nanmean(transition_scores)),
        "td_result": result,  # pass to other experiments
        "topo_interp": topo_interp,
        "h1_entropy": h1_entropy,
        "centers": centers,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Surrogate testing
# ---------------------------------------------------------------------------

def _compute_cloud_ph_stats(signal: np.ndarray, subsample: int, seed: int) -> dict:
    """Compute PH stats on Takens-embedded point cloud (fast, no sliding window)."""
    embedder = TakensEmbedder(delay=1, dimension=5)
    cloud = embedder.fit_transform(signal)
    pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
    result = pa.fit_transform(cloud, subsample=subsample, seed=seed)
    h0_ent = result["persistence_entropy"][0]
    h1_ent = result["persistence_entropy"][1] if len(result["persistence_entropy"]) > 1 else 0.0
    h1_count = len(pa.diagrams_[1]) if len(pa.diagrams_) > 1 else 0
    total_pers = sum(
        np.sum(dgm[:, 1] - dgm[:, 0]) for dgm in pa.diagrams_ if len(dgm) > 0
    )
    return {
        "h0_entropy": float(h0_ent),
        "h1_entropy": float(h1_ent),
        "h1_count": int(h1_count),
        "total_persistence": float(total_pers),
    }


def run_exp2(returns: np.ndarray, exp1_result: dict,
             subsample: int, n_surrogates: int, seed: int,
             fig_dir: Path) -> dict:
    """Surrogate testing: does topology capture nonlinear structure?

    Uses full-cloud PH comparison (fast) rather than sliding-window
    TransitionDetector per surrogate (~150s each would be prohibitive).
    """
    pprint("\n=== Experiment 2: Surrogate testing ===")

    # Real data PH stats
    pprint("  Computing PH on real embedded returns...")
    real_stats = _compute_cloud_ph_stats(returns, subsample=subsample, seed=seed)
    pprint(f"  Real: H0_ent={real_stats['h0_entropy']:.4f}, H1_ent={real_stats['h1_entropy']:.4f}, "
           f"H1_count={real_stats['h1_count']}, total_pers={real_stats['total_persistence']:.2f}")

    results = {}
    for surr_type in ["phase_randomize", "time_shuffle"]:
        pprint(f"\n  --- {surr_type} surrogates (n={n_surrogates}) ---")
        if surr_type == "phase_randomize":
            surrogates = phase_randomize(returns, n_surrogates=n_surrogates, seed=seed)
        else:
            surrogates = time_shuffle(returns, n_surrogates=n_surrogates, seed=seed)

        surr_h1_ent = []
        surr_h0_ent = []
        surr_total_pers = []
        surr_h1_count = []
        for i in range(n_surrogates):
            if i % 10 == 0:
                pprint(f"    Surrogate {i+1}/{n_surrogates}...")
            try:
                s = _compute_cloud_ph_stats(surrogates[i], subsample=subsample, seed=seed + i + 1)
                surr_h0_ent.append(s["h0_entropy"])
                surr_h1_ent.append(s["h1_entropy"])
                surr_h1_count.append(s["h1_count"])
                surr_total_pers.append(s["total_persistence"])
            except Exception as e:
                pprint(f"    Surrogate {i} failed: {e}")
                continue

        surr_h1_ent = np.array(surr_h1_ent)
        surr_total_pers = np.array(surr_total_pers)

        # Z-scores for each metric
        def _zscore(real_val, surr_vals):
            if len(surr_vals) > 2 and np.std(surr_vals) > 1e-12:
                return (real_val - np.mean(surr_vals)) / np.std(surr_vals)
            return 0.0

        z_h1_ent = _zscore(real_stats["h1_entropy"], surr_h1_ent)
        z_total_pers = _zscore(real_stats["total_persistence"], surr_total_pers)
        z_h0_ent = _zscore(real_stats["h0_entropy"], np.array(surr_h0_ent))

        pprint(f"    H1 entropy: real={real_stats['h1_entropy']:.4f}, surr={np.mean(surr_h1_ent):.4f}±{np.std(surr_h1_ent):.4f}, z={z_h1_ent:.2f}")
        pprint(f"    Total persistence: real={real_stats['total_persistence']:.2f}, surr={np.mean(surr_total_pers):.2f}±{np.std(surr_total_pers):.2f}, z={z_total_pers:.2f}")
        pprint(f"    H0 entropy: real={real_stats['h0_entropy']:.4f}, surr={np.mean(surr_h0_ent):.4f}±{np.std(surr_h0_ent):.4f}, z={z_h0_ent:.2f}")

        results[surr_type] = {
            "z_h1_entropy": float(z_h1_ent),
            "z_total_persistence": float(z_total_pers),
            "z_h0_entropy": float(z_h0_ent),
            "z_mean": float(z_total_pers),  # primary z-score
            "real_h1_entropy": real_stats["h1_entropy"],
            "real_total_persistence": real_stats["total_persistence"],
            "surr_mean_h1_ent": float(np.mean(surr_h1_ent)),
            "surr_std_h1_ent": float(np.std(surr_h1_ent)),
            "surr_mean_total_pers": float(np.mean(surr_total_pers)),
            "surr_std_total_pers": float(np.std(surr_total_pers)),
            "n_valid": len(surr_h1_ent),
            "all_surr_total_pers": surr_total_pers.tolist(),
        }

    # Interpretation
    pr_z = results["phase_randomize"]["z_total_persistence"]
    ts_z = results["time_shuffle"]["z_total_persistence"]
    if abs(pr_z) < 2.0:
        explanation = "Phase-randomized surrogates produce similar topology (|z|<2). Signal is likely LINEAR (driven by power spectrum)."
    elif abs(pr_z) >= 2.0 and abs(ts_z) >= 2.0:
        explanation = "Both surrogate types produce different topology (|z|>=2). Signal has NONLINEAR component beyond power spectrum."
    else:
        explanation = f"Mixed result: phase_randomize z={pr_z:.2f}, time_shuffle z={ts_z:.2f}. Partial nonlinear structure."

    pprint(f"\n  Interpretation: {explanation}")
    results["linear_explanation"] = explanation

    # --- Figure: surrogate distributions ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, surr_type in zip(axes, ["phase_randomize", "time_shuffle"]):
            r = results[surr_type]
            surr_vals = np.array(r["all_surr_total_pers"])
            ax.hist(surr_vals, bins=20, alpha=0.5, color="steelblue", label="Surrogate dist.")
            ax.axvline(r["real_total_persistence"], color="red", linewidth=2,
                       label=f"Real (total_pers={r['real_total_persistence']:.1f})")
            ax.set_title(f"{surr_type}\nz={r['z_total_persistence']:.2f}")
            ax.set_xlabel("Total Persistence")
            ax.legend(fontsize=8)
        plt.suptitle("Surrogate Testing: Real vs Null Topology", fontsize=14)
        plt.tight_layout()
        fig.savefig(fig_dir / "exp2_surrogate_testing.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'exp2_surrogate_testing.png'}")
    except Exception as e:
        pprint(f"  Warning: figure failed: {e}")

    # Remove bulky list before returning
    for k in ["phase_randomize", "time_shuffle"]:
        results[k].pop("all_surr_total_pers", None)

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Topology vs VIX correlation
# ---------------------------------------------------------------------------

def run_exp3(returns: np.ndarray, vix: np.ndarray, dates,
             exp1_result: dict, ews_window: int,
             fig_dir: Path) -> dict:
    """Correlate rolling H1 entropy with VIX."""
    pprint("\n=== Experiment 3: Topology vs VIX correlation ===")

    h1_entropy = exp1_result["h1_entropy"]
    centers = exp1_result["centers"]

    # Interpolate H1 entropy to daily resolution (returns length)
    n_ret = len(returns)
    ent_centers = centers[:len(h1_entropy)]
    h1_daily = np.full(n_ret, np.nan)
    for i, c in enumerate(ent_centers):
        if 0 <= c < n_ret:
            h1_daily[c] = h1_entropy[i]
    valid = ~np.isnan(h1_daily)
    if np.sum(valid) > 2:
        interp_fn = interp1d(
            np.where(valid)[0], h1_daily[valid],
            kind="linear", fill_value="extrapolate", bounds_error=False,
        )
        h1_interp = interp_fn(np.arange(n_ret))
    else:
        pprint("  WARNING: too few H1 entropy values to interpolate")
        h1_interp = h1_daily

    # Align VIX (vix has len(spy) = len(returns)+1 entries)
    vix_aligned = vix[1:]  # drop first to align with returns

    # Compute rolling H1 entropy from the interpolated series
    # Also compute rolling variance of returns for comparison
    roll_var = rolling_stat(returns, ews_window, "variance")

    # Correlate: only where both are valid
    valid_mask = ~np.isnan(h1_interp) & ~np.isnan(vix_aligned)
    if np.sum(valid_mask) > 30:
        h1_v = h1_interp[valid_mask]
        vix_v = vix_aligned[valid_mask]
        pearson_r, pearson_p = stats.pearsonr(h1_v, vix_v)
        spearman_r, spearman_p = stats.spearmanr(h1_v, vix_v)
    else:
        pearson_r = spearman_r = 0.0
        pearson_p = spearman_p = 1.0

    pprint(f"  H1 entropy vs VIX: Pearson r={pearson_r:.4f} (p={pearson_p:.4e})")
    pprint(f"  H1 entropy vs VIX: Spearman r={spearman_r:.4f} (p={spearman_p:.4e})")

    # Also correlate transition scores with VIX
    topo_interp = exp1_result["topo_interp"]
    valid_mask2 = ~np.isnan(topo_interp) & ~np.isnan(vix_aligned)
    if np.sum(valid_mask2) > 30:
        topo_v = topo_interp[valid_mask2]
        vix_v2 = vix_aligned[valid_mask2]
        ts_pearson, ts_pearson_p = stats.pearsonr(topo_v, vix_v2)
        ts_spearman, ts_spearman_p = stats.spearmanr(topo_v, vix_v2)
    else:
        ts_pearson = ts_spearman = 0.0
        ts_pearson_p = ts_spearman_p = 1.0

    pprint(f"  Transition score vs VIX: Pearson r={ts_pearson:.4f} (p={ts_pearson_p:.4e})")
    pprint(f"  Transition score vs VIX: Spearman r={ts_spearman:.4f} (p={ts_spearman_p:.4e})")

    # --- Figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        ret_dates = dates[1:]
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

        # Panel 1: VIX
        axes[0].plot(ret_dates, vix_aligned, color="purple", linewidth=0.8, label="VIX")
        axes[0].set_ylabel("VIX")
        axes[0].set_title("VIX vs Topological Signals")
        axes[0].legend(fontsize=8)

        # Panel 2: H1 entropy
        axes[1].plot(ret_dates, h1_interp, color="darkorange", linewidth=0.8,
                     label=f"H1 Entropy (Pearson r={pearson_r:.3f})")
        axes[1].set_ylabel("H1 Persistence Entropy")
        axes[1].legend(fontsize=8)

        # Panel 3: Transition score
        axes[2].plot(ret_dates, topo_interp, color="steelblue", linewidth=0.8,
                     label=f"Transition Score (Pearson r={ts_pearson:.3f})")
        axes[2].set_ylabel("Transition Score")
        axes[2].legend(fontsize=8)

        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
        plt.tight_layout()
        fig.savefig(fig_dir / "exp3_topology_vix_correlation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'exp3_topology_vix_correlation.png'}")

        # Scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if np.sum(valid_mask) > 30:
            axes[0].scatter(h1_interp[valid_mask], vix_aligned[valid_mask],
                           alpha=0.1, s=2, color="darkorange")
            axes[0].set_xlabel("H1 Persistence Entropy")
            axes[0].set_ylabel("VIX")
            axes[0].set_title(f"H1 Entropy vs VIX (r={pearson_r:.3f})")
        if np.sum(valid_mask2) > 30:
            axes[1].scatter(topo_interp[valid_mask2], vix_aligned[valid_mask2],
                           alpha=0.1, s=2, color="steelblue")
            axes[1].set_xlabel("Transition Score")
            axes[1].set_ylabel("VIX")
            axes[1].set_title(f"Transition Score vs VIX (r={ts_pearson:.3f})")
        plt.tight_layout()
        fig.savefig(fig_dir / "exp3_scatter_correlation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'exp3_scatter_correlation.png'}")
    except Exception as e:
        pprint(f"  Warning: figure failed: {e}")

    return {
        "h1_vix_pearson": float(pearson_r),
        "h1_vix_pearson_p": float(pearson_p),
        "h1_vix_spearman": float(spearman_r),
        "h1_vix_spearman_p": float(spearman_p),
        "ts_vix_pearson": float(ts_pearson),
        "ts_vix_pearson_p": float(ts_pearson_p),
        "ts_vix_spearman": float(ts_spearman),
        "ts_vix_spearman_p": float(ts_spearman_p),
    }


# ---------------------------------------------------------------------------
# Experiment 4: Bull vs Bear attractor topology
# ---------------------------------------------------------------------------

def run_exp4(returns: np.ndarray, vix: np.ndarray, dates,
             seed: int, cloud_size: int, n_permutations: int,
             fig_dir: Path) -> dict:
    """Compare point cloud topology of bull vs bear market regimes."""
    pprint("\n=== Experiment 4: Bull vs Bear attractor topology ===")

    vix_aligned = vix[1:]  # align with returns

    # Identify sustained bull (VIX < 20 for 21+ consecutive days)
    # and bear (VIX > 30 for 21+ consecutive days) periods
    bull_mask = np.zeros(len(returns), dtype=bool)
    bear_mask = np.zeros(len(returns), dtype=bool)

    # Rolling 21-day minimum VIX for bull, maximum VIX for bear
    for i in range(21, len(returns)):
        window_vix = vix_aligned[i-21:i]
        if np.all(window_vix < 20):
            bull_mask[i] = True
        if np.all(window_vix > 30):
            bear_mask[i] = True

    n_bull = np.sum(bull_mask)
    n_bear = np.sum(bear_mask)
    pprint(f"  Bull days (sustained VIX<20): {n_bull}")
    pprint(f"  Bear days (sustained VIX>30): {n_bear}")

    if n_bull < 100 or n_bear < 100:
        pprint("  WARNING: insufficient bull/bear days for robust comparison")
        # Relax: use instantaneous VIX threshold
        bull_mask = vix_aligned < 20
        bear_mask = vix_aligned > 30
        n_bull = np.sum(bull_mask)
        n_bear = np.sum(bear_mask)
        pprint(f"  Relaxed — Bull days (VIX<20): {n_bull}, Bear days (VIX>30): {n_bear}")

    # Embed returns from each regime
    embedder = TakensEmbedder(delay=1, dimension=5)

    # Bull cloud
    bull_returns = returns[bull_mask]
    if len(bull_returns) > 100:
        bull_cloud = embedder.fit_transform(bull_returns)
        rng = np.random.default_rng(seed)
        if len(bull_cloud) > cloud_size:
            idx = rng.choice(len(bull_cloud), cloud_size, replace=False)
            bull_cloud = bull_cloud[idx]
    else:
        bull_cloud = None

    # Bear cloud
    bear_returns = returns[bear_mask]
    if len(bear_returns) > 100:
        bear_cloud = embedder.fit_transform(bear_returns)
        rng = np.random.default_rng(seed + 1)
        if len(bear_cloud) > cloud_size:
            idx = rng.choice(len(bear_cloud), cloud_size, replace=False)
            bear_cloud = bear_cloud[idx]
    else:
        bear_cloud = None

    if bull_cloud is None or bear_cloud is None:
        pprint("  SKIP: insufficient data for one regime")
        return {"skipped": True, "n_bull": int(n_bull), "n_bear": int(n_bear)}

    pprint(f"  Bull cloud shape: {bull_cloud.shape}")
    pprint(f"  Bear cloud shape: {bear_cloud.shape}")

    # PH on each
    pa_bull = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa_bear = PersistenceAnalyzer(max_dim=1, backend="ripser")

    pprint("  Computing PH for bull regime...")
    res_bull = pa_bull.fit_transform(bull_cloud, subsample=min(500, len(bull_cloud)), seed=seed)
    pprint("  Computing PH for bear regime...")
    res_bear = pa_bear.fit_transform(bear_cloud, subsample=min(500, len(bear_cloud)), seed=seed)

    # Compare
    wass_dist = pa_bull.distance(pa_bear, metric="wasserstein_1")
    pprint(f"  Wasserstein-1 distance (bull vs bear): {wass_dist:.4f}")

    bull_h1_ent = res_bull["persistence_entropy"][1] if len(res_bull["persistence_entropy"]) > 1 else 0.0
    bear_h1_ent = res_bear["persistence_entropy"][1] if len(res_bear["persistence_entropy"]) > 1 else 0.0
    bull_h0_ent = res_bull["persistence_entropy"][0]
    bear_h0_ent = res_bear["persistence_entropy"][0]

    pprint(f"  Bull H0 entropy: {bull_h0_ent:.4f}, H1 entropy: {bull_h1_ent:.4f}")
    pprint(f"  Bear H0 entropy: {bear_h0_ent:.4f}, H1 entropy: {bear_h1_ent:.4f}")

    # H1 feature counts
    bull_h1_count = len(pa_bull.diagrams_[1]) if len(pa_bull.diagrams_) > 1 else 0
    bear_h1_count = len(pa_bear.diagrams_[1]) if len(pa_bear.diagrams_) > 1 else 0
    pprint(f"  Bull H1 features: {bull_h1_count}, Bear H1 features: {bear_h1_count}")

    # Permutation test: pool all returns, randomly assign to bull/bear
    pprint(f"  Permutation test ({n_permutations} permutations)...")
    all_returns = np.concatenate([bull_returns, bear_returns])
    all_cloud_data = embedder.fit_transform(all_returns)
    rng = np.random.default_rng(seed + 100)

    perm_dists = []
    n_b = len(bull_cloud)
    n_total = len(all_cloud_data)
    for p in range(n_permutations):
        if p % 25 == 0:
            pprint(f"    Permutation {p+1}/{n_permutations}...")
        idx_perm = rng.permutation(n_total)
        perm_bull = all_cloud_data[idx_perm[:n_b]]
        perm_bear = all_cloud_data[idx_perm[n_b:n_b + len(bear_cloud)]]

        # Subsample for speed
        if len(perm_bull) > 500:
            sub_idx = rng.choice(len(perm_bull), 500, replace=False)
            perm_bull = perm_bull[sub_idx]
        if len(perm_bear) > 500:
            sub_idx = rng.choice(len(perm_bear), 500, replace=False)
            perm_bear = perm_bear[sub_idx]

        pa1 = PersistenceAnalyzer(max_dim=1, backend="ripser")
        pa2 = PersistenceAnalyzer(max_dim=1, backend="ripser")
        try:
            pa1.fit_transform(perm_bull, seed=seed + p)
            pa2.fit_transform(perm_bear, seed=seed + p + 1000)
            perm_dists.append(pa1.distance(pa2, metric="wasserstein_1"))
        except Exception:
            continue

    perm_dists = np.array(perm_dists)
    if len(perm_dists) > 0:
        p_value = np.mean(perm_dists >= wass_dist)
        z_score = (wass_dist - np.mean(perm_dists)) / (np.std(perm_dists) + 1e-12)
    else:
        p_value = 1.0
        z_score = 0.0

    pprint(f"  Permutation p-value: {p_value:.4f}")
    pprint(f"  Permutation z-score: {z_score:.2f}")

    # --- Figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Persistence diagrams
        for ax, pa, label, color in [
            (axes[0], pa_bull, "Bull (VIX<20)", "green"),
            (axes[1], pa_bear, "Bear (VIX>30)", "red"),
        ]:
            for dim_idx, marker in [(0, "o"), (1, "^")]:
                if dim_idx < len(pa.diagrams_) and len(pa.diagrams_[dim_idx]) > 0:
                    dgm = pa.diagrams_[dim_idx]
                    ax.scatter(dgm[:, 0], dgm[:, 1], marker=marker, alpha=0.4, s=10,
                              label=f"H{dim_idx} ({len(dgm)} features)")
            max_val = max(
                np.max(pa.diagrams_[0][:, 1]) if len(pa.diagrams_[0]) > 0 else 1,
                np.max(pa.diagrams_[1][:, 1]) if len(pa.diagrams_) > 1 and len(pa.diagrams_[1]) > 0 else 1,
            )
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)
            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")
            ax.set_title(f"{label}\nH1 ent={res_bull['persistence_entropy'][1] if label.startswith('Bull') else res_bear['persistence_entropy'][1]:.3f}" if len(res_bull['persistence_entropy']) > 1 else label)
            ax.legend(fontsize=7)

        # Permutation distribution
        if len(perm_dists) > 0:
            axes[2].hist(perm_dists, bins=30, alpha=0.6, color="steelblue", label="Null distribution")
            axes[2].axvline(wass_dist, color="red", linewidth=2,
                           label=f"Observed W1={wass_dist:.3f}\np={p_value:.4f}, z={z_score:.2f}")
            axes[2].set_xlabel("Wasserstein-1 Distance")
            axes[2].set_ylabel("Count")
            axes[2].set_title("Permutation Test")
            axes[2].legend(fontsize=8)

        plt.suptitle("Bull vs Bear Attractor Topology", fontsize=14)
        plt.tight_layout()
        fig.savefig(fig_dir / "exp4_bull_bear_topology.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'exp4_bull_bear_topology.png'}")
    except Exception as e:
        pprint(f"  Warning: figure failed: {e}")

    return {
        "skipped": False,
        "n_bull": int(n_bull),
        "n_bear": int(n_bear),
        "bull_h1_entropy": float(bull_h1_ent),
        "bear_h1_entropy": float(bear_h1_ent),
        "bull_h0_entropy": float(bull_h0_ent),
        "bear_h0_entropy": float(bear_h0_ent),
        "bull_h1_count": bull_h1_count,
        "bear_h1_count": bear_h1_count,
        "wasserstein_1": float(wass_dist),
        "perm_p_value": float(p_value),
        "perm_z_score": float(z_score),
        "n_permutations": len(perm_dists),
    }


# ---------------------------------------------------------------------------
# Overview figure
# ---------------------------------------------------------------------------

def plot_overview(spy: np.ndarray, vix: np.ndarray, returns: np.ndarray,
                  dates, fig_dir: Path) -> None:
    """Plot overview: SPY price, VIX, and log returns."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

        axes[0].plot(dates, spy, color="steelblue", linewidth=0.8)
        axes[0].set_ylabel("SPY Close ($)")
        axes[0].set_title("S&P 500 (SPY) — 20 Years of Market Data")
        axes[0].set_yscale("log")

        axes[1].plot(dates, vix, color="purple", linewidth=0.8)
        axes[1].axhline(20, color="orange", linestyle=":", alpha=0.5)
        axes[1].axhline(30, color="red", linestyle=":", alpha=0.5)
        axes[1].set_ylabel("VIX")

        ret_dates = dates[1:]
        axes[2].plot(ret_dates, returns, color="steelblue", alpha=0.5, linewidth=0.3)
        axes[2].set_ylabel("Log Returns")

        for ax in axes:
            for key, event in CRISIS_EVENTS.items():
                idx = find_date_index(dates, event["onset"])
                ax.axvline(dates[idx], color="red", linestyle="--", alpha=0.5)

        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
        plt.tight_layout()
        fig.savefig(fig_dir / "overview.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        pprint(f"  Saved {fig_dir / 'overview.png'}")
    except Exception as e:
        pprint(f"  Warning: overview figure failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Branch 8: Financial Regime Topology")
    parser.add_argument("--window_size", type=int, default=252, help="TD window size (1 trading year)")
    parser.add_argument("--step_size", type=int, default=21, help="TD step size (1 trading month)")
    parser.add_argument("--subsample", type=int, default=200, help="PH subsample size")
    parser.add_argument("--ews_window", type=int, default=252, help="Rolling EWS window")
    parser.add_argument("--cloud_size", type=int, default=2000, help="Point cloud size for Exp4")
    parser.add_argument("--n_surrogates", type=int, default=50, help="Number of surrogates for Exp2")
    parser.add_argument("--n_permutations", type=int, default=100, help="Number of permutations for Exp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=str, default="2005-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    args = parser.parse_args()

    pprint("=" * 70)
    pprint("Branch 8: Financial Regime Detection — Market State Topology")
    pprint("=" * 70)

    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data" / "finance"
    fig_dir = base_dir / "figures" / "finance"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Download data
    data = download_financial_data(start=args.start, end=args.end, seed=args.seed)

    # Overview figure
    plot_overview(data["spy"], data["vix"], data["returns"], data["dates"], fig_dir)

    # Experiment 1
    exp1 = run_exp1(
        data["returns"], data["dates"], data["vix"],
        window_size=args.window_size, step_size=args.step_size,
        subsample=args.subsample, seed=args.seed, fig_dir=fig_dir,
    )

    # Experiment 2
    exp2 = run_exp2(
        data["returns"], exp1,
        subsample=args.subsample, n_surrogates=args.n_surrogates,
        seed=args.seed, fig_dir=fig_dir,
    )

    # Experiment 3
    exp3 = run_exp3(
        data["returns"], data["vix"], data["dates"],
        exp1, ews_window=args.ews_window, fig_dir=fig_dir,
    )

    # Experiment 4
    exp4 = run_exp4(
        data["returns"], data["vix"], data["dates"],
        seed=args.seed, cloud_size=args.cloud_size,
        n_permutations=args.n_permutations, fig_dir=fig_dir,
    )

    # Determine overall verdict
    pr_z = abs(exp2.get("phase_randomize", {}).get("z_mean", 0))
    ts_z = abs(exp2.get("time_shuffle", {}).get("z_mean", 0))
    exp4_sig = exp4.get("perm_p_value", 1.0) < 0.05 if not exp4.get("skipped") else False

    n_detected = sum(1 for v in exp1["crisis_detection"].values() if v["detected"])
    h1_corr = abs(exp3.get("h1_vix_pearson", 0))
    ts_corr = abs(exp3.get("ts_vix_pearson", 0))

    if pr_z < 2.0 and n_detected <= 1 and not exp4_sig:
        verdict = "negative"
    elif pr_z >= 2.0 and n_detected >= 2 and exp4_sig:
        verdict = "positive signal"
    else:
        verdict = "linear-only" if pr_z < 2.0 else "mixed"

    pprint(f"\n{'=' * 70}")
    pprint(f"OVERALL VERDICT: {verdict}")
    pprint(f"  Crises detected: {n_detected}/3")
    pprint(f"  Phase-randomize z: {exp2.get('phase_randomize', {}).get('z_mean', 0):.2f}")
    pprint(f"  Time-shuffle z: {exp2.get('time_shuffle', {}).get('z_mean', 0):.2f}")
    pprint(f"  H1-VIX Pearson: {exp3.get('h1_vix_pearson', 0):.4f}")
    pprint(f"  Bull/Bear p-value: {exp4.get('perm_p_value', 'N/A')}")
    pprint(f"{'=' * 70}")

    total_time = time.time() - t_start
    pprint(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save results
    results = {
        "branch": "experiment/tda-finance",
        "ticker": "SPY",
        "date_range": f"{data['date_start']} to {data['date_end']}",
        "n_days": len(data["spy"]),
        "config": {
            "window_size": args.window_size,
            "step_size": args.step_size,
            "subsample": args.subsample,
            "ews_window": args.ews_window,
            "cloud_size": args.cloud_size,
            "n_surrogates": args.n_surrogates,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "embedding_delay": 1,
            "embedding_dim": 5,
        },
        "exp1_crisis_detection": exp1["crisis_detection"],
        "exp1_n_changepoints": exp1["n_changepoints"],
        "exp1_threshold": exp1["threshold"],
        "exp2_surrogate_z_scores": {
            "phase_randomize": exp2.get("phase_randomize", {}).get("z_mean", 0),
            "time_shuffle": exp2.get("time_shuffle", {}).get("z_mean", 0),
        },
        "exp2_linear_explanation": exp2.get("linear_explanation", ""),
        "exp2_details": {k: v for k, v in exp2.items() if k != "linear_explanation"},
        "exp3_h1_vix_pearson": exp3.get("h1_vix_pearson", 0),
        "exp3_h1_vix_spearman": exp3.get("h1_vix_spearman", 0),
        "exp3_ts_vix_pearson": exp3.get("ts_vix_pearson", 0),
        "exp3_ts_vix_spearman": exp3.get("ts_vix_spearman", 0),
        "exp3_details": exp3,
        "exp4_bull_h1_entropy": exp4.get("bull_h1_entropy", 0),
        "exp4_bear_h1_entropy": exp4.get("bear_h1_entropy", 0),
        "exp4_wasserstein_p": exp4.get("perm_p_value", 1.0),
        "exp4_details": exp4,
        "overall_verdict": verdict,
        "runtime_seconds": round(total_time, 1),
    }

    results_path = data_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    pprint(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
