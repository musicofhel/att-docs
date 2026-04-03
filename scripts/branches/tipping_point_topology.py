#!/usr/bin/env python3
"""Branch 3: Topological Early Warning Signals for Ecosystem Tipping Points.

Generates synthetic dynamical systems with catastrophic bifurcations and tests
whether ATT's topological tools detect the approach of the tipping point BEFORE
the system flips. Three models (saddle-node, Hopf, double-well) and three
experiments (sliding-window topology, classical EWS comparison, pre/post
tipping cloud topology).
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

# ATT imports
from att.topology.persistence import PersistenceAnalyzer
from att.transitions.detector import TransitionDetector
from att.embedding.takens import TakensEmbedder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def pprint(msg):
    print(msg)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Synthetic Models
# ---------------------------------------------------------------------------

def saddle_node_model(n_steps=20000, dt=0.01, r_start=-0.5, r_end=0.5,
                      noise=0.05, seed=42):
    """Lake eutrophication: dx/dt = -x^3 + x + r(t).

    Tipping at r = 2/(3*sqrt(3)) ≈ 0.385 where the stable equilibrium vanishes.
    """
    rng = np.random.default_rng(seed)
    r = np.linspace(r_start, r_end, n_steps)
    x = np.zeros(n_steps)
    x[0] = -1.0
    for i in range(1, n_steps):
        x[i] = (x[i-1]
                 + (-x[i-1]**3 + x[i-1] + r[i]) * dt
                 + noise * np.sqrt(dt) * rng.standard_normal())
    r_crit = 2 / (3 * np.sqrt(3))  # ≈ 0.3849
    tip_idx = int((r_crit - r_start) / (r_end - r_start) * n_steps)
    return x, r, tip_idx, r_crit


def hopf_model(n_steps=20000, dt=0.01, mu_start=-0.5, mu_end=0.5,
               noise=0.02, seed=42):
    """Hopf bifurcation: oscillation onset at mu=0."""
    rng = np.random.default_rng(seed)
    mu = np.linspace(mu_start, mu_end, n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    x[0], y[0] = 0.1, 0.1
    for i in range(1, n_steps):
        r2 = x[i-1]**2 + y[i-1]**2
        x[i] = (x[i-1]
                 + (mu[i] * x[i-1] - y[i-1] - x[i-1] * r2) * dt
                 + noise * np.sqrt(dt) * rng.standard_normal())
        y[i] = (y[i-1]
                 + (x[i-1] + mu[i] * y[i-1] - y[i-1] * r2) * dt
                 + noise * np.sqrt(dt) * rng.standard_normal())
    tip_idx = int((0.0 - mu_start) / (mu_end - mu_start) * n_steps)
    return np.column_stack([x, y]), mu, tip_idx, 0.0


def double_well_model(n_steps=20000, dt=0.01, noise=0.3, seed=42):
    """Double-well with slow tilt: dx/dt = x - x^3 + r(t)."""
    rng = np.random.default_rng(seed)
    r = 0.2 * np.sin(2 * np.pi * np.arange(n_steps) / 20000)
    x = np.zeros(n_steps)
    x[0] = -1.0
    for i in range(1, n_steps):
        x[i] = (x[i-1]
                 + (x[i-1] - x[i-1]**3 + r[i]) * dt
                 + noise * np.sqrt(dt) * rng.standard_normal())
    crossings = np.where(np.diff(np.sign(x)))[0]
    tip_idx = crossings[0] if len(crossings) > 0 else n_steps // 2
    return x, r, tip_idx, r[tip_idx] if tip_idx < n_steps else 0.0


# ---------------------------------------------------------------------------
# Classical Early Warning Signals
# ---------------------------------------------------------------------------

def rolling_stat(x, window, stat_fn):
    """Compute rolling statistic with given window size."""
    n = len(x)
    result = np.full(n, np.nan)
    half_w = window // 2
    for i in range(half_w, n - half_w):
        chunk = x[i - half_w:i + half_w]
        result[i] = stat_fn(chunk)
    return result


def rolling_variance(x, window=500):
    return rolling_stat(x, window, np.var)


def rolling_autocorr(x, window=500, lag=1):
    def ac(chunk):
        if len(chunk) < lag + 2:
            return np.nan
        return np.corrcoef(chunk[:-lag], chunk[lag:])[0, 1]
    return rolling_stat(x, window, ac)


def rolling_skewness(x, window=500):
    return rolling_stat(x, window, lambda c: stats.skew(c))


def detect_ews_lead_time(signal_ts, tip_idx, pre_end_frac=0.6):
    """Find when signal exceeds 2sigma above pre-bifurcation mean.

    Returns lead time in indices (positive = before tipping).
    """
    pre_end = int(tip_idx * pre_end_frac)
    if pre_end < 10:
        return 0

    baseline = signal_ts[:pre_end]
    baseline = baseline[~np.isnan(baseline)]
    if len(baseline) < 10:
        return 0

    mean_b = np.mean(baseline)
    std_b = np.std(baseline)
    if std_b < 1e-12:
        return 0

    threshold = mean_b + 2 * std_b

    consecutive = 0
    for i in range(pre_end, min(tip_idx + 1, len(signal_ts))):
        if not np.isnan(signal_ts[i]) and signal_ts[i] > threshold:
            consecutive += 1
            if consecutive >= 3:
                return tip_idx - (i - 2)
        else:
            consecutive = 0
    return 0


# ---------------------------------------------------------------------------
# Run TransitionDetector once per model (shared by Exp1 and Exp2)
# ---------------------------------------------------------------------------

def run_transition_detector(ts_1d, model_name, window_size, step_size,
                            subsample, seed):
    """Run TransitionDetector and return results dict."""
    pprint(f"  [{model_name}] TransitionDetector "
           f"(win={window_size}, step={step_size})...")
    detector = TransitionDetector(
        window_size=window_size,
        step_size=step_size,
        max_dim=1,
        backend="ripser",
        subsample=subsample,
    )
    result = detector.fit_transform(
        ts_1d, seed=seed, embedding_dim=5, embedding_delay=4,
    )

    # Extract H1 entropy per window
    topo_ts = result["topology_timeseries"]
    h1_entropy = []
    for wt in topo_ts:
        if wt is not None and "persistence_entropy" in wt:
            pe = wt["persistence_entropy"]
            h1_entropy.append(pe[1] if len(pe) > 1 else 0.0)
        else:
            h1_entropy.append(0.0)

    changepoints = detector.detect_changepoints(method="cusum")
    centers = result["window_centers"]
    cp_indices = [centers[cp] for cp in changepoints]

    return {
        "centers": centers,
        "distances": result["distances"],
        "transition_scores": result["transition_scores"],
        "h1_entropy": np.array(h1_entropy),
        "changepoints": changepoints,
        "cp_indices": cp_indices,
    }


# ---------------------------------------------------------------------------
# Experiment 1: Sliding-window topology approaching tipping
# ---------------------------------------------------------------------------

def analyze_exp1(ts_1d, tip_idx, model_name, td_result, step_size, fig_dir):
    """Analyze topology lead time from pre-computed TransitionDetector."""
    centers = td_result["centers"]
    transition_scores = td_result["transition_scores"]
    h1_entropy = td_result["h1_entropy"]
    cp_indices = td_result["cp_indices"]

    # Interpolate transition scores to full-length for lead time detection
    score_centers = centers[:len(transition_scores)]
    topo_full = np.full(len(ts_1d), np.nan)
    for i, c in enumerate(score_centers):
        if 0 <= c < len(ts_1d):
            topo_full[c] = transition_scores[i]
    valid = ~np.isnan(topo_full)
    if np.sum(valid) > 2:
        interp_fn = interp1d(
            np.where(valid)[0], topo_full[valid],
            kind="linear", fill_value="extrapolate", bounds_error=False
        )
        topo_interp = interp_fn(np.arange(len(ts_1d)))
    else:
        topo_interp = topo_full
    lead_time_steps = detect_ews_lead_time(topo_interp, tip_idx)

    # --- Figures ---
    if fig_dir is not None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        ax = axes[0]
        plot_step = max(1, len(ts_1d) // 5000)
        ax.plot(np.arange(len(ts_1d))[::plot_step], ts_1d[::plot_step],
                color="steelblue", linewidth=0.5, alpha=0.8)
        ax.axvline(tip_idx, color="red", linewidth=2, linestyle="--",
                   label=f"Tipping point (idx={tip_idx})")
        ax.set_ylabel("State x(t)")
        ax.set_title(f"{model_name} — Time Series")
        ax.legend()

        ax = axes[1]
        score_centers = centers[:len(transition_scores)]
        ax.plot(score_centers, transition_scores, color="darkorange", lw=1.0)
        ax.axvline(tip_idx, color="red", linewidth=2, linestyle="--")
        for cp in cp_indices:
            ax.axvline(cp, color="green", linewidth=1, alpha=0.5, ls=":")
        ax.set_ylabel("Transition Score\n(PI distance)")
        ax.set_title("Sliding-Window Transition Scores")

        ax = axes[2]
        ax.plot(centers[:len(h1_entropy)], h1_entropy, color="purple", lw=1.0)
        ax.axvline(tip_idx, color="red", linewidth=2, linestyle="--")
        ax.set_ylabel("H1 Entropy")
        ax.set_xlabel("Timestep")
        ax.set_title("H1 Persistence Entropy")

        plt.tight_layout()
        fname = fig_dir / f"exp1_{model_name}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        pprint(f"    Saved {fname}")

    tip_win = max(1, tip_idx // step_size)
    return {
        "model": model_name,
        "lead_time_steps": int(lead_time_steps),
        "n_changepoints": len(td_result["changepoints"]),
        "tip_idx": int(tip_idx),
        "mean_transition_score_pre": float(
            np.nanmean(transition_scores[:tip_win])
        ),
        "mean_transition_score_post": float(
            np.nanmean(transition_scores[tip_win:])
            if tip_win < len(transition_scores) else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Experiment 2: Topology vs Classical EWS
# ---------------------------------------------------------------------------

def analyze_exp2(ts_1d, tip_idx, model_name, td_result, step_size,
                 ews_window, fig_dir):
    """Compare topology vs classical EWS using pre-computed topology."""
    # Classical EWS
    pprint(f"  [{model_name}] Computing classical EWS...")
    var_ts = rolling_variance(ts_1d, ews_window)
    ac_ts = rolling_autocorr(ts_1d, ews_window, lag=1)
    skew_ts = rolling_skewness(ts_1d, ews_window)

    var_lead = detect_ews_lead_time(var_ts, tip_idx)
    ac_lead = detect_ews_lead_time(ac_ts, tip_idx)
    skew_lead = detect_ews_lead_time(skew_ts, tip_idx)

    # Interpolate topology scores to full-length for comparable lead time
    centers = td_result["centers"]
    transition_scores = td_result["transition_scores"]
    score_centers = centers[:len(transition_scores)]

    topo_full = np.full(len(ts_1d), np.nan)
    for i, c in enumerate(score_centers):
        if 0 <= c < len(ts_1d):
            topo_full[c] = transition_scores[i]
    valid = ~np.isnan(topo_full)
    if np.sum(valid) > 2:
        interp_fn = interp1d(
            np.where(valid)[0], topo_full[valid],
            kind="linear", fill_value="extrapolate", bounds_error=False
        )
        topo_interp = interp_fn(np.arange(len(ts_1d)))
    else:
        topo_interp = topo_full

    topo_lead = detect_ews_lead_time(topo_interp, tip_idx)

    lead_times = {
        "topology": int(topo_lead),
        "variance": int(var_lead),
        "autocorr": int(ac_lead),
        "skewness": int(skew_lead),
    }

    # --- Figure ---
    if fig_dir is not None:
        fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

        ax = axes[0]
        plot_step = max(1, len(ts_1d) // 5000)
        ax.plot(np.arange(len(ts_1d))[::plot_step], ts_1d[::plot_step],
                color="steelblue", linewidth=0.5)
        ax.axvline(tip_idx, color="red", linewidth=2, linestyle="--",
                   label="Tipping point")
        ax.set_ylabel("x(t)")
        ax.set_title(f"{model_name} — EWS Comparison")
        ax.legend()

        for ax, (signal, color, label, lead) in zip(axes[1:], [
            (transition_scores, "darkorange", "Topology (PI dist)", topo_lead),
            (var_ts, "teal", "Variance", var_lead),
            (ac_ts, "coral", "Lag-1 AC", ac_lead),
            (skew_ts, "mediumpurple", "Skewness", skew_lead),
        ]):
            if signal is transition_scores:
                ax.plot(score_centers, signal, color=color, linewidth=1.0)
            else:
                ax.plot(np.arange(len(signal)), signal, color=color, lw=0.8)
            ax.axvline(tip_idx, color="red", linewidth=2, linestyle="--")
            if lead > 0:
                ax.axvline(tip_idx - lead, color="green", linewidth=2,
                           linestyle=":", label=f"Lead: {lead}")
                ax.legend()
            ax.set_ylabel(label)
        axes[-1].set_xlabel("Timestep")

        plt.tight_layout()
        fname = fig_dir / f"exp2_{model_name}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        pprint(f"    Saved {fname}")

    return lead_times


# ---------------------------------------------------------------------------
# Experiment 3: Pre-tipping vs Post-tipping point cloud topology
# ---------------------------------------------------------------------------

def run_exp3(ts_1d, tip_idx, model_name, cloud_size=5000,
             subsample=500, seed=42, fig_dir=None):
    """Compare PH of point clouds just before and after tipping."""
    pprint(f"  [Exp3] {model_name}: Pre/post tipping cloud topology...")

    pre_start = max(0, tip_idx - cloud_size)
    pre_ts = ts_1d[pre_start:tip_idx]
    post_ts = ts_1d[tip_idx:min(len(ts_1d), tip_idx + cloud_size)]

    if len(pre_ts) < 200 or len(post_ts) < 200:
        pprint(f"    WARNING: Insufficient data (pre={len(pre_ts)}, "
               f"post={len(post_ts)}). Skipping.")
        return {
            "pre_post_wasserstein": 0.0, "pre_h1_count": 0,
            "post_h1_count": 0, "pre_h0_entropy": 0.0,
            "post_h0_entropy": 0.0, "pre_h1_entropy": 0.0,
            "post_h1_entropy": 0.0, "skipped": True,
        }

    embedder = TakensEmbedder(delay=4, dimension=5)
    pre_cloud = embedder.fit_transform(pre_ts)
    post_cloud = embedder.fit_transform(post_ts)
    pprint(f"    Pre cloud: {pre_cloud.shape}, Post cloud: {post_cloud.shape}")

    pa_pre = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa_post = PersistenceAnalyzer(max_dim=1, backend="ripser")
    res_pre = pa_pre.fit_transform(pre_cloud, subsample=subsample, seed=seed)
    res_post = pa_post.fit_transform(post_cloud, subsample=subsample, seed=seed)

    wass = pa_pre.distance(pa_post, metric="wasserstein_1")

    def count_finite_h1(pa):
        if len(pa.diagrams_) > 1:
            diag = pa.diagrams_[1]
            return sum(1 for b, d in diag if np.isfinite(d)) if len(diag) > 0 else 0
        return 0

    pre_pe = res_pre["persistence_entropy"]
    post_pe = res_post["persistence_entropy"]

    result = {
        "pre_post_wasserstein": float(wass),
        "pre_h1_count": count_finite_h1(pa_pre),
        "post_h1_count": count_finite_h1(pa_post),
        "pre_h0_entropy": float(pre_pe[0]) if len(pre_pe) > 0 else 0.0,
        "post_h0_entropy": float(post_pe[0]) if len(post_pe) > 0 else 0.0,
        "pre_h1_entropy": float(pre_pe[1]) if len(pre_pe) > 1 else 0.0,
        "post_h1_entropy": float(post_pe[1]) if len(post_pe) > 1 else 0.0,
        "skipped": False,
    }

    if fig_dir is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for idx, (pa, label) in enumerate([
            (pa_pre, "Pre-tipping"), (pa_post, "Post-tipping"),
        ]):
            ax = axes[idx]
            max_val = 0
            for dim, marker in [(0, "o"), (1, "^")]:
                if dim < len(pa.diagrams_):
                    diag = pa.diagrams_[dim]
                    finite = diag[np.isfinite(diag[:, 1])] if len(diag) > 0 else np.empty((0, 2))
                    if len(finite) > 0:
                        ax.scatter(finite[:, 0], finite[:, 1], marker=marker,
                                   alpha=0.5, s=20,
                                   label=f"H{dim} ({len(finite)})")
                        max_val = max(max_val, finite.max())
            if max_val > 0:
                ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)
            ax.set_xlabel("Birth"); ax.set_ylabel("Death")
            ax.set_title(label); ax.legend(fontsize=8)

        ax = axes[2]
        pre_imgs = pa_pre.to_image(resolution=50)
        post_imgs = pa_post.to_image(resolution=50)
        if len(pre_imgs) > 1 and len(post_imgs) > 1:
            diff = post_imgs[1] - pre_imgs[1]
            im = ax.imshow(diff, cmap="RdBu_r", origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax)
            ax.set_title("H1 PI Difference\n(Post - Pre)")
        else:
            ax.text(0.5, 0.5, "No H1 data", ha="center", va="center",
                    transform=ax.transAxes)

        fig.suptitle(f"{model_name} — Pre vs Post Tipping "
                     f"(Wass={wass:.2f})", fontsize=13)
        plt.tight_layout()
        fname = fig_dir / f"exp3_{model_name}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        pprint(f"    Saved {fname}")

    return result


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def plot_lead_time_comparison(exp2_results, fig_dir):
    models = list(exp2_results.keys())
    methods = ["topology", "variance", "autocorr", "skewness"]
    colors = ["darkorange", "teal", "coral", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.2
    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = [exp2_results[m][method] for m in models]
        ax.bar(x + i * width, vals, width, label=method.capitalize(),
               color=color, alpha=0.85)
    ax.set_xlabel("Model")
    ax.set_ylabel("Lead Time (timesteps)")
    ax.set_title("Early Warning Signal Lead Times — Topology vs Classical")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fname = fig_dir / "lead_time_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    pprint(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--window_size", type=int, default=500)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--subsample", type=int, default=200)
    parser.add_argument("--ews_window", type=int, default=500,
                        help="Rolling window for classical EWS")
    parser.add_argument("--cloud_size", type=int, default=5000,
                        help="Points before/after tipping for Exp 3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data/tipping")
    parser.add_argument("--fig_dir", type=str, default="figures/tipping")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pprint("=" * 60)
    pprint("Branch 3: Topological Early Warning Signals for Tipping Points")
    pprint("=" * 60)

    # --- Generate synthetic data ---
    pprint("\n[1/5] Generating synthetic time series...")
    sn_x, sn_r, sn_tip, sn_r_crit = saddle_node_model(
        n_steps=args.n_steps, seed=args.seed)
    pprint(f"  Saddle-node: {len(sn_x)} steps, tipping at idx={sn_tip} "
           f"(r_crit={sn_r_crit:.4f})")

    hopf_xy, hopf_mu, hopf_tip, hopf_mu_crit = hopf_model(
        n_steps=args.n_steps, seed=args.seed)
    hopf_amp = np.sqrt(hopf_xy[:, 0]**2 + hopf_xy[:, 1]**2)
    pprint(f"  Hopf: {len(hopf_amp)} steps, tipping at idx={hopf_tip} "
           f"(mu_crit={hopf_mu_crit:.4f})")

    dw_x, dw_r, dw_tip, dw_r_crit = double_well_model(
        n_steps=args.n_steps, seed=args.seed)
    pprint(f"  Double-well: {len(dw_x)} steps, first crossing at idx={dw_tip} "
           f"(r={dw_r_crit:.4f})")

    models = {
        "saddle_node": (sn_x, sn_tip),
        "hopf": (hopf_amp, hopf_tip),
        "double_well": (dw_x, dw_tip),
    }

    # --- Time series overview figure ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, (name, (ts, tip)) in zip(axes, models.items()):
        plot_step = max(1, len(ts) // 5000)
        ax.plot(np.arange(len(ts))[::plot_step], ts[::plot_step],
                color="steelblue", linewidth=0.5)
        ax.axvline(tip, color="red", linewidth=2, linestyle="--",
                   label=f"Tipping (idx={tip})")
        ax.set_ylabel("State")
        ax.set_title(name.replace("_", " ").title())
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(fig_dir / "time_series_overview.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    pprint("  Saved time_series_overview.png")

    # --- Run TransitionDetector once per model (shared by Exp1 + Exp2) ---
    pprint("\n[2/5] Running TransitionDetector on all models...")
    td_results = {}
    for name, (ts, tip) in models.items():
        td_results[name] = run_transition_detector(
            ts, name, args.window_size, args.step_size,
            args.subsample, args.seed,
        )
        n_windows = len(td_results[name]["centers"])
        pprint(f"    {name}: {n_windows} windows, "
               f"{len(td_results[name]['changepoints'])} changepoints")

    # --- Experiment 1 ---
    pprint("\n[3/5] Experiment 1: Topology lead times...")
    exp1_results = {}
    for name, (ts, tip) in models.items():
        res = analyze_exp1(ts, tip, name, td_results[name],
                           args.step_size, fig_dir)
        exp1_results[name] = res
        pprint(f"    {name}: lead={res['lead_time_steps']} steps, "
               f"changepoints={res['n_changepoints']}")

    # --- Experiment 2 ---
    pprint("\n[4/5] Experiment 2: Classical EWS comparison...")
    exp2_results = {}
    for name, (ts, tip) in models.items():
        lead_times = analyze_exp2(ts, tip, name, td_results[name],
                                  args.step_size, args.ews_window, fig_dir)
        exp2_results[name] = lead_times
        pprint(f"    {name}: topo={lead_times['topology']}, "
               f"var={lead_times['variance']}, "
               f"ac={lead_times['autocorr']}, "
               f"skew={lead_times['skewness']}")

    topo_earliest = sum(
        1 for name in models
        if exp2_results[name]["topology"] > 0
        and exp2_results[name]["topology"] >= max(
            exp2_results[name]["variance"],
            exp2_results[name]["autocorr"],
            exp2_results[name]["skewness"],
        )
    )
    plot_lead_time_comparison(exp2_results, fig_dir)

    # --- Experiment 3 ---
    pprint("\n[5/5] Experiment 3: Pre/post tipping cloud (saddle-node)...")
    exp3_result = run_exp3(
        sn_x, sn_tip, "saddle_node",
        cloud_size=args.cloud_size, subsample=args.subsample,
        seed=args.seed, fig_dir=fig_dir,
    )
    pprint(f"    Wasserstein: {exp3_result['pre_post_wasserstein']:.4f}")
    pprint(f"    Pre H1: {exp3_result['pre_h1_count']} features, "
           f"entropy={exp3_result['pre_h1_entropy']:.4f}")
    pprint(f"    Post H1: {exp3_result['post_h1_count']} features, "
           f"entropy={exp3_result['post_h1_entropy']:.4f}")

    # --- Save results ---
    results = {
        "branch": "experiment/tda-tipping",
        "models": ["saddle_node", "hopf", "double_well"],
        "n_steps": args.n_steps,
        "config": {
            "window_size": args.window_size,
            "step_size": args.step_size,
            "subsample": args.subsample,
            "ews_window": args.ews_window,
            "cloud_size": args.cloud_size,
            "seed": args.seed,
            "embedding_delay": 4,
            "embedding_dim": 5,
        },
        "tipping_points": {
            "saddle_node": {"idx": int(sn_tip), "r_crit": float(sn_r_crit)},
            "hopf": {"idx": int(hopf_tip), "mu_crit": float(hopf_mu_crit)},
            "double_well": {"idx": int(dw_tip),
                            "r_at_crossing": float(dw_r_crit)},
        },
        "exp1_lead_times": {
            name: exp1_results[name]["lead_time_steps"] for name in models
        },
        "exp1_lead_time_unit": "timesteps",
        "exp1_details": exp1_results,
        "exp2_lead_times": exp2_results,
        "exp2_topology_earliest_count": topo_earliest,
        "exp3_pre_post_wasserstein": exp3_result["pre_post_wasserstein"],
        "exp3_pre_h1_count": exp3_result["pre_h1_count"],
        "exp3_post_h1_count": exp3_result["post_h1_count"],
        "exp3_details": exp3_result,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    pprint(f"\nResults saved to {results_path}")

    # --- Summary ---
    pprint("\n" + "=" * 60)
    pprint("SUMMARY")
    pprint("=" * 60)
    pprint(f"\nExp 1 — Topological lead times (timesteps before tipping):")
    for name in models:
        lt = exp1_results[name]["lead_time_steps"]
        pprint(f"  {name:15s}: {lt:6d} steps")
    pprint(f"\nExp 2 — EWS comparison (lead in timesteps):")
    for name in models:
        lt = exp2_results[name]
        pprint(f"  {name:15s}: topo={lt['topology']:6d}  "
               f"var={lt['variance']:6d}  "
               f"ac={lt['autocorr']:6d}  skew={lt['skewness']:6d}")
    pprint(f"  Topology earliest in {topo_earliest}/{len(models)} models")
    pprint(f"\nExp 3 — Pre/post tipping (saddle-node):")
    pprint(f"  Wasserstein-1: {exp3_result['pre_post_wasserstein']:.4f}")
    pprint(f"  Pre H1: {exp3_result['pre_h1_count']} features, "
           f"entropy={exp3_result['pre_h1_entropy']:.4f}")
    pprint(f"  Post H1: {exp3_result['post_h1_count']} features, "
           f"entropy={exp3_result['post_h1_entropy']:.4f}")
    pprint("")


if __name__ == "__main__":
    main()
