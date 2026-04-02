"""Direction 4: Zigzag persistence across transformer layers.

Computes zigzag persistent homology across layers for each difficulty level.
Tracks topological features born and dying as representations evolve through
the network. Tests whether harder problems produce longer-lived features
(Datta et al. prediction).

Usage:
    python scripts/run_zigzag.py
    python scripts/run_zigzag.py --subsample 50 --layers 0,5,10,15,20,25,28
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from att.config import set_seed
from att.llm import HiddenStateLoader
from att.llm.zigzag import (
    ZigzagLayerAnalyzer,
    compare_zigzag_levels,
    zigzag_feature_lifetime_stats,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "transformer", "math500_hidden_states.npz")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "llm")
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "transformer")


def plot_zigzag_barcode(result, dim=1, ax=None, title=None):
    """Plot zigzag barcode as horizontal bars."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    bars = result.barcodes.get(dim, np.empty((0, 2)))
    if len(bars) == 0:
        ax.text(0.5, 0.5, f"No H{dim} features", ha="center", va="center",
                transform=ax.transAxes)
        if title:
            ax.set_title(title)
        return ax

    # Sort by birth time
    order = np.argsort(bars[:, 0])
    bars = bars[order]

    lifetimes = bars[:, 1] - bars[:, 0]
    colors = plt.cm.viridis(lifetimes / max(lifetimes.max(), 1e-10))

    for i, (bar, color) in enumerate(zip(bars, colors)):
        ax.barh(i, bar[1] - bar[0], left=bar[0], height=0.8, color=color, alpha=0.8)

    ax.set_xlabel("Layer (zigzag time)")
    ax.set_ylabel("Feature index")
    ax.set_title(title or f"H{dim} Zigzag Barcode — Level {result.level}")
    ax.grid(True, alpha=0.2, axis="x")
    return ax


def main():
    parser = argparse.ArgumentParser(description="Zigzag persistence across layers")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample", type=int, default=100)
    parser.add_argument("--max-dim", type=int, default=1)
    parser.add_argument("--n-pca", type=int, default=50)
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices (default: every 3rd layer + last)"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    loader = HiddenStateLoader(args.data)
    print(loader)

    # Determine layer indices
    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(",")]
    else:
        n_layers = loader.num_layers
        step = max(1, n_layers // 10)
        layer_indices = list(range(0, n_layers, step)) + [n_layers - 1]
        layer_indices = sorted(set(layer_indices))

    print(f"Using {len(layer_indices)} layers: {layer_indices}")

    zza = ZigzagLayerAnalyzer(
        max_dim=args.max_dim,
        n_pca_components=args.n_pca,
        subsample=args.subsample,
        seed=args.seed,
    )

    # --- Compute zigzag per level ---
    levels = sorted(loader.unique_levels)
    results = {}
    all_stats = {}

    for level in levels:
        print(f"\n=== Level {level} ===")
        result = zza.fit(loader, level=level, layer_indices=layer_indices)
        results[level] = result

        for dim in range(args.max_dim + 1):
            stats = zigzag_feature_lifetime_stats(result, dim=dim)
            all_stats[(level, dim)] = stats
            n_bars = len(result.barcodes.get(dim, []))
            print(f"  H{dim}: {n_bars} total bars, {stats['n_features']} non-trivial, "
                  f"mean_lifetime={stats['mean_lifetime']:.2f}, "
                  f"n_long_lived={stats['n_long_lived']}")

    # --- Plot barcodes: Level 1 vs Level 5 ---
    if 1 in results and 5 in results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        plot_zigzag_barcode(results[1], dim=1, ax=axes[0],
                            title="H1 Zigzag Barcode — Level 1 (Easy)")
        plot_zigzag_barcode(results[5], dim=1, ax=axes[1],
                            title="H1 Zigzag Barcode — Level 5 (Hard)")
        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, "zigzag_barcode_easy_vs_hard.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nSaved: {path}")

    # --- Lifetime distribution by difficulty ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    for level, color in zip(levels, colors):
        bars = results[level].barcodes.get(1, np.empty((0, 2)))
        if len(bars) > 0:
            lifetimes = bars[:, 1] - bars[:, 0]
            lifetimes = lifetimes[lifetimes > 0]
            if len(lifetimes) > 0:
                ax.hist(lifetimes, bins=20, alpha=0.5, color=color,
                        label=f"Level {level}", density=True)
    ax.set_xlabel("Feature Lifetime (layers)")
    ax.set_ylabel("Density")
    ax.set_title("H1 Zigzag Feature Lifetime Distribution by Difficulty")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(FIGURES_DIR, "zigzag_lifetime_distribution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # --- Summary bar chart: mean lifetime by level ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mean_lifetimes = [all_stats.get((l, 1), {}).get("mean_lifetime", 0) for l in levels]
    n_features = [all_stats.get((l, 1), {}).get("n_features", 0) for l in levels]

    axes[0].bar(levels, mean_lifetimes, color=colors, alpha=0.8)
    axes[0].set_xlabel("Difficulty Level")
    axes[0].set_ylabel("Mean H1 Lifetime (layers)")
    axes[0].set_title("Mean Zigzag H1 Lifetime by Difficulty")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(levels, n_features, color=colors, alpha=0.8)
    axes[1].set_xlabel("Difficulty Level")
    axes[1].set_ylabel("Number of H1 Features")
    axes[1].set_title("Zigzag H1 Feature Count by Difficulty")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "zigzag_summary_by_difficulty.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # --- Pairwise level comparisons ---
    print("\n--- Pairwise Level Comparisons (H1) ---")
    comparisons = {}
    for i, la in enumerate(levels):
        for lb in levels[i + 1:]:
            comp = compare_zigzag_levels(results[la], results[lb], dim=1)
            key = f"{la}_vs_{lb}"
            comparisons[key] = comp
            sig = "***" if comp["ks_pvalue"] < 0.001 else "**" if comp["ks_pvalue"] < 0.01 else "*" if comp["ks_pvalue"] < 0.05 else "ns"
            print(f"  Level {la} vs {lb}: KS={comp['ks_statistic']:.3f}, "
                  f"p={comp['ks_pvalue']:.4f} {sig}, "
                  f"Δmean_lifetime={comp['mean_lifetime_diff']:+.3f}")

    # --- Save results ---
    output = {
        "layer_indices": layer_indices,
        "n_layers_used": len(layer_indices),
        "config": {
            "seed": args.seed,
            "subsample": args.subsample,
            "max_dim": args.max_dim,
            "n_pca": args.n_pca,
        },
        "lifetime_stats": {
            f"level_{l}_h{d}": all_stats[(l, d)]
            for l in levels for d in range(args.max_dim + 1)
            if (l, d) in all_stats
        },
        "comparisons": comparisons,
    }
    out_path = os.path.join(RESULTS_DIR, "zigzag_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
