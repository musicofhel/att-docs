"""Direction 9: Topological compression vs resistance to simplification.

Decomposes total persistence into feature count × mean lifetime per difficulty
per layer. Tests whether harder problems show compression (Fay et al.) or
resistance to simplification (Datta et al.).

Usage:
    python scripts/run_compression_resistance.py
    python scripts/run_compression_resistance.py --layers 0 5 10 15 20 25 28
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from att.config import set_seed
from att.llm import HiddenStateLoader
from att.topology import PersistenceAnalyzer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "transformer", "math500_hidden_states.npz")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "llm")
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "transformer")


def compute_persistence_decomposition(diagrams: list[np.ndarray]) -> dict:
    """Decompose total persistence into count and mean lifetime.

    Returns per-dimension dict with:
        total_persistence, n_features, mean_lifetime, max_lifetime
    """
    result = {}
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            result[dim] = {
                "total_persistence": 0.0,
                "n_features": 0,
                "mean_lifetime": 0.0,
                "max_lifetime": 0.0,
            }
            continue
        lifetimes = dgm[:, 1] - dgm[:, 0]
        lifetimes = lifetimes[lifetimes > 0]
        n = len(lifetimes)
        result[dim] = {
            "total_persistence": float(lifetimes.sum()),
            "n_features": int(n),
            "mean_lifetime": float(lifetimes.mean()) if n > 0 else 0.0,
            "max_lifetime": float(lifetimes.max()) if n > 0 else 0.0,
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compression vs resistance to simplification analysis"
    )
    parser.add_argument(
        "--data", default=DATA_PATH, help="Path to hidden states .npz"
    )
    parser.add_argument(
        "--layers", nargs="*", type=int, default=None,
        help="Specific layer indices to analyze (default: all)"
    )
    parser.add_argument(
        "--n-pca", type=int, default=50, help="PCA components"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    loader = HiddenStateLoader(args.data)
    print(loader)

    levels = sorted(loader.unique_levels.tolist())
    layer_indices = args.layers if args.layers else list(range(loader.num_layers))
    print(f"Analyzing {len(layer_indices)} layers for {len(levels)} levels")

    # Compute decomposition at each layer for each level
    results = {}  # (level, layer) -> decomposition dict

    for level in levels:
        for layer_idx in layer_indices:
            cloud = loader.get_level_cloud(level, layer=layer_idx)
            n_pts = cloud.shape[0]
            if n_pts < 3:
                continue

            n_comp = min(args.n_pca, n_pts - 1, cloud.shape[1])
            pca = PCA(n_components=n_comp)
            cloud_pca = pca.fit_transform(cloud)

            pa = PersistenceAnalyzer(max_dim=2, backend="ripser")
            result = pa.fit_transform(
                cloud_pca, subsample=min(n_pts, 200), seed=args.seed
            )

            decomp = compute_persistence_decomposition(result["diagrams"])
            results[(level, layer_idx)] = decomp

            if layer_idx == layer_indices[-1]:
                print(
                    f"  Level {level}, final layer: "
                    f"H1 total={decomp[1]['total_persistence']:.4f}, "
                    f"n={decomp[1]['n_features']}, "
                    f"mean={decomp[1]['mean_lifetime']:.4f}"
                )

    # Summary: aggregate across layers for each level
    print("\n--- Per-level Summary (averaged across layers) ---")
    for level in levels:
        total_p = []
        n_feat = []
        mean_lt = []
        for layer_idx in layer_indices:
            key = (level, layer_idx)
            if key in results:
                total_p.append(results[key][1]["total_persistence"])
                n_feat.append(results[key][1]["n_features"])
                mean_lt.append(results[key][1]["mean_lifetime"])
        print(
            f"Level {level}: H1 avg_total={np.mean(total_p):.4f}, "
            f"avg_n={np.mean(n_feat):.1f}, avg_mean_lt={np.mean(mean_lt):.4f}"
        )

    # --- Figures ---

    # Figure 1: Total persistence by difficulty (at final layer)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    final_layer = layer_indices[-1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(levels)))

    for dim, dim_label in enumerate(["H0", "H1", "H2"]):
        ax = axes[dim]
        tp = [results.get((lv, final_layer), {}).get(dim, {}).get("total_persistence", 0) for lv in levels]
        ax.bar(levels, tp, color=colors, edgecolor="black", alpha=0.8)
        ax.set_xlabel("Difficulty Level")
        ax.set_ylabel("Total Persistence")
        ax.set_title(f"{dim_label} Total Persistence (Layer {final_layer})")
        ax.set_xticks(levels)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_DIR, "compression_total_persistence.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {os.path.join(FIGURES_DIR, 'compression_total_persistence.png')}")

    # Figure 2: Feature count vs mean lifetime (H1, dual axis)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    n_feats = [results.get((lv, final_layer), {}).get(1, {}).get("n_features", 0) for lv in levels]
    mean_lts = [results.get((lv, final_layer), {}).get(1, {}).get("mean_lifetime", 0) for lv in levels]

    ax1.bar(
        [lv - 0.15 for lv in levels], n_feats, width=0.3,
        color="steelblue", alpha=0.8, label="Feature Count",
    )
    ax2.bar(
        [lv + 0.15 for lv in levels], mean_lts, width=0.3,
        color="coral", alpha=0.8, label="Mean Lifetime",
    )

    ax1.set_xlabel("Difficulty Level")
    ax1.set_ylabel("H1 Feature Count", color="steelblue")
    ax2.set_ylabel("H1 Mean Lifetime", color="coral")
    ax1.set_xticks(levels)
    ax1.set_title("H1 Decomposition: Count vs Mean Lifetime")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURES_DIR, "compression_count_vs_lifetime.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {os.path.join(FIGURES_DIR, 'compression_count_vs_lifetime.png')}")

    # Figure 3: Layer-wise total persistence for easy vs hard
    if len(layer_indices) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        for level in [1, 5]:
            tp = [
                results.get((level, li), {}).get(1, {}).get("total_persistence", 0)
                for li in layer_indices
            ]
            ax.plot(layer_indices, tp, "o-", linewidth=2, markersize=4, label=f"Level {level}")

        n_layers = loader.num_layers
        shade_start = max(0, n_layers - 6)
        ax.axvspan(shade_start, n_layers - 1, alpha=0.1, color="gray", label="Final 5 layers")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("H1 Total Persistence")
        ax.set_title("Layer-wise H1 Total Persistence: Easy vs Hard")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            os.path.join(FIGURES_DIR, "compression_layerwise.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(f"Saved: {os.path.join(FIGURES_DIR, 'compression_layerwise.png')}")

    # Save JSON results
    serializable = {}
    for (level, layer_idx), decomp in results.items():
        key = f"L{level}_Ly{layer_idx}"
        serializable[key] = decomp
    out_path = os.path.join(RESULTS_DIR, "compression_resistance_results.json")
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
