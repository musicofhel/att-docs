"""Direction 1: Per-layer decomposition of the z=8.11 topological signal.

Runs permutation test at each transformer layer independently, producing a
"topological discriminability profile" — z-score as a function of layer index.

Usage:
    python scripts/run_perlayer_zscore.py
    python scripts/run_perlayer_zscore.py --n-perms 100  # faster for testing
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from att.config import set_seed
from att.llm import HiddenStateLoader, LayerwiseAnalyzer
from att.viz.plotting import plot_zscore_profile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "transformer", "math500_hidden_states.npz")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "llm")
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "transformer")


def main():
    parser = argparse.ArgumentParser(
        description="Per-layer z-score decomposition of topological signal"
    )
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--n-perms", type=int, default=200)
    parser.add_argument("--n-pca", type=int, default=50)
    parser.add_argument("--max-dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    loader = HiddenStateLoader(args.data)
    print(loader)
    print()

    # Step 1: Fit PH at every layer for every level
    print("Step 1: Per-layer PH computation...")
    t0 = time.time()
    analyzer = LayerwiseAnalyzer(
        n_pca_components=args.n_pca,
        max_dim=args.max_dim,
        subsample=200,
        n_permutations=args.n_perms,
        seed=args.seed,
    )
    analyzer.fit(loader)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Step 2: Entropy profiles
    print("\nStep 2: Entropy profiles...")
    entropy = analyzer.entropy_profile()
    for level in sorted(entropy.keys()):
        h1 = entropy[level][:, 1] if entropy[level].shape[1] > 1 else entropy[level][:, 0]
        print(f"  Level {level}: H1 entropy range [{h1.min():.3f}, {h1.max():.3f}]")

    # Step 3: Bottleneck profiles
    print("\nStep 3: Bottleneck profiles...")
    bottleneck = analyzer.bottleneck_profile()
    for level in sorted(bottleneck.keys()):
        dists = bottleneck[level]
        if len(dists) >= 6:
            terminal = np.mean(dists[-5:])
            nonterminal = np.mean(dists[:-5])
            ratio = terminal / nonterminal if nonterminal > 0 else float("inf")
            print(f"  Level {level}: terminal/non-terminal ratio = {ratio:.2f}x")

    # Step 4: Permutation z-score profiles
    print(f"\nStep 4: Permutation z-score profiles ({args.n_perms} permutations)...")
    print("  This may take a while...")
    t0 = time.time()
    zscore_result = analyzer.zscore_profile(loader, metric="wasserstein_1")
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    z = zscore_result["z_scores"]
    p = zscore_result["p_values"]
    print(f"\n--- Z-score Profile ---")
    print(f"  Max z-score: {z.max():.2f} at layer {z.argmax()}")
    print(f"  Mean z-score: {z.mean():.2f}")
    print(f"  Significant layers (p<0.05): {(p < 0.05).sum()}/{len(p)}")

    # Terminal vs non-terminal
    if len(z) > 6:
        terminal_z = z[-5:].mean()
        nonterminal_z = z[:-5].mean()
        print(f"  Terminal 5 mean z: {terminal_z:.2f}")
        print(f"  Non-terminal mean z: {nonterminal_z:.2f}")
        print(f"  Ratio: {terminal_z / nonterminal_z:.2f}x" if nonterminal_z > 0 else "  Ratio: inf")

    # Per-dimension breakdown
    for dim, dim_z in zscore_result["per_dim"].items():
        print(f"  H{dim}: max z={dim_z.max():.2f} at layer {dim_z.argmax()}")

    # --- Figures ---

    # Figure 1: Z-score profile with per-dim overlay
    fig = plot_zscore_profile(
        z_scores=zscore_result["z_scores"],
        p_values=zscore_result["p_values"],
        per_dim_z_scores=zscore_result["per_dim"],
    )
    path = os.path.join(FIGURES_DIR, "perlayer_zscore_profile.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")

    # Figure 2: Entropy heatmap (levels × layers)
    fig, ax = plt.subplots(figsize=(14, 4))
    levels_sorted = sorted(entropy.keys())
    # H1 entropy matrix: (n_levels, n_layers)
    ent_matrix = np.array([entropy[lv][:, 1] for lv in levels_sorted])
    im = ax.imshow(ent_matrix, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_yticks(range(len(levels_sorted)))
    ax.set_yticklabels([f"Level {lv}" for lv in levels_sorted])
    ax.set_xlabel("Layer Index")
    ax.set_title("H1 Persistence Entropy by Difficulty Level and Layer")
    fig.colorbar(im, ax=ax, label="Entropy", fraction=0.02, pad=0.04)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "perlayer_entropy_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Figure 3: Bottleneck profile (easy vs hard)
    fig, ax = plt.subplots(figsize=(10, 5))
    for level in [1, 5]:
        if level in bottleneck:
            dists = bottleneck[level]
            ax.plot(range(len(dists)), dists, "o-", linewidth=2, markersize=4, label=f"Level {level}")
    n_layers = loader.num_layers
    if n_layers > 6:
        ax.axvspan(n_layers - 6, n_layers - 2, alpha=0.1, color="gray", label="Terminal 5 layers")
    ax.set_xlabel("Layer Transition")
    ax.set_ylabel("Bottleneck Distance")
    ax.set_title("Layer-wise Topological Transition")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "perlayer_bottleneck.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Save JSON results
    results = {
        "z_scores": zscore_result["z_scores"].tolist(),
        "p_values": zscore_result["p_values"].tolist(),
        "observed": zscore_result["observed"].tolist(),
        "null_mean": zscore_result["null_mean"].tolist(),
        "null_std": zscore_result["null_std"].tolist(),
        "per_dim": {str(k): v.tolist() for k, v in zscore_result["per_dim"].items()},
        "entropy": {str(k): v.tolist() for k, v in entropy.items()},
        "bottleneck": {str(k): v.tolist() for k, v in bottleneck.items()},
        "config": {
            "n_permutations": args.n_perms,
            "n_pca_components": args.n_pca,
            "max_dim": args.max_dim,
            "seed": args.seed,
        },
    }
    out_path = os.path.join(RESULTS_DIR, "perlayer_zscore_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
