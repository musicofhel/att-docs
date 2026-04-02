"""Direction 7: Intrinsic dimension profiles across layers and difficulty.

Computes TwoNN intrinsic dimension estimates at each transformer layer for
each difficulty level. Overlays with bottleneck profile from Wave 1 to
correlate topological complexity with representation dimensionality.

Usage:
    python scripts/run_intrinsic_dim.py
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
from att.llm.intrinsic_dim import id_profile, twonn_dimension

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "transformer", "math500_hidden_states.npz")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "llm")
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "transformer")


def main():
    parser = argparse.ArgumentParser(description="Intrinsic dimension profile analysis")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["twonn", "phd"], default="twonn")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    loader = HiddenStateLoader(args.data)
    print(loader)

    # --- TwoNN ID profile ---
    print(f"\nComputing {args.method} ID profile...")
    profiles = id_profile(loader, method=args.method)

    levels = sorted(profiles.keys())
    n_layers = loader.num_layers
    layer_indices = list(range(n_layers))

    # Plot ID profiles
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    for level, color in zip(levels, colors):
        ax.plot(layer_indices, profiles[level], "-o", color=color,
                label=f"Level {level}", markersize=3, linewidth=1.5)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Intrinsic Dimension")
    ax.set_title(f"{args.method.upper()} Intrinsic Dimension by Layer and Difficulty")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shade terminal layer region
    terminal_start = max(0, n_layers - 5)
    ax.axvspan(terminal_start, n_layers - 1, alpha=0.1, color="red",
               label="Terminal layers")

    path = os.path.join(FIGURES_DIR, f"id_profile_{args.method}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # --- ID difference between easy and hard ---
    if 1 in profiles and 5 in profiles:
        diff = profiles[5] - profiles[1]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(layer_indices, diff, color=["red" if d > 0 else "blue" for d in diff], alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("ID(Level 5) - ID(Level 1)")
        ax.set_title("Intrinsic Dimension Difference: Hard - Easy")
        ax.grid(True, alpha=0.3, axis="y")
        path = os.path.join(FIGURES_DIR, f"id_diff_hard_vs_easy_{args.method}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # --- Summary statistics ---
    print(f"\n{args.method.upper()} ID Summary:")
    for level in levels:
        ids = profiles[level]
        print(f"  Level {level}: mean={ids.mean():.2f}, "
              f"min={ids.min():.2f} (layer {ids.argmin()}), "
              f"max={ids.max():.2f} (layer {ids.argmax()})")

    # --- ID at terminal layer ---
    print("\nTerminal layer IDs:")
    for level in levels:
        terminal_id = profiles[level][-1]
        print(f"  Level {level}: {terminal_id:.2f}")

    # Save results
    results = {
        "method": args.method,
        "profiles": {str(k): v.tolist() for k, v in profiles.items()},
        "n_layers": n_layers,
        "levels": [int(l) for l in levels],
        "config": {"seed": args.seed, "method": args.method},
    }
    out_path = os.path.join(RESULTS_DIR, f"intrinsic_dim_{args.method}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
