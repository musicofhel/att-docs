"""Direction 5: CROCKER plots for visualizing the topological landscape of difficulty.

Computes CROCKER matrices (Betti number heatmaps) with difficulty level and
layer index as the varying parameter. Resolves non-monotonic H1 entropy by
revealing scale-dependent topological structure.

Usage:
    python scripts/run_crocker.py
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
from att.llm import HiddenStateLoader, CROCKERMatrix
from att.viz.plotting import plot_crocker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "transformer", "math500_hidden_states.npz")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "llm")
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "transformer")


def main():
    parser = argparse.ArgumentParser(description="CROCKER plot analysis")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--n-grid", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    loader = HiddenStateLoader(args.data)
    print(loader)

    # --- CROCKER by difficulty level ---
    print("\nComputing CROCKER by difficulty level (at final layer)...")
    cm_diff = CROCKERMatrix(
        n_filtration_steps=args.n_grid, max_dim=1, seed=args.seed
    )
    cm_diff.fit_by_difficulty(loader, layer=-1)

    # H1 CROCKER
    fig = plot_crocker(
        cm_diff.betti_matrices[1],
        parameter_labels=cm_diff.parameter_labels,
        filtration_range=cm_diff.filtration_range,
        title="H1 CROCKER: Difficulty Level (Final Layer)",
    )
    path = os.path.join(FIGURES_DIR, "crocker_difficulty_h1.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # H0 CROCKER
    fig = plot_crocker(
        cm_diff.betti_matrices[0],
        parameter_labels=cm_diff.parameter_labels,
        filtration_range=cm_diff.filtration_range,
        title="H0 CROCKER: Difficulty Level (Final Layer)",
        colormap="YlOrRd",
    )
    path = os.path.join(FIGURES_DIR, "crocker_difficulty_h0.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # L1 distances
    l1 = cm_diff.pairwise_l1_distances(dim=1)
    print("\nH1 CROCKER L1 distances between difficulty levels:")
    labels = cm_diff.parameter_labels
    header = "      " + "  ".join([f"{l:>6}" for l in labels])
    print(header)
    for i, lab in enumerate(labels):
        row = f"{lab:>5} " + "  ".join([f"{l1[i, j]:6.2f}" for j in range(len(labels))])
        print(row)

    # --- CROCKER by layer index ---
    for level in [1, 5]:
        print(f"\nComputing CROCKER by layer for Level {level}...")
        cm_layer = CROCKERMatrix(
            n_filtration_steps=args.n_grid, max_dim=1, seed=args.seed
        )
        cm_layer.fit_by_layer(loader, level=level)

        fig = plot_crocker(
            cm_layer.betti_matrices[1],
            parameter_labels=cm_layer.parameter_labels,
            filtration_range=cm_layer.filtration_range,
            title=f"H1 CROCKER: Layer Index (Level {level})",
        )
        path = os.path.join(FIGURES_DIR, f"crocker_layer_h1_level{level}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # --- Side-by-side: all 5 difficulty levels ---
    print("\nGenerating side-by-side CROCKER comparison...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    for i, level in enumerate(sorted(loader.unique_levels)):
        cm = CROCKERMatrix(
            n_filtration_steps=args.n_grid, max_dim=1, seed=args.seed
        )
        cm.fit_by_layer(loader, level=int(level))
        mat = cm.betti_matrices[1]
        ax = axes[i]
        im = ax.imshow(
            mat, aspect="auto", origin="lower", cmap="viridis",
            extent=[-0.5, mat.shape[1] - 0.5, cm.filtration_range[0], cm.filtration_range[1]],
            interpolation="nearest",
        )
        ax.set_title(f"Level {int(level)}")
        ax.set_xlabel("Layer")
        if i == 0:
            ax.set_ylabel("Filtration scale (ε)")

    fig.colorbar(im, ax=axes.tolist(), label="β₁", fraction=0.02, pad=0.04)
    fig.suptitle("H1 CROCKER by Layer: All Difficulty Levels", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "crocker_all_levels_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Save L1 distances
    results = {
        "l1_distances_difficulty_h1": l1.tolist(),
        "difficulty_labels": cm_diff.parameter_labels,
        "filtration_range": list(cm_diff.filtration_range),
        "config": {
            "n_filtration_steps": args.n_grid,
            "seed": args.seed,
        },
    }
    out_path = os.path.join(RESULTS_DIR, "crocker_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
