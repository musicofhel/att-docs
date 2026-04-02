"""Direction 8: Token-position-resolved topological analysis.

Computes persistent homology on token-position point clouds partitioned into
functional regions (instruction, problem, operator, answer). Tests whether
specific token regions carry more difficulty-dependent topological signal.

Usage:
    python scripts/run_token_topology.py
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

from sklearn.decomposition import PCA

from att.config import set_seed
from att.llm import HiddenStateLoader
from att.llm.token_partition import TokenPartitioner
from att.topology.persistence import PersistenceAnalyzer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "transformer", "math500_hidden_states.npz")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "llm")
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "transformer")


def compute_region_ph(
    token_traj: np.ndarray,
    region_indices: np.ndarray,
    max_dim: int = 1,
    n_pca: int = 30,
    subsample: int = 100,
    seed: int = 42,
):
    """Compute PH on a region's token embeddings."""
    if len(region_indices) < 5:
        return None

    cloud = token_traj[region_indices]
    n_comp = min(n_pca, cloud.shape[0] - 1, cloud.shape[1])
    if n_comp < 2:
        return None

    pca = PCA(n_components=n_comp)
    cloud_pca = pca.fit_transform(cloud)

    if subsample and len(cloud_pca) > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(cloud_pca), size=subsample, replace=False)
        cloud_pca = cloud_pca[idx]

    pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser")
    return pa.fit_transform(cloud_pca)


def main():
    parser = argparse.ArgumentParser(description="Token-region topological analysis")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pca", type=int, default=30)
    parser.add_argument("--subsample", type=int, default=100)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    loader = HiddenStateLoader(args.data)
    print(loader)

    # Fixed instruction token counts for Qwen2.5-1.5B-Instruct prompt template.
    # "You are a helpful math assistant. Provide the final answer.\n\n" ≈ 15 tokens
    # "\n\nPlease provide the final answer." ≈ 9 tokens
    # These are stable across tokenizers (±2 tokens).
    INSTRUCTION_PREFIX_TOKENS = 15
    INSTRUCTION_SUFFIX_TOKENS = 9

    # Regions to analyze
    main_regions = ["instruction_prefix", "problem", "instruction_suffix"]
    levels = sorted(loader.unique_levels)

    # --- Per-level, per-region persistence entropy ---
    print("\n=== Per-Region Persistence Entropy by Difficulty ===")
    region_entropy = {r: {l: [] for l in levels} for r in main_regions}

    for level in levels:
        mask = loader.get_level_mask(level)
        indices = np.where(mask)[0]
        print(f"\nLevel {level}: {len(indices)} problems")

        for i, problem_idx in enumerate(indices):
            token_traj = loader.token_trajectories[problem_idx]
            seq_len = int(loader.seq_lengths[problem_idx])

            # Partition using fixed token counts for instruction template
            prefix_end = min(INSTRUCTION_PREFIX_TOKENS, seq_len)
            suffix_start = max(prefix_end, seq_len - INSTRUCTION_SUFFIX_TOKENS)
            parts = {
                "instruction_prefix": np.arange(0, prefix_end, dtype=np.intp),
                "problem": np.arange(prefix_end, suffix_start, dtype=np.intp),
                "instruction_suffix": np.arange(suffix_start, seq_len, dtype=np.intp),
            }

            for region in main_regions:
                region_idx = parts[region]
                if len(region_idx) < 5:
                    continue

                result = compute_region_ph(
                    token_traj, region_idx,
                    n_pca=args.n_pca, subsample=args.subsample, seed=args.seed,
                )
                if result is not None:
                    pe = result["persistence_entropy"]
                    for dim in [0, 1]:
                        if isinstance(pe, dict) and dim in pe:
                            region_entropy[region][level].append(pe[dim])
                        elif isinstance(pe, (list, np.ndarray)) and dim < len(pe):
                            region_entropy[region][level].append(float(pe[dim]))

        for region in main_regions:
            vals = region_entropy[region][level]
            if vals:
                print(f"  {region:25s}: entropy={np.mean(vals):.3f} "
                      f"(±{np.std(vals):.3f}, n={len(vals)})")
            else:
                print(f"  {region:25s}: too few tokens")

    # --- Plot: region entropy by difficulty ---
    fig, axes = plt.subplots(1, len(main_regions), figsize=(5 * len(main_regions), 5),
                              sharey=True)
    if len(main_regions) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))

    for ax, region in zip(axes, main_regions):
        means = []
        stds = []
        for level in levels:
            vals = region_entropy[region][level]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(levels, means, yerr=stds, color=colors, alpha=0.8, capsize=3)
        ax.set_xlabel("Difficulty Level")
        ax.set_ylabel("Mean Persistence Entropy (H0)")
        ax.set_title(region.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Persistence Entropy by Token Region and Difficulty", fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "token_region_entropy.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")

    # --- Aggregate: which region carries most difficulty signal? ---
    print("\n=== Difficulty Signal Strength by Region ===")
    signal_strength = {}
    for region in main_regions:
        if region_entropy[region].get(1) and region_entropy[region].get(5):
            easy_mean = np.mean(region_entropy[region][1])
            hard_mean = np.mean(region_entropy[region][5])
            diff = abs(hard_mean - easy_mean)
            signal_strength[region] = {
                "easy_mean": float(easy_mean),
                "hard_mean": float(hard_mean),
                "abs_diff": float(diff),
                "direction": "increase" if hard_mean > easy_mean else "decrease",
            }
            print(f"  {region:25s}: |Δ|={diff:.4f} "
                  f"({signal_strength[region]['direction']})")

    # --- Plot comparison: easy vs hard per region ---
    if signal_strength:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(signal_strength))
        width = 0.35
        regions_sorted = sorted(signal_strength.keys())
        easy_vals = [signal_strength[r]["easy_mean"] for r in regions_sorted]
        hard_vals = [signal_strength[r]["hard_mean"] for r in regions_sorted]

        ax.bar(x - width / 2, easy_vals, width, label="Level 1 (Easy)", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, hard_vals, width, label="Level 5 (Hard)", color="firebrick", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", "\n") for r in regions_sorted], fontsize=9)
        ax.set_ylabel("Persistence Entropy")
        ax.set_title("Token Region Entropy: Easy vs Hard")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        path = os.path.join(FIGURES_DIR, "token_region_easy_vs_hard.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # --- Save results ---
    output = {
        "config": {"seed": args.seed, "n_pca": args.n_pca, "subsample": args.subsample},
        "region_entropy": {
            region: {
                str(level): {
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                    "n": len(vals),
                }
                for level, vals in level_data.items()
            }
            for region, level_data in region_entropy.items()
        },
        "signal_strength": signal_strength,
    }
    out_path = os.path.join(RESULTS_DIR, "token_topology_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
