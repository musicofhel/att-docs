"""Direction 10: Attention-hidden topological coupling analysis.

Computes per-difficulty per-layer coupling scores between attention topology
and hidden-state topology. Uses permutation surrogates for significance.

Usage:
    python scripts/run_attention_binding.py
    python scripts/run_attention_binding.py --n-permutations 50
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
from att.llm.attention_binding import AttentionHiddenBinding

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "transformer", "math500_hidden_states.npz")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "llm")
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "transformer")


def compute_proxy_binding(loader, ahb, levels, layer_indices, max_problems=20):
    """Compute binding scores using token-cloud self-coupling proxy.

    When attention matrices are not available, we use split-half token cloud
    coupling as a proxy: split each problem's token cloud into two halves,
    compute distance matrix on one half, and measure topological coupling
    with the other half.
    """
    from scipy.spatial.distance import cdist
    from sklearn.decomposition import PCA

    scores = {}
    for level in levels:
        mask = loader.get_level_mask(level)
        indices = np.where(mask)[0][:max_problems]
        print(f"\nLevel {level}: {len(indices)} problems")

        for layer_idx in layer_indices:
            level_scores = []
            for problem_idx in indices:
                token_traj = loader.token_trajectories[problem_idx]
                n_tokens = token_traj.shape[0]

                if n_tokens < 20:
                    continue

                # PCA reduce first
                n_comp = min(30, n_tokens - 1, token_traj.shape[1])
                if n_comp < 2:
                    continue
                pca = PCA(n_components=n_comp)
                reduced = pca.fit_transform(token_traj)

                # Split into two halves
                mid = n_tokens // 2
                cloud_a = reduced[:mid]
                cloud_b = reduced[mid:]

                # Create distance matrix from cloud_a as "attention proxy"
                sub = min(50, len(cloud_a))
                cloud_a_sub = cloud_a[:sub]
                cloud_b_sub = cloud_b[:min(sub, len(cloud_b))]

                dists = cdist(cloud_a_sub, cloud_a_sub)
                if dists.max() > 0:
                    dists /= dists.max()

                # Convert distance to similarity (attention-like) since
                # compute_binding calls attention_to_distance internally
                similarity = 1.0 - dists
                result = ahb.compute_binding(similarity, cloud_b_sub)
                level_scores.append(result.binding_score)

            mean_score = float(np.mean(level_scores)) if level_scores else 0.0
            std_score = float(np.std(level_scores)) if level_scores else 0.0
            scores[(level, layer_idx)] = {
                "mean": mean_score,
                "std": std_score,
                "n": len(level_scores),
            }
            print(f"  Layer {layer_idx}: binding={mean_score:.4f} (±{std_score:.4f}, n={len(level_scores)})")

    return scores


def compute_significance_sample(loader, ahb, level, n_problems=5, n_permutations=50):
    """Run significance test on a small sample for one difficulty level."""
    from scipy.spatial.distance import cdist
    from sklearn.decomposition import PCA

    mask = loader.get_level_mask(level)
    indices = np.where(mask)[0][:n_problems]

    sig_results = []
    for problem_idx in indices:
        token_traj = loader.token_trajectories[problem_idx]
        n_tokens = token_traj.shape[0]
        if n_tokens < 20:
            continue

        n_comp = min(30, n_tokens - 1, token_traj.shape[1])
        if n_comp < 2:
            continue
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(token_traj)

        mid = n_tokens // 2
        cloud_a = reduced[:min(40, mid)]
        cloud_b = reduced[mid:mid + min(40, n_tokens - mid)]

        dists = cdist(cloud_a, cloud_a)
        if dists.max() > 0:
            dists /= dists.max()

        # Convert distance to similarity (attention-like)
        similarity = 1.0 - dists
        result = ahb.test_significance(similarity, cloud_b, n_permutations=n_permutations)
        sig_results.append({
            "observed": result.observed_score,
            "p_value": result.p_value,
            "z_score": result.z_score,
        })

    return sig_results


def plot_binding_heatmap(scores, levels, layer_indices, save_path):
    """Plot binding score heatmap (difficulty × layer)."""
    n_levels = len(levels)
    n_layers = len(layer_indices)
    matrix = np.zeros((n_levels, n_layers))

    for i, level in enumerate(levels):
        for j, layer in enumerate(layer_indices):
            entry = scores.get((level, layer), {})
            matrix[i, j] = entry.get("mean", 0.0) if isinstance(entry, dict) else entry

    fig, ax = plt.subplots(figsize=(max(8, n_layers), max(4, n_levels * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(l) for l in layer_indices])
    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([f"Level {l}" for l in levels])
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Difficulty Level")
    ax.set_title("Attention-Hidden Binding Score (difficulty × layer)")

    # Add text annotations
    for i in range(n_levels):
        for j in range(n_layers):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i, j] > matrix.mean() else "black")

    fig.colorbar(im, ax=ax, label="Binding Score")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_binding_easy_vs_hard(scores, levels, layer_indices, save_path):
    """Plot binding scores: easy vs hard across layers."""
    if 1 not in levels or 5 not in levels:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for level, color, label in [(1, "steelblue", "Level 1 (Easy)"), (5, "firebrick", "Level 5 (Hard)")]:
        means = []
        stds = []
        for layer in layer_indices:
            entry = scores.get((level, layer), {})
            means.append(entry.get("mean", 0.0) if isinstance(entry, dict) else 0.0)
            stds.append(entry.get("std", 0.0) if isinstance(entry, dict) else 0.0)

        ax.errorbar(layer_indices, means, yerr=stds, fmt="o-", color=color,
                     label=label, linewidth=2, markersize=6, capsize=3, alpha=0.8)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Binding Score")
    ax.set_title("Attention-Hidden Coupling: Easy vs Hard")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_surrogate_null(sig_results, level, save_path):
    """Plot surrogate null distribution for a difficulty level."""
    if not sig_results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    observed = [r["observed"] for r in sig_results]
    p_values = [r["p_value"] for r in sig_results]

    ax.bar(range(len(observed)), observed, color="steelblue", alpha=0.8, label="Observed")
    for i, p in enumerate(p_values):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(i, observed[i] + 0.01, sig, ha="center", fontsize=10)

    ax.set_xlabel("Problem Index")
    ax.set_ylabel("Binding Score")
    ax.set_title(f"Binding Significance — Level {level}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def compute_real_attention_binding(loader, ahb, attention_ph_path, max_problems=20):
    """Compute binding scores using real pre-extracted attention PH diagrams."""
    data = np.load(attention_ph_path, allow_pickle=True)
    attn_ph_all = data["attention_ph"]  # array of dicts per problem
    attn_levels = data["difficulty_levels"]
    terminal_layers = data["terminal_layers"].tolist()

    print(f"Loaded attention PH: {len(attn_ph_all)} problems, layers {terminal_layers}")
    print(f"Hidden states: {loader}")

    # Match by difficulty level — both use seed=42 sampling
    hidden_levels = loader.levels
    unique_levels = sorted(set(attn_levels.tolist()))

    scores = {}
    for level in unique_levels:
        attn_idx = np.where(attn_levels == level)[0]
        hidden_idx = np.where(hidden_levels == level)[0]
        n_match = min(len(attn_idx), len(hidden_idx), max_problems)

        print(f"\nLevel {level}: matching {n_match} problems (attn={len(attn_idx)}, hidden={len(hidden_idx)})")

        for layer_idx in terminal_layers:
            level_scores = []
            for i in range(n_match):
                problem_ph = attn_ph_all[attn_idx[i]]
                if not isinstance(problem_ph, dict):
                    problem_ph = problem_ph.item()  # np scalar → dict

                layer_ph = problem_ph.get(layer_idx)
                if layer_ph is None:
                    continue

                diagrams = [np.array(d) for d in layer_ph["diagrams"]]

                # Get hidden-state token trajectory for matched problem
                h_idx = hidden_idx[i]
                token_traj = loader.token_trajectories[h_idx]

                if token_traj.shape[0] < 10:
                    continue

                try:
                    result = ahb.compute_binding_from_diagrams(diagrams, token_traj)
                    level_scores.append(result.binding_score)
                except Exception as e:
                    print(f"  Warning: problem {i} layer {layer_idx}: {e}")
                    continue

            mean_score = float(np.mean(level_scores)) if level_scores else 0.0
            std_score = float(np.std(level_scores)) if level_scores else 0.0
            scores[(level, layer_idx)] = {
                "mean": mean_score,
                "std": std_score,
                "n": len(level_scores),
            }
            print(f"  Layer {layer_idx}: binding={mean_score:.4f} (±{std_score:.4f}, n={len(level_scores)})")

    return scores, unique_levels, terminal_layers


def main():
    parser = argparse.ArgumentParser(description="Attention-hidden binding analysis")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--attention-ph", default=os.path.join(RESULTS_DIR, "attention_ph_diagrams.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-problems", type=int, default=20)
    parser.add_argument("--n-permutations", type=int, default=50)
    parser.add_argument("--n-sig-problems", type=int, default=5)
    parser.add_argument("--mode", choices=["proxy", "real", "both"], default="both",
                        help="proxy=split-half, real=pre-extracted attention PH, both=run both")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    loader = HiddenStateLoader(args.data)
    print(loader)

    ahb = AttentionHiddenBinding(
        max_dim=1, image_resolution=30, image_sigma=0.1,
        n_pca_components=30, subsample=50, seed=args.seed,
    )

    levels = sorted(loader.unique_levels.tolist())
    n_layers = loader.num_layers
    # Terminal 5 layers + a few early/mid reference points
    layer_indices = sorted(set(
        [0, n_layers // 4, n_layers // 2] +
        list(range(max(0, n_layers - 5), n_layers))
    ))

    output = {
        "config": {
            "seed": args.seed,
            "max_problems": args.max_problems,
            "n_permutations": args.n_permutations,
            "mode": args.mode,
        },
    }

    # --- Real attention binding ---
    if args.mode in ("real", "both") and os.path.exists(args.attention_ph):
        print("\n=== Real Attention Binding ===")
        real_scores, real_levels, real_layers = compute_real_attention_binding(
            loader, ahb, args.attention_ph, max_problems=args.max_problems,
        )

        plot_binding_heatmap(
            real_scores, real_levels, real_layers,
            os.path.join(FIGURES_DIR, "attention_binding_real_heatmap.png"),
        )
        plot_binding_easy_vs_hard(
            real_scores, real_levels, real_layers,
            os.path.join(FIGURES_DIR, "attention_binding_real_easy_vs_hard.png"),
        )

        output["real_binding_scores"] = {
            f"level{level}_layer{layer}": data
            for (level, layer), data in real_scores.items()
        }
        output["real_layers"] = real_layers
    elif args.mode in ("real", "both"):
        print(f"\nWARNING: No attention PH data at {args.attention_ph}, skipping real mode")

    # --- Proxy binding ---
    if args.mode in ("proxy", "both"):
        print(f"\nLayers: {layer_indices}")
        print(f"Levels: {levels}")

        print("\n=== Binding Scores (proxy mode) ===")
        scores = compute_proxy_binding(loader, ahb, levels, layer_indices, max_problems=args.max_problems)

        plot_binding_heatmap(
            scores, levels, layer_indices,
            os.path.join(FIGURES_DIR, "attention_binding_heatmap.png"),
        )
        plot_binding_easy_vs_hard(
            scores, levels, layer_indices,
            os.path.join(FIGURES_DIR, "attention_binding_easy_vs_hard.png"),
        )

        output["binding_scores"] = {
            f"level{level}_layer{layer}": data
            for (level, layer), data in scores.items()
        }
        output["config"]["layer_indices"] = layer_indices

    # --- Significance tests (proxy) ---
    if args.mode in ("proxy", "both"):
        print("\n=== Significance Tests ===")
        sig_all = {}
        for level in [1, 5]:
            if level not in levels:
                continue
            print(f"\nLevel {level}:")
            sig = compute_significance_sample(
                loader, ahb, level,
                n_problems=args.n_sig_problems,
                n_permutations=args.n_permutations,
            )
            sig_all[level] = sig
            for i, r in enumerate(sig):
                sig_marker = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
                print(f"  Problem {i}: score={r['observed']:.4f}, "
                      f"p={r['p_value']:.4f} {sig_marker}, z={r['z_score']:.2f}")

            plot_surrogate_null(
                sig, level,
                os.path.join(FIGURES_DIR, f"attention_binding_significance_level{level}.png"),
            )
        output["significance"] = {
            str(level): sig
            for level, sig in sig_all.items()
        }

    # --- Save results ---
    out_path = os.path.join(RESULTS_DIR, "attention_binding_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
