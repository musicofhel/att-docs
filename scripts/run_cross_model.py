"""Direction 6: Cross-model topological replication analysis.

Runs the Wave 1+2 analysis pipeline (z-score profiles, CROCKER, ID profiles)
on each extracted model and compares results for universality claims.

Usage:
    python scripts/run_cross_model.py
    python scripts/run_cross_model.py --models qwen phi2 pythia
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
from att.llm import HiddenStateLoader, LayerwiseAnalyzer, CROCKERMatrix
from att.llm.intrinsic_dim import id_profile, twonn_dimension

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "transformer")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "llm")

MODEL_FILES = {
    "qwen": "math500_hidden_states.npz",
    "phi2": "phi2_hidden_states.npz",
    "pythia": "pythia14b_hidden_states.npz",
    "stablelm": "stablelm16b_hidden_states.npz",
}

MODEL_LABELS = {
    "qwen": "Qwen2.5-1.5B",
    "phi2": "Phi-2 (2.7B)",
    "pythia": "Pythia-1.4B",
    "stablelm": "StableLM-2-1.6B",
}

MODEL_COLORS = {
    "qwen": "#1f77b4",
    "phi2": "#ff7f0e",
    "pythia": "#2ca02c",
    "stablelm": "#d62728",
}


def load_available_models(model_keys: list[str]) -> dict[str, HiddenStateLoader]:
    """Load available model archives."""
    loaders = {}
    for key in model_keys:
        path = os.path.join(DATA_DIR, MODEL_FILES[key])
        if os.path.exists(path):
            loaders[key] = HiddenStateLoader(path)
            print(f"Loaded {key}: {loaders[key]}")
        else:
            print(f"Skipping {key}: {path} not found")
    return loaders


def run_zscore_profiles(loaders: dict[str, HiddenStateLoader], seed: int = 42) -> dict:
    """Compute z-score profiles for each model."""
    results = {}
    for key, loader in loaders.items():
        print(f"\n--- Z-score profile: {key} ---")
        lw = LayerwiseAnalyzer(
            n_pca_components=50, max_dim=1, subsample=200,
            n_permutations=100, seed=seed,
        )
        lw.fit(loader)
        zs = lw.zscore_profile(loader, metric="wasserstein_1")
        results[key] = {
            "z_scores": zs["z_scores"],
            "p_values": zs["p_values"],
            "n_layers": loader.num_layers,
        }
        peak = np.argmax(zs["z_scores"])
        print(f"  Peak z-score: {zs['z_scores'][peak]:.2f} at layer {peak}/{loader.num_layers - 1}")
    return results


def run_entropy_profiles(loaders: dict[str, HiddenStateLoader], seed: int = 42) -> dict:
    """Compute H1 entropy profiles for each model."""
    results = {}
    for key, loader in loaders.items():
        print(f"\n--- Entropy profile: {key} ---")
        lw = LayerwiseAnalyzer(
            n_pca_components=50, max_dim=1, subsample=200, seed=seed,
        )
        lw.fit(loader)
        ent = lw.entropy_profile()
        results[key] = {
            level: ent[level][:, 1].tolist()  # H1 entropy per layer
            for level in ent
        }
    return results


def run_id_profiles(loaders: dict[str, HiddenStateLoader]) -> dict:
    """Compute intrinsic dimension profiles for each model."""
    results = {}
    for key, loader in loaders.items():
        print(f"\n--- ID profile: {key} ---")
        profiles = id_profile(loader, n_pca_components=50, method="twonn")
        results[key] = {
            level: profiles[level].tolist()
            for level in profiles
        }
    return results


def run_terminal_entropy(loaders: dict[str, HiddenStateLoader], seed: int = 42) -> dict:
    """Compute terminal-layer H1 entropy by difficulty for each model."""
    from att.topology.persistence import PersistenceAnalyzer
    from sklearn.decomposition import PCA

    results = {}
    for key, loader in loaders.items():
        print(f"\n--- Terminal entropy: {key} ---")
        levels = sorted(loader.unique_levels.tolist())
        entropies = {}
        for level in levels:
            cloud = loader.get_level_cloud(level, layer=-1)
            n_pts = cloud.shape[0]
            n_comp = min(50, n_pts - 1, cloud.shape[1])
            pca = PCA(n_components=n_comp)
            cloud_pca = pca.fit_transform(cloud)
            pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
            result = pa.fit_transform(cloud_pca, subsample=min(n_pts, 200), seed=seed)
            pe = result["persistence_entropy"]
            h1_ent = pe[1] if isinstance(pe, list) and len(pe) > 1 else (pe.get(1, 0.0) if isinstance(pe, dict) else 0.0)
            entropies[level] = float(h1_ent)
            print(f"  Level {level}: H1 entropy = {h1_ent:.3f}")
        results[key] = entropies
    return results


def plot_cross_model_zscore(zscore_results: dict, save_path: str):
    """Overlay z-score profiles from all models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for key, data in zscore_results.items():
        z = data["z_scores"]
        n = len(z)
        x = np.linspace(0, 1, n)  # normalized layer position
        label = MODEL_LABELS.get(key, key)
        color = MODEL_COLORS.get(key, None)
        ax.plot(x, z, label=label, color=color, linewidth=2, alpha=0.8)

    ax.axhline(y=1.96, color="gray", linestyle="--", alpha=0.5, label="p<0.05")
    ax.axhline(y=2.58, color="gray", linestyle=":", alpha=0.5, label="p<0.01")
    ax.set_xlabel("Normalized Layer Position (0=embedding, 1=final)")
    ax.set_ylabel("Z-Score")
    ax.set_title("Cross-Model Z-Score Profiles — Universality Test")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_cross_model_entropy(terminal_entropy: dict, save_path: str):
    """Compare H1 entropy vs difficulty across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, entropies in terminal_entropy.items():
        levels = sorted(entropies.keys())
        vals = [entropies[l] for l in levels]
        label = MODEL_LABELS.get(key, key)
        color = MODEL_COLORS.get(key, None)
        ax.plot(levels, vals, "o-", label=label, color=color, linewidth=2, markersize=8, alpha=0.8)

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("H1 Persistence Entropy")
    ax.set_title("Terminal-Layer H1 Entropy vs Difficulty — Cross-Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_cross_model_id(id_results: dict, save_path: str):
    """Overlay ID profiles across models (level 1 vs 5)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, level, title in zip(axes, [1, 5], ["Level 1 (Easy)", "Level 5 (Hard)"]):
        for key, profiles in id_results.items():
            if level not in profiles:
                continue
            ids = profiles[level]
            x = np.linspace(0, 1, len(ids))
            label = MODEL_LABELS.get(key, key)
            color = MODEL_COLORS.get(key, None)
            ax.plot(x, ids, label=label, color=color, linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Normalized Layer Position")
        ax.set_ylabel("Intrinsic Dimension (TwoNN)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Model Intrinsic Dimension Profiles", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-model topological analysis")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_FILES.keys()),
        choices=list(MODEL_FILES.keys()),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-zscore", action="store_true", help="Skip z-score (slow)")
    parser.add_argument("--skip-id", action="store_true", help="Skip ID profiles")
    parser.add_argument("--qwen-data", type=str, default=None,
                        help="Override Qwen data path (e.g. for aligned extraction)")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output JSON path")
    args = parser.parse_args()

    if args.qwen_data:
        MODEL_FILES["qwen"] = os.path.basename(args.qwen_data)

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    loaders = load_available_models(args.models)
    if not loaders:
        print("No model archives found. Run extract_hidden_states_multimodel.py first.")
        sys.exit(1)

    results = {"models": list(loaders.keys())}

    # --- Terminal-layer entropy (fast) ---
    terminal_entropy = run_terminal_entropy(loaders, seed=args.seed)
    results["terminal_entropy"] = {
        k: {str(l): v for l, v in ent.items()}
        for k, ent in terminal_entropy.items()
    }
    plot_cross_model_entropy(
        terminal_entropy,
        os.path.join(FIGURES_DIR, "cross_model_h1_entropy.png"),
    )

    # --- Z-score profiles (slow) ---
    if not args.skip_zscore:
        zscore_results = run_zscore_profiles(loaders, seed=args.seed)
        results["zscore_profiles"] = {
            k: {
                "z_scores": v["z_scores"].tolist(),
                "peak_layer": int(np.argmax(v["z_scores"])),
                "peak_zscore": float(np.max(v["z_scores"])),
            }
            for k, v in zscore_results.items()
        }
        plot_cross_model_zscore(
            zscore_results,
            os.path.join(FIGURES_DIR, "cross_model_zscore.png"),
        )

    # --- ID profiles ---
    if not args.skip_id:
        id_results = run_id_profiles(loaders)
        results["id_profiles"] = {
            k: {str(l): v for l, v in profiles.items()}
            for k, profiles in id_results.items()
        }
        plot_cross_model_id(
            id_results,
            os.path.join(FIGURES_DIR, "cross_model_id_profiles.png"),
        )

    # --- Universality summary ---
    print("\n" + "=" * 60)
    print("UNIVERSALITY SUMMARY")
    print("=" * 60)

    if "zscore_profiles" in results:
        print("\nTerminal-layer effect replication:")
        for key, data in results["zscore_profiles"].items():
            n_layers = loaders[key].num_layers
            peak_frac = data["peak_layer"] / (n_layers - 1)
            terminal = "YES" if peak_frac > 0.8 else "NO"
            print(f"  {MODEL_LABELS.get(key, key)}: peak at layer "
                  f"{data['peak_layer']}/{n_layers-1} ({peak_frac:.0%}) "
                  f"z={data['peak_zscore']:.2f} — terminal: {terminal}")

    print("\nH1 entropy non-monotonicity:")
    for key, ent in terminal_entropy.items():
        levels = sorted(ent.keys())
        vals = [ent[l] for l in levels]
        is_monotonic = all(vals[i] <= vals[i+1] for i in range(len(vals)-1)) or \
                       all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
        print(f"  {MODEL_LABELS.get(key, key)}: {['non-monotonic', 'MONOTONIC'][is_monotonic]} "
              f"— {[f'{v:.3f}' for v in vals]}")

    # Save results
    out_path = args.output or os.path.join(DATA_DIR, "cross_model_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
