"""Direction 3: Euclidean vs spectral PH comparison across layers.

Compares persistent homology computed on Euclidean distances vs spectral
(effective resistance) distances at each transformer layer. Spectral PH
should better capture intrinsic geometry in high-dimensional spaces.

Usage:
    python scripts/run_spectral_ph.py
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
from att.topology.persistence import PersistenceAnalyzer
from att.topology.spectral import spectral_distance_matrix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "transformer", "math500_hidden_states.npz")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "llm")
RESULTS_DIR = os.path.join(REPO_ROOT, "data", "transformer")


def compute_ph(cloud, max_dim=1, subsample=200, seed=42, method="euclidean", k=15, n_pca=50):
    """Compute PH on a point cloud using Euclidean or spectral distances."""
    n_pts = cloud.shape[0]
    if n_pts < 5:
        return None

    n_comp = min(n_pca, n_pts - 1, cloud.shape[1])
    pca = PCA(n_components=n_comp)
    cloud_pca = pca.fit_transform(cloud)

    sub = min(n_pts, subsample) if subsample else None
    if sub and sub < n_pts:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_pts, size=sub, replace=False)
        cloud_pca = cloud_pca[idx]

    if method == "spectral":
        D = spectral_distance_matrix(cloud_pca, k=min(k, len(cloud_pca) - 1))
        pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser", metric="precomputed")
        result = pa.fit_transform(D)
    else:
        pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser")
        result = pa.fit_transform(cloud_pca)

    return result


def main():
    parser = argparse.ArgumentParser(description="Spectral vs Euclidean PH comparison")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=15, help="kNN parameter for spectral")
    parser.add_argument("--subsample", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    loader = HiddenStateLoader(args.data)
    print(loader)

    levels_to_test = [1, 5]
    # Select a subset of layers to keep runtime manageable
    n_layers = loader.num_layers
    layer_indices = list(range(0, n_layers, max(1, n_layers // 10))) + [n_layers - 1]
    layer_indices = sorted(set(layer_indices))

    results = {}

    for level in levels_to_test:
        print(f"\n=== Level {level} ===")
        euc_entropy = {0: [], 1: []}
        spec_entropy = {0: [], 1: []}
        euc_n_features = {0: [], 1: []}
        spec_n_features = {0: [], 1: []}
        layer_labels = []

        for layer_idx in layer_indices:
            cloud = loader.get_level_cloud(level, layer=layer_idx)
            print(f"  Layer {layer_idx}: {cloud.shape[0]} points, dim={cloud.shape[1]}")

            res_euc = compute_ph(cloud, subsample=args.subsample, seed=args.seed, method="euclidean")
            res_spec = compute_ph(cloud, subsample=args.subsample, seed=args.seed, method="spectral", k=args.k)

            layer_labels.append(layer_idx)

            for dim in [0, 1]:
                if res_euc:
                    euc_entropy[dim].append(res_euc["persistence_entropy"][dim])
                    euc_n_features[dim].append(len(res_euc["diagrams"][dim]))
                else:
                    euc_entropy[dim].append(0)
                    euc_n_features[dim].append(0)

                if res_spec:
                    spec_entropy[dim].append(res_spec["persistence_entropy"][dim])
                    spec_n_features[dim].append(len(res_spec["diagrams"][dim]))
                else:
                    spec_entropy[dim].append(0)
                    spec_n_features[dim].append(0)

        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for dim in [0, 1]:
            ax = axes[0, dim]
            ax.plot(layer_labels, euc_entropy[dim], "b-o", label="Euclidean", markersize=4)
            ax.plot(layer_labels, spec_entropy[dim], "r-s", label="Spectral", markersize=4)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Persistence Entropy")
            ax.set_title(f"H{dim} Entropy — Level {level}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1, dim]
            ax.plot(layer_labels, euc_n_features[dim], "b-o", label="Euclidean", markersize=4)
            ax.plot(layer_labels, spec_n_features[dim], "r-s", label="Spectral", markersize=4)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Feature Count")
            ax.set_title(f"H{dim} Features — Level {level}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Euclidean vs Spectral PH — Level {level}", fontsize=14)
        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, f"spectral_comparison_level{level}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

        results[f"level_{level}"] = {
            "layers": layer_labels,
            "euclidean_entropy_h0": euc_entropy[0],
            "euclidean_entropy_h1": euc_entropy[1],
            "spectral_entropy_h0": spec_entropy[0],
            "spectral_entropy_h1": spec_entropy[1],
            "euclidean_n_features_h1": euc_n_features[1],
            "spectral_n_features_h1": spec_n_features[1],
        }

    # Save results
    out_path = os.path.join(RESULTS_DIR, "spectral_ph_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
