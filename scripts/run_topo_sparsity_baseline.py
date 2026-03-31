"""Baseline topological sparsity analysis (old 1e4 threshold behavior).

Runs all 4 experiments from topo_sparsity_analysis.ipynb with
condition_threshold=1e4 forced in Experiment 2. Saves figures to
figures/baseline/ and prints all metrics for capture.
"""

import os
import sys
import json
import warnings
from itertools import combinations

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure att is importable from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from att.config import set_seed
from att.topology import PersistenceAnalyzer
from att.embedding import validate_embedding

set_seed(42)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "baseline")
os.makedirs(FIGURES_DIR, exist_ok=True)
DATA_PATH = os.path.join(REPO_ROOT, "data", "transformer", "math500_hidden_states.npz")
RESULTS_PATH = os.path.join(REPO_ROOT, "data", "transformer", "baseline_results.json")

# ========== LOAD DATA ==========
print("=" * 60)
print("BASELINE RUN (condition_threshold=1e4)")
print("=" * 60)

data = np.load(DATA_PATH, allow_pickle=True)
last_hidden = data["last_hidden_states"]
levels = data["difficulty_levels"]
layer_hidden = data["layer_hidden_states"]
token_trajs = data["token_trajectories"]
seq_lengths = data["seq_lengths"]
model_name = str(data["model_name"])
hidden_dim = int(data["hidden_dim"])
num_layers = int(data["num_layers"])

print(f"Model: {model_name}")
print(f"Hidden dim: {hidden_dim}, Layers (incl. embedding): {num_layers}")
print(f"Total problems: {len(last_hidden)}")
for lv in range(1, 6):
    print(f"  Level {lv}: {(levels == lv).sum()} problems")


# ========== CELL 0: SPARSITY BASELINES ==========
print("\n--- Cell 0: Sparsity Baselines ---")

def l1_norm(h):
    return np.sum(np.abs(h)) / len(h)

def top_k_energy(h, k_pct=0.1):
    h2 = h ** 2
    total = np.sum(h2)
    if total < 1e-15:
        return 0.0
    k = max(1, int(len(h) * k_pct))
    top_k = np.sort(h2)[-k:]
    return np.sum(top_k) / total

def effective_rank(h):
    h2 = h ** 2
    total = np.sum(h2)
    if total < 1e-15:
        return 0.0
    p = h2 / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy) / len(h)

sparsity_metrics = {}
for level in range(1, 6):
    mask = levels == level
    hs = last_hidden[mask]
    l1s = [l1_norm(h) for h in hs]
    t10s = [top_k_energy(h, 0.1) for h in hs]
    ers = [effective_rank(h) for h in hs]
    sparsity_metrics[level] = {
        "l1_norm": float(np.mean(l1s)),
        "top10_energy": float(np.mean(t10s)),
        "eff_rank": float(np.mean(ers)),
        "n_problems": int(mask.sum()),
    }
    print(f"Level {level}: n={mask.sum()}, L1={np.mean(l1s):.4f}, Top10E={np.mean(t10s):.4f}, EffRank={np.mean(ers):.4f}")

t10_values = [sparsity_metrics[lv]["top10_energy"] for lv in range(1, 6)]
r_t10, p_t10 = stats.spearmanr(range(1, 6), t10_values)
print(f"\nSpearman(difficulty, Top10E): r={r_t10:.3f}, p={p_t10:.4f}")
if r_t10 > 0 and p_t10 < 0.1:
    print("Sparsity-difficulty trend CONFIRMED.")
else:
    print("WARNING: Sparsity-difficulty trend NOT confirmed.")


# ========== EXPERIMENT 1: Point Cloud PH ==========
print("\n--- Experiment 1: Point Cloud Topology ---")

N_PCA_COMPONENTS = 50
exp1_results = {}
exp1_analyzers = {}
exp1_pca_info = {}

for level in range(1, 6):
    mask = levels == level
    cloud = last_hidden[mask]
    n_k = cloud.shape[0]
    n_components = min(N_PCA_COMPONENTS, n_k - 1, cloud.shape[1])
    pca = PCA(n_components=n_components)
    cloud_pca = pca.fit_transform(cloud)
    cumvar = pca.explained_variance_ratio_.cumsum()
    exp1_pca_info[level] = cumvar
    print(f"Level {level}: {n_k} points -> PCA to {n_components}d, var explained: {cumvar[-1]:.4f} (@10={cumvar[min(9, len(cumvar)-1)]:.4f})")

    pa = PersistenceAnalyzer(max_dim=2, backend="ripser")
    result = pa.fit_transform(cloud_pca, subsample=min(n_k, 200), seed=42)
    exp1_results[level] = result
    exp1_analyzers[level] = pa
    pe = result["persistence_entropy"]
    print(f"  PE: H0={pe[0]:.4f}, H1={pe[1]:.4f}, H2={pe[2]:.4f}")

# Experiment 1 correlations
level_pe_h1 = np.array([exp1_results[lv]["persistence_entropy"][1] for lv in range(1, 6)])
level_l1 = np.array([sparsity_metrics[lv]["l1_norm"] for lv in range(1, 6)])
level_t10 = np.array([sparsity_metrics[lv]["top10_energy"] for lv in range(1, 6)])
level_er = np.array([sparsity_metrics[lv]["eff_rank"] for lv in range(1, 6)])

print("\nPearson correlations (5 data points, one per difficulty level):")
for name, vals in [("L1 Norm", level_l1), ("Top10% Energy", level_t10), ("Eff Rank", level_er)]:
    r, p = stats.pearsonr(level_pe_h1, vals)
    print(f"  PE(H1) vs {name}: r={r:.3f}, p={p:.4f}")

# --- Exp 1 Plots ---
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for i, level in enumerate(range(1, 6)):
    ax = axes[i]
    diagrams = exp1_results[level]["diagrams"]
    colors = ["tab:blue", "tab:orange"]
    for dim in range(2):
        dgm = diagrams[dim]
        if len(dgm) > 0:
            ax.scatter(dgm[:, 0], dgm[:, 1], c=colors[dim], label=f"H{dim}", s=15, alpha=0.7, edgecolors="k", linewidths=0.3)
    all_vals = np.concatenate([d.ravel() for d in diagrams[:2] if len(d) > 0])
    if len(all_vals) > 0:
        vmin, vmax = all_vals.min(), all_vals.max()
        ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.3)
    ax.set_title(f"Level {level}")
    ax.set_xlabel("Birth")
    if i == 0:
        ax.set_ylabel("Death")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
fig.suptitle("Persistence Diagrams by Difficulty Level (BASELINE)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "exp1_persistence_diagrams.png"), dpi=300, bbox_inches="tight")
plt.close()

pe_h1_plot = [exp1_results[lv]["persistence_entropy"][1] for lv in range(1, 6)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, 6), pe_h1_plot, "o-", color="tab:red", linewidth=2, markersize=8)
ax.set_xlabel("Difficulty Level")
ax.set_ylabel("Persistence Entropy (H1)")
ax.set_title("H1 Persistence Entropy vs Task Difficulty (BASELINE)")
ax.set_xticks(range(1, 6))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "exp1_entropy_vs_difficulty.png"), dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
for n_comp, color, label in [(10, "tab:blue", "@10"), (20, "tab:orange", "@20"), (50, "tab:green", "@50")]:
    vals = []
    for lv in range(1, 6):
        cv = exp1_pca_info[lv]
        idx = min(n_comp - 1, len(cv) - 1)
        vals.append(cv[idx])
    ax.plot(range(1, 6), vals, "o-", color=color, linewidth=2, markersize=8, label=label)
ax.set_xlabel("Difficulty Level")
ax.set_ylabel("Cumulative Variance Explained")
ax.set_title("PCA Variance Concentration vs Difficulty (BASELINE)")
ax.set_xticks(range(1, 6))
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "exp1_pca_variance.png"), dpi=300, bbox_inches="tight")
plt.close()

# PI difference
all_h1_births, all_h1_pers = [], []
for lv in range(1, 6):
    dgm = exp1_results[lv]["diagrams"][1]
    if len(dgm) > 0:
        all_h1_births.extend(dgm[:, 0])
        all_h1_pers.extend(dgm[:, 1] - dgm[:, 0])

if all_h1_births:
    shared_birth_range = (min(all_h1_births), max(all_h1_births))
    shared_pers_range = (0.0, max(all_h1_pers))
else:
    shared_birth_range = (0.0, 1.0)
    shared_pers_range = (0.0, 1.0)

pi_level1 = exp1_analyzers[1].to_image(resolution=50, sigma=0.1, birth_range=shared_birth_range, persistence_range=shared_pers_range)
pi_level5 = exp1_analyzers[5].to_image(resolution=50, sigma=0.1, birth_range=shared_birth_range, persistence_range=shared_pers_range)
diff_h1 = pi_level5[1] - pi_level1[1]

fig, ax = plt.subplots(figsize=(6, 5))
vmax = max(abs(diff_h1.min()), abs(diff_h1.max())) or 1.0
im = ax.imshow(diff_h1, cmap="RdBu_r", origin="lower", aspect="auto", vmin=-vmax, vmax=vmax)
ax.set_title("PI(Level 5) - PI(Level 1), H1 (BASELINE)")
ax.set_xlabel("Birth")
ax.set_ylabel("Persistence")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "exp1_pi_difference_h1.png"), dpi=300, bbox_inches="tight")
plt.close()


# ========== EXPERIMENT 2: Trajectory Topology (BASELINE: threshold=1e4) ==========
print("\n--- Experiment 2: Token Trajectory Topology (BASELINE threshold=1e4) ---")

N_PROBLEMS_PER_LEVEL = 20
N_PCA_TRAJ = 10
BASELINE_THRESHOLD = 1e4

exp2_results = {lv: [] for lv in range(1, 6)}

for level in range(1, 6):
    mask = levels == level
    level_indices = np.where(mask)[0][:N_PROBLEMS_PER_LEVEL]
    n_degen = 0
    n_total = 0

    for idx in level_indices:
        traj = token_trajs[idx]
        if traj.shape[0] < 5:
            continue
        n_total += 1
        n_comp = min(N_PCA_TRAJ, traj.shape[0] - 1, traj.shape[1])
        pca = PCA(n_components=n_comp)
        traj_pca = pca.fit_transform(traj)

        # BASELINE: force old threshold
        val = validate_embedding(traj_pca, condition_threshold=BASELINE_THRESHOLD)
        is_degen = val["degenerate"]
        if is_degen:
            n_degen += 1

        pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
        result = pa.fit_transform(traj_pca, subsample=min(traj_pca.shape[0], 500), seed=42)

        exp2_results[level].append({
            "persistence_entropy_h1": float(result["persistence_entropy"][1]),
            "condition_number": float(val["condition_number"]),
            "effective_rank": int(val["effective_rank"]),
            "degenerate": bool(is_degen),
            "n_tokens": int(traj.shape[0]),
            "n_h1_features": int(len(result["diagrams"][1])),
            "threshold_used": float(val["threshold_used"]),
        })

    degen_rate = n_degen / n_total if n_total > 0 else 0
    mean_cond = np.mean([r["condition_number"] for r in exp2_results[level]]) if exp2_results[level] else 0
    mean_pe = np.mean([r["persistence_entropy_h1"] for r in exp2_results[level]]) if exp2_results[level] else 0
    print(f"Level {level}: {n_total} traj, {degen_rate:.0%} degen, mean cond={mean_cond:.1f}, mean PE(H1)={mean_pe:.4f}")

# Exp 2 summary table
print("\nLevel | Degen Rate | Mean Cond | Mean PE(H1) | Mean #H1 Features")
print("------|------------|-----------|-------------|------------------")
for lv in range(1, 6):
    results = exp2_results[lv]
    if results:
        dr = np.mean([r["degenerate"] for r in results])
        mc = np.mean([r["condition_number"] for r in results])
        mp = np.mean([r["persistence_entropy_h1"] for r in results])
        mf = np.mean([r["n_h1_features"] for r in results])
        print(f"  {lv}   | {dr:>9.0%} | {mc:>9.1f} | {mp:>11.4f} | {mf:>17.1f}")

# Exp 2 plots
level_labels = list(range(1, 6))
exp2_pe_means, exp2_pe_stds, exp2_cond_means, exp2_cond_stds, exp2_degen_rates = [], [], [], [], []
for lv in level_labels:
    results = exp2_results[lv]
    if results:
        pes = [r["persistence_entropy_h1"] for r in results]
        conds = [r["condition_number"] for r in results]
        degens = [r["degenerate"] for r in results]
        exp2_pe_means.append(np.mean(pes))
        exp2_pe_stds.append(np.std(pes) / np.sqrt(len(pes)))
        exp2_cond_means.append(np.mean(conds))
        exp2_cond_stds.append(np.std(conds) / np.sqrt(len(conds)))
        exp2_degen_rates.append(np.mean(degens))
    else:
        exp2_pe_means.append(0)
        exp2_pe_stds.append(0)
        exp2_cond_means.append(0)
        exp2_cond_stds.append(0)
        exp2_degen_rates.append(0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax = axes[0]
ax.errorbar(level_labels, exp2_pe_means, yerr=exp2_pe_stds, fmt="o-", color="tab:red", linewidth=2, markersize=8, capsize=4)
ax.set_xlabel("Difficulty Level"); ax.set_ylabel("Persistence Entropy (H1)")
ax.set_title("Trajectory H1 Entropy vs Difficulty (BASELINE)"); ax.set_xticks(level_labels); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.errorbar(level_labels, exp2_cond_means, yerr=exp2_cond_stds, fmt="s-", color="tab:purple", linewidth=2, markersize=8, capsize=4)
ax.set_xlabel("Difficulty Level"); ax.set_ylabel("Condition Number")
ax.set_title("Trajectory Condition Number vs Difficulty (BASELINE)"); ax.set_xticks(level_labels); ax.grid(True, alpha=0.3)

ax = axes[2]
bars = ax.bar(level_labels, exp2_degen_rates, color="tab:gray", edgecolor="black")
ax.set_xlabel("Difficulty Level"); ax.set_ylabel("Fraction Degenerate")
ax.set_title("Trajectory Degeneracy Rate vs Difficulty (BASELINE)"); ax.set_xticks(level_labels)
ax.set_ylim(0, 1); ax.grid(True, alpha=0.3, axis="y")
for bar, rate in zip(bars, exp2_degen_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{rate:.0%}", ha="center", va="bottom", fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "exp2_trajectory_topology.png"), dpi=300, bbox_inches="tight")
plt.close()


# ========== EXPERIMENT 3: Layer-wise Transition ==========
print("\n--- Experiment 3: Layer-wise Topological Transition ---")

N_PROBLEMS_EXP3 = 30
N_PCA_LAYER = 20
LEVELS_EXP3 = [1, 5]

exp3_distances = {}
exp3_entropies = {}

for level in LEVELS_EXP3:
    mask = levels == level
    level_indices = np.where(mask)[0][:N_PROBLEMS_EXP3]
    n_problems = len(level_indices)
    n_total_layers = layer_hidden.shape[1]
    print(f"Level {level}: {n_problems} problems, {n_total_layers} layers")

    analyzers_per_layer = []
    layer_entropies = []

    for ell in range(n_total_layers):
        cloud = layer_hidden[level_indices, ell, :]
        n_comp = min(N_PCA_LAYER, n_problems - 1, cloud.shape[1])
        pca = PCA(n_components=n_comp)
        cloud_pca = pca.fit_transform(cloud)
        pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
        result = pa.fit_transform(cloud_pca, seed=42)
        analyzers_per_layer.append(pa)
        layer_entropies.append(result["persistence_entropy"][1])

    distances = []
    for ell in range(n_total_layers - 1):
        d = analyzers_per_layer[ell].distance(analyzers_per_layer[ell + 1], metric="bottleneck")
        distances.append(d)

    exp3_distances[level] = distances
    exp3_entropies[level] = layer_entropies
    print(f"  Bottleneck (last 5): {[f'{d:.4f}' for d in distances[-5:]]}")

# Terminal vs non-terminal ratios
for level in LEVELS_EXP3:
    dists = exp3_distances[level]
    if len(dists) >= 6:
        terminal = np.mean(dists[-5:])
        nonterminal = np.mean(dists[:-5])
        ratio = terminal / nonterminal if nonterminal > 0 else float("inf")
        print(f"Level {level}: terminal={terminal:.4f}, non-terminal={nonterminal:.4f}, ratio={ratio:.2f}x")

# Exp 3 plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_map = {"1": "tab:blue", "5": "tab:red"}
ax = axes[0]
for level in LEVELS_EXP3:
    dists = exp3_distances[level]
    ax.plot(range(len(dists)), dists, "o-", color=colors_map[str(level)], linewidth=2, markersize=4, label=f"Level {level}")
n_total_layers = layer_hidden.shape[1]
shade_start = max(0, n_total_layers - 6)
ax.axvspan(shade_start, n_total_layers - 2, alpha=0.1, color="gray", label="Final 5 layers")
ax.set_xlabel("Layer Transition"); ax.set_ylabel("Bottleneck Distance")
ax.set_title("Layer-wise Topological Transition (BASELINE)"); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
for level in LEVELS_EXP3:
    ents = exp3_entropies[level]
    ax.plot(range(len(ents)), ents, "o-", color=colors_map[str(level)], linewidth=2, markersize=4, label=f"Level {level}")
ax.axvspan(shade_start, n_total_layers - 1, alpha=0.1, color="gray", label="Final 5 layers")
ax.set_xlabel("Layer Index"); ax.set_ylabel("Persistence Entropy (H1)")
ax.set_title("Layer-wise H1 Complexity (BASELINE)"); ax.legend(); ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "exp3_layerwise_transition.png"), dpi=300, bbox_inches="tight")
plt.close()


# ========== EXPERIMENT 4: Pairwise Distance + Permutation Test ==========
print("\n--- Experiment 4: Pairwise Wasserstein Distance ---")

topo_distance_matrix = np.zeros((5, 5))
for i, j in combinations(range(5), 2):
    d = exp1_analyzers[i + 1].distance(exp1_analyzers[j + 1], metric="wasserstein_1")
    topo_distance_matrix[i, j] = d
    topo_distance_matrix[j, i] = d

print("Observed Wasserstein distance matrix:")
header = "      " + "  ".join([f"Lv{lv}" for lv in range(1, 6)])
print(header)
for i in range(5):
    row = f"Lv{i+1}  " + "  ".join([f"{topo_distance_matrix[i, j]:.4f}" for j in range(5)])
    print(row)

adj_dists = [topo_distance_matrix[i, i + 1] for i in range(4)]
d_1_5 = topo_distance_matrix[0, 4]
print(f"\nMean adjacent-level distance: {np.mean(adj_dists):.4f}")
print(f"Level 1 vs Level 5 distance: {d_1_5:.4f}")

# Permutation test
print("\nRunning permutation test (N=200)...")
N_PERMUTATIONS = 200
observed_mean_dist = topo_distance_matrix[np.triu_indices(5, k=1)].mean()
rng = np.random.default_rng(42)
null_mean_dists = []

for perm_idx in range(N_PERMUTATIONS):
    if perm_idx % 50 == 0:
        print(f"  Permutation {perm_idx}/{N_PERMUTATIONS}...")
    shuffled_levels = rng.permutation(levels)
    perm_analyzers = {}
    for level in range(1, 6):
        mask = shuffled_levels == level
        cloud = last_hidden[mask]
        n_k = cloud.shape[0]
        if n_k < 3:
            perm_analyzers[level] = None
            continue
        n_comp = min(N_PCA_COMPONENTS, n_k - 1, cloud.shape[1])
        pca = PCA(n_components=n_comp)
        cloud_pca = pca.fit_transform(cloud)
        pa = PersistenceAnalyzer(max_dim=2, backend="ripser")
        pa.fit_transform(cloud_pca, subsample=min(n_k, 200), seed=42)
        perm_analyzers[level] = pa

    perm_dists = []
    for i, j in combinations(range(5), 2):
        pa_i = perm_analyzers[i + 1]
        pa_j = perm_analyzers[j + 1]
        if pa_i is not None and pa_j is not None:
            d = pa_i.distance(pa_j, metric="wasserstein_1")
            perm_dists.append(d)
    if perm_dists:
        null_mean_dists.append(np.mean(perm_dists))

null_mean_dists = np.array(null_mean_dists)
p_value = (np.sum(null_mean_dists >= observed_mean_dist) + 1) / (len(null_mean_dists) + 1)
z_score = (observed_mean_dist - null_mean_dists.mean()) / (null_mean_dists.std() + 1e-15)

print(f"\nObserved mean Wasserstein: {observed_mean_dist:.4f}")
print(f"Null: mean={null_mean_dists.mean():.4f}, std={null_mean_dists.std():.4f}")
print(f"z-score: {z_score:.2f}")
print(f"Permutation p-value: {p_value:.4f}")
print(f"Significant at alpha=0.05: {p_value < 0.05}")

# Exp 4 plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
im = ax.imshow(topo_distance_matrix, cmap="YlOrRd", origin="lower")
ax.set_xticks(range(5)); ax.set_xticklabels([f"Level {i}" for i in range(1, 6)])
ax.set_yticks(range(5)); ax.set_yticklabels([f"Level {i}" for i in range(1, 6)])
ax.set_title("Pairwise Wasserstein Distance (BASELINE)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for i in range(5):
    for j in range(5):
        val = topo_distance_matrix[i, j]
        color = "white" if val > topo_distance_matrix.max() * 0.6 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

ax = axes[1]
ax.hist(null_mean_dists, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
ax.axvline(observed_mean_dist, color="red", linewidth=2, label=f"Observed = {observed_mean_dist:.4f}")
p95 = np.percentile(null_mean_dists, 95)
ax.axvline(p95, color="orange", linewidth=1.5, linestyle="--", label=f"95th pctile = {p95:.4f}")
ax.set_xlabel("Mean Pairwise Wasserstein Distance"); ax.set_ylabel("Count")
ax.set_title(f"Permutation Null (n={N_PERMUTATIONS}), p={p_value:.4f} (BASELINE)"); ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "exp4_distance_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()


# ========== FINAL SUMMARY ==========
print("\n--- Final Summary Table (BASELINE) ---")

summary = {}
for level in range(1, 6):
    sm = sparsity_metrics[level]
    pe_h1 = exp1_results[level]["persistence_entropy"][1]
    cv = exp1_pca_info[level]
    pca_var_50 = cv[min(49, len(cv) - 1)]
    e2 = exp2_results[level]
    mean_cond = float(np.mean([r["condition_number"] for r in e2])) if e2 else float("nan")
    degen_rate = float(np.mean([r["degenerate"] for r in e2])) if e2 else float("nan")
    mean_pe_h1 = float(np.mean([r["persistence_entropy_h1"] for r in e2])) if e2 else float("nan")
    mean_h1_feat = float(np.mean([r["n_h1_features"] for r in e2])) if e2 else float("nan")

    summary[level] = {
        "l1_norm": sm["l1_norm"],
        "top10_energy": sm["top10_energy"],
        "eff_rank": sm["eff_rank"],
        "pers_entropy_h1": float(pe_h1),
        "pca_var_50": float(pca_var_50),
        "cond_number": mean_cond,
        "degen_rate": degen_rate,
        "traj_pe_h1": mean_pe_h1,
        "traj_h1_features": mean_h1_feat,
    }

cols = ["L1 Norm", "Top10% E", "Eff Rank", "PE(H1)", "PCA@50", "Cond #", "Degen %"]
keys = ["l1_norm", "top10_energy", "eff_rank", "pers_entropy_h1", "pca_var_50", "cond_number", "degen_rate"]
header = "Level | " + " | ".join(f"{c:>9s}" for c in cols)
sep = "------|" + "|".join(["-" * 11] * len(cols))
print(header)
print(sep)
for level in range(1, 6):
    vals = []
    for k in keys:
        v = summary[level][k]
        if k == "degen_rate":
            vals.append(f"{v:>8.0%} " if not np.isnan(v) else "      n/a ")
        elif k == "cond_number":
            vals.append(f"{v:>9.1f}" if not np.isnan(v) else "      n/a")
        else:
            vals.append(f"{v:>9.4f}")
    print(f"  {level}   | " + " | ".join(vals))

# Correlation matrix
print("\nPearson Correlation Matrix:")
metric_names = ["L1 Norm", "Top10% Energy", "Eff Rank", "PE(H1)", "PCA Var@50", "Cond Number"]
metric_keys = ["l1_norm", "top10_energy", "eff_rank", "pers_entropy_h1", "pca_var_50", "cond_number"]
metric_arrays = {}
for k in metric_keys:
    metric_arrays[k] = np.array([summary[lv][k] for lv in range(1, 6)])

print(f"\n{'':>16s}", end="")
for name in metric_names:
    print(f" {name:>13s}", end="")
print()
for i, (name_i, key_i) in enumerate(zip(metric_names, metric_keys)):
    print(f"{name_i:>16s}", end="")
    for j, (name_j, key_j) in enumerate(zip(metric_names, metric_keys)):
        arr_i = metric_arrays[key_i]
        arr_j = metric_arrays[key_j]
        if np.any(np.isnan(arr_i)) or np.any(np.isnan(arr_j)):
            print(f" {'n/a':>13s}", end="")
        else:
            r, _ = stats.pearsonr(arr_i, arr_j)
            print(f" {r:>13.3f}", end="")
    print()

# Save results to JSON
results_out = {
    "run_type": "baseline",
    "condition_threshold": BASELINE_THRESHOLD,
    "sparsity_metrics": sparsity_metrics,
    "spearman_difficulty_top10e": {"r": float(r_t10), "p": float(p_t10)},
    "exp1_pe_h1": {str(lv): float(exp1_results[lv]["persistence_entropy"][1]) for lv in range(1, 6)},
    "exp2_summary": {str(lv): summary[lv] for lv in range(1, 6)},
    "exp3_terminal_ratios": {},
    "exp4_observed_mean_dist": float(observed_mean_dist),
    "exp4_p_value": float(p_value),
    "exp4_z_score": float(z_score),
    "exp4_significant": bool(p_value < 0.05),
}
for level in LEVELS_EXP3:
    dists = exp3_distances[level]
    if len(dists) >= 6:
        terminal = float(np.mean(dists[-5:]))
        nonterminal = float(np.mean(dists[:-5]))
        ratio = terminal / nonterminal if nonterminal > 0 else float("inf")
        exp3_terminal_ratios = {"terminal": terminal, "nonterminal": nonterminal, "ratio": ratio}
    else:
        exp3_terminal_ratios = {}
    results_out["exp3_terminal_ratios"][str(level)] = exp3_terminal_ratios

with open(RESULTS_PATH, "w") as f:
    json.dump(results_out, f, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")
print(f"Figures saved to {FIGURES_DIR}")
print("\nBASELINE RUN COMPLETE.")
