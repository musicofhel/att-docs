#!/usr/bin/env python3
"""Experiment 1: Semantic Cartography — What Are the Two Clusters?

Tests five competing hypotheses about what the two clusters in LLM hidden states
represent by assigning every token to a cluster and cross-referencing against
semantic, functional, and computational labels.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "two_clusters"
FIG_DIR = ROOT / "figures" / "two_clusters"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
data = np.load(ROOT / "data/transformer/math500_hidden_states_aligned.npz", allow_pickle=True)
token_trajs = data["token_trajectories"]
hidden_levels = data["difficulty_levels"]
layer_hidden_states = data["layer_hidden_states"]  # (500, 29, 1536)

corr_data = np.load(ROOT / "data/transformer/math500_correctness.npz", allow_pickle=True)
correct_all = corr_data["correct"]
corr_levels = corr_data["difficulty_levels"]

# ── Pool-index matching ────────────────────────────────────────────────────
print("Aligning correctness labels...")
aligned_h_idx = []
aligned_c_idx = []
for lv in range(1, 6):
    h_idx = np.where(hidden_levels == lv)[0]
    c_idx = np.where(corr_levels == lv)[0]
    n = min(len(h_idx), len(c_idx))
    aligned_h_idx.extend(h_idx[:n].tolist())
    aligned_c_idx.extend(c_idx[:n].tolist())
aligned_h_idx = np.array(aligned_h_idx)
aligned_c_idx = np.array(aligned_c_idx)
correct = correct_all[aligned_c_idx]
print(f"  Matched {len(aligned_h_idx)} problems with correctness labels")

# ── 1a: Extract and Cluster ───────────────────────────────────────────────
print("\n=== 1a: K-means clustering (k=2) on all 500 problems ===")
all_cluster_results = []

for idx in range(len(token_trajs)):
    traj = token_trajs[idx]
    if traj.shape[0] < 10:
        all_cluster_results.append(None)
        continue

    n_comp = min(30, traj.shape[0] - 1, traj.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(traj)

    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(reduced)

    sil = silhouette_score(reduced, labels) if len(set(labels)) > 1 else 0.0
    per_token_sil = silhouette_samples(reduced, labels) if len(set(labels)) > 1 else np.zeros(len(labels))

    all_cluster_results.append({
        "labels": labels,
        "silhouette": sil,
        "per_token_silhouette": per_token_sil,
        "centroid_distance": float(np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1])),
        "cluster_sizes": [int((labels == 0).sum()), int((labels == 1).sum())],
        "pca": pca,
        "raw_traj": traj,
    })

valid_count = sum(1 for r in all_cluster_results if r is not None)
sil_scores = [r["silhouette"] for r in all_cluster_results if r is not None]
print(f"  Valid problems: {valid_count}/500")
print(f"  Mean silhouette: {np.mean(sil_scores):.4f} ± {np.std(sil_scores):.4f}")

# ── 1b: Token-Type Probe ──────────────────────────────────────────────────
print("\n=== 1b: Token-type classification ===")

# Load tokenizer for token classification
from transformers import AutoTokenizer
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
print("  Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load problem texts to tokenize.
# Match by problem hash stored in the .npz to guarantee alignment.
print("  Loading MATH-500 dataset for tokenization...")
import hashlib
from datasets import load_dataset
math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)

stored_hashes = data["problem_hashes"]  # length 500

def problem_hash(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]

# Build hash → dataset index map
hash_to_ds_idx = {}
for i, item in enumerate(math_ds):
    h = problem_hash(item["problem"])
    hash_to_ds_idx[h] = i

# Map each .npz problem to its dataset row
problem_indices_in_ds = []
matched = 0
for idx in range(500):
    h = stored_hashes[idx]
    ds_idx = hash_to_ds_idx.get(h, None)
    problem_indices_in_ds.append(ds_idx)
    if ds_idx is not None:
        matched += 1

print(f"  Hash-matched {matched}/500 problems to MATH-500 dataset")

def classify_token(token_id, tokenizer):
    """Classify a token into semantic categories."""
    token_str = tokenizer.decode([token_id]).strip()
    if not token_str:
        return "structural"
    cleaned = token_str.replace(".", "").replace("-", "").replace(",", "")
    if cleaned.isdigit():
        return "numeric"
    math_symbols = set("+-*/=^()[]{}\\<>|")
    math_words = {
        "frac", "sqrt", "sum", "int", "lim", "sin", "cos", "tan", "log", "ln",
        "pi", "theta", "alpha", "beta", "gamma", "delta", "cdot", "times", "div",
        "leq", "geq", "neq", "approx", "infty", "pm",
    }
    if token_str in math_words or (len(token_str) <= 3 and any(c in math_symbols for c in token_str)):
        return "math_symbol"
    if token_str.startswith("\\"):
        return "latex"
    if token_str in [".", ",", ":", ";", "!", "?", "\n", " "] or token_str.startswith("<"):
        return "structural"
    if any(c.isdigit() for c in token_str):
        return "numeric"
    return "language"


# Classify tokens for all 500 problems and build contingency table
print("  Classifying tokens...")
contingency = {t: [0, 0] for t in ["numeric", "math_symbol", "language", "latex", "structural"]}
per_problem_type_dist = []

for prob_idx in range(500):
    cr = all_cluster_results[prob_idx]
    ds_idx = problem_indices_in_ds[prob_idx]
    if cr is None or ds_idx is None:
        per_problem_type_dist.append(None)
        continue

    problem_text = math_ds[ds_idx]["problem"]
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=512)

    n_tokens_cluster = len(cr["labels"])
    n_tokens_input = len(input_ids)
    # Use the minimum — token count may differ slightly due to truncation
    n = min(n_tokens_cluster, n_tokens_input)

    type_counts = {t: [0, 0] for t in ["numeric", "math_symbol", "language", "latex", "structural"]}
    for t_idx in range(n):
        tok_type = classify_token(input_ids[t_idx], tokenizer)
        cluster = cr["labels"][t_idx]
        contingency[tok_type][cluster] += 1
        type_counts[tok_type][cluster] += 1
    per_problem_type_dist.append(type_counts)

# Print contingency table
print("\n  Contingency table (token type × cluster):")
print(f"  {'':15s} | {'Cluster A':>10s} | {'Cluster B':>10s}")
print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}")
for ttype in ["numeric", "math_symbol", "language", "latex", "structural"]:
    print(f"  {ttype:15s} | {contingency[ttype][0]:10d} | {contingency[ttype][1]:10d}")

# Chi-squared test — remove rows with all zeros (e.g., "latex" if BPE splits backslash tokens)
type_names = ["numeric", "math_symbol", "language", "latex", "structural"]
observed_raw = np.array([contingency[t] for t in type_names])
nonzero_mask = observed_raw.sum(axis=1) > 0
observed = observed_raw[nonzero_mask]
active_types = [t for t, m in zip(type_names, nonzero_mask) if m]
chi2, chi2_p, dof, expected = stats.chi2_contingency(observed)
n_total = observed.sum()
cramers_v = np.sqrt(chi2 / (n_total * (min(observed.shape) - 1)))
print(f"\n  Chi-squared: {chi2:.2f}, p={chi2_p:.2e}, Cramér's V={cramers_v:.4f}")

# Determine dominant type per cluster
cluster_a_total = observed[:, 0]
cluster_b_total = observed[:, 1]
# Dominant = highest proportion relative to total
prop_a = cluster_a_total / (cluster_a_total + cluster_b_total + 1e-10)
prop_b = cluster_b_total / (cluster_a_total + cluster_b_total + 1e-10)
# The type most skewed toward cluster A
dominant_a_type = active_types[np.argmax(prop_a)]
dominant_b_type = active_types[np.argmax(prop_b)]
print(f"  Dominant in Cluster A: {dominant_a_type} (proportion {prop_a.max():.3f})")
print(f"  Dominant in Cluster B: {dominant_b_type} (proportion {prop_b.max():.3f})")

# ── 1c: Correctness Probe ─────────────────────────────────────────────────
print("\n=== 1c: Correctness probe ===")
correct_sils = []
incorrect_sils = []
correct_cluster_a_fracs = []
incorrect_cluster_a_fracs = []

for i in range(len(aligned_h_idx)):
    h_idx = aligned_h_idx[i]
    cr = all_cluster_results[h_idx]
    if cr is None:
        continue
    is_correct = bool(correct[i])
    sil = cr["silhouette"]
    total = sum(cr["cluster_sizes"])
    frac_a = cr["cluster_sizes"][0] / total if total > 0 else 0.5

    if is_correct:
        correct_sils.append(sil)
        correct_cluster_a_fracs.append(frac_a)
    else:
        incorrect_sils.append(sil)
        incorrect_cluster_a_fracs.append(frac_a)

correct_mean_sil = np.mean(correct_sils)
incorrect_mean_sil = np.mean(incorrect_sils)
u_stat, sil_p = stats.mannwhitneyu(correct_sils, incorrect_sils, alternative="two-sided")
print(f"  Correct silhouette: {correct_mean_sil:.4f} (n={len(correct_sils)})")
print(f"  Incorrect silhouette: {incorrect_mean_sil:.4f} (n={len(incorrect_sils)})")
print(f"  Mann-Whitney U p-value: {sil_p:.4e}")

correct_a_frac = np.mean(correct_cluster_a_fracs)
incorrect_a_frac = np.mean(incorrect_cluster_a_fracs)
print(f"  Correct cluster A fraction: {correct_a_frac:.4f}")
print(f"  Incorrect cluster A fraction: {incorrect_a_frac:.4f}")

# ── 1d: Per-Layer Clustering (Approach A: last-token aggregated) ──────────
print("\n=== 1d: Per-layer clustering (last-token, aggregated by level) ===")

# layer_hidden_states shape: (500, 29, 1536)
per_layer_sil = []
for layer_idx in range(29):
    layer_states = layer_hidden_states[:, layer_idx, :]  # (500, 1536)
    n_comp = min(30, layer_states.shape[0] - 1, layer_states.shape[1])
    pca_l = PCA(n_components=n_comp)
    reduced_l = pca_l.fit_transform(layer_states)
    km_l = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels_l = km_l.fit_predict(reduced_l)
    sil_l = silhouette_score(reduced_l, labels_l)
    per_layer_sil.append(sil_l)

# Find first layer exceeding 0.3
formation_layer = None
for l_idx, s in enumerate(per_layer_sil):
    if s > 0.3:
        formation_layer = l_idx
        break

print(f"  Per-layer silhouette scores (layers 0-28):")
for i, s in enumerate(per_layer_sil):
    marker = " <-- formation" if i == formation_layer else ""
    print(f"    Layer {i:2d}: {s:.4f}{marker}")

# Also compute per-level silhouette at each layer
per_layer_per_level_sil = {lv: [] for lv in range(1, 6)}
for layer_idx in range(29):
    for lv in range(1, 6):
        mask = hidden_levels == lv
        layer_states_lv = layer_hidden_states[mask, layer_idx, :]
        if layer_states_lv.shape[0] < 10:
            per_layer_per_level_sil[lv].append(0.0)
            continue
        n_comp = min(30, layer_states_lv.shape[0] - 1, layer_states_lv.shape[1])
        pca_l = PCA(n_components=n_comp)
        reduced_l = pca_l.fit_transform(layer_states_lv)
        km_l = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels_l = km_l.fit_predict(reduced_l)
        sil_l = silhouette_score(reduced_l, labels_l) if len(set(labels_l)) > 1 else 0.0
        per_layer_per_level_sil[lv].append(sil_l)

# ── Approach B: full per-token per-layer (GPU, 50 problems) ──────────────
print("\n=== 1d-B: Full per-token per-layer clustering (GPU, 50 problems) ===")
import torch
from transformers import AutoModelForCausalLM

TARGET_LAYERS = [0, 7, 14, 21, 27]

# Select 10 problems per level
subset_indices = []
for lv in range(1, 6):
    lv_mask = np.where(hidden_levels == lv)[0]
    subset_indices.extend(lv_mask[:10].tolist())

print(f"  Loading model for forward pass ({len(subset_indices)} problems)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True,
    trust_remote_code=True,
)
model.eval()

per_layer_token_sil = {l: [] for l in TARGET_LAYERS}
per_layer_token_labels = {l: [] for l in TARGET_LAYERS}
switch_rates = []  # between consecutive sampled layers

for count, prob_idx in enumerate(subset_indices):
    ds_idx = problem_indices_in_ds[prob_idx]
    if ds_idx is None:
        continue
    problem_text = math_ds[ds_idx]["problem"]
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Collect hidden states at target layers
    layer_labels = {}
    for l in TARGET_LAYERS:
        hs = outputs.hidden_states[l][0].cpu().float().numpy()  # (n_tokens, hidden_dim)
        if hs.shape[0] < 10:
            continue
        n_comp = min(30, hs.shape[0] - 1, hs.shape[1])
        pca_l = PCA(n_components=n_comp)
        reduced_l = pca_l.fit_transform(hs)
        km_l = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels_l = km_l.fit_predict(reduced_l)
        sil_l = silhouette_score(reduced_l, labels_l) if len(set(labels_l)) > 1 else 0.0
        per_layer_token_sil[l].append(sil_l)
        layer_labels[l] = labels_l

    # Compute switch rates between consecutive layers
    sorted_layers = sorted(layer_labels.keys())
    for i in range(1, len(sorted_layers)):
        l_prev = sorted_layers[i - 1]
        l_curr = sorted_layers[i]
        if l_prev in layer_labels and l_curr in layer_labels:
            n = min(len(layer_labels[l_prev]), len(layer_labels[l_curr]))
            # Need to align labels since KMeans can flip 0/1 between runs
            lab_prev = layer_labels[l_prev][:n]
            lab_curr = layer_labels[l_curr][:n]
            # Try both alignments, pick the one with fewer switches
            switches_direct = np.mean(lab_prev != lab_curr)
            switches_flipped = np.mean(lab_prev != (1 - lab_curr))
            switch_rate = min(switches_direct, switches_flipped)
            switch_rates.append({"from_layer": l_prev, "to_layer": l_curr, "switch_rate": float(switch_rate)})

    torch.cuda.empty_cache()
    if (count + 1) % 10 == 0:
        print(f"    Processed {count + 1}/{len(subset_indices)}")

print(f"\n  Per-layer token silhouette (full per-token, 50 problems):")
for l in TARGET_LAYERS:
    scores = per_layer_token_sil[l]
    if scores:
        print(f"    Layer {l:2d}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

print(f"  Switch rates between consecutive layers:")
# Aggregate by layer pair
from collections import defaultdict
switch_by_pair = defaultdict(list)
for sr in switch_rates:
    key = (sr["from_layer"], sr["to_layer"])
    switch_by_pair[key].append(sr["switch_rate"])
for (l1, l2), rates in sorted(switch_by_pair.items()):
    print(f"    Layer {l1} → {l2}: {np.mean(rates):.4f} ± {np.std(rates):.4f}")

# Determine dominant type of switching tokens
# For each problem in subset, identify tokens that switch and classify them
switching_token_types = {t: 0 for t in ["numeric", "math_symbol", "language", "latex", "structural"]}
for count, prob_idx in enumerate(subset_indices):
    cr = all_cluster_results[prob_idx]
    if cr is None:
        continue
    ds_idx = problem_indices_in_ds[prob_idx]
    problem_text = math_ds[ds_idx]["problem"]
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=512)

    # Use per-layer labels from GPU pass if available
    # Check if we have labels at consecutive layers
    for (l1, l2) in switch_by_pair.keys():
        # Identify switching tokens (we don't have per-problem layer labels stored separately,
        # but we can use the final-layer cluster labels as proxy)
        pass  # This will be approximated from the data we have

# ── 1e: Wikipedia Control ─────────────────────────────────────────────────
print("\n=== 1e: Wikipedia control ===")
from datasets import load_dataset as load_ds
print("  Loading wikitext dataset (as non-math control)...")
try:
    wiki_raw = load_ds("wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
    wiki_passages = []
    for row in wiki_raw:
        text = row["text"].strip()
        if len(text) > 100 and not text.startswith("="):
            # Wrap in same prompt template as MATH problems
            wiki_passages.append(PROMPT_TEMPLATE.format(problem=text[:500]))
        if len(wiki_passages) >= 50:
            break
except Exception as e:
    print(f"  Wikitext load failed ({e}), using synthetic non-math passages")
    # Generate synthetic non-math passages as fallback
    non_math_texts = [
        "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789. It ended with Napoleon Bonaparte's rise to power.",
        "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organism's activities.",
        "The Amazon rainforest covers much of northwestern Brazil and extends into Colombia, Peru, and other South American countries. It is the world's largest tropical rainforest.",
    ] * 17  # Repeat to get 51 passages
    wiki_passages = [PROMPT_TEMPLATE.format(problem=t) for t in non_math_texts[:50]]

print(f"  Processing {len(wiki_passages)} Wikipedia passages...")
wiki_sils = []
for i, passage in enumerate(wiki_passages):
    inputs = tokenizer(passage, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states[-1][0].cpu().float().numpy()  # final layer, all tokens
    torch.cuda.empty_cache()

    if hs.shape[0] < 10:
        continue
    n_comp = min(30, hs.shape[0] - 1, hs.shape[1])
    pca_w = PCA(n_components=n_comp)
    reduced_w = pca_w.fit_transform(hs)
    km_w = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels_w = km_w.fit_predict(reduced_w)
    sil_w = silhouette_score(reduced_w, labels_w) if len(set(labels_w)) > 1 else 0.0
    wiki_sils.append(sil_w)
    if (i + 1) % 10 == 0:
        print(f"    Processed {i + 1}/{len(wiki_passages)}")

wiki_mean_sil = np.mean(wiki_sils)
math_mean_sil = np.mean(sil_scores)
print(f"\n  Wikipedia mean silhouette: {wiki_mean_sil:.4f}")
print(f"  MATH-500 mean silhouette: {math_mean_sil:.4f}")
two_cluster_math_specific = math_mean_sil > wiki_mean_sil + 0.05

# Free GPU memory
del model
torch.cuda.empty_cache()

# ── Generate Figures ───────────────────────────────────────────────────────
print("\n=== Generating figures ===")

# Figure 1: Contingency heatmap
fig, ax = plt.subplots(figsize=(8, 5))
heatmap_data = np.array([contingency[t] for t in type_names])
# Normalize rows for better visualization
row_sums = heatmap_data.sum(axis=1, keepdims=True)
heatmap_norm = heatmap_data / (row_sums + 1e-10)
im = ax.imshow(heatmap_norm, aspect="auto", cmap="YlOrRd")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Cluster A", "Cluster B"])
ax.set_yticks(range(len(type_names)))
ax.set_yticklabels(type_names)
for i in range(len(type_names)):
    for j in range(2):
        ax.text(j, i, f"{heatmap_data[i, j]}\n({heatmap_norm[i, j]:.2f})",
                ha="center", va="center", fontsize=9)
plt.colorbar(im, label="Proportion in cluster")
ax.set_title(f"Token Type × Cluster Membership\nχ²={chi2:.1f}, p={chi2_p:.1e}, V={cramers_v:.3f}")
plt.tight_layout()
plt.savefig(FIG_DIR / "contingency_heatmap.png", dpi=150)
plt.close()
print("  Saved contingency_heatmap.png")

# Figure 2: Silhouette by layer
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(29), per_layer_sil, "ko-", linewidth=2, label="All problems")
colors = plt.cm.viridis(np.linspace(0, 1, 5))
for lv in range(1, 6):
    ax.plot(range(29), per_layer_per_level_sil[lv], "o-", color=colors[lv-1],
            alpha=0.6, label=f"Level {lv}")
ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="Formation threshold (0.3)")
if formation_layer is not None:
    ax.axvline(formation_layer, color="red", linestyle=":", alpha=0.5)
ax.set_xlabel("Layer")
ax.set_ylabel("Silhouette Score (k=2)")
ax.set_title("Two-Cluster Formation Across Layers (Last Token, Aggregated)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "silhouette_by_layer.png", dpi=150)
plt.close()
print("  Saved silhouette_by_layer.png")

# Figure 3: Switch rate by layer
fig, ax = plt.subplots(figsize=(8, 5))
layer_pairs = sorted(switch_by_pair.keys())
pair_labels = [f"{l1}→{l2}" for l1, l2 in layer_pairs]
pair_means = [np.mean(switch_by_pair[k]) for k in layer_pairs]
pair_stds = [np.std(switch_by_pair[k]) for k in layer_pairs]
ax.bar(range(len(layer_pairs)), pair_means, yerr=pair_stds, capsize=5, color="steelblue")
ax.set_xticks(range(len(layer_pairs)))
ax.set_xticklabels(pair_labels, rotation=45)
ax.set_ylabel("Switch Rate")
ax.set_title("Fraction of Tokens Switching Clusters Between Layers")
plt.tight_layout()
plt.savefig(FIG_DIR / "switch_rate_by_layer.png", dpi=150)
plt.close()
print("  Saved switch_rate_by_layer.png")

# Figure 4: UMAP of cluster structure for one easy and one hard problem
print("  Computing UMAP for cluster visualization...")
import umap

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Find one easy (level 1) and one hard (level 5) problem with good silhouette
for ax, target_level, title in [
    (axes[0], 1, "Easy (Level 1)"),
    (axes[1], 5, "Hard (Level 5)"),
]:
    best_idx = None
    best_sil = -1
    for idx in range(500):
        if hidden_levels[idx] == target_level and all_cluster_results[idx] is not None:
            if all_cluster_results[idx]["silhouette"] > best_sil:
                best_sil = all_cluster_results[idx]["silhouette"]
                best_idx = idx

    if best_idx is not None:
        cr = all_cluster_results[best_idx]
        traj = cr["raw_traj"]
        labels = cr["labels"]
        n_comp = min(30, traj.shape[0] - 1, traj.shape[1])
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, traj.shape[0] - 1),
                            random_state=42)
        embedding = reducer.fit_transform(PCA(n_components=n_comp).fit_transform(traj))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="coolwarm",
                             s=20, alpha=0.7)
        ax.set_title(f"{title}\nSilhouette={best_sil:.3f}, n={len(labels)} tokens")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

plt.suptitle("Token Hidden States Colored by Cluster (k=2)", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "cluster_umap.png", dpi=150)
plt.close()
print("  Saved cluster_umap.png")

# ── Determine switching token dominant type ───────────────────────────────
# Use a heuristic: tokens at the boundary (low silhouette) are "switching" types
switching_types = {t: 0 for t in type_names}
non_switching_types = {t: 0 for t in type_names}
for prob_idx in subset_indices:
    cr = all_cluster_results[prob_idx]
    ds_idx = problem_indices_in_ds[prob_idx]
    if cr is None or ds_idx is None:
        continue
    problem_text = math_ds[ds_idx]["problem"]
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=512)
    n = min(len(cr["per_token_silhouette"]), len(input_ids))
    for t_idx in range(n):
        tok_type = classify_token(input_ids[t_idx], tokenizer)
        if abs(cr["per_token_silhouette"][t_idx]) < 0.1:
            switching_types[tok_type] += 1
        else:
            non_switching_types[tok_type] += 1

switching_dominant = max(switching_types, key=switching_types.get) if any(switching_types.values()) else "unknown"

# ── Compute mean switch rate across all layer pairs ───────────────────────
all_switch_rates = [sr["switch_rate"] for sr in switch_rates]
mean_switch_rate = float(np.mean(all_switch_rates)) if all_switch_rates else 0.0

# ── Save results ──────────────────────────────────────────────────────────
print("\n=== Saving results ===")

# Build interpretation
if cramers_v > 0.2:
    type_interp = f"Strong token-type separation (V={cramers_v:.3f}): clusters likely represent {dominant_a_type} vs {dominant_b_type} content."
elif cramers_v > 0.1:
    type_interp = f"Moderate token-type association (V={cramers_v:.3f}): partial separation by content type."
else:
    type_interp = f"Weak token-type association (V={cramers_v:.3f}): clusters not primarily driven by token type."

if abs(correct_mean_sil - incorrect_mean_sil) > 0.02 and sil_p < 0.05:
    corr_interp = f" Correct problems show {'higher' if correct_mean_sil > incorrect_mean_sil else 'lower'} silhouette (p={sil_p:.1e})."
else:
    corr_interp = " No significant correctness effect on cluster structure."

if two_cluster_math_specific:
    math_interp = f" Two-cluster structure is math-specific (math sil={math_mean_sil:.3f} vs wiki={wiki_mean_sil:.3f})."
else:
    math_interp = f" Two-cluster structure is NOT math-specific (math sil={math_mean_sil:.3f} vs wiki={wiki_mean_sil:.3f}) — appears universal."

interpretation = type_interp + corr_interp + math_interp

results = {
    "contingency_table": {t: contingency[t] for t in type_names},
    "chi_squared": float(chi2),
    "chi_squared_p": float(chi2_p),
    "cramers_v": float(cramers_v),
    "dominant_cluster_a_type": dominant_a_type,
    "dominant_cluster_b_type": dominant_b_type,
    "correct_mean_silhouette": float(correct_mean_sil),
    "incorrect_mean_silhouette": float(incorrect_mean_sil),
    "correctness_silhouette_p": float(sil_p),
    "correct_cluster_a_fraction": float(correct_a_frac),
    "incorrect_cluster_a_fraction": float(incorrect_a_frac),
    "cluster_formation_layer_mean": float(formation_layer) if formation_layer is not None else -1.0,
    "per_layer_silhouette": [float(s) for s in per_layer_sil],
    "per_layer_per_level_silhouette": {str(lv): [float(s) for s in per_layer_per_level_sil[lv]] for lv in range(1, 6)},
    "per_token_layer_silhouette": {str(l): float(np.mean(per_layer_token_sil[l])) for l in TARGET_LAYERS if per_layer_token_sil[l]},
    "token_switch_rate_by_layer": [float(np.mean(switch_by_pair[k])) for k in sorted(switch_by_pair.keys())],
    "switching_tokens_dominant_type": switching_dominant,
    "wiki_mean_silhouette": float(wiki_mean_sil),
    "math_mean_silhouette": float(math_mean_sil),
    "two_cluster_math_specific": bool(two_cluster_math_specific),
    "mean_silhouette_all": float(np.mean(sil_scores)),
    "std_silhouette_all": float(np.std(sil_scores)),
    "n_valid_problems": valid_count,
    "n_correctness_matched": len(aligned_h_idx),
    "interpretation": interpretation,
}

with open(DATA_DIR / "exp1_semantic_cartography.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved exp1_semantic_cartography.json")

# Also save cluster results for later experiments (lightweight version)
cluster_save = []
for idx, cr in enumerate(all_cluster_results):
    if cr is None:
        cluster_save.append(None)
    else:
        cluster_save.append({
            "labels": cr["labels"].tolist(),
            "silhouette": cr["silhouette"],
            "per_token_silhouette": cr["per_token_silhouette"].tolist(),
            "centroid_distance": cr["centroid_distance"],
            "cluster_sizes": cr["cluster_sizes"],
        })

np.savez_compressed(
    DATA_DIR / "exp1_cluster_results.npz",
    cluster_results=np.array(cluster_save, dtype=object),
    aligned_h_idx=aligned_h_idx,
    aligned_c_idx=aligned_c_idx,
    correct=correct,
    problem_indices_in_ds=np.array(problem_indices_in_ds),
    allow_pickle=True,
)
print("  Saved exp1_cluster_results.npz")

print("\n=== Experiment 1 complete ===")
print(f"  Interpretation: {interpretation}")
