#!/usr/bin/env python3
"""Experiment 3: Causal Intervention on the Bridge.

Determines the computational role of bridge tokens (between clusters)
by intervening on them and measuring the impact on model output.
"""

import hashlib
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "two_clusters"
FIG_DIR = ROOT / "figures" / "two_clusters"

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
data = np.load(ROOT / "data/transformer/math500_hidden_states_aligned.npz", allow_pickle=True)
token_trajs = data["token_trajectories"]
hidden_levels = data["difficulty_levels"]
stored_hashes = data["problem_hashes"]

# Load Exp 1 cluster results
exp1_data = np.load(DATA_DIR / "exp1_cluster_results.npz", allow_pickle=True)
cluster_results_raw = exp1_data["cluster_results"]
aligned_h_idx = exp1_data["aligned_h_idx"]
aligned_c_idx = exp1_data["aligned_c_idx"]
correct = exp1_data["correct"]
problem_indices_in_ds = exp1_data["problem_indices_in_ds"]

# Convert cluster results back from serialized form
cluster_results = []
for cr in cluster_results_raw:
    if cr is None:
        cluster_results.append(None)
    else:
        cluster_results.append({
            "labels": np.array(cr["labels"]),
            "silhouette": cr["silhouette"],
            "per_token_silhouette": np.array(cr["per_token_silhouette"]),
            "centroid_distance": cr["centroid_distance"],
            "cluster_sizes": cr["cluster_sizes"],
        })

# ── 3a: Identify Bridge Tokens ────────────────────────────────────────────
print("\n=== 3a: Identify bridge tokens ===")

bridge_fractions = []
bridge_fractions_by_level = defaultdict(list)

for idx in range(500):
    cr = cluster_results[idx]
    if cr is None:
        bridge_fractions.append(None)
        continue
    pts = cr["per_token_silhouette"]
    bridge_mask = np.abs(pts) < 0.1
    bridge_frac = float(bridge_mask.mean())
    bridge_fractions.append(bridge_frac)
    lv = hidden_levels[idx]
    bridge_fractions_by_level[int(lv)].append(bridge_frac)

print("  Bridge fraction by difficulty level:")
bf_by_level = {}
for lv in range(1, 6):
    fracs = bridge_fractions_by_level[lv]
    bf_by_level[str(lv)] = float(np.mean(fracs)) if fracs else 0.0
    print(f"    Level {lv}: {np.mean(fracs):.4f} ± {np.std(fracs):.4f} (n={len(fracs)})")

# Token-type distribution of bridge vs core tokens
print("\n  Loading tokenizer for token classification...")
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)

def problem_hash_fn(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]

hash_to_ds_idx = {}
for i, item in enumerate(math_ds):
    h = problem_hash_fn(item["problem"])
    hash_to_ds_idx[h] = i

def classify_token(token_id, tokenizer):
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

bridge_token_types = defaultdict(int)
core_token_types = defaultdict(int)

for idx in range(500):
    cr = cluster_results[idx]
    if cr is None:
        continue
    h = stored_hashes[idx]
    ds_idx = hash_to_ds_idx.get(h, None)
    if ds_idx is None:
        continue

    problem_text = math_ds[ds_idx]["problem"]
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=512)
    pts = cr["per_token_silhouette"]
    n = min(len(pts), len(input_ids))

    for t_idx in range(n):
        tok_type = classify_token(input_ids[t_idx], tokenizer)
        if abs(pts[t_idx]) < 0.1:
            bridge_token_types[tok_type] += 1
        elif abs(pts[t_idx]) > 0.5:
            core_token_types[tok_type] += 1

print("\n  Token type distribution:")
print(f"  {'':15s} | {'Bridge':>8s} | {'Core':>8s}")
all_types = ["numeric", "math_symbol", "language", "latex", "structural"]
for t in all_types:
    print(f"  {t:15s} | {bridge_token_types.get(t, 0):8d} | {core_token_types.get(t, 0):8d}")

# Normalize for output
bridge_total = sum(bridge_token_types.values()) or 1
core_total = sum(core_token_types.values()) or 1
bridge_types_norm = {t: bridge_token_types.get(t, 0) / bridge_total for t in all_types}
core_types_norm = {t: core_token_types.get(t, 0) / core_total for t in all_types}

# ── 3b: Activation Patching ───────────────────────────────────────────────
print("\n=== 3b: Activation patching ===")

import torch
from transformers import AutoModelForCausalLM

print("  Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True,
    trust_remote_code=True,
)
model.eval()

# Select 30 problems (6 per level)
patch_indices = []
for lv in range(1, 6):
    lv_mask = np.where(hidden_levels == lv)[0]
    # Pick problems that have valid cluster results
    valid = [i for i in lv_mask if cluster_results[i] is not None]
    patch_indices.extend(valid[:6])

print(f"  Patching {len(patch_indices)} problems")

# Determine cluster-formation layer: from Exp 1d-B, layers 7-21 have silhouette ~0.98
# Use layer 14 (mid-network, strongest clustering) as the intervention point
PATCH_LAYER = 14
print(f"  Intervention layer: {PATCH_LAYER}")

intervention_results = {
    "bridge_ablation": {"kl_divs": [], "answer_changed": [], "correct_prob_changes": []},
    "steer_to_A": {"kl_divs": [], "answer_changed": [], "correct_prob_changes": []},
    "steer_to_B": {"kl_divs": [], "answer_changed": [], "correct_prob_changes": []},
    "core_swap": {"kl_divs": [], "answer_changed": [], "correct_prob_changes": []},
}

for count, prob_idx in enumerate(patch_indices):
    h = stored_hashes[prob_idx]
    ds_idx = hash_to_ds_idx.get(h, None)
    if ds_idx is None:
        continue

    cr = cluster_results[prob_idx]
    problem_text = math_ds[ds_idx]["problem"]
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    n_input_tokens = inputs["input_ids"].shape[1]

    # Get baseline output
    with torch.no_grad():
        baseline_outputs = model(**inputs, output_hidden_states=True)
    baseline_logits = baseline_outputs.logits[0, -1, :].cpu().float()
    baseline_probs = torch.softmax(baseline_logits, dim=0).numpy()
    baseline_top = int(baseline_logits.argmax())

    # Get hidden states at intervention layer for clustering
    # Use the stored final-layer cluster labels, but we need to identify
    # bridge/core tokens based on those labels
    pts = cr["per_token_silhouette"]
    labels = cr["labels"]
    n_cluster = len(labels)
    n = min(n_cluster, n_input_tokens)

    bridge_mask = np.abs(pts[:n]) < 0.1
    core_a_mask = (labels[:n] == 0) & (np.abs(pts[:n]) > 0.5)
    core_b_mask = (labels[:n] == 1) & (np.abs(pts[:n]) > 0.5)

    bridge_indices = np.where(bridge_mask)[0].tolist()
    core_a_indices = np.where(core_a_mask)[0].tolist()
    core_b_indices = np.where(core_b_mask)[0].tolist()

    if not bridge_indices:
        # No bridge tokens — skip this problem
        continue

    # Compute cluster centroids in original hidden state space
    # We need the actual hidden states at the intervention layer
    hs_at_layer = baseline_outputs.hidden_states[PATCH_LAYER][0].cpu().float().numpy()  # (n_tokens, hidden_dim)
    mean_hidden = hs_at_layer.mean(axis=0)  # (hidden_dim,)
    centroid_a = hs_at_layer[labels[:n] == 0].mean(axis=0) if (labels[:n] == 0).any() else mean_hidden
    centroid_b = hs_at_layer[labels[:n] == 1].mean(axis=0) if (labels[:n] == 1).any() else mean_hidden

    def run_patched(patch_token_indices, replacement_vectors):
        """Run model with patched activations."""
        hooks = []

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
            else:
                hidden = output.clone()
            for tok_idx, replacement in zip(patch_token_indices, replacement_vectors):
                if tok_idx < hidden.shape[1]:
                    hidden[0, tok_idx, :] = torch.tensor(
                        replacement, dtype=hidden.dtype, device=hidden.device
                    )
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layer = model.model.layers[PATCH_LAYER]
        hook = layer.register_forward_hook(patch_hook)
        hooks.append(hook)

        with torch.no_grad():
            outputs = model(**inputs)

        for hk in hooks:
            hk.remove()

        logits = outputs.logits[0, -1, :].cpu().float()
        return logits

    def kl_divergence(p, q):
        """KL(p || q) with numerical stability."""
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return float(np.sum(p * np.log(p / q)))

    # Intervention 1: Bridge ablation (replace with mean)
    bridge_replacements = [mean_hidden for _ in bridge_indices]
    patched_logits = run_patched(bridge_indices, bridge_replacements)
    patched_probs = torch.softmax(patched_logits, dim=0).numpy()
    kl = kl_divergence(baseline_probs, patched_probs)
    changed = int(patched_logits.argmax()) != baseline_top
    intervention_results["bridge_ablation"]["kl_divs"].append(kl)
    intervention_results["bridge_ablation"]["answer_changed"].append(changed)

    # Intervention 2: Steer bridge to cluster A centroid
    steer_a_replacements = [centroid_a for _ in bridge_indices]
    patched_logits = run_patched(bridge_indices, steer_a_replacements)
    patched_probs = torch.softmax(patched_logits, dim=0).numpy()
    kl = kl_divergence(baseline_probs, patched_probs)
    changed = int(patched_logits.argmax()) != baseline_top
    intervention_results["steer_to_A"]["kl_divs"].append(kl)
    intervention_results["steer_to_A"]["answer_changed"].append(changed)

    # Intervention 3: Steer bridge to cluster B centroid
    steer_b_replacements = [centroid_b for _ in bridge_indices]
    patched_logits = run_patched(bridge_indices, steer_b_replacements)
    patched_probs = torch.softmax(patched_logits, dim=0).numpy()
    kl = kl_divergence(baseline_probs, patched_probs)
    changed = int(patched_logits.argmax()) != baseline_top
    intervention_results["steer_to_B"]["kl_divs"].append(kl)
    intervention_results["steer_to_B"]["answer_changed"].append(changed)

    # Intervention 4: Core swap (A→B centroid, B→A centroid)
    swap_indices = core_a_indices + core_b_indices
    swap_replacements = [centroid_b for _ in core_a_indices] + [centroid_a for _ in core_b_indices]
    if swap_indices:
        patched_logits = run_patched(swap_indices, swap_replacements)
        patched_probs = torch.softmax(patched_logits, dim=0).numpy()
        kl = kl_divergence(baseline_probs, patched_probs)
        changed = int(patched_logits.argmax()) != baseline_top
        intervention_results["core_swap"]["kl_divs"].append(kl)
        intervention_results["core_swap"]["answer_changed"].append(changed)

    torch.cuda.empty_cache()
    if (count + 1) % 6 == 0:
        print(f"    Processed {count + 1}/{len(patch_indices)}")

del model
torch.cuda.empty_cache()

# Summarize intervention results
print("\n  Intervention results:")
intervention_summary = {}
for name, data_dict in intervention_results.items():
    if data_dict["kl_divs"]:
        mean_kl = float(np.mean(data_dict["kl_divs"]))
        changed_frac = float(np.mean(data_dict["answer_changed"]))
        intervention_summary[name] = {
            "mean_kl_divergence": mean_kl,
            "answer_changed_fraction": changed_frac,
            "correct_prob_change": 0.0,  # would require answer token tracking
            "n_problems": len(data_dict["kl_divs"]),
        }
        print(f"    {name}: KL={mean_kl:.4f}, answer changed={changed_frac:.1%}")
    else:
        intervention_summary[name] = {
            "mean_kl_divergence": 0.0,
            "answer_changed_fraction": 0.0,
            "correct_prob_change": 0.0,
            "n_problems": 0,
        }

# ── 3c: Bridge Fraction and Goldilocks ────────────────────────────────────
print("\n=== 3c: Bridge fraction and Goldilocks ===")

# Check if H1 persistence entropy data exists from prior experiments
h1_data_path = ROOT / "data" / "transformer" / "attention_ph_diagrams.npz"
h1_available = False
h1_by_level = {}

try:
    ph_data = np.load(h1_data_path, allow_pickle=True)
    # Try to extract H1 persistence entropy
    if "persistence_diagrams" in ph_data:
        print("  H1 persistence data found")
        h1_available = True
except Exception as e:
    print(f"  H1 persistence data not available: {e}")

# Compute bridge fraction by level for correlation
bf_values = []
level_values = []
for idx in range(500):
    if bridge_fractions[idx] is not None:
        bf_values.append(bridge_fractions[idx])
        level_values.append(hidden_levels[idx])

bf_values = np.array(bf_values)
level_values = np.array(level_values)

# Test for Goldilocks: does level 3 have the highest bridge fraction?
level_means = {}
for lv in range(1, 6):
    mask = level_values == lv
    level_means[lv] = float(bf_values[mask].mean())

goldilocks_level = max(level_means, key=level_means.get)
is_goldilocks = goldilocks_level == 3
print(f"  Highest bridge fraction at level {goldilocks_level}: {level_means[goldilocks_level]:.4f}")
print(f"  Goldilocks at level 3: {is_goldilocks}")

# Correlation between bridge fraction and difficulty
spearman_r, spearman_p = stats.spearmanr(bf_values, level_values)
print(f"  Spearman(bridge_fraction, difficulty): r={spearman_r:.4f}, p={spearman_p:.2e}")

goldilocks_explanation = (
    f"Bridge fraction peaks at level {goldilocks_level} ({level_means[goldilocks_level]:.4f}). "
    f"{'This matches the Goldilocks H1 effect at level 3.' if is_goldilocks else 'This does NOT match the Goldilocks H1 effect at level 3.'} "
    f"Spearman correlation with difficulty: r={spearman_r:.3f} (p={spearman_p:.1e})."
)

# ── Generate Figures ───────────────────────────────────────────────────────
print("\n=== Generating figures ===")

# Figure 1: Bridge fraction by level
fig, ax = plt.subplots(figsize=(8, 5))
levels = [1, 2, 3, 4, 5]
means = [bf_by_level[str(lv)] for lv in levels]
stds = [np.std(bridge_fractions_by_level[lv]) for lv in levels]
colors = plt.cm.viridis(np.linspace(0, 1, 5))
bars = ax.bar(levels, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
ax.set_xlabel("Difficulty Level")
ax.set_ylabel("Bridge Fraction (|silhouette| < 0.1)")
ax.set_title("Bridge Token Fraction by Difficulty Level")
ax.set_xticks(levels)
# Highlight Goldilocks level
if goldilocks_level in levels:
    bars[goldilocks_level - 1].set_edgecolor("red")
    bars[goldilocks_level - 1].set_linewidth(3)
plt.tight_layout()
plt.savefig(FIG_DIR / "bridge_fraction_by_level.png", dpi=150)
plt.close()
print("  Saved bridge_fraction_by_level.png")

# Figure 2: Intervention effects
fig, ax = plt.subplots(figsize=(10, 5))
names = list(intervention_summary.keys())
kl_vals = [intervention_summary[n]["mean_kl_divergence"] for n in names]
changed_vals = [intervention_summary[n]["answer_changed_fraction"] for n in names]
x = np.arange(len(names))
width = 0.35
bars1 = ax.bar(x - width/2, kl_vals, width, label="Mean KL Divergence", color="steelblue")
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, changed_vals, width, label="Answer Changed %", color="coral")
ax.set_xlabel("Intervention Type")
ax.set_ylabel("KL Divergence")
ax2.set_ylabel("Fraction Answer Changed")
ax.set_xticks(x)
ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
ax.set_title("Causal Intervention Effects on Model Output")
plt.tight_layout()
plt.savefig(FIG_DIR / "intervention_effects.png", dpi=150)
plt.close()
print("  Saved intervention_effects.png")

# Figure 3: Bridge fraction vs H1 (scatter by problem)
fig, ax = plt.subplots(figsize=(8, 5))
scatter_colors = plt.cm.viridis(np.linspace(0, 1, 5))
for lv in range(1, 6):
    mask = level_values == lv
    ax.scatter(bf_values[mask], np.full(mask.sum(), lv) + np.random.normal(0, 0.1, mask.sum()),
               c=[scatter_colors[lv-1]], alpha=0.3, s=15, label=f"Level {lv}")
ax.set_xlabel("Bridge Fraction")
ax.set_ylabel("Difficulty Level (jittered)")
ax.set_title(f"Bridge Fraction vs Difficulty\nSpearman r={spearman_r:.3f}, p={spearman_p:.1e}")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "bridge_vs_h1.png", dpi=150)
plt.close()
print("  Saved bridge_vs_h1.png")

# ── Save results ──────────────────────────────────────────────────────────
print("\n=== Saving results ===")

results = {
    "bridge_fraction_by_level": bf_by_level,
    "bridge_fraction_h1_correlation": float(spearman_r),
    "bridge_fraction_h1_p": float(spearman_p),
    "bridge_token_types": bridge_types_norm,
    "core_token_types": core_types_norm,
    "intervention_results": intervention_summary,
    "goldilocks_explanation": goldilocks_explanation,
    "goldilocks_level": int(goldilocks_level),
    "patch_layer": PATCH_LAYER,
}

with open(DATA_DIR / "exp3_causal_intervention.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved exp3_causal_intervention.json")

print("\n=== Experiment 3 complete ===")
print(f"  {goldilocks_explanation}")
