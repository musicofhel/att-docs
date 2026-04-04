#!/usr/bin/env python3
"""Experiment 2: The 1D Reaction Coordinate.

Identifies what single variable parameterizes the 1D flow through the residual
stream and tests whether it represents computational progress from
'problem encoded' to 'answer formed.'
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
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "two_clusters"
FIG_DIR = ROOT / "figures" / "two_clusters"

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
data = np.load(ROOT / "data/transformer/math500_hidden_states_aligned.npz", allow_pickle=True)
layer_hidden_states = data["layer_hidden_states"]  # (500, 29, 1536)
hidden_levels = data["difficulty_levels"]
stored_hashes = data["problem_hashes"]

# ── 2a: Extract the Principal Flow Axis ────────────────────────────────────
print("\n=== 2a: Principal flow axis extraction ===")

all_displacements = []
for idx in range(len(layer_hidden_states)):
    states = layer_hidden_states[idx]  # (29, 1536)
    if states.ndim == 2 and states.shape[0] >= 2:
        deltas = np.diff(states, axis=0)  # (28, 1536)
        all_displacements.append(deltas)

all_displacements = np.vstack(all_displacements)  # (500*28, 1536)
print(f"  Total displacement vectors: {all_displacements.shape}")

pca_flow = PCA(n_components=10)
pca_flow.fit(all_displacements)
flow_axis = pca_flow.components_[0]  # the 1D flow direction
var_ratios = pca_flow.explained_variance_ratio_

print(f"  Flow axis variance explained: {var_ratios[0]:.4f}")
print(f"  Top-5 variance explained: {var_ratios[:5]}")
print(f"  Cumulative top-5: {var_ratios[:5].sum():.4f}")

# ── 2b: Logit Lens Along the Flow ─────────────────────────────────────────
print("\n=== 2b: Logit lens along the flow ===")

import hashlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
print("  Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True,
    trust_remote_code=True,
)
model.eval()

# Extract norm weights and unembed matrix for logit lens
final_norm_weight = model.model.norm.weight.float().detach().cpu().numpy()
unembed = model.lm_head.weight.float().detach().cpu().numpy()  # (vocab_size, hidden_dim)
print(f"  Unembed matrix shape: {unembed.shape}")

def rms_norm(x, weight, eps=1e-6):
    """Apply RMSNorm."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight

def logit_lens(hidden_state, final_norm_weight, unembed):
    """Project hidden state to logits via norm + unembedding."""
    normed = rms_norm(hidden_state, final_norm_weight)
    logits = normed @ unembed.T
    # Numerically stable softmax
    logits_shifted = logits - logits.max()
    probs = np.exp(logits_shifted)
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-15))
    top5_idx = np.argsort(logits)[-5:][::-1]
    top_token = int(top5_idx[0])
    return {
        "entropy": float(entropy),
        "top_token": top_token,
        "top_prob": float(probs[top_token]),
        "top5_tokens": top5_idx.tolist(),
    }

# Load MATH-500 for ground truth answers
math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)

def problem_hash(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]

hash_to_ds_idx = {}
for i, item in enumerate(math_ds):
    h = problem_hash(item["problem"])
    hash_to_ds_idx[h] = i

# Select 50 problems (10 per level)
subset_indices = []
for lv in range(1, 6):
    lv_mask = np.where(hidden_levels == lv)[0]
    subset_indices.extend(lv_mask[:10].tolist())

print(f"  Running logit lens on {len(subset_indices)} problems...")

# Verify pre-norm state
sample_states = layer_hidden_states[0]  # (29, 1536) for problem 0
print(f"  Max abs at layer 14: {np.max(np.abs(sample_states[14])):.1f} (>50 means pre-norm, expected)")

all_entropies = []  # shape: (n_problems, 29)
all_convergence_layers = []  # layer where correct answer first in top-5
flow_projections = []  # projection onto flow axis at each layer
flow_entropies = []  # corresponding entropy

for count, prob_idx in enumerate(subset_indices):
    h = stored_hashes[prob_idx]
    ds_idx = hash_to_ds_idx.get(h, None)
    if ds_idx is None:
        continue

    problem_text = math_ds[ds_idx]["problem"]
    answer_text = math_ds[ds_idx].get("answer", "")
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)

    # Run model forward pass to get fresh hidden states at all layers
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get last-token hidden states at each layer
    n_layers = len(outputs.hidden_states)
    entropies = []
    convergence_layer = None

    # Tokenize answer to find target tokens
    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False) if answer_text else []

    for l_idx in range(n_layers):
        hs = outputs.hidden_states[l_idx][0, -1, :].cpu().float().numpy()  # last token
        result = logit_lens(hs, final_norm_weight, unembed)
        entropies.append(result["entropy"])

        # Check if answer token appears in top-5
        if convergence_layer is None and answer_tokens:
            if answer_tokens[0] in result["top5_tokens"]:
                convergence_layer = l_idx

        # Track flow projection vs entropy
        proj = float(np.dot(hs, flow_axis))
        flow_projections.append(proj)
        flow_entropies.append(result["entropy"])

    all_entropies.append(entropies)
    all_convergence_layers.append(convergence_layer if convergence_layer is not None else n_layers)
    torch.cuda.empty_cache()

    if (count + 1) % 10 == 0:
        print(f"    Processed {count + 1}/{len(subset_indices)}")

del model
torch.cuda.empty_cache()

all_entropies = np.array(all_entropies)  # (n_problems, n_layers)
print(f"\n  Entropy matrix shape: {all_entropies.shape}")
mean_convergence = np.mean(all_convergence_layers)
print(f"  Mean convergence layer (answer in top-5): {mean_convergence:.1f}")

# Check monotonic decrease
monotonic_count = 0
for i in range(len(all_entropies)):
    diffs = np.diff(all_entropies[i])
    if np.all(diffs <= 0):
        monotonic_count += 1
monotonic_fraction = monotonic_count / len(all_entropies)
print(f"  Fraction with monotonically decreasing entropy: {monotonic_fraction:.3f}")

# Flow projection vs entropy correlation
flow_entropy_corr, flow_entropy_p = stats.pearsonr(flow_projections, flow_entropies)
print(f"  Flow projection-entropy correlation: r={flow_entropy_corr:.4f}, p={flow_entropy_p:.2e}")

# ── 2c: Flow Coordinate vs Difficulty ──────────────────────────────────────
print("\n=== 2c: Flow coordinate vs difficulty ===")

displacement_by_level = {}
start_pos_by_level = {}
end_pos_by_level = {}
speed_profiles_by_level = {}

for lv in range(1, 6):
    lv_mask = hidden_levels == lv
    lv_states = layer_hidden_states[lv_mask]  # (n_lv, 29, 1536)

    starts = []
    ends = []
    displacements = []
    speeds = []

    for idx in range(lv_states.shape[0]):
        trajectory = lv_states[idx]  # (29, 1536)
        projections = trajectory @ flow_axis  # (29,)
        starts.append(projections[0])
        ends.append(projections[-1])
        displacements.append(projections[-1] - projections[0])

        # Speed profile: |delta projection| per layer
        delta_proj = np.abs(np.diff(projections))  # (28,)
        speeds.append(delta_proj)

    displacement_by_level[str(lv)] = float(np.mean(displacements))
    start_pos_by_level[str(lv)] = float(np.mean(starts))
    end_pos_by_level[str(lv)] = float(np.mean(ends))
    speed_profiles_by_level[str(lv)] = np.mean(speeds, axis=0).tolist()

    print(f"  Level {lv}: start={np.mean(starts):.2f}, end={np.mean(ends):.2f}, "
          f"displacement={np.mean(displacements):.2f}")

# Per-level entropy profiles from logit lens
entropy_by_level = {}
for i, prob_idx in enumerate(subset_indices):
    if i >= len(all_entropies):
        break
    lv = hidden_levels[prob_idx]
    entropy_by_level.setdefault(int(lv), []).append(all_entropies[i])

# ── Generate Figures ───────────────────────────────────────────────────────
print("\n=== Generating figures ===")

# Figure 1: Flow axis displacement by difficulty
fig, ax = plt.subplots(figsize=(8, 5))
levels = [1, 2, 3, 4, 5]
disps = [displacement_by_level[str(lv)] for lv in levels]
colors = plt.cm.viridis(np.linspace(0, 1, 5))
bars = ax.bar(levels, disps, color=colors, edgecolor="black")
ax.set_xlabel("Difficulty Level")
ax.set_ylabel("Total Displacement Along Flow Axis")
ax.set_title(f"Flow Axis Displacement by Difficulty\n(PC1 explains {var_ratios[0]:.1%} of displacement variance)")
ax.set_xticks(levels)
for bar, d in zip(bars, disps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{d:.1f}",
            ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "flow_axis_displacement.png", dpi=150)
plt.close()
print("  Saved flow_axis_displacement.png")

# Figure 2: Logit lens entropy by layer, colored by difficulty
fig, ax = plt.subplots(figsize=(10, 6))
for lv in range(1, 6):
    if lv in entropy_by_level and entropy_by_level[lv]:
        mean_ent = np.mean(entropy_by_level[lv], axis=0)
        std_ent = np.std(entropy_by_level[lv], axis=0)
        layers = np.arange(len(mean_ent))
        ax.plot(layers, mean_ent, "o-", color=colors[lv-1], label=f"Level {lv}", linewidth=2)
        ax.fill_between(layers, mean_ent - std_ent, mean_ent + std_ent,
                        color=colors[lv-1], alpha=0.15)
ax.set_xlabel("Layer")
ax.set_ylabel("Logit Lens Entropy (nats)")
ax.set_title("Predicted Distribution Entropy Across Layers")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "entropy_by_layer.png", dpi=150)
plt.close()
print("  Saved entropy_by_layer.png")

# Figure 3: Speed profiles by difficulty
fig, ax = plt.subplots(figsize=(10, 6))
for lv in range(1, 6):
    speeds = speed_profiles_by_level[str(lv)]
    ax.plot(range(len(speeds)), speeds, "o-", color=colors[lv-1],
            label=f"Level {lv}", linewidth=2, alpha=0.8)
ax.set_xlabel("Layer Transition (l → l+1)")
ax.set_ylabel("|Displacement Along Flow Axis|")
ax.set_title("Speed Profile: Rate of Progress Along 1D Flow")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "speed_profiles.png", dpi=150)
plt.close()
print("  Saved speed_profiles.png")

# ── Save results ──────────────────────────────────────────────────────────
print("\n=== Saving results ===")

results = {
    "flow_axis_variance_explained": float(var_ratios[0]),
    "top5_variance_explained": var_ratios[:5].tolist(),
    "top10_variance_explained": var_ratios[:10].tolist(),
    "entropy_decrease_monotonic_fraction": float(monotonic_fraction),
    "mean_convergence_layer": float(mean_convergence),
    "flow_entropy_correlation": float(flow_entropy_corr),
    "flow_entropy_correlation_p": float(flow_entropy_p),
    "displacement_by_level": displacement_by_level,
    "start_position_by_level": start_pos_by_level,
    "end_position_by_level": end_pos_by_level,
    "speed_profile_by_level": speed_profiles_by_level,
    "mean_entropy_by_layer": np.mean(all_entropies, axis=0).tolist(),
}

with open(DATA_DIR / "exp2_reaction_coordinate.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved exp2_reaction_coordinate.json")

# Save flow axis for later experiments
np.savez_compressed(
    DATA_DIR / "exp2_flow_axis.npz",
    flow_axis=flow_axis,
    pca_components=pca_flow.components_,
    pca_explained_variance_ratio=pca_flow.explained_variance_ratio_,
)
print("  Saved exp2_flow_axis.npz")

print("\n=== Experiment 2 complete ===")
print(f"  Flow axis explains {var_ratios[0]:.1%} of displacement variance")
print(f"  Flow-entropy correlation: r={flow_entropy_corr:.3f}")
print(f"  Entropy monotonically decreasing: {monotonic_fraction:.1%} of problems")
