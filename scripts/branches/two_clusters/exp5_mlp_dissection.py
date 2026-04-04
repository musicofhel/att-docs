#!/usr/bin/env python3
"""Experiment 5: MLP Dissection — Which Neurons Create the Clusters?

Identifies the specific MLP layers and neurons responsible for the
two-cluster structure through ablation and attribution analysis.
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
from scipy.special import expit  # sigmoid
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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
problem_indices_in_ds = exp1_data["problem_indices_in_ds"]

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

# Load flow axis from Exp 2
flow_data = np.load(DATA_DIR / "exp2_flow_axis.npz")
flow_axis = flow_data["flow_axis"]

# Load model and dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)

print("Loading model...")
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

math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

def problem_hash_fn(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]

hash_to_ds_idx = {}
for i, item in enumerate(math_ds):
    h = problem_hash_fn(item["problem"])
    hash_to_ds_idx[h] = i

# Number of layers
N_LAYERS = model.config.num_hidden_layers  # 28 for Qwen2.5-1.5B
print(f"Model has {N_LAYERS} layers")

# ── 5a: Layer-by-Layer MLP Ablation ───────────────────────────────────────
print("\n=== 5a: Layer-by-layer MLP ablation ===")

# Select 30 problems (6 per level)
ablation_indices = []
for lv in range(1, 6):
    lv_mask = np.where(hidden_levels == lv)[0]
    valid = [i for i in lv_mask if cluster_results[i] is not None]
    ablation_indices.extend(valid[:6])

print(f"  Using {len(ablation_indices)} problems for ablation")

# First, compute baseline silhouette for each problem
baseline_sils = {}
print("  Computing baselines...")
for count, prob_idx in enumerate(ablation_indices):
    h = stored_hashes[prob_idx]
    ds_idx = hash_to_ds_idx.get(h, None)
    if ds_idx is None:
        continue

    problem_text = math_ds[ds_idx]["problem"]
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hs = outputs.hidden_states[-1][0].cpu().float().numpy()
    torch.cuda.empty_cache()

    if hs.shape[0] >= 10:
        n_comp = min(30, hs.shape[0] - 1, hs.shape[1])
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(hs)
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = km.fit_predict(reduced)
        sil = silhouette_score(reduced, labels) if len(set(labels)) > 1 else 0.0
        baseline_sils[prob_idx] = sil

print(f"  Baseline mean silhouette: {np.mean(list(baseline_sils.values())):.4f}")

# MLP ablation at each layer
print("  Running MLP ablation across layers...")
mlp_sil_drops = []  # per layer
attn_sil_drops = []  # per layer

for layer_idx in range(N_LAYERS):
    mlp_sils = []
    attn_sils = []

    for prob_idx in ablation_indices:
        if prob_idx not in baseline_sils:
            continue

        h = stored_hashes[prob_idx]
        ds_idx = hash_to_ds_idx.get(h, None)
        if ds_idx is None:
            continue

        problem_text = math_ds[ds_idx]["problem"]
        prompt = PROMPT_TEMPLATE.format(problem=problem_text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        # MLP ablation
        hooks = []
        def make_mlp_ablate_hook():
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    tensor = output[0]
                    mean_out = tensor.mean(dim=1, keepdim=True).expand_as(tensor)
                    return (mean_out,) + output[1:]
                else:
                    mean_out = output.mean(dim=1, keepdim=True).expand_as(output)
                    return mean_out
            return hook_fn

        layer = model.model.layers[layer_idx]
        hook = layer.mlp.register_forward_hook(make_mlp_ablate_hook())
        hooks.append(hook)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for hk in hooks:
            hk.remove()

        hs = outputs.hidden_states[-1][0].cpu().float().numpy()
        torch.cuda.empty_cache()

        if hs.shape[0] >= 10:
            n_comp = min(30, hs.shape[0] - 1, hs.shape[1])
            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(hs)
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            labels = km.fit_predict(reduced)
            sil = silhouette_score(reduced, labels) if len(set(labels)) > 1 else 0.0
            mlp_sils.append(baseline_sils[prob_idx] - sil)

        # Attention ablation
        hooks = []
        def make_attn_ablate_hook():
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    tensor = output[0]
                    mean_out = tensor.mean(dim=1, keepdim=True).expand_as(tensor)
                    return (mean_out,) + output[1:]
                else:
                    mean_out = output.mean(dim=1, keepdim=True).expand_as(output)
                    return mean_out
            return hook_fn

        hook = layer.self_attn.register_forward_hook(make_attn_ablate_hook())
        hooks.append(hook)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        for hk in hooks:
            hk.remove()

        hs = outputs.hidden_states[-1][0].cpu().float().numpy()
        torch.cuda.empty_cache()

        if hs.shape[0] >= 10:
            n_comp = min(30, hs.shape[0] - 1, hs.shape[1])
            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(hs)
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            labels = km.fit_predict(reduced)
            sil = silhouette_score(reduced, labels) if len(set(labels)) > 1 else 0.0
            attn_sils.append(baseline_sils[prob_idx] - sil)

    mlp_drop = float(np.mean(mlp_sils)) if mlp_sils else 0.0
    attn_drop = float(np.mean(attn_sils)) if attn_sils else 0.0
    mlp_sil_drops.append(mlp_drop)
    attn_sil_drops.append(attn_drop)

    if (layer_idx + 1) % 7 == 0 or layer_idx == N_LAYERS - 1:
        print(f"    Layer {layer_idx}: MLP drop={mlp_drop:.4f}, Attn drop={attn_drop:.4f}")

# Identify critical layers
mlp_drops_arr = np.array(mlp_sil_drops)
attn_drops_arr = np.array(attn_sil_drops)
critical_mlp_layers = np.argsort(mlp_drops_arr)[-3:][::-1].tolist()
critical_attn_layers = np.argsort(attn_drops_arr)[-3:][::-1].tolist()

print(f"\n  Top-3 critical MLP layers: {critical_mlp_layers}")
print(f"  Top-3 critical Attn layers: {critical_attn_layers}")

# ── 5b: Neuron-Level Attribution ──────────────────────────────────────────
print("\n=== 5b: Neuron-level attribution ===")

# Compute cluster axis for each problem (in original hidden-state space)
# Use all 500 problems
cluster_axes = []
cluster_axis_by_level = defaultdict(list)
for idx in range(500):
    cr = cluster_results[idx]
    if cr is None:
        cluster_axes.append(None)
        continue
    traj = token_trajs[idx]
    labels = cr["labels"]
    n = min(len(labels), traj.shape[0])
    mask_a = labels[:n] == 0
    mask_b = labels[:n] == 1
    if mask_a.any() and mask_b.any():
        centroid_a = traj[:n][mask_a].mean(axis=0)
        centroid_b = traj[:n][mask_b].mean(axis=0)
        axis = centroid_a - centroid_b
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 0:
            cluster_axes.append(axis / axis_norm)  # unit vector
            cluster_axis_by_level[int(hidden_levels[idx])].append(axis_norm)
        else:
            cluster_axes.append(None)
    else:
        cluster_axes.append(None)

valid_axes = [a for a in cluster_axes if a is not None]
print(f"  Valid cluster axes: {len(valid_axes)}/500")

# Compute mean cluster axis as the attribution direction
mean_cluster_axis = np.mean(valid_axes, axis=0)
mean_cluster_axis /= np.linalg.norm(mean_cluster_axis)

# Neuron attribution at the most critical MLP layer
critical_layer = critical_mlp_layers[0]
print(f"  Computing neuron attribution at layer {critical_layer}...")

# Select 20 problems for detailed attribution
attr_indices = []
for lv in range(1, 6):
    lv_mask = np.where(hidden_levels == lv)[0]
    valid = [i for i in lv_mask if cluster_results[i] is not None and hash_to_ds_idx.get(stored_hashes[i]) is not None]
    attr_indices.extend(valid[:4])

all_neuron_importances = []
all_neuron_alignments = []

for count, prob_idx in enumerate(attr_indices):
    h = stored_hashes[prob_idx]
    ds_idx = hash_to_ds_idx[h]
    problem_text = math_ds[ds_idx]["problem"]
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)

    # Hook to capture intermediate MLP activations
    activations = {}

    def capture_hook(name):
        def hook_fn(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            activations[name] = tensor.detach().cpu().float()
        return hook_fn

    layer = model.model.layers[critical_layer]
    hooks = []
    hooks.append(layer.mlp.gate_proj.register_forward_hook(capture_hook("gate")))
    hooks.append(layer.mlp.up_proj.register_forward_hook(capture_hook("up")))

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        model(**inputs)

    for hk in hooks:
        hk.remove()

    # MLP output: down_proj(silu(gate_proj(x)) * up_proj(x))
    gate_out = activations["gate"][0].numpy()  # (n_tokens, intermediate_dim)
    up_out = activations["up"][0].numpy()

    silu_output = gate_out * expit(gate_out)  # silu(gate_proj(x))
    intermediate = silu_output * up_out  # element-wise product

    # Per-neuron contribution to cluster axis
    down_weight = layer.mlp.down_proj.weight.float().detach().cpu().numpy()  # (hidden_dim, intermediate_dim)
    neuron_alignment = down_weight.T @ mean_cluster_axis  # (intermediate_dim,)
    mean_intermediate = np.mean(np.abs(intermediate), axis=0)  # (intermediate_dim,)
    neuron_importance = np.abs(neuron_alignment) * mean_intermediate

    all_neuron_importances.append(neuron_importance)
    all_neuron_alignments.append(neuron_alignment)

    torch.cuda.empty_cache()

# Aggregate across problems
mean_importance = np.mean(all_neuron_importances, axis=0)
mean_alignment = np.mean(all_neuron_alignments, axis=0)

# Top 50 neurons
top50_idx = np.argsort(mean_importance)[-50:][::-1]
top50_neurons = []
for ni in top50_idx:
    top50_neurons.append({
        "layer": int(critical_layer),
        "neuron": int(ni),
        "alignment": float(mean_alignment[ni]),
        "importance": float(mean_importance[ni]),
    })

print(f"  Top-10 neurons at layer {critical_layer}:")
for n in top50_neurons[:10]:
    print(f"    Neuron {n['neuron']}: importance={n['importance']:.4f}, alignment={n['alignment']:.4f}")

# Also check top neurons at the other critical layers
for other_layer in critical_mlp_layers[1:3]:
    print(f"\n  Attribution at layer {other_layer}...")
    layer = model.model.layers[other_layer]
    down_weight = layer.mlp.down_proj.weight.float().detach().cpu().numpy()
    neuron_alignment_other = down_weight.T @ mean_cluster_axis

    # Run one problem to get activation magnitudes
    prob_idx = attr_indices[0]
    h = stored_hashes[prob_idx]
    ds_idx = hash_to_ds_idx[h]
    prompt = PROMPT_TEMPLATE.format(problem=math_ds[ds_idx]["problem"])

    activations = {}
    hooks = []
    hooks.append(layer.mlp.gate_proj.register_forward_hook(capture_hook("gate")))
    hooks.append(layer.mlp.up_proj.register_forward_hook(capture_hook("up")))

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        model(**inputs)
    for hk in hooks:
        hk.remove()

    gate_out = activations["gate"][0].numpy()
    up_out = activations["up"][0].numpy()
    intermediate = (gate_out * expit(gate_out)) * up_out
    mag = np.mean(np.abs(intermediate), axis=0)
    imp_other = np.abs(neuron_alignment_other) * mag

    top5_other = np.argsort(imp_other)[-5:][::-1]
    for ni in top5_other:
        top50_neurons.append({
            "layer": int(other_layer),
            "neuron": int(ni),
            "alignment": float(neuron_alignment_other[ni]),
            "importance": float(imp_other[ni]),
        })
        print(f"    Neuron {ni}: importance={imp_other[ni]:.4f}")

    torch.cuda.empty_cache()

# ── 5c: Cluster Axis Geometry ─────────────────────────────────────────────
print("\n=== 5c: Cluster axis geometry ===")

# Pairwise cosine similarity between cluster axes
valid_axes_arr = np.array(valid_axes)
n_valid = len(valid_axes)

# Sample for efficiency if too many
if n_valid > 200:
    sample_idx = np.random.RandomState(42).choice(n_valid, 200, replace=False)
    axes_sample = valid_axes_arr[sample_idx]
else:
    axes_sample = valid_axes_arr

# Cosine similarity matrix
cosines = axes_sample @ axes_sample.T
# Extract upper triangle
upper = cosines[np.triu_indices(len(axes_sample), k=1)]
mean_pairwise_cos = float(np.mean(np.abs(upper)))
std_pairwise_cos = float(np.std(np.abs(upper)))
consistent = mean_pairwise_cos > 0.5

print(f"  Mean pairwise |cosine|: {mean_pairwise_cos:.4f} ± {std_pairwise_cos:.4f}")
print(f"  Cluster axis consistent: {consistent}")

# Cosine between cluster axis and flow axis
cluster_flow_cosine = float(np.abs(np.dot(mean_cluster_axis, flow_axis)))
print(f"  Cluster axis–flow axis |cosine|: {cluster_flow_cosine:.4f}")

# Cluster axis norm by difficulty level
axis_norm_by_level = {}
for lv in range(1, 6):
    norms = cluster_axis_by_level.get(lv, [])
    axis_norm_by_level[str(lv)] = float(np.mean(norms)) if norms else 0.0
    print(f"  Level {lv} mean cluster axis norm: {axis_norm_by_level[str(lv)]:.2f}")

del model
torch.cuda.empty_cache()

# ── Generate Figures ───────────────────────────────────────────────────────
print("\n=== Generating figures ===")

# Figure 1: MLP vs Attention ablation silhouette drop
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(N_LAYERS)
width = 0.35
ax.bar(x - width/2, mlp_sil_drops, width, label="MLP Ablation", color="steelblue", alpha=0.8)
ax.bar(x + width/2, attn_sil_drops, width, label="Attention Ablation", color="coral", alpha=0.8)
ax.set_xlabel("Layer Index")
ax.set_ylabel("Silhouette Drop (baseline - ablated)")
ax.set_title("Cluster Destruction: MLP vs Attention Ablation by Layer")
ax.legend()
for cl in critical_mlp_layers[:3]:
    ax.axvline(cl, color="blue", linestyle=":", alpha=0.5)
ax.set_xticks(range(0, N_LAYERS, 2))
plt.tight_layout()
plt.savefig(FIG_DIR / "mlp_ablation_silhouette.png", dpi=150)
plt.close()
print("  Saved mlp_ablation_silhouette.png")

# Figure 2: Top-50 neuron importances
fig, ax = plt.subplots(figsize=(10, 5))
# Color by layer
layer_colors = {"steelblue": [], "coral": [], "green": []}
neuron_layers = [n["layer"] for n in top50_neurons]
unique_layers = sorted(set(neuron_layers))
cmap = plt.cm.Set2(np.linspace(0, 1, max(len(unique_layers), 1)))
layer_to_color = {l: cmap[i] for i, l in enumerate(unique_layers)}

importances = [n["importance"] for n in top50_neurons]
colors = [layer_to_color[n["layer"]] for n in top50_neurons]
bars = ax.bar(range(len(top50_neurons)), importances, color=colors, edgecolor="gray", linewidth=0.5)
ax.set_xlabel("Neuron Rank")
ax.set_ylabel("Importance Score")
ax.set_title(f"Top Neurons Contributing to Cluster Separation\n(Layers: {unique_layers})")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=layer_to_color[l], label=f"Layer {l}") for l in unique_layers]
ax.legend(handles=legend_elements, fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "neuron_importance.png", dpi=150)
plt.close()
print("  Saved neuron_importance.png")

# Figure 3: Pairwise cosine similarity histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(np.abs(upper), bins=50, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(mean_pairwise_cos, color="red", linestyle="--", label=f"Mean |cos|={mean_pairwise_cos:.3f}")
ax.set_xlabel("|Cosine Similarity| Between Problem Cluster Axes")
ax.set_ylabel("Count")
ax.set_title(f"Cluster Axis Consistency Across Problems\n({'Consistent' if consistent else 'Inconsistent'}: mean={mean_pairwise_cos:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "cluster_axis_alignment.png", dpi=150)
plt.close()
print("  Saved cluster_axis_alignment.png")

# ── Save results ──────────────────────────────────────────────────────────
print("\n=== Saving results ===")

interpretation = (
    f"MLP ablation at layers {critical_mlp_layers} causes the largest cluster disruption. "
    f"Top neurons are concentrated at layer {critical_mlp_layers[0]}. "
    f"Cluster axes across problems have mean |cosine|={mean_pairwise_cos:.3f} — "
    f"{'a consistent universal direction' if consistent else 'problem-specific directions'}. "
    f"Cluster axis–flow axis alignment: |cos|={cluster_flow_cosine:.3f} — "
    f"{'they are aligned (same direction)' if cluster_flow_cosine > 0.5 else 'they are NOT aligned (different directions)'}."
)

results = {
    "ablation_results": {
        "mlp_silhouette_drop_by_layer": mlp_sil_drops,
        "attn_silhouette_drop_by_layer": attn_sil_drops,
        "critical_mlp_layers": critical_mlp_layers,
        "critical_attn_layers": critical_attn_layers,
    },
    "top_50_neurons": top50_neurons,
    "cluster_axis_consistency": {
        "mean_pairwise_cosine": mean_pairwise_cos,
        "std_pairwise_cosine": std_pairwise_cos,
        "consistent": consistent,
    },
    "cluster_axis_flow_axis_cosine": cluster_flow_cosine,
    "cluster_axis_norm_by_level": axis_norm_by_level,
    "interpretation": interpretation,
}

with open(DATA_DIR / "exp5_mlp_dissection.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved exp5_mlp_dissection.json")

print(f"\n=== Experiment 5 complete ===")
print(f"  {interpretation}")
