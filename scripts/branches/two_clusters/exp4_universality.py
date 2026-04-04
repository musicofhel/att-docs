#!/usr/bin/env python3
"""Experiment 4: Cross-Architecture Universality.

Tests whether the two-cluster structure and 1D flow appear in other models,
tasks, and training checkpoints.
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
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "two_clusters"
FIG_DIR = ROOT / "figures" / "two_clusters"

# ── 4a: Load Existing Multi-Model Data ────────────────────────────────────
print("=== 4a: Load multi-model data ===")

model_files = {
    "qwen": ROOT / "data/transformer/math500_hidden_states_aligned.npz",
    "phi2": ROOT / "data/transformer/phi2_hidden_states.npz",
    "pythia": ROOT / "data/transformer/pythia14b_hidden_states.npz",
    "stablelm": ROOT / "data/transformer/stablelm16b_hidden_states.npz",
}

models_data = {}
for name, path in model_files.items():
    try:
        d = np.load(path, allow_pickle=True)
        models_data[name] = d
        print(f"  Loaded {name}: keys={list(d.keys())}")
        # Check for token_trajectories vs last_hidden_states
        if "token_trajectories" in d:
            tt = d["token_trajectories"]
            print(f"    token_trajectories: {tt.shape}, first item shape: {tt[0].shape if hasattr(tt[0], 'shape') else type(tt[0])}")
        if "last_hidden_states" in d:
            lhs = d["last_hidden_states"]
            print(f"    last_hidden_states: {lhs.shape}")
        if "layer_hidden_states" in d:
            ls = d["layer_hidden_states"]
            print(f"    layer_hidden_states: {ls.shape}")
        if "difficulty_levels" in d:
            dl = d["difficulty_levels"]
            print(f"    difficulty_levels: {np.unique(dl)}")
    except FileNotFoundError:
        print(f"  Missing: {path}")

# ── 4b: Per-Model Analysis ────────────────────────────────────────────────
print("\n=== 4b: Per-model two-cluster analysis ===")

per_model_results = {}

for name, d in models_data.items():
    print(f"\n  --- {name} ---")
    result = {}

    # Token-level clustering (if token_trajectories available)
    if "token_trajectories" in d:
        tt = d["token_trajectories"]
        sil_scores = []
        for idx in range(len(tt)):
            traj = tt[idx]
            if hasattr(traj, 'shape') and traj.ndim == 2 and traj.shape[0] >= 10:
                n_comp = min(30, traj.shape[0] - 1, traj.shape[1])
                pca = PCA(n_components=n_comp)
                reduced = pca.fit_transform(traj)
                km = KMeans(n_clusters=2, n_init=10, random_state=42)
                labels = km.fit_predict(reduced)
                sil = silhouette_score(reduced, labels) if len(set(labels)) > 1 else 0.0
                sil_scores.append(sil)
        result["mean_silhouette"] = float(np.mean(sil_scores)) if sil_scores else 0.0
        result["std_silhouette"] = float(np.std(sil_scores)) if sil_scores else 0.0
        result["n_problems"] = len(sil_scores)
        print(f"    Token-level k=2 silhouette: {result['mean_silhouette']:.4f} ± {result['std_silhouette']:.4f} (n={result['n_problems']})")

        # Optimal k search
        best_k = 2
        best_sil = result["mean_silhouette"]
        for k in [3, 4, 5]:
            k_sils = []
            for idx in range(min(100, len(tt))):
                traj = tt[idx]
                if hasattr(traj, 'shape') and traj.ndim == 2 and traj.shape[0] >= k + 5:
                    n_comp = min(30, traj.shape[0] - 1, traj.shape[1])
                    pca = PCA(n_components=n_comp)
                    reduced = pca.fit_transform(traj)
                    km = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = km.fit_predict(reduced)
                    sil = silhouette_score(reduced, labels) if len(set(labels)) > 1 else 0.0
                    k_sils.append(sil)
            if k_sils and np.mean(k_sils) > best_sil:
                best_sil = np.mean(k_sils)
                best_k = k
        result["k_opt"] = best_k
        print(f"    Optimal k: {best_k}")

    elif "last_hidden_states" in d:
        # Cluster per-problem last hidden states grouped by level
        lhs = d["last_hidden_states"]
        if lhs.ndim == 2 and lhs.shape[0] >= 10:
            n_comp = min(30, lhs.shape[0] - 1, lhs.shape[1])
            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(lhs)
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            labels = km.fit_predict(reduced)
            sil = silhouette_score(reduced, labels) if len(set(labels)) > 1 else 0.0
            result["mean_silhouette"] = float(sil)
            result["n_problems"] = lhs.shape[0]
            print(f"    Problem-level k=2 silhouette: {sil:.4f} (n={lhs.shape[0]})")

            best_k = 2
            best_sil = sil
            for k in [3, 4, 5]:
                if lhs.shape[0] >= k + 5:
                    km = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = km.fit_predict(reduced)
                    sil_k = silhouette_score(reduced, labels) if len(set(labels)) > 1 else 0.0
                    if sil_k > best_sil:
                        best_sil = sil_k
                        best_k = k
            result["k_opt"] = best_k
            print(f"    Optimal k: {best_k}")

    # Layer trajectory analysis (correlation dimension proxy, flow PCA)
    if "layer_hidden_states" in d:
        ls = d["layer_hidden_states"]
        if ls.ndim == 3:
            n_problems, n_layers, hidden_dim = ls.shape
            print(f"    Layer hidden states: {ls.shape}")

            # Displacement PCA for 1D flow check
            all_displacements = []
            for idx in range(n_problems):
                states = ls[idx]
                if states.ndim == 2 and states.shape[0] >= 2:
                    deltas = np.diff(states, axis=0)
                    all_displacements.append(deltas)

            if all_displacements:
                all_displacements = np.vstack(all_displacements)
                pca_flow = PCA(n_components=min(5, all_displacements.shape[1]))
                pca_flow.fit(all_displacements)
                result["flow_pc1_var"] = float(pca_flow.explained_variance_ratio_[0])
                print(f"    Flow PC1 variance: {result['flow_pc1_var']:.4f}")

            # Correlation dimension proxy: mean pairwise distance ratio
            # For a 1D manifold, intrinsic dim ~ 1.0
            corr_dims = []
            for idx in range(min(100, n_problems)):
                traj = ls[idx]
                if traj.shape[0] >= 5:
                    pca = PCA(n_components=min(5, traj.shape[0] - 1, traj.shape[1]))
                    reduced = pca.fit_transform(traj)
                    # Simple ratio-based estimate: var explained by PC1 / total
                    if pca.explained_variance_ratio_[0] > 0:
                        # Effective dimensionality = 1 / sum(p^2) where p = normalized eigenvalues
                        evr = pca.explained_variance_ratio_
                        participation_ratio = 1.0 / np.sum(evr ** 2)
                        corr_dims.append(participation_ratio)

            if corr_dims:
                result["corr_dim"] = float(np.mean(corr_dims))
                print(f"    Correlation dimension (participation ratio): {result['corr_dim']:.2f}")

        # Per-layer silhouette (aggregated)
        if "difficulty_levels" in d:
            levels = d["difficulty_levels"]
            per_layer_sil = []
            for layer_idx in range(ls.shape[1]):
                layer_states = ls[:, layer_idx, :]
                n_comp = min(30, layer_states.shape[0] - 1, layer_states.shape[1])
                pca = PCA(n_components=n_comp)
                reduced = pca.fit_transform(layer_states)
                km = KMeans(n_clusters=2, n_init=10, random_state=42)
                labels = km.fit_predict(reduced)
                sil = silhouette_score(reduced, labels) if len(set(labels)) > 1 else 0.0
                per_layer_sil.append(sil)
            result["per_layer_silhouette"] = [float(s) for s in per_layer_sil]

    # Goldilocks check
    if "difficulty_levels" in d and "token_trajectories" in d:
        tt = d["token_trajectories"]
        levels = d["difficulty_levels"]
        sil_by_level = {}
        for lv in np.unique(levels):
            lv_mask = levels == lv
            lv_sils = []
            for idx in np.where(lv_mask)[0]:
                traj = tt[idx]
                if hasattr(traj, 'shape') and traj.ndim == 2 and traj.shape[0] >= 10:
                    n_comp = min(30, traj.shape[0] - 1, traj.shape[1])
                    pca = PCA(n_components=n_comp)
                    reduced = pca.fit_transform(traj)
                    km = KMeans(n_clusters=2, n_init=10, random_state=42)
                    labels = km.fit_predict(reduced)
                    sil = silhouette_score(reduced, labels) if len(set(labels)) > 1 else 0.0
                    lv_sils.append(sil)
            sil_by_level[int(lv)] = float(np.mean(lv_sils)) if lv_sils else 0.0
        result["silhouette_by_level"] = sil_by_level
        # Peak level
        if sil_by_level:
            peak_lv = max(sil_by_level, key=sil_by_level.get)
            result["goldilocks"] = peak_lv == 3

    per_model_results[name] = result

# ── 4c: Pythia Training Checkpoints ───────────────────────────────────────
print("\n=== 4c: Pythia training checkpoints ===")

# Try to load mechanistic branch data
mech_results_path = ROOT / "data" / "mechanistic" / "results.json"
cluster_emergence_step = None

if mech_results_path.exists():
    with open(mech_results_path) as f:
        mech_results = json.load(f)
    print(f"  Loaded mechanistic results: {list(mech_results.keys())[:5]}...")
    # Look for per-checkpoint silhouette data
    if "checkpoints" in mech_results:
        for ckpt in mech_results["checkpoints"]:
            step = ckpt.get("step", 0)
            sil = ckpt.get("silhouette_k2", 0)
            if sil > 0.3 and cluster_emergence_step is None:
                cluster_emergence_step = step
                print(f"    Two-cluster emergence at step {step} (silhouette={sil:.3f})")
else:
    # Try git checkout
    import subprocess
    try:
        result = subprocess.run(
            ["git", "checkout", "experiment/tda-mechanistic", "--", "data/mechanistic/results.json"],
            cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and mech_results_path.exists():
            with open(mech_results_path) as f:
                mech_results = json.load(f)
            print(f"  Loaded from mechanistic branch: {list(mech_results.keys())[:5]}...")
        else:
            print(f"  Mechanistic branch data not available: {result.stderr[:200]}")
    except Exception as e:
        print(f"  Could not load mechanistic data: {e}")

if cluster_emergence_step is None:
    print("  No per-checkpoint silhouette data found; skipping emergence analysis")

# ── 4d: Cross-Task Analysis ───────────────────────────────────────────────
print("\n=== 4d: Cross-task analysis (GSM8K, TriviaQA) ===")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)

print("  Loading model...")
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

cross_task_results = {}

# GSM8K
print("  Loading GSM8K...")
try:
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_sils = []
    for i in range(min(50, len(gsm))):
        prompt = PROMPT_TEMPLATE.format(problem=gsm[i]["question"])
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
            gsm_sils.append(sil)

        if (i + 1) % 10 == 0:
            print(f"    GSM8K: {i + 1}/50")

    cross_task_results["gsm8k_silhouette"] = float(np.mean(gsm_sils)) if gsm_sils else 0.0
    print(f"    GSM8K mean silhouette: {cross_task_results['gsm8k_silhouette']:.4f}")
except Exception as e:
    print(f"    GSM8K failed: {e}")
    cross_task_results["gsm8k_silhouette"] = None

# TriviaQA
print("  Loading TriviaQA...")
try:
    trivia = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="validation")
    trivia_sils = []
    for i in range(min(50, len(trivia))):
        prompt = PROMPT_TEMPLATE.format(problem=trivia[i]["question"])
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
            trivia_sils.append(sil)

        if (i + 1) % 10 == 0:
            print(f"    TriviaQA: {i + 1}/50")

    cross_task_results["triviaqa_silhouette"] = float(np.mean(trivia_sils)) if trivia_sils else 0.0
    print(f"    TriviaQA mean silhouette: {cross_task_results['triviaqa_silhouette']:.4f}")
except Exception as e:
    print(f"    TriviaQA failed: {e}")
    cross_task_results["triviaqa_silhouette"] = None

# MATH-500 baseline (from Exp 1)
exp1_path = DATA_DIR / "exp1_semantic_cartography.json"
if exp1_path.exists():
    with open(exp1_path) as f:
        exp1 = json.load(f)
    cross_task_results["math500_silhouette"] = exp1.get("mean_silhouette_all", 0.0)
else:
    cross_task_results["math500_silhouette"] = 0.0

del model
torch.cuda.empty_cache()

# ── Determine universality ────────────────────────────────────────────────
print("\n=== Universality assessment ===")

# k=2 universal?
k2_models = [per_model_results[m].get("k_opt", None) for m in per_model_results]
k2_universal = all(k == 2 for k in k2_models if k is not None)
print(f"  k=2 optimal across all models: {k2_universal}")

# 1D flow universal?
flow_models = [per_model_results[m].get("flow_pc1_var", 0) for m in per_model_results]
flow_1d_universal = all(v > 0.5 for v in flow_models if v > 0)
print(f"  1D flow (PC1 > 50%): {flow_1d_universal}")

interpretation = (
    f"k=2 is {'universally' if k2_universal else 'NOT universally'} optimal. "
    f"1D flow is {'universal' if flow_1d_universal else 'NOT universal'} (PC1 > 50% in all models). "
    f"Cross-task: MATH={cross_task_results.get('math500_silhouette', 0):.3f}, "
    f"GSM8K={cross_task_results.get('gsm8k_silhouette', 'N/A')}, "
    f"TriviaQA={cross_task_results.get('triviaqa_silhouette', 'N/A')}."
)
print(f"  {interpretation}")

# ── Save results ──────────────────────────────────────────────────────────
print("\n=== Saving results ===")

results = {
    "per_model": per_model_results,
    "k2_universal": k2_universal,
    "flow_1d_universal": flow_1d_universal,
    "cluster_emergence_step": cluster_emergence_step,
    "cross_task": cross_task_results,
    "interpretation": interpretation,
}

# Convert any numpy types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

results = convert_numpy(results)

with open(DATA_DIR / "exp4_universality.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved exp4_universality.json")

# Print comparison table
print("\n  Comparison table:")
print(f"  {'Model':10s} | {'k_opt':>5s} | {'Mean Sil':>8s} | {'Corr Dim':>8s} | {'Flow PC1':>8s} | {'Goldilocks':>10s}")
print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
for name in per_model_results:
    r = per_model_results[name]
    k = str(r.get("k_opt", "?"))
    sil = f"{r.get('mean_silhouette', 0):.4f}" if "mean_silhouette" in r else "?"
    cd = f"{r.get('corr_dim', 0):.2f}" if "corr_dim" in r else "?"
    fpc = f"{r.get('flow_pc1_var', 0):.4f}" if "flow_pc1_var" in r else "?"
    gold = str(r.get("goldilocks", "?"))
    print(f"  {name:10s} | {k:>5s} | {sil:>8s} | {cd:>8s} | {fpc:>8s} | {gold:>10s}")

print("\n=== Experiment 4 complete ===")
