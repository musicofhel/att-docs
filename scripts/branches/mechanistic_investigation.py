"""Mechanistic investigation: where does the topological structure come from?

5 experiments:
1. Training dynamics — when does H1 non-monotonicity emerge in Pythia-1.4B checkpoints?
2. Per-head topology — which attention heads carry the difficulty signal in Qwen2.5-1.5B?
3. Residual stream decomposition — MLP vs attention contribution at terminal layer
4. What makes Level 1 special — geometric analysis of easy vs hard representations
5. Layer-by-layer H1 trajectory — at which layer do difficulty levels separate?

Usage:
    python scripts/branches/mechanistic_investigation.py
    python scripts/branches/mechanistic_investigation.py --exp 1      # single experiment
    python scripts/branches/mechanistic_investigation.py --exp 1,2,3  # specific experiments
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "mechanistic")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "mechanistic")

PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)

# ─── Utility functions ───────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_math500(n_per_level=25, seed=42):
    """Load MATH-500 problems, sampling n_per_level per difficulty."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Loaded {len(ds)} problems from MATH-500")

    problems_by_level = {k: [] for k in range(1, 6)}
    for row in ds:
        level_raw = row.get("level", "")
        if not level_raw and level_raw != 0:
            continue
        try:
            level = int(level_raw) if isinstance(level_raw, int) else int(str(level_raw).replace("Level ", ""))
        except (ValueError, AttributeError):
            continue
        if 1 <= level <= 5:
            problems_by_level[level].append({"problem": row["problem"], "level": level})

    rng = np.random.default_rng(seed)
    sampled = []
    for level in range(1, 6):
        pool = problems_by_level[level]
        n_sample = min(n_per_level, len(pool))
        indices = rng.choice(len(pool), size=n_sample, replace=False)
        sampled.extend([pool[i] for i in indices])
        print(f"  Level {level}: {n_sample}/{len(pool)}")
    print(f"  Total: {len(sampled)} problems")
    return sampled


def persistence_entropy(dgm):
    """Compute persistence entropy from a persistence diagram (n,2) array."""
    if len(dgm) == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    total = lifetimes.sum()
    if total == 0:
        return 0.0
    probs = lifetimes / total
    return float(-np.sum(probs * np.log(probs + 1e-16)))


def compute_vr_ph(cloud, max_dim=1, subsample=25, seed=42):
    """Compute VR persistence on a point cloud. Returns dict with diagrams and entropy."""
    import ripser

    if cloud.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(cloud.shape[0], size=subsample, replace=False)
        cloud = cloud[idx]

    # PCA to 50 dims if needed
    if cloud.shape[1] > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, cloud.shape[0] - 1))
        cloud = pca.fit_transform(cloud)

    result = ripser.ripser(cloud, maxdim=max_dim)
    diagrams = result["dgms"]

    entropies = {}
    for dim, dgm in enumerate(diagrams):
        # Remove infinite points
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
        entropies[f"H{dim}"] = persistence_entropy(finite)

    return {"diagrams": diagrams, "entropy": entropies}


def compute_vr_ph_distance(dist_matrix, max_dim=1):
    """Compute VR persistence from a precomputed distance matrix."""
    import ripser
    result = ripser.ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
    diagrams = result["dgms"]
    entropies = {}
    for dim, dgm in enumerate(diagrams):
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
        entropies[f"H{dim}"] = persistence_entropy(finite)
    return {"diagrams": diagrams, "entropy": entropies}


def permutation_zscore(values_by_level, n_perms=100, seed=42):
    """Compute permutation z-score for difficulty discrimination.

    values_by_level: dict {level: [list of scalar values]}
    Tests whether the variance across level means exceeds chance.
    """
    rng = np.random.default_rng(seed)
    all_values = []
    labels = []
    for level in sorted(values_by_level.keys()):
        vals = values_by_level[level]
        all_values.extend(vals)
        labels.extend([level] * len(vals))
    all_values = np.array(all_values)
    labels = np.array(labels)

    level_means = [all_values[labels == lv].mean() for lv in sorted(set(labels))]
    observed = np.var(level_means)

    null_dist = []
    for _ in range(n_perms):
        perm_labels = rng.permutation(labels)
        perm_means = [all_values[perm_labels == lv].mean() for lv in sorted(set(labels))]
        null_dist.append(np.var(perm_means))
    null_dist = np.array(null_dist)

    z = (observed - null_dist.mean()) / (null_dist.std() + 1e-16)
    return float(z)


def free_gpu():
    """Force free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Experiment 1: Training Dynamics ─────────────────────────────────────────

def run_exp1(problems_by_level, results):
    """When does the H1 non-monotonicity emerge during Pythia-1.4B training?"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoints = [0, 8, 64, 512, 2000, 8000, 32000, 64000, 128000, 143000]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use 25 problems per level (same as standard)
    # problems_by_level already structured
    h1_entropy_map = {}  # step -> {level: entropy}

    for step_idx, step in enumerate(checkpoints):
        revision = f"step{step}"
        print(f"\n{'='*60}")
        print(f"Checkpoint {step_idx+1}/{len(checkpoints)}: step {step} (revision={revision})")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/pythia-1.4b",
                revision=revision,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/pythia-1.4b",
                revision=revision,
                torch_dtype=torch.float16,
                device_map="auto",
                output_hidden_states=True,
                trust_remote_code=True,
            )
            model.eval()
        except Exception as e:
            print(f"  Failed to load step {step}: {e}")
            h1_entropy_map[f"step{step}"] = {str(lv): -1.0 for lv in range(1, 6)}
            continue

        print(f"  Model loaded in {time.time()-t0:.1f}s")

        # Extract last-token hidden states per level
        step_entropies = {}
        for level in range(1, 6):
            level_problems = problems_by_level[level]
            hidden_states_list = []

            for prob in level_problems:
                prompt = PROMPT_TEMPLATE.format(problem=prob["problem"])
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512,
                ).to(device)

                try:
                    with torch.no_grad():
                        outputs = model(**inputs)
                    # Last token of final layer
                    hs = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
                    hidden_states_list.append(hs)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue

            if len(hidden_states_list) < 5:
                print(f"  Level {level}: only {len(hidden_states_list)} states, skipping")
                step_entropies[str(level)] = -1.0
                continue

            cloud = np.array(hidden_states_list)
            ph = compute_vr_ph(cloud, max_dim=1, subsample=min(25, len(cloud)))
            h1_ent = ph["entropy"].get("H1", 0.0)
            step_entropies[str(level)] = h1_ent
            print(f"  Level {level}: {len(hidden_states_list)} problems, H1={h1_ent:.4f}")

        h1_entropy_map[f"step{step}"] = step_entropies

        # Free model
        del model, tokenizer
        free_gpu()

    # Analysis
    # Check if random init is non-monotonic
    step0 = h1_entropy_map.get("step0", {})
    step0_vals = [step0.get(str(lv), -1) for lv in range(1, 6)]
    random_nonmono = False
    if all(v >= 0 for v in step0_vals):
        # Non-monotonic = Level 1 is NOT the maximum
        random_nonmono = step0_vals[0] < max(step0_vals)

    # When does Level 1 minimum emerge?
    level1_min_step = None
    for step in checkpoints:
        vals = h1_entropy_map.get(f"step{step}", {})
        v = [vals.get(str(lv), -1) for lv in range(1, 6)]
        if all(x >= 0 for x in v):
            if v[0] == min(v):
                level1_min_step = step
                break

    results["exp1_training_dynamics"] = {
        "checkpoints": checkpoints,
        "h1_entropy_by_checkpoint_level": h1_entropy_map,
        "level1_minimum_emerges_at_step": level1_min_step,
        "random_init_nonmonotonic": random_nonmono,
        "terminal_effect_transient": False,  # updated after plotting
    }

    # Plot heatmap
    _plot_exp1(checkpoints, h1_entropy_map)
    return results


def _plot_exp1(checkpoints, h1_map):
    """Plot training dynamics heatmap and dual-line plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    matrix = []
    valid_steps = []
    for step in checkpoints:
        key = f"step{step}"
        if key in h1_map:
            row = [h1_map[key].get(str(lv), np.nan) for lv in range(1, 6)]
            row = [np.nan if v < 0 else v for v in row]
            matrix.append(row)
            valid_steps.append(step)

    if not matrix:
        print("  No data for Exp 1 heatmap")
        return

    matrix = np.array(matrix)
    im = ax1.imshow(matrix.T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax1.set_xticks(range(len(valid_steps)))
    ax1.set_xticklabels([str(s) for s in valid_steps], rotation=45, fontsize=8)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels([f"Level {i}" for i in range(1, 6)])
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Difficulty Level")
    ax1.set_title("H1 Persistence Entropy Across Training")
    plt.colorbar(im, ax=ax1, label="H1 Entropy")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax1.text(i, j, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v < np.nanmedian(matrix) else "black")

    # Dual line plot: Level 1 vs Level 5
    l1_vals = [h1_map.get(f"step{s}", {}).get("1", np.nan) for s in valid_steps]
    l5_vals = [h1_map.get(f"step{s}", {}).get("5", np.nan) for s in valid_steps]
    l1_vals = [np.nan if v < 0 else v for v in l1_vals]
    l5_vals = [np.nan if v < 0 else v for v in l5_vals]

    x = np.arange(len(valid_steps))
    ax2.plot(x, l1_vals, "o-", color="#2196F3", label="Level 1 (easy)", linewidth=2)
    ax2.plot(x, l5_vals, "s-", color="#F44336", label="Level 5 (hard)", linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in valid_steps], rotation=45, fontsize=8)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("H1 Persistence Entropy")
    ax2.set_title("Level 1 vs Level 5 H1 Through Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "training_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Experiment 2: Per-Head Topology ─────────────────────────────────────────

def run_exp2(problems, results):
    """Which attention heads carry the difficulty signal?"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Qwen2.5-1.5B with eager attention (float32 for stable attn)...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # float32 required: float16 produces NaN attention weights with eager attention
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    # Qwen2.5-1.5B: 28 layers, 12 heads
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    # GQA: num_key_value_heads may differ from num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    print(f"  KV heads: {n_kv_heads} (GQA)" if n_kv_heads != n_heads else "")
    print(f"  {n_layers} layers, {n_heads} heads = {n_layers * n_heads} total")

    # Use 50 problems (10 per level) — reduce if OOM
    n_per_level = 10
    rng = np.random.default_rng(42)
    problems_subset = []
    by_level = {lv: [p for p in problems if p["level"] == lv] for lv in range(1, 6)}
    for lv in range(1, 6):
        pool = by_level[lv]
        n = min(n_per_level, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False)
        problems_subset.extend([pool[i] for i in idx])

    # For each problem, get attention at terminal layer, per head
    # Store per-head H1 entropy keyed by (level, head_idx)
    terminal_layer = n_layers - 1
    # Use n_heads for indexing — actual head count may differ with GQA
    head_entropies = {lv: {h: [] for h in range(n_heads)} for lv in range(1, 6)}

    for pi, prob in enumerate(problems_subset):
        if pi % 10 == 0:
            print(f"  Problem {pi}/{len(problems_subset)}")

        prompt = PROMPT_TEMPLATE.format(problem=prob["problem"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256,
        ).to(device)

        try:
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        if outputs.attentions is None or len(outputs.attentions) == 0:
            print(f"  WARNING: No attentions returned for problem {pi}")
            continue

        # outputs.attentions[layer] shape: (1, n_heads, seq_len, seq_len)
        attn = outputs.attentions[terminal_layer][0].cpu().float().numpy()  # (n_heads, T, T)
        actual_heads = attn.shape[0]
        if pi == 0:
            print(f"  Attention tensor shape at terminal layer: {attn.shape}")

        for head_idx in range(actual_heads):
            attn_head = attn[head_idx]  # (T, T)
            n_tok = attn_head.shape[0]

            # Subsample tokens if too many
            subsample_tok = min(n_tok, 64)
            if n_tok > subsample_tok:
                tok_idx = np.sort(rng.choice(n_tok, size=subsample_tok, replace=False))
                attn_head = attn_head[np.ix_(tok_idx, tok_idx)]

            # Symmetrize → distance
            sym = (attn_head + attn_head.T) / 2.0
            dist = 1.0 - sym
            np.clip(dist, 0.0, 1.0, out=dist)
            np.fill_diagonal(dist, 0.0)

            # Check for NaN (float16 attention issue)
            if np.any(np.isnan(dist)):
                continue

            try:
                ph = compute_vr_ph_distance(dist, max_dim=1)
                head_entropies[prob["level"]][head_idx].append(ph["entropy"].get("H1", 0.0))
            except Exception:
                continue

        torch.cuda.empty_cache()

    # Determine actual number of heads from data
    actual_n_heads = max(
        max(h for h in head_entropies[lv].keys() if head_entropies[lv][h])
        for lv in range(1, 6) if any(head_entropies[lv][h] for h in head_entropies[lv])
    ) + 1 if any(head_entropies[lv][h] for lv in range(1, 6) for h in head_entropies[lv]) else n_heads
    print(f"  Actual heads with data: {actual_n_heads}")

    # Compute z-scores per head: can this head discriminate difficulty?
    head_zscores = []
    for head_idx in range(actual_n_heads):
        values_by_level = {}
        for lv in range(1, 6):
            vals = head_entropies[lv][head_idx]
            if vals:
                values_by_level[lv] = vals
        if len(values_by_level) >= 3:
            z = permutation_zscore(values_by_level, n_perms=100)
        else:
            z = 0.0
        head_zscores.append(z)
        print(f"  Head {head_idx}: z={z:.2f}")

    # For full heatmap, we need all layers. Let's also do a quick scan of all layers.
    # But with 28 layers × 12 heads × 50 problems this is very expensive.
    # Instead, scan 5 representative layers: 0, 7, 14, 21, 27
    scan_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    scan_layers = sorted(set(scan_layers))  # deduplicate

    # Detect actual head count from first run
    actual_n_heads = len(head_zscores)  # may differ from config n_heads due to GQA
    full_z_matrix = np.zeros((len(scan_layers), actual_n_heads))
    for li, layer_idx in enumerate(scan_layers):
        if layer_idx == terminal_layer:
            full_z_matrix[li] = head_zscores
            continue

        print(f"  Scanning layer {layer_idx}...")
        layer_head_ent = {lv: {h: [] for h in range(actual_n_heads)} for lv in range(1, 6)}
        for pi, prob in enumerate(problems_subset[:25]):  # reduced for speed
            prompt = PROMPT_TEMPLATE.format(problem=prob["problem"])
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256,
            ).to(device)
            try:
                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

            if outputs.attentions is None:
                continue
            attn = outputs.attentions[layer_idx][0].cpu().float().numpy()
            for head_idx in range(min(actual_n_heads, attn.shape[0])):
                attn_head = attn[head_idx]
                n_tok = attn_head.shape[0]
                subsample_tok = min(n_tok, 64)
                if n_tok > subsample_tok:
                    tok_idx = np.sort(rng.choice(n_tok, size=subsample_tok, replace=False))
                    attn_head = attn_head[np.ix_(tok_idx, tok_idx)]
                sym = (attn_head + attn_head.T) / 2.0
                dist = 1.0 - sym
                np.clip(dist, 0.0, 1.0, out=dist)
                np.fill_diagonal(dist, 0.0)
                if np.any(np.isnan(dist)):
                    continue
                try:
                    ph = compute_vr_ph_distance(dist, max_dim=1)
                    layer_head_ent[prob["level"]][head_idx].append(ph["entropy"].get("H1", 0.0))
                except Exception:
                    continue
            torch.cuda.empty_cache()

        for head_idx in range(actual_n_heads):
            values_by_level = {}
            for lv in range(1, 6):
                vals = layer_head_ent[lv][head_idx]
                if vals:
                    values_by_level[lv] = vals
            if len(values_by_level) >= 3:
                full_z_matrix[li, head_idx] = permutation_zscore(values_by_level, n_perms=100)

    # Top and bottom heads at terminal layer
    sorted_heads = np.argsort(head_zscores)
    n_top = min(3, len(sorted_heads))
    top_heads = [{"layer": terminal_layer, "head": int(sorted_heads[-(i+1)]),
                  "z": float(head_zscores[sorted_heads[-(i+1)]])} for i in range(n_top)]
    bottom_heads = [{"layer": terminal_layer, "head": int(sorted_heads[i]),
                     "z": float(head_zscores[sorted_heads[i]])} for i in range(n_top)]

    n_significant = sum(1 for z in head_zscores if z > 2.0)
    concentrated_terminal = sum(1 for z in head_zscores if z > 2.0) > n_significant * 0.5 if n_significant > 0 else False

    results["exp2_per_head"] = {
        "scan_layers": scan_layers,
        "z_scores_by_scan_layer_head": full_z_matrix.tolist(),
        "terminal_layer_zscores": [float(z) for z in head_zscores],
        "top_heads": top_heads,
        "bottom_heads": bottom_heads,
        "n_significant_heads": n_significant,
        "n_actual_heads": actual_n_heads,
        "concentrated_in_terminal": concentrated_terminal,
    }

    _plot_exp2(scan_layers, full_z_matrix, actual_n_heads, head_entropies, top_heads, bottom_heads)

    del model, tokenizer
    free_gpu()
    return results


def _plot_exp2(scan_layers, z_matrix, n_heads, head_entropies, top_heads, bottom_heads):
    """Plot per-head z-score heatmap and top/bottom persistence diagrams."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(z_matrix.T, aspect="auto", cmap="RdYlBu_r", interpolation="nearest",
                   vmin=-2, vmax=6)
    ax.set_xticks(range(len(scan_layers)))
    ax.set_xticklabels([f"L{l}" for l in scan_layers])
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Head")
    ax.set_title("Per-Head Difficulty Discrimination (z-score)")
    plt.colorbar(im, ax=ax, label="z-score")

    # Top/bottom head entropy comparison
    ax2 = axes[1]
    terminal_layer = scan_layers[-1]
    colors_top = ["#F44336", "#E91E63", "#9C27B0"]
    colors_bot = ["#2196F3", "#03A9F4", "#00BCD4"]

    for i, h_info in enumerate(top_heads):
        h = h_info["head"]
        l1_vals = head_entropies.get(1, {}).get(h, [])
        l5_vals = head_entropies.get(5, {}).get(h, [])
        x_pos = i
        if l1_vals:
            ax2.scatter([x_pos - 0.15] * len(l1_vals), l1_vals, c=colors_top[i], marker="o",
                       alpha=0.5, s=30)
            ax2.plot(x_pos - 0.15, np.mean(l1_vals), "o", c=colors_top[i], markersize=10,
                    markeredgecolor="black")
        if l5_vals:
            ax2.scatter([x_pos + 0.15] * len(l5_vals), l5_vals, c=colors_top[i], marker="s",
                       alpha=0.5, s=30)
            ax2.plot(x_pos + 0.15, np.mean(l5_vals), "s", c=colors_top[i], markersize=10,
                    markeredgecolor="black")

    for i, h_info in enumerate(bottom_heads):
        h = h_info["head"]
        l1_vals = head_entropies.get(1, {}).get(h, [])
        l5_vals = head_entropies.get(5, {}).get(h, [])
        x_pos = i + 4
        if l1_vals:
            ax2.scatter([x_pos - 0.15] * len(l1_vals), l1_vals, c=colors_bot[i], marker="o",
                       alpha=0.5, s=30)
            ax2.plot(x_pos - 0.15, np.mean(l1_vals), "o", c=colors_bot[i], markersize=10,
                    markeredgecolor="black")
        if l5_vals:
            ax2.scatter([x_pos + 0.15] * len(l5_vals), l5_vals, c=colors_bot[i], marker="s",
                       alpha=0.5, s=30)
            ax2.plot(x_pos + 0.15, np.mean(l5_vals), "s", c=colors_bot[i], markersize=10,
                    markeredgecolor="black")

    top_labels = [f"T{i+1}:H{h['head']}\nz={h['z']:.1f}" for i, h in enumerate(top_heads)]
    bot_labels = [f"B{i+1}:H{h['head']}\nz={h['z']:.1f}" for i, h in enumerate(bottom_heads)]
    ax2.set_xticks(list(range(3)) + list(range(4, 7)))
    ax2.set_xticklabels(top_labels + bot_labels, fontsize=8)
    ax2.set_ylabel("H1 Entropy")
    ax2.set_title("Top vs Bottom Heads: Level 1 (o) vs Level 5 (s)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "head_zscore_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Experiment 3: Residual Stream Decomposition ────────────────────────────

def run_exp3(problems, results):
    """MLP vs attention contribution to topological structure at terminal layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Qwen2.5-1.5B for residual decomposition...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()

    terminal_layer_idx = model.config.num_hidden_layers - 1
    terminal_layer = model.model.layers[terminal_layer_idx]

    # Hook storage
    hook_data = {}

    def make_pre_hook(name):
        def hook_fn(module, args, kwargs=None):
            # Find the first tensor in args
            for a in (args if isinstance(args, (tuple, list)) else [args]):
                if isinstance(a, torch.Tensor):
                    hook_data[name] = a.detach().cpu().float()
                    return
            # Check kwargs for hidden_states
            if kwargs and "hidden_states" in kwargs:
                hook_data[name] = kwargs["hidden_states"].detach().cpu().float()
        return hook_fn

    def make_post_hook(name):
        def hook_fn(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                hook_data[name] = out.detach().cpu().float()
        return hook_fn

    # Register hooks — use with_kwargs=True for pre_hooks since Qwen uses kwargs
    handles = []
    handles.append(terminal_layer.self_attn.register_forward_pre_hook(make_pre_hook("pre_attn_input"), with_kwargs=True))
    handles.append(terminal_layer.self_attn.register_forward_hook(make_post_hook("attn_out")))
    handles.append(terminal_layer.mlp.register_forward_hook(make_post_hook("mlp_out")))
    handles.append(terminal_layer.register_forward_hook(make_post_hook("layer_out")))
    handles.append(terminal_layer.mlp.register_forward_pre_hook(make_pre_hook("pre_mlp_input"), with_kwargs=True))

    # Collect per-problem last-token states for each component
    components = ["pre_attn", "attn_out", "post_attn", "mlp_out", "post_mlp"]
    comp_states = {c: {lv: [] for lv in range(1, 6)} for c in components}

    # Use ALL 25 problems per level for a richer point cloud
    problems_subset = problems  # all 125 problems (25/level)

    for pi, prob in enumerate(problems_subset):
        if pi % 10 == 0:
            print(f"  Problem {pi}/{len(problems_subset)}")

        hook_data.clear()
        prompt = PROMPT_TEMPLATE.format(problem=prob["problem"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(device)

        try:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        lv = prob["level"]

        # Extract last-token states from hooks
        # pre_attn: input to self_attn
        if "pre_attn_input" in hook_data:
            comp_states["pre_attn"][lv].append(hook_data["pre_attn_input"][0, -1].numpy())

        # attn_out: output of self_attn (just the attention contribution)
        if "attn_out" in hook_data:
            comp_states["attn_out"][lv].append(hook_data["attn_out"][0, -1].numpy())

        # post_attn = pre_mlp_input (residual + attn_out, after layernorm is applied for MLP)
        # Actually pre_mlp_input is AFTER the post_attention_layernorm
        # The true post_attn (before layernorm) = pre_attn + attn_out
        if "pre_attn_input" in hook_data and "attn_out" in hook_data:
            post_attn = hook_data["pre_attn_input"][0, -1].numpy() + hook_data["attn_out"][0, -1].numpy()
            comp_states["post_attn"][lv].append(post_attn)

        # mlp_out: output of MLP (just the MLP contribution)
        if "mlp_out" in hook_data:
            comp_states["mlp_out"][lv].append(hook_data["mlp_out"][0, -1].numpy())

        # post_mlp: full layer output = post_attn + mlp_out
        if "layer_out" in hook_data:
            comp_states["post_mlp"][lv].append(hook_data["layer_out"][0, -1].numpy())

        torch.cuda.empty_cache()

    # Remove hooks
    for h in handles:
        h.remove()

    # Compute H1 entropy and z-scores per component
    comp_results = {}
    for comp in components:
        h1_by_level = {}
        values_by_level = {}
        for lv in range(1, 6):
            states = comp_states[comp][lv]
            if len(states) < 3:
                continue
            cloud = np.array(states)
            ph = compute_vr_ph(cloud, max_dim=1, subsample=min(25, len(cloud)))
            h1 = ph["entropy"].get("H1", 0.0)
            h1_by_level[str(lv)] = h1
            values_by_level[lv] = [h1]  # single value per level for this component
            print(f"  {comp} Level {lv}: H1={h1:.4f} ({len(states)} problems)")

        # For z-score, we need per-problem values. Use individual problem clouds differently.
        # Alternative: compute H1 per level and use the entropy values directly.
        # Actually, z-score from permutation test needs multiple samples per level.
        # Let's do a simplified version: bootstrap resample within each level.
        boot_values = {}
        for lv in range(1, 6):
            states = comp_states[comp][lv]
            if len(states) < 5:
                continue
            cloud = np.array(states)
            boot_vals = []
            for b in range(10):
                boot_rng = np.random.default_rng(42 + b)
                idx = boot_rng.choice(len(cloud), size=len(cloud), replace=True)
                ph = compute_vr_ph(cloud[idx], max_dim=1, subsample=min(25, len(cloud[idx])))
                boot_vals.append(ph["entropy"].get("H1", 0.0))
            boot_values[lv] = boot_vals

        z = permutation_zscore(boot_values, n_perms=100) if len(boot_values) >= 3 else 0.0

        comp_results[comp] = {
            "h1_by_level": h1_by_level,
            "z_score": z,
        }

    # Determine dominant component
    z_scores = {c: comp_results[c]["z_score"] for c in components}
    dominant = max(z_scores, key=z_scores.get)

    if z_scores.get("attn_out", 0) > 2 * z_scores.get("mlp_out", 0.01):
        finding = "Attention creates the topological structure — MLP contribution is secondary."
    elif z_scores.get("mlp_out", 0) > 2 * z_scores.get("attn_out", 0.01):
        finding = "MLP creates the topological structure — attention contribution is secondary."
    elif z_scores.get("pre_attn", 0) > max(z_scores.get("attn_out", 0), z_scores.get("mlp_out", 0)):
        finding = "Structure already present before terminal layer processing — created in earlier layers."
    else:
        finding = "Structure emerges from interaction of attention and MLP components."

    results["exp3_residual_decomposition"] = {
        "components": components,
        "h1_entropy_by_component_level": {c: comp_results[c]["h1_by_level"] for c in components},
        "z_scores_by_component": z_scores,
        "dominant_component": dominant,
        "finding": finding,
    }

    del model, tokenizer
    free_gpu()
    return results


# ─── Experiment 4: What Makes Level 1 Special? ──────────────────────────────

def run_exp4(problems, results):
    """Geometric analysis of Level 1 vs Level 5 hidden states."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Qwen2.5-1.5B for geometric analysis...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()

    # Extract hidden states for ALL problems at Level 1 and Level 5
    level_states = {1: [], 5: []}
    for prob in problems:
        if prob["level"] not in (1, 5):
            continue
        prompt = PROMPT_TEMPLATE.format(problem=prob["problem"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(device)

        try:
            with torch.no_grad():
                outputs = model(**inputs)
            hs = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
            level_states[prob["level"]].append(hs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()

    print(f"  Level 1: {len(level_states[1])} states, Level 5: {len(level_states[5])} states")

    def twonn_id(X):
        """TwoNN intrinsic dimension estimator (Facco et al. 2017)."""
        n = X.shape[0]
        if n < 4:
            return float("nan")
        nn = NearestNeighbors(n_neighbors=3).fit(X)
        dists, _ = nn.kneighbors(X)
        mu = dists[:, 2] / (dists[:, 1] + 1e-16)
        mu = mu[mu > 1]
        if len(mu) == 0:
            return float("nan")
        return 1.0 / np.mean(np.log(mu))

    def isotropy(X):
        """Variance of singular value spectrum (higher = more isotropic)."""
        if X.shape[0] < 2:
            return float("nan")
        # PCA on centered data
        X_centered = X - X.mean(axis=0)
        svs = np.linalg.svd(X_centered, compute_uv=False)
        svs = svs / (svs.sum() + 1e-16)
        # Isotropy: 1 - std(normalized SVs). More isotropic = lower std = higher isotropy
        return float(1.0 - np.std(svs))

    def clustering_score(X, k=2):
        """Silhouette score with k clusters. Lower = more uniform blob."""
        if X.shape[0] < k + 1:
            return float("nan")
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        if len(set(labels)) < 2:
            return float("nan")
        return float(silhouette_score(X, labels))

    level_results = {}
    for lv in [1, 5]:
        if len(level_states[lv]) < 5:
            print(f"  Level {lv}: insufficient data")
            continue

        X = np.array(level_states[lv])
        # PCA to 50 dims for all metrics
        pca = PCA(n_components=min(50, X.shape[0] - 1))
        X_pca = pca.fit_transform(X)

        id_val = twonn_id(X_pca)
        iso_val = isotropy(X_pca)
        sil_val = clustering_score(X_pca)

        # VR PH
        ph = compute_vr_ph(X, max_dim=1, subsample=min(25, len(X)))
        h1 = ph["entropy"].get("H1", 0.0)
        h0 = ph["entropy"].get("H0", 0.0)

        level_results[str(lv)] = {
            "intrinsic_dim": float(id_val),
            "isotropy": float(iso_val),
            "silhouette_k2": float(sil_val),
            "h1_entropy": float(h1),
            "h0_entropy": float(h0),
            "n_problems": len(level_states[lv]),
        }
        print(f"  Level {lv}: ID={id_val:.2f}, iso={iso_val:.4f}, sil={sil_val:.3f}, H1={h1:.4f}")

    # Generate finding
    if "1" in level_results and "5" in level_results:
        l1, l5 = level_results["1"], level_results["5"]
        findings = []
        if l1["intrinsic_dim"] < l5["intrinsic_dim"]:
            findings.append(f"Level 1 has lower intrinsic dimension ({l1['intrinsic_dim']:.1f} vs {l5['intrinsic_dim']:.1f})")
        if l1["isotropy"] > l5["isotropy"]:
            findings.append("Level 1 is more isotropic")
        if l1["silhouette_k2"] < l5["silhouette_k2"]:
            findings.append("Level 1 is less clustered (more blob-like)")
        if l1["h1_entropy"] < l5["h1_entropy"]:
            findings.append(f"Level 1 has lower H1 entropy ({l1['h1_entropy']:.3f} vs {l5['h1_entropy']:.3f})")
        finding = ". ".join(findings) + "." if findings else "No clear geometric distinction found."
    else:
        finding = "Insufficient data for comparison."

    results["exp4_level1_special"] = {
        "level1": level_results.get("1", {}),
        "level5": level_results.get("5", {}),
        "finding": finding,
    }

    del model, tokenizer
    free_gpu()
    return results


# ─── Experiment 5: Layer-by-Layer H1 Trajectory ─────────────────────────────

def run_exp5(problems, results):
    """H1 persistence entropy at every layer for all 5 difficulty levels."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Qwen2.5-1.5B for layer trajectory...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    # Use ALL 25 problems per level for meaningful H1 persistence
    problems_subset = problems  # all 125 problems (25/level)

    # layer_states[layer_idx][level] = list of last-token hidden states
    layer_states = {li: {lv: [] for lv in range(1, 6)} for li in range(n_layers)}

    for pi, prob in enumerate(problems_subset):
        if pi % 10 == 0:
            print(f"  Problem {pi}/{len(problems_subset)}")

        prompt = PROMPT_TEMPLATE.format(problem=prob["problem"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512,
        ).to(device)

        try:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        if outputs.hidden_states is None:
            continue
        for li in range(min(n_layers, len(outputs.hidden_states))):
            hs = outputs.hidden_states[li][0, -1, :].cpu().float().numpy()
            layer_states[li][prob["level"]].append(hs)

        torch.cuda.empty_cache()

    # Compute H1 entropy at each layer for each level
    h1_matrix = np.zeros((n_layers, 5))  # (layers, levels)
    for li in range(n_layers):
        for lv_idx, lv in enumerate(range(1, 6)):
            states = layer_states[li][lv]
            if len(states) < 5:
                h1_matrix[li, lv_idx] = np.nan
                continue
            cloud = np.array(states)
            ph = compute_vr_ph(cloud, max_dim=1, subsample=min(25, len(cloud)))
            h1_matrix[li, lv_idx] = ph["entropy"].get("H1", 0.0)
        print(f"  Layer {li}/{n_layers-1}: "
              + ", ".join(f"L{lv}={h1_matrix[li, lv-1]:.3f}" for lv in range(1, 6)))

    # Find separation layer
    sep_layer = None
    sep_z = 0.0
    for li in range(n_layers):
        l1 = h1_matrix[li, 0]
        l5 = h1_matrix[li, 4]
        if np.isnan(l1) or np.isnan(l5):
            continue
        diff = abs(l5 - l1)
        col_std = np.nanstd(h1_matrix[li, :])
        if col_std > 0 and diff / col_std > 1.0:
            if sep_layer is None:
                sep_layer = li
                sep_z = diff / col_std

    results["exp5_layer_trajectory"] = {
        "h1_entropy_by_layer_level": h1_matrix.tolist(),
        "separation_layer": sep_layer,
        "separation_zscore": float(sep_z) if sep_z else 0.0,
        "n_layers": n_layers,
    }

    _plot_exp5(h1_matrix, n_layers, sep_layer)

    del model, tokenizer
    free_gpu()
    return results


def _plot_exp5(h1_matrix, n_layers, sep_layer):
    """Plot layer-by-layer H1 trajectory for all 5 levels."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    for lv_idx in range(5):
        vals = h1_matrix[:, lv_idx]
        ax.plot(range(n_layers), vals, "o-", color=colors[lv_idx],
                label=f"Level {lv_idx + 1}", linewidth=1.5, markersize=3, alpha=0.8)

    if sep_layer is not None:
        ax.axvline(x=sep_layer, color="gray", linestyle="--", alpha=0.5,
                   label=f"Separation layer ({sep_layer})")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("H1 Persistence Entropy")
    ax.set_title("H1 Trajectory Across Layers by Difficulty Level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "h1_layer_trajectory.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mechanistic investigation of topological structure")
    parser.add_argument("--exp", default="1,2,3,4,5", help="Comma-separated experiment numbers to run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exps = [int(x.strip()) for x in args.exp.split(",")]
    print(f"Running experiments: {exps}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ensure_dirs()
    np.random.seed(args.seed)

    # Load MATH-500 problems (shared across experiments)
    print("\n" + "=" * 60)
    print("Loading MATH-500 problems")
    print("=" * 60)
    all_problems = load_math500(n_per_level=25, seed=args.seed)
    problems_by_level = {lv: [p for p in all_problems if p["level"] == lv] for lv in range(1, 6)}

    # Load existing results if any
    results_path = os.path.join(DATA_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded existing results from {results_path}")
    else:
        results = {}

    t_total = time.time()

    if 1 in exps:
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Training Dynamics (Pythia-1.4B checkpoints)")
        print("=" * 60)
        t0 = time.time()
        results = run_exp1(problems_by_level, results)
        print(f"Exp 1 done in {time.time()-t0:.0f}s")
        # Save intermediate
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    if 2 in exps:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Per-Head Topology (Qwen2.5-1.5B)")
        print("=" * 60)
        t0 = time.time()
        results = run_exp2(all_problems, results)
        print(f"Exp 2 done in {time.time()-t0:.0f}s")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    if 3 in exps:
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Residual Stream Decomposition")
        print("=" * 60)
        t0 = time.time()
        results = run_exp3(all_problems, results)
        print(f"Exp 3 done in {time.time()-t0:.0f}s")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    if 4 in exps:
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: What Makes Level 1 Special?")
        print("=" * 60)
        t0 = time.time()
        results = run_exp4(all_problems, results)
        print(f"Exp 4 done in {time.time()-t0:.0f}s")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    if 5 in exps:
        print("\n" + "=" * 60)
        print("EXPERIMENT 5: Layer-by-Layer H1 Trajectory")
        print("=" * 60)
        t0 = time.time()
        results = run_exp5(all_problems, results)
        print(f"Exp 5 done in {time.time()-t0:.0f}s")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"All experiments done in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Results: {results_path}")
    print(f"Figures: {FIGURES_DIR}/")

    # Save final results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
