#!/usr/bin/env python3
"""Branch 6: Multilingual Math — Topology of LLM Representations Across Languages.

Tests whether mathematical reasoning representations are language-agnostic by
comparing hidden-state topology across English, Chinese, Spanish, and Japanese
on the same math problems (MGSM dataset, 250 problems × 4 languages).

Three experiments:
1. Cross-language topological distance (4×4 Wasserstein matrix + permutation test)
2. Per-problem topological fingerprint consistency (cross-language correlation)
3. Cross-lingual binding (EN-ZH persistence image subtraction + surrogate test)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from scipy import stats
from sklearn.decomposition import PCA

from att.llm.features import TopologicalFeatureExtractor
from att.topology.persistence import PersistenceAnalyzer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "multilingual")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "multilingual")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 512
LANGUAGES = ["en", "zh", "es", "ja"]
LANG_NAMES = {"en": "English", "zh": "Chinese", "es": "Spanish", "ja": "Japanese"}

N_PCA = 50
SUBSAMPLE = 200
N_PERMS = 200
N_SURROGATES = 50
SEED = 42


def load_mgsm(languages):
    """Load MGSM dataset for specified languages from TSV files."""
    import csv

    from huggingface_hub import hf_hub_download

    all_data = {}
    for lang in languages:
        print(f"  Loading MGSM {lang}...")
        tsv_path = hf_hub_download(
            "juletxara/mgsm", f"mgsm_{lang}.tsv", repo_type="dataset"
        )

        problems = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    problems.append({"question": row[0], "answer": row[1].strip()})

        all_data[lang] = problems
        print(f"    {lang}: {len(problems)} problems")

    # Use the minimum count across all languages
    counts = {lang: len(probs) for lang, probs in all_data.items()}
    n = min(counts.values())
    for lang in languages:
        all_data[lang] = all_data[lang][:n]

    print(f"  Using {n} problems per language")
    return all_data, n


def extract_hidden_states(problems_by_lang, seed=42):
    """Extract hidden states from Qwen2.5-1.5B-Instruct for all languages."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()

    results = {}
    for lang, problems in problems_by_lang.items():
        print(f"\n  Extracting {lang} ({len(problems)} problems)...")
        t0 = time.time()

        all_last_hidden = []
        all_layer_hidden = []
        all_token_traj = []
        all_seq_lengths = []

        for i, prob in enumerate(problems):
            prompt = prob["question"]

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(device)
            seq_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.hidden_states  # tuple of (1, seq, d)

            # Last token, final layer
            final_layer = hidden_states[-1][0]  # (seq, d)
            last_hidden = final_layer[-1].cpu().float().numpy()  # (d,)

            # All layers at last token
            layer_states = (
                torch.stack(
                    [hidden_states[j][0, -1, :] for j in range(len(hidden_states))]
                )
                .cpu()
                .float()
                .numpy()  # (L+1, d)
            )

            # Token trajectory (all tokens at final layer)
            token_traj = final_layer.cpu().float().numpy()  # (T, d)

            all_last_hidden.append(last_hidden)
            all_layer_hidden.append(layer_states)
            all_token_traj.append(token_traj)
            all_seq_lengths.append(seq_len)

            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(problems) - i - 1) / rate
                print(
                    f"    [{i+1}/{len(problems)}] {elapsed:.0f}s elapsed, "
                    f"~{remaining:.0f}s remaining"
                )

        last_hidden_arr = np.array(all_last_hidden)  # (N, d)
        layer_hidden_arr = np.array(all_layer_hidden)  # (N, L+1, d)

        # Save per-language NPZ
        npz_path = os.path.join(DATA_DIR, f"{lang}_hidden_states.npz")
        np.savez_compressed(
            npz_path,
            last_hidden_states=last_hidden_arr,
            layer_hidden_states=layer_hidden_arr,
            token_trajectories=np.array(all_token_traj, dtype=object),
            seq_lengths=np.array(all_seq_lengths),
            model_name=np.array(MODEL_ID),
            language=np.array(lang),
            hidden_dim=np.array(last_hidden_arr.shape[1]),
            num_layers=np.array(layer_hidden_arr.shape[1]),
        )

        elapsed = time.time() - t0
        print(f"  {lang} done: {last_hidden_arr.shape}, {elapsed:.1f}s")
        print(f"  Saved: {npz_path}")

        results[lang] = {
            "last_hidden": last_hidden_arr,
            "layer_hidden": layer_hidden_arr,
            "token_trajectories": all_token_traj,
        }

    # Clean up GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def load_cached_hidden_states(languages):
    """Load previously extracted hidden states."""
    results = {}
    for lang in languages:
        npz_path = os.path.join(DATA_DIR, f"{lang}_hidden_states.npz")
        if not os.path.exists(npz_path):
            return None
        data = np.load(npz_path, allow_pickle=True)
        results[lang] = {
            "last_hidden": data["last_hidden_states"],
            "layer_hidden": data["layer_hidden_states"],
            "token_trajectories": data["token_trajectories"],
        }
        print(f"  Loaded {lang}: {data['last_hidden_states'].shape}")
    return results


def run_ph(cloud, max_dim=1, subsample=200, n_pca=50, seed=42):
    """PCA + PH on a point cloud."""
    n_comp = min(n_pca, cloud.shape[0] - 1, cloud.shape[1])
    pca = PCA(n_components=n_comp)
    cloud_pca = pca.fit_transform(cloud)

    pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser")
    result = pa.fit_transform(cloud_pca, subsample=subsample, seed=seed)
    return pa, result


# ── Experiment 1: Cross-language topological distance ──────────────────────


def run_exp1(hidden_by_lang, seed=42):
    """Cross-language Wasserstein distances + permutation test."""
    print("\n=== Experiment 1: Cross-language topological distance ===")

    languages = list(hidden_by_lang.keys())
    n_langs = len(languages)

    # PH per language
    analyzers = {}
    entropies = {}
    for lang in languages:
        cloud = hidden_by_lang[lang]["last_hidden"]
        print(f"  PH for {lang}: {cloud.shape}")
        pa, result = run_ph(cloud, max_dim=1, subsample=SUBSAMPLE, n_pca=N_PCA, seed=seed)
        analyzers[lang] = pa
        entropies[lang] = {
            f"H{dim}": float(result["persistence_entropy"][dim])
            for dim in range(len(result["persistence_entropy"]))
        }
        print(
            f"    H0={entropies[lang]['H0']:.3f}, H1={entropies[lang]['H1']:.3f}"
        )

    # 4×4 Wasserstein matrix
    wass_matrix = np.zeros((n_langs, n_langs))
    for i, lang_i in enumerate(languages):
        for j, lang_j in enumerate(languages):
            if i < j:
                d = analyzers[lang_i].distance(
                    analyzers[lang_j], metric="wasserstein_1"
                )
                wass_matrix[i, j] = d
                wass_matrix[j, i] = d
                print(f"    {lang_i}-{lang_j}: {d:.2f}")

    # Closest / farthest pairs
    min_d, max_d = np.inf, -np.inf
    closest, farthest = "", ""
    for i in range(n_langs):
        for j in range(i + 1, n_langs):
            if wass_matrix[i, j] < min_d:
                min_d = wass_matrix[i, j]
                closest = f"{languages[i]}-{languages[j]}"
            if wass_matrix[i, j] > max_d:
                max_d = wass_matrix[i, j]
                farthest = f"{languages[i]}-{languages[j]}"

    # Permutation test: shuffle language labels, compare mean off-diagonal Wasserstein
    observed_mean = float(np.mean(wass_matrix[np.triu_indices(n_langs, k=1)]))

    all_hidden = np.vstack(
        [hidden_by_lang[lang]["last_hidden"] for lang in languages]
    )
    n_per_lang = hidden_by_lang[languages[0]]["last_hidden"].shape[0]

    rng = np.random.default_rng(seed)
    null_means = np.zeros(N_PERMS)

    print(f"\n  {N_PERMS}-permutation test...")
    for perm_i in range(N_PERMS):
        shuffled_idx = rng.permutation(len(all_hidden))
        perm_analyzers = {}
        for k, lang in enumerate(languages):
            perm_cloud = all_hidden[
                shuffled_idx[k * n_per_lang : (k + 1) * n_per_lang]
            ]
            pa, _ = run_ph(
                perm_cloud, max_dim=1, subsample=SUBSAMPLE, n_pca=N_PCA, seed=seed
            )
            perm_analyzers[lang] = pa

        perm_wass = []
        for i in range(n_langs):
            for j in range(i + 1, n_langs):
                d = perm_analyzers[languages[i]].distance(
                    perm_analyzers[languages[j]], metric="wasserstein_1"
                )
                perm_wass.append(d)
        null_means[perm_i] = np.mean(perm_wass)

        if (perm_i + 1) % 50 == 0:
            print(f"    Permutation {perm_i+1}/{N_PERMS}")

    p_value = float((np.sum(null_means >= observed_mean) + 1) / (N_PERMS + 1))
    null_mean = float(np.mean(null_means))
    null_std = float(np.std(null_means, ddof=1))
    z_score = (
        float((observed_mean - null_mean) / null_std) if null_std > 1e-10 else 0.0
    )

    print(f"\n  Observed mean Wasserstein: {observed_mean:.2f}")
    print(f"  Null: {null_mean:.2f} ± {null_std:.2f}")
    print(f"  z={z_score:.2f}, p={p_value:.4f}")
    print(f"  Closest: {closest} ({min_d:.2f}), Farthest: {farthest} ({max_d:.2f})")

    return {
        "wasserstein_matrix": wass_matrix.tolist(),
        "languages": languages,
        "entropies": entropies,
        "observed_mean_wasserstein": float(observed_mean),
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p_value,
        "z_score": z_score,
        "closest_pair": closest,
        "farthest_pair": farthest,
        "null_scores": null_means,
    }


# ── Experiment 2: Per-problem topological fingerprint consistency ──────────


def run_exp2(hidden_by_lang, seed=42):
    """Per-problem topological feature correlation across language pairs."""
    print("\n=== Experiment 2: Per-problem topological fingerprint consistency ===")

    languages = list(hidden_by_lang.keys())
    n_problems = hidden_by_lang[languages[0]]["last_hidden"].shape[0]

    tfe = TopologicalFeatureExtractor(
        max_dim=1,
        n_pca_components=30,
        subsample=100,
        feature_set="summary",
        seed=seed,
    )

    features_by_lang = {}
    for lang in languages:
        print(f"  Extracting features for {lang}...")
        trajectories = hidden_by_lang[lang]["token_trajectories"]
        lang_features = []

        for i in range(n_problems):
            traj = trajectories[i]  # (T_i, d)
            if traj.shape[0] < 5:
                lang_features.append(np.zeros(tfe.n_features))
                continue
            try:
                feat = tfe.extract_single(traj)
                lang_features.append(feat)
            except Exception:
                lang_features.append(np.zeros(tfe.n_features))

        features_by_lang[lang] = np.array(lang_features)  # (N, n_features)
        print(f"    Shape: {features_by_lang[lang].shape}")

    # Pairwise per-problem correlation
    pair_correlations = {}
    for i, lang_i in enumerate(languages):
        for j, lang_j in enumerate(languages):
            if i < j:
                pair_key = f"{lang_i}-{lang_j}"
                corrs = []
                for k in range(n_problems):
                    f_i = features_by_lang[lang_i][k]
                    f_j = features_by_lang[lang_j][k]
                    if np.std(f_i) < 1e-10 or np.std(f_j) < 1e-10:
                        continue
                    r, _ = stats.pearsonr(f_i, f_j)
                    if not np.isnan(r):
                        corrs.append(r)

                mean_corr = float(np.mean(corrs)) if corrs else 0.0
                pair_correlations[pair_key] = mean_corr
                print(
                    f"    {pair_key}: r={mean_corr:.6f} ({len(corrs)} valid pairs)"
                )

    mean_cross_lang = float(np.mean(list(pair_correlations.values())))
    print(f"\n  Overall mean cross-language correlation: {mean_cross_lang:.6f}")

    return {
        "per_pair_correlation": pair_correlations,
        "mean_cross_lang_correlation": mean_cross_lang,
        "n_features": tfe.n_features,
        "feature_names": tfe.feature_names,
        "features_by_lang": features_by_lang,  # kept for plotting only
    }


# ── Experiment 3: Cross-lingual binding (EN-ZH) ───────────────────────────


def compute_binding(en_cloud, zh_cloud, seed=42):
    """Binding score via persistence image subtraction (joint vs marginals)."""
    n = en_cloud.shape[0]
    sub = min(SUBSAMPLE, n)

    # Joint: concatenate paired features → (N, 2d)
    joint = np.hstack([en_cloud, zh_cloud])
    n_pca_j = min(50, joint.shape[0] - 1, joint.shape[1])
    pca_j = PCA(n_components=n_pca_j)
    joint_pca = pca_j.fit_transform(joint)
    pa_joint = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa_joint.fit_transform(joint_pca, subsample=sub, seed=seed)
    pi_joint = pa_joint.to_image(resolution=50, sigma=0.1)

    # Marginal EN
    n_pca_m = min(50, en_cloud.shape[0] - 1, en_cloud.shape[1])
    pca_en = PCA(n_components=n_pca_m)
    en_pca = pca_en.fit_transform(en_cloud)
    pa_en = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa_en.fit_transform(en_pca, subsample=sub, seed=seed)
    pi_en = pa_en.to_image(resolution=50, sigma=0.1)

    # Marginal ZH
    pca_zh = PCA(n_components=n_pca_m)
    zh_pca = pca_zh.fit_transform(zh_cloud)
    pa_zh = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa_zh.fit_transform(zh_pca, subsample=sub, seed=seed)
    pi_zh = pa_zh.to_image(resolution=50, sigma=0.1)

    # Binding = sum of ||PI_joint[d] - avg(PI_en[d], PI_zh[d])|| over dimensions
    binding = 0.0
    for dim in range(min(len(pi_joint), len(pi_en), len(pi_zh))):
        avg_marginal = (pi_en[dim] + pi_zh[dim]) / 2.0
        residual = pi_joint[dim] - avg_marginal
        binding += float(np.linalg.norm(residual))

    return binding


def run_exp3(hidden_by_lang, seed=42):
    """Cross-lingual binding: EN-ZH matched vs shuffled."""
    print("\n=== Experiment 3: Cross-lingual binding (EN-ZH) ===")

    en_hidden = hidden_by_lang["en"]["last_hidden"]
    zh_hidden = hidden_by_lang["zh"]["last_hidden"]
    n = en_hidden.shape[0]

    observed_binding = compute_binding(en_hidden, zh_hidden, seed=seed)
    print(f"  Observed EN-ZH binding: {observed_binding:.4f}")

    # Surrogate: shuffle ZH to break problem correspondence
    rng = np.random.default_rng(seed)
    null_bindings = np.zeros(N_SURROGATES)

    print(f"  {N_SURROGATES}-surrogate test...")
    for s in range(N_SURROGATES):
        shuffled_zh = zh_hidden[rng.permutation(n)]
        null_bindings[s] = compute_binding(en_hidden, shuffled_zh, seed=seed + s + 1)
        if (s + 1) % 10 == 0:
            print(f"    Surrogate {s+1}/{N_SURROGATES}")

    null_mean = float(np.mean(null_bindings))
    null_std = float(np.std(null_bindings, ddof=1))
    z_score = (
        float((observed_binding - null_mean) / null_std) if null_std > 1e-10 else 0.0
    )
    p_value = float(
        (np.sum(null_bindings >= observed_binding) + 1) / (N_SURROGATES + 1)
    )

    print(f"  Null: {null_mean:.4f} ± {null_std:.4f}")
    print(f"  z={z_score:.2f}, p={p_value:.4f}")

    return {
        "en_zh_binding": float(observed_binding),
        "en_zh_null_binding": null_mean,
        "en_zh_binding_z": z_score,
        "en_zh_binding_p": p_value,
        "null_bindings": null_bindings,
    }


# ── Plotting ───────────────────────────────────────────────────────────────


def plot_exp1(exp1_results, save_dir):
    """Cross-language Wasserstein matrix + entropy + permutation test."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    languages = exp1_results["languages"]
    lang_labels = [LANG_NAMES[l] for l in languages]
    wass = np.array(exp1_results["wasserstein_matrix"])

    # Panel 1: Wasserstein heatmap
    ax = axes[0]
    im = ax.imshow(wass, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels(lang_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(lang_labels)
    for i in range(len(languages)):
        for j in range(len(languages)):
            ax.text(
                j,
                i,
                f"{wass[i,j]:.1f}",
                ha="center",
                va="center",
                color="white" if wass[i, j] > wass.max() * 0.6 else "black",
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, label="Wasserstein-1 distance")
    ax.set_title("Cross-Language Wasserstein Distances")

    # Panel 2: H1 entropy per language
    ax = axes[1]
    h1_vals = [exp1_results["entropies"][l]["H1"] for l in languages]
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    bars = ax.bar(lang_labels, h1_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("H1 Persistence Entropy")
    ax.set_title("H1 Entropy by Language")
    for bar, val in zip(bars, h1_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel 3: Permutation test
    ax = axes[2]
    null_scores = exp1_results["null_scores"]
    ax.hist(null_scores, bins=30, density=True, alpha=0.7, color="gray", label="Null")
    ax.axvline(
        exp1_results["observed_mean_wasserstein"],
        color="red",
        linewidth=2,
        label=f"Observed ({exp1_results['observed_mean_wasserstein']:.2f})",
    )
    ax.set_xlabel("Mean Wasserstein Distance")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Permutation Test (z={exp1_results['z_score']:.2f}, "
        f"p={exp1_results['p_value']:.3f})"
    )
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "exp1_cross_language_topology.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_exp2(exp2_results, save_dir):
    """Cross-language correlation matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    languages = LANGUAGES
    lang_labels = [LANG_NAMES[l] for l in languages]
    n = len(languages)

    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{languages[i]}-{languages[j]}"
            r = exp2_results["per_pair_correlation"].get(key, 0)
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    im = ax.imshow(corr_matrix, cmap="RdYlBu_r", vmin=-0.5, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(lang_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(lang_labels)

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{corr_matrix[i,j]:.3f}",
                ha="center",
                va="center",
                color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                fontsize=10,
            )

    plt.colorbar(im, ax=ax, label="Mean Pearson Correlation")
    ax.set_title(
        f"Per-Problem Topological Feature Correlation\n"
        f"(mean across pairs: {exp2_results['mean_cross_lang_correlation']:.3f})"
    )

    plt.tight_layout()
    path = os.path.join(save_dir, "exp2_cross_lang_correlation.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_exp3(exp3_results, save_dir):
    """Binding score bar chart + surrogate distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    observed = exp3_results["en_zh_binding"]
    null_mean = exp3_results["en_zh_null_binding"]

    # Bar chart
    ax = axes[0]
    ax.bar(
        ["Matched\n(EN-ZH)", "Shuffled\n(null)"],
        [observed, null_mean],
        color=["#2196F3", "#9E9E9E"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Binding Score")
    ax.set_title("Cross-Lingual Binding: Matched vs Shuffled")
    ax.text(0, observed + 0.005, f"{observed:.4f}", ha="center", va="bottom")
    ax.text(1, null_mean + 0.005, f"{null_mean:.4f}", ha="center", va="bottom")

    # Surrogate distribution
    ax = axes[1]
    null_bindings = exp3_results["null_bindings"]
    ax.hist(null_bindings, bins=20, density=True, alpha=0.7, color="gray", label="Null")
    ax.axvline(
        observed,
        color="red",
        linewidth=2,
        label=f"Observed ({observed:.4f})",
    )
    ax.set_xlabel("Binding Score")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Surrogate Test (z={exp3_results['en_zh_binding_z']:.2f}, "
        f"p={exp3_results['en_zh_binding_p']:.3f})"
    )
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "exp3_cross_lingual_binding.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_overview(exp1_results, exp2_results, exp3_results, save_dir):
    """4-panel overview."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    languages = exp1_results["languages"]
    lang_labels = [LANG_NAMES[l] for l in languages]

    # A — Wasserstein matrix
    ax = axes[0, 0]
    wass = np.array(exp1_results["wasserstein_matrix"])
    im = ax.imshow(wass, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels(lang_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(lang_labels)
    for i in range(len(languages)):
        for j in range(len(languages)):
            ax.text(
                j,
                i,
                f"{wass[i,j]:.1f}",
                ha="center",
                va="center",
                color="white" if wass[i, j] > wass.max() * 0.6 else "black",
                fontsize=8,
            )
    plt.colorbar(im, ax=ax, label="W-1")
    ax.set_title("A. Cross-Language Wasserstein")

    # B — Correlation matrix
    ax = axes[0, 1]
    n = len(languages)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{languages[i]}-{languages[j]}"
            r = exp2_results["per_pair_correlation"].get(key, 0)
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
    im = ax.imshow(corr_matrix, cmap="RdYlBu_r", vmin=-0.5, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(lang_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(lang_labels)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{corr_matrix[i,j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
                fontsize=8,
            )
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("B. Per-Problem Feature Correlation")

    # C — Permutation test
    ax = axes[1, 0]
    null_scores = exp1_results["null_scores"]
    ax.hist(null_scores, bins=25, density=True, alpha=0.7, color="gray")
    ax.axvline(exp1_results["observed_mean_wasserstein"], color="red", linewidth=2)
    ax.set_xlabel("Mean Wasserstein Distance")
    ax.set_ylabel("Density")
    ax.set_title(
        f"C. Permutation Test (z={exp1_results['z_score']:.2f}, "
        f"p={exp1_results['p_value']:.3f})"
    )

    # D — Binding
    ax = axes[1, 1]
    null_bindings = exp3_results["null_bindings"]
    ax.hist(null_bindings, bins=20, density=True, alpha=0.7, color="gray")
    ax.axvline(exp3_results["en_zh_binding"], color="red", linewidth=2)
    ax.set_xlabel("Binding Score")
    ax.set_ylabel("Density")
    ax.set_title(
        f"D. EN-ZH Binding (z={exp3_results['en_zh_binding_z']:.2f}, "
        f"p={exp3_results['en_zh_binding_p']:.3f})"
    )

    plt.tight_layout()
    path = os.path.join(save_dir, "overview.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Branch 6: Multilingual math topology"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Use cached NPZ files instead of re-extracting",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    t_start = time.time()

    # ── Load or extract hidden states ──
    if args.skip_extraction:
        print("Loading cached hidden states...")
        hidden_by_lang = load_cached_hidden_states(LANGUAGES)
        if hidden_by_lang is None:
            print("ERROR: Cached files not found. Run without --skip-extraction.")
            sys.exit(1)
    else:
        print("Loading MGSM dataset...")
        problems_by_lang, n_problems = load_mgsm(LANGUAGES)

        print(f"\nExtracting hidden states ({MODEL_ID})...")
        hidden_by_lang = extract_hidden_states(problems_by_lang, seed=args.seed)

    t_extract = time.time()
    extract_time = t_extract - t_start
    print(f"\nExtraction: {extract_time:.0f}s")

    n_problems = hidden_by_lang[LANGUAGES[0]]["last_hidden"].shape[0]
    hidden_dim = int(hidden_by_lang[LANGUAGES[0]]["last_hidden"].shape[1])

    # ── Run experiments ──
    exp1_results = run_exp1(hidden_by_lang, seed=args.seed)
    exp2_results = run_exp2(hidden_by_lang, seed=args.seed)
    exp3_results = run_exp3(hidden_by_lang, seed=args.seed)

    t_analysis = time.time()
    analysis_time = t_analysis - t_extract

    # ── Figures ──
    print("\nGenerating figures...")
    plot_exp1(exp1_results, FIGURES_DIR)
    plot_exp2(exp2_results, FIGURES_DIR)
    plot_exp3(exp3_results, FIGURES_DIR)
    plot_overview(exp1_results, exp2_results, exp3_results, FIGURES_DIR)

    # ── Save results ──
    results = {
        "branch": "experiment/tda-multilingual",
        "languages": LANGUAGES,
        "n_problems": n_problems,
        "model": MODEL_ID,
        "hidden_dim": hidden_dim,
        "exp1_wasserstein_matrix": exp1_results["wasserstein_matrix"],
        "exp1_permutation_p": exp1_results["p_value"],
        "exp1_z_score": exp1_results["z_score"],
        "exp1_closest_pair": exp1_results["closest_pair"],
        "exp1_farthest_pair": exp1_results["farthest_pair"],
        "exp1_observed_mean_wasserstein": exp1_results["observed_mean_wasserstein"],
        "exp1_entropies": exp1_results["entropies"],
        "exp2_mean_cross_lang_correlation": exp2_results["mean_cross_lang_correlation"],
        "exp2_per_pair_correlation": exp2_results["per_pair_correlation"],
        "exp3_en_zh_binding": exp3_results["en_zh_binding"],
        "exp3_en_zh_binding_p": exp3_results["en_zh_binding_p"],
        "exp3_en_zh_null_binding": exp3_results["en_zh_null_binding"],
        "runtime_seconds": float(time.time() - t_start),
        "extraction_seconds": float(extract_time),
        "analysis_seconds": float(analysis_time),
    }

    results_path = os.path.join(DATA_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset: MGSM, {n_problems} problems x {len(LANGUAGES)} languages")
    print(f"Model: {MODEL_ID}")
    print(f"\nExp 1 — Cross-language topology:")
    print(f"  Closest: {exp1_results['closest_pair']}")
    print(f"  Farthest: {exp1_results['farthest_pair']}")
    print(f"  Permutation: z={exp1_results['z_score']:.2f}, p={exp1_results['p_value']:.4f}")
    print(f"\nExp 2 — Per-problem fingerprint consistency:")
    print(f"  Mean cross-lang correlation: {exp2_results['mean_cross_lang_correlation']:.6f}")
    for pair, corr in exp2_results["per_pair_correlation"].items():
        print(f"    {pair}: r={corr:.6f}")
    print(f"\nExp 3 — Cross-lingual binding (EN-ZH):")
    print(f"  Binding: {exp3_results['en_zh_binding']:.4f}")
    print(f"  Null: {exp3_results['en_zh_null_binding']:.4f}")
    print(f"  z={exp3_results['en_zh_binding_z']:.2f}, p={exp3_results['en_zh_binding_p']:.3f}")
    print(f"\nTotal runtime: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
