#!/usr/bin/env python3
"""Branch 5: Code Comprehension — Topology of LLM Representations on Programming Tasks.

Replicates the MATH-500 topology analysis on code problems (HumanEval) to test
whether non-monotonic H1 entropy and difficulty-dependent topology transfer to
a different reasoning domain.

Three experiments:
  1. Point cloud PH by code difficulty — PCA→50, PH max_dim=2, persistence
     entropy by difficulty bin. Tests non-monotonicity and level-1 minimum.
  2. MATH vs Code topology comparison — Wasserstein distances between domains
     to test whether hard-math and hard-code share topological structure.
  3. Correctness prediction — logistic regression on topological features to
     predict code generation success. Reports AUROC.

Usage:
    python scripts/branches/code_comprehension_topology.py
    python scripts/branches/code_comprehension_topology.py --skip-extraction
"""

import argparse
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from itertools import combinations

import numpy as np

print = functools.partial(print, flush=True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from att.topology.persistence import PersistenceAnalyzer

DATA_DIR = os.path.join(REPO_ROOT, "data", "code")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "code")
MATH_DATA_DIR = os.path.join(REPO_ROOT, "data", "transformer")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 512
MAX_NEW_TOKENS = 256
N_PCA = 50
SUBSAMPLE = 200
N_PERMS = 200
SEED = 42

PROMPT_TEMPLATE = "{prompt}"  # raw function signature — model continues the body


# ============================================================================
# Difficulty binning
# ============================================================================

def compute_complexity(canonical_solution: str) -> float:
    """Compute complexity score from canonical solution text."""
    lines = [l for l in canonical_solution.strip().split("\n") if l.strip()]
    n_lines = len(lines)
    code = canonical_solution
    n_control = sum(
        code.count(kw)
        for kw in ["if ", "for ", "while ", "try:", "except ", "with ", "elif "]
    )
    # Count nesting depth — lines with 2+ levels of indent inside solution body
    n_nested = 0
    for line in lines:
        stripped = line.rstrip()
        if stripped:
            leading = len(stripped) - len(stripped.lstrip())
            if leading >= 8:  # double indent = nested control flow
                n_nested += 1
    n_calls = len(re.findall(r"\w+\(", code))
    return n_lines + 2.0 * n_control + 3.0 * n_nested + 0.3 * n_calls


def load_humaneval_with_difficulty():
    """Load HumanEval and assign 3-level difficulty from solution complexity."""
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    print(f"Loaded {len(ds)} problems from openai/openai_humaneval")

    problems = []
    for row in ds:
        problems.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        })

    # Compute complexity scores and bin into 3 difficulty levels
    scores = [compute_complexity(p["canonical_solution"]) for p in problems]
    order = np.argsort(scores)
    n = len(problems)
    for rank, idx in enumerate(order):
        if rank < n // 3:
            problems[idx]["difficulty"] = 1  # easy
        elif rank < 2 * n // 3:
            problems[idx]["difficulty"] = 2  # medium
        else:
            problems[idx]["difficulty"] = 3  # hard
        problems[idx]["complexity_score"] = scores[idx]

    for d, label in [(1, "easy"), (2, "medium"), (3, "hard")]:
        count = sum(1 for p in problems if p["difficulty"] == d)
        mean_score = np.mean([p["complexity_score"] for p in problems if p["difficulty"] == d])
        print(f"  {label} (level {d}): {count} problems, mean complexity={mean_score:.1f}")

    return problems


# ============================================================================
# Code execution
# ============================================================================

def execute_code_safe(code: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Execute generated code in subprocess with timeout. Returns (success, msg)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        output = (result.stdout + result.stderr)[:500]
        return success, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ============================================================================
# Phase 1: Extraction
# ============================================================================

def extract_and_evaluate(problems, seed=SEED):
    """Extract hidden states from Qwen and evaluate code correctness."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    print(f"Model loaded: {MODEL_ID}")

    all_last_hidden = []
    all_layer_hidden = []
    all_token_trajectories = []
    all_levels = []
    all_seq_lengths = []
    all_hashes = []
    all_correct = []
    all_generated = []
    skipped = []

    t0 = time.time()
    for idx, prob in enumerate(problems):
        if idx % 20 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(problems) - idx) / rate if rate > 0 else 0
            print(f"  [{idx}/{len(problems)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s left")

        prompt_text = PROMPT_TEMPLATE.format(prompt=prob["prompt"])
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)
        seq_len = inputs["input_ids"].shape[1]

        try:
            # -- Hidden state extraction --
            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.hidden_states  # tuple of (1, seq, d)
            n_layers_p1 = len(hidden_states)

            final_layer = hidden_states[-1][0]  # (seq, d)
            last_hidden = final_layer[-1].cpu().float().numpy()  # (d,)
            layer_states = (
                torch.stack([hidden_states[i][0, -1, :] for i in range(n_layers_p1)])
                .cpu()
                .float()
                .numpy()
            )  # (L+1, d)
            token_traj = final_layer.cpu().float().numpy()  # (seq, d)

            # -- Code generation (use raw prompt — model continues function body) --
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                )
            generated = tokenizer.decode(
                gen_out[0][seq_len:], skip_special_tokens=True
            )

            # Post-process: strip markdown, stop at next top-level def/class
            gen_clean = generated
            if "```" in gen_clean:
                # Extract code from markdown block
                parts = gen_clean.split("```")
                if len(parts) >= 2:
                    code_block = parts[1]
                    if code_block.startswith("python"):
                        code_block = code_block[6:]
                    gen_clean = code_block
            # Stop at next top-level definition (new function/class)
            lines = gen_clean.split("\n")
            cut_lines = []
            for line in lines:
                if cut_lines and (
                    line.startswith("def ") or line.startswith("class ")
                ):
                    break
                cut_lines.append(line)
            gen_clean = "\n".join(cut_lines)

            # -- Correctness check via execution --
            full_code = (
                prob["prompt"]
                + gen_clean
                + "\n"
                + prob["test"]
                + f"\ncheck({prob['entry_point']})\n"
            )
            is_correct, _ = execute_code_safe(full_code, timeout=10.0)

            all_last_hidden.append(last_hidden)
            all_layer_hidden.append(layer_states)
            all_token_trajectories.append(token_traj)
            all_levels.append(prob["difficulty"])
            all_seq_lengths.append(seq_len)
            all_hashes.append(
                hashlib.sha256(prob["prompt"].encode()).hexdigest()[:16]
            )
            all_correct.append(is_correct)
            all_generated.append(generated[:500])

        except torch.cuda.OutOfMemoryError:
            print(f"    OOM on {idx} (seq_len={seq_len}), skipping")
            skipped.append(idx)
            torch.cuda.empty_cache()
            continue

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    n_ok = len(all_last_hidden)
    n_correct = sum(all_correct)
    print(
        f"\nExtraction done: {n_ok}/{len(problems)} in {elapsed:.1f}s. "
        f"Skipped: {len(skipped)}. Correct: {n_correct}/{n_ok} = {n_correct/n_ok:.1%}"
    )

    d = all_last_hidden[0].shape[0]
    n_layers = all_layer_hidden[0].shape[0]
    print(f"Hidden dim: {d}, Layers (incl. embedding): {n_layers}")

    levels_arr = np.array(all_levels)
    correct_arr = np.array(all_correct)
    for lv, label in [(1, "easy"), (2, "medium"), (3, "hard")]:
        mask = levels_arr == lv
        if mask.sum():
            acc = correct_arr[mask].mean()
            print(f"  {label}: {correct_arr[mask].sum()}/{mask.sum()} = {acc:.1%}")

    npz_path = os.path.join(DATA_DIR, "code_hidden_states.npz")
    np.savez_compressed(
        npz_path,
        last_hidden_states=np.array(all_last_hidden),
        difficulty_levels=levels_arr,
        layer_hidden_states=np.array(all_layer_hidden),
        token_trajectories=np.array(all_token_trajectories, dtype=object),
        seq_lengths=np.array(all_seq_lengths),
        problem_hashes=np.array(all_hashes),
        correct=correct_arr,
        model_name=np.array(MODEL_ID),
        skipped_indices=np.array(skipped),
        hidden_dim=np.array(d),
        num_layers=np.array(n_layers),
        dataset_source=np.array("openai/openai_humaneval"),
    )
    print(f"Saved: {npz_path}")
    return npz_path


# ============================================================================
# TDA helpers
# ============================================================================

def run_ph(cloud, max_dim=2, subsample=SUBSAMPLE, n_pca=N_PCA, seed=SEED):
    """PCA + PH on a point cloud. Returns PersistenceAnalyzer."""
    n = cloud.shape[0]
    if n < 3:
        return None
    n_comp = min(n_pca, n - 1, cloud.shape[1])
    pca = PCA(n_components=n_comp)
    cloud_pca = pca.fit_transform(cloud)
    pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser")
    pa.fit_transform(cloud_pca, subsample=min(n, subsample), seed=seed)
    return pa


def persistence_entropy(diagrams, dim):
    """Persistence entropy for a single homology dimension."""
    dgm = diagrams[dim] if dim < len(diagrams) else np.empty((0, 2))
    if len(dgm) == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    total = lifetimes.sum()
    probs = lifetimes / total
    return float(-np.sum(probs * np.log(probs + 1e-15)))


def n_features(diagrams, dim):
    """Number of H_dim features with positive lifetime."""
    dgm = diagrams[dim] if dim < len(diagrams) else np.empty((0, 2))
    if len(dgm) == 0:
        return 0
    return int(np.sum((dgm[:, 1] - dgm[:, 0]) > 0))


def total_persistence(diagrams, dim):
    """Sum of lifetimes for H_dim."""
    dgm = diagrams[dim] if dim < len(diagrams) else np.empty((0, 2))
    if len(dgm) == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    return float(lifetimes.sum())


# ============================================================================
# Experiment 1: Point cloud PH by code difficulty
# ============================================================================

def run_exp1(last_hidden, levels, n_perms=N_PERMS, seed=SEED):
    """PH per difficulty level with permutation test."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Point cloud PH by code difficulty")
    print("=" * 60)

    unique_levels = sorted(np.unique(levels))
    n_levels = len(unique_levels)
    label_map = {1: "easy", 2: "medium", 3: "hard"}

    # PH per level
    analyzers = {}
    results = {}
    for lv in unique_levels:
        mask = levels == lv
        cloud = last_hidden[mask]
        print(f"\n  Level {lv} ({label_map.get(lv, lv)}): {cloud.shape[0]} points in R^{cloud.shape[1]}")
        pa = run_ph(cloud, max_dim=2)
        analyzers[lv] = pa
        if pa is not None and pa.diagrams_ is not None:
            dgms = pa.diagrams_
            h0_e = persistence_entropy(dgms, 0)
            h1_e = persistence_entropy(dgms, 1)
            h2_e = persistence_entropy(dgms, 2) if len(dgms) > 2 else 0.0
            h1_n = n_features(dgms, 1)
            h1_tp = total_persistence(dgms, 1)
            results[lv] = {
                "h0_entropy": h0_e,
                "h1_entropy": h1_e,
                "h2_entropy": h2_e,
                "h1_features": h1_n,
                "h1_total_persistence": h1_tp,
            }
            print(f"    H0 entropy={h0_e:.3f}, H1 entropy={h1_e:.3f}, H2 entropy={h2_e:.3f}")
            print(f"    H1 features={h1_n}, H1 total persistence={h1_tp:.3f}")

    # Check non-monotonicity
    h1_entropies = [results[lv]["h1_entropy"] for lv in unique_levels]
    is_non_monotonic = not (
        all(a <= b for a, b in zip(h1_entropies, h1_entropies[1:]))
        or all(a >= b for a, b in zip(h1_entropies, h1_entropies[1:]))
    )
    level1_min = h1_entropies[0] == min(h1_entropies)
    print(f"\n  H1 entropy sequence: {['%.3f' % e for e in h1_entropies]}")
    print(f"  Non-monotonic: {is_non_monotonic}")
    print(f"  Level-1 (easy) is minimum: {level1_min}")

    # Pairwise Wasserstein distance matrix
    dist_matrix = np.zeros((n_levels, n_levels))
    for i, j in combinations(range(n_levels), 2):
        li, lj = unique_levels[i], unique_levels[j]
        if analyzers[li] is not None and analyzers[lj] is not None:
            d = analyzers[li].distance(analyzers[lj], metric="wasserstein_1")
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    print("\n  Wasserstein distance matrix:")
    header = "        " + "  ".join([f"{label_map[lv]:>8s}" for lv in unique_levels])
    print(header)
    for i, li in enumerate(unique_levels):
        row = f"  {label_map[li]:>6s}" + "  ".join(
            [f"{dist_matrix[i, j]:>8.4f}" for j in range(n_levels)]
        )
        print(row)

    # Permutation test
    print(f"\n  Permutation test (N={n_perms})...")
    observed_mean = dist_matrix[np.triu_indices(n_levels, k=1)].mean()
    rng = np.random.default_rng(seed)
    null_dists = []

    for pi in range(n_perms):
        if pi % 50 == 0:
            print(f"    Perm {pi}/{n_perms}...")
        shuffled = rng.permutation(levels)
        perm_analyzers = {}
        for lv in unique_levels:
            mask = shuffled == lv
            cloud = last_hidden[mask]
            perm_analyzers[lv] = run_ph(cloud, max_dim=2, seed=seed)

        perm_dists_list = []
        for i, j in combinations(range(n_levels), 2):
            li, lj = unique_levels[i], unique_levels[j]
            if perm_analyzers[li] is not None and perm_analyzers[lj] is not None:
                d = perm_analyzers[li].distance(perm_analyzers[lj], metric="wasserstein_1")
                perm_dists_list.append(d)
        if perm_dists_list:
            null_dists.append(np.mean(perm_dists_list))

    null_dists = np.array(null_dists)
    p_value = (np.sum(null_dists >= observed_mean) + 1) / (len(null_dists) + 1)
    z_score = (observed_mean - null_dists.mean()) / (null_dists.std() + 1e-15)

    print(f"\n  Observed mean Wasserstein: {observed_mean:.4f}")
    print(f"  Null: mean={null_dists.mean():.4f}, std={null_dists.std():.4f}")
    print(f"  z-score: {z_score:.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant (p<0.05): {p_value < 0.05}")

    return {
        "per_level": results,
        "h1_entropies": {label_map[lv]: h1_entropies[i] for i, lv in enumerate(unique_levels)},
        "non_monotonic": is_non_monotonic,
        "level1_minimum": level1_min,
        "wasserstein_matrix": dist_matrix.tolist(),
        "observed_mean_wasserstein": float(observed_mean),
        "null_mean": float(null_dists.mean()),
        "null_std": float(null_dists.std()),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "null_dists": null_dists,
        "analyzers": analyzers,
    }


# ============================================================================
# Experiment 2: MATH vs Code topology comparison
# ============================================================================

def run_exp2(code_last_hidden, code_levels, seed=SEED):
    """Cross-domain Wasserstein comparison between MATH and Code."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: MATH vs Code topology comparison")
    print("=" * 60)

    math_path = os.path.join(MATH_DATA_DIR, "math500_hidden_states_aligned.npz")
    if not os.path.exists(math_path):
        math_path = os.path.join(MATH_DATA_DIR, "math500_hidden_states.npz")
    if not os.path.exists(math_path):
        print("  WARNING: No MATH hidden states found. Skipping Exp 2.")
        return None

    math_data = np.load(math_path, allow_pickle=True)
    math_last_hidden = math_data["last_hidden_states"]
    math_levels = math_data["difficulty_levels"]
    print(f"  MATH: {math_last_hidden.shape[0]} problems, dim={math_last_hidden.shape[1]}")
    print(f"  Code: {code_last_hidden.shape[0]} problems, dim={code_last_hidden.shape[1]}")

    # Define groups
    math_easy_mask = math_levels == 1
    math_hard_mask = math_levels == 5
    code_easy_mask = code_levels == 1
    code_hard_mask = code_levels == 3

    groups = {
        "math_easy": math_last_hidden[math_easy_mask],
        "math_hard": math_last_hidden[math_hard_mask],
        "code_easy": code_last_hidden[code_easy_mask],
        "code_hard": code_last_hidden[code_hard_mask],
    }

    for name, cloud in groups.items():
        print(f"  {name}: {cloud.shape[0]} points")

    # Fit PCA on combined data (shared embedding space)
    combined = np.vstack(list(groups.values()))
    n_comp = min(N_PCA, combined.shape[0] - 1, combined.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(combined)
    print(f"  PCA: {n_comp} components on {combined.shape[0]} combined points")

    # PH on each group in shared PCA space
    group_analyzers = {}
    for name, cloud in groups.items():
        cloud_pca = pca.transform(cloud)
        pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
        pa.fit_transform(cloud_pca, subsample=min(cloud.shape[0], SUBSAMPLE), seed=seed)
        group_analyzers[name] = pa
        dgms = pa.diagrams_
        h1_e = persistence_entropy(dgms, 1)
        h1_n = n_features(dgms, 1)
        print(f"  {name}: H1 entropy={h1_e:.3f}, H1 features={h1_n}")

    # Compute Wasserstein distances for key comparisons
    comparisons = [
        ("math_easy", "code_easy", "Same difficulty, different domain"),
        ("math_hard", "code_hard", "Same difficulty, different domain"),
        ("math_easy", "code_hard", "Cross: easy math vs hard code"),
        ("math_hard", "code_easy", "Cross: hard math vs easy code"),
        ("math_easy", "math_hard", "Within MATH"),
        ("code_easy", "code_hard", "Within Code"),
    ]

    comp_results = {}
    print("\n  Cross-domain Wasserstein distances:")
    for g1, g2, desc in comparisons:
        d = group_analyzers[g1].distance(group_analyzers[g2], metric="wasserstein_1")
        comp_results[f"{g1}_vs_{g2}"] = float(d)
        print(f"    {g1} vs {g2}: {d:.4f}  ({desc})")

    return {
        "comparisons": comp_results,
        "n_per_group": {name: int(cloud.shape[0]) for name, cloud in groups.items()},
        "pca_components": n_comp,
        "group_analyzers": group_analyzers,
    }


# ============================================================================
# Experiment 3: Correctness prediction
# ============================================================================

def run_exp3(code_data_path, seed=SEED):
    """Logistic regression on TDA features → predict code correctness."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Correctness prediction on code")
    print("=" * 60)

    from att.llm.features import TopologicalFeatureExtractor

    data = np.load(code_data_path, allow_pickle=True)
    correct = data["correct"]
    token_trajectories = data["token_trajectories"]
    levels = data["difficulty_levels"]

    n_correct = correct.sum()
    n_total = len(correct)
    print(f"  {n_correct}/{n_total} correct ({n_correct / n_total:.1%})")

    if n_correct < 5 or (n_total - n_correct) < 5:
        print("  Too few of one class for meaningful prediction. Skipping.")
        return {
            "auroc": 0.5,
            "accuracy": float(max(n_correct, n_total - n_correct) / n_total),
            "n_problems": int(n_total),
            "n_correct": int(n_correct),
            "skipped": True,
            "top_feature": "N/A",
        }

    # Extract TDA features per problem from token trajectories
    print("  Extracting topological features per problem...")
    tfe = TopologicalFeatureExtractor(
        max_dim=1,
        n_pca_components=30,
        subsample=100,
        feature_set="summary",
        seed=seed,
    )

    n = len(token_trajectories)
    n_feat = tfe.n_features
    feature_names = tfe.feature_names
    X = np.zeros((n, n_feat))

    for i in range(n):
        traj = token_trajectories[i]
        if traj is not None and len(traj) >= 3:
            X[i] = tfe.extract_single(traj)
        if i % 30 == 0:
            print(f"    [{i}/{n}]")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = correct.astype(bool)

    print(f"  Feature matrix: {X.shape}, {n_feat} features")

    # 5-fold stratified CV
    print("  Running 5-fold CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aurocs, accs = [], []
    all_y_true, all_y_prob = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=seed)
        clf.fit(X_tr_s, y_tr)

        y_prob = clf.predict_proba(X_te_s)[:, 1]
        y_pred = clf.predict(X_te_s)

        if len(np.unique(y_te)) > 1:
            aurocs.append(roc_auc_score(y_te, y_prob))
        accs.append(accuracy_score(y_te, y_pred))
        all_y_true.extend(y_te)
        all_y_prob.extend(y_prob)

    mean_auroc = float(np.mean(aurocs)) if aurocs else 0.5
    mean_acc = float(np.mean(accs))
    print(f"  AUROC: {mean_auroc:.3f} (±{np.std(aurocs):.3f})")
    print(f"  Accuracy: {mean_acc:.3f} (±{np.std(accs):.3f})")

    # Feature importance from full model
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf_full = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=seed)
    clf_full.fit(X_s, y)
    importance = np.abs(clf_full.coef_[0])
    top_idx = np.argsort(importance)[::-1]

    print("  Top 5 features:")
    for i in top_idx[:5]:
        print(f"    {feature_names[i]}: {importance[i]:.4f}")

    top_feature = feature_names[top_idx[0]]

    return {
        "auroc": mean_auroc,
        "auroc_std": float(np.std(aurocs)) if aurocs else 0.0,
        "accuracy": mean_acc,
        "accuracy_std": float(np.std(accs)),
        "n_problems": int(n_total),
        "n_correct": int(n_correct),
        "skipped": False,
        "top_feature": top_feature,
        "feature_importance": {
            feature_names[i]: float(importance[i]) for i in top_idx[:10]
        },
        "all_y_true": np.array(all_y_true),
        "all_y_prob": np.array(all_y_prob),
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_exp1(exp1_results, out_dir):
    """Plot Exp 1: PH by difficulty."""
    per_level = exp1_results["per_level"]
    levels_sorted = sorted(per_level.keys())
    label_map = {1: "easy", 2: "medium", 3: "hard"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: H0/H1/H2 entropy by difficulty
    ax = axes[0, 0]
    for dim, color, marker in [(0, "tab:blue", "o"), (1, "tab:red", "s"), (2, "tab:green", "^")]:
        key = f"h{dim}_entropy"
        vals = [per_level[lv][key] for lv in levels_sorted]
        ax.plot(
            [label_map[lv] for lv in levels_sorted],
            vals,
            f"-{marker}",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"H{dim}",
        )
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Persistence Entropy")
    ax.set_title("Persistence Entropy by Difficulty (Code)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: H1 features and total persistence
    ax = axes[0, 1]
    h1_feat = [per_level[lv]["h1_features"] for lv in levels_sorted]
    h1_tp = [per_level[lv]["h1_total_persistence"] for lv in levels_sorted]
    x = range(len(levels_sorted))
    ax.bar(
        [i - 0.2 for i in x], h1_feat, 0.35, label="H1 features", color="tab:blue", alpha=0.8
    )
    ax2 = ax.twinx()
    ax2.bar(
        [i + 0.2 for i in x],
        h1_tp,
        0.35,
        label="H1 total persistence",
        color="tab:orange",
        alpha=0.8,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([label_map[lv] for lv in levels_sorted])
    ax.set_ylabel("H1 Feature Count", color="tab:blue")
    ax2.set_ylabel("H1 Total Persistence", color="tab:orange")
    ax.set_title("H1 Topological Complexity (Code)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Panel 3: Wasserstein distance matrix
    ax = axes[1, 0]
    wass_mat = np.array(exp1_results["wasserstein_matrix"])
    n = len(levels_sorted)
    im = ax.imshow(wass_mat, cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(n))
    ax.set_xticklabels([label_map[lv] for lv in levels_sorted])
    ax.set_yticks(range(n))
    ax.set_yticklabels([label_map[lv] for lv in levels_sorted])
    ax.set_title("Pairwise Wasserstein Distance (Code)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(n):
            val = wass_mat[i, j]
            color = "white" if val > wass_mat.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10, color=color)

    # Panel 4: Permutation test null
    ax = axes[1, 1]
    null_dists = exp1_results["null_dists"]
    observed = exp1_results["observed_mean_wasserstein"]
    ax.hist(null_dists, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(observed, color="red", linewidth=2, label=f"Observed={observed:.4f}")
    p95 = np.percentile(null_dists, 95)
    ax.axvline(p95, color="orange", linewidth=1.5, linestyle="--", label=f"95th={p95:.4f}")
    ax.set_xlabel("Mean Pairwise Wasserstein Distance")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Permutation Null (n={len(null_dists)}), p={exp1_results['p_value']:.4f}"
    )
    ax.legend()

    fig.suptitle("Exp 1: Code Difficulty Topology", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "exp1_code_difficulty_topology.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_exp2(exp2_results, out_dir):
    """Plot Exp 2: Cross-domain comparison."""
    if exp2_results is None:
        return

    comps = exp2_results["comparisons"]

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    values = []
    colors = []

    color_map = {
        "same_diff": "tab:blue",     # same difficulty, different domain
        "cross": "tab:red",          # cross difficulty + domain
        "within": "tab:green",       # within domain
    }

    for key, val in comps.items():
        labels.append(key.replace("_vs_", "\nvs\n"))
        values.append(val)
        if "math_easy_vs_code_easy" in key or "math_hard_vs_code_hard" in key:
            colors.append(color_map["same_diff"])
        elif "math" in key and "code" in key:
            colors.append(color_map["cross"])
        else:
            colors.append(color_map["within"])

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Wasserstein-1 Distance")
    ax.set_title("Cross-Domain Topology: MATH vs Code (H1)")
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map["same_diff"], label="Same difficulty, diff domain"),
        Patch(facecolor=color_map["cross"], label="Cross difficulty + domain"),
        Patch(facecolor=color_map["within"], label="Within domain"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    path = os.path.join(out_dir, "exp2_math_vs_code_topology.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_exp3(exp3_results, out_dir):
    """Plot Exp 3: correctness prediction ROC + feature importance."""
    if exp3_results.get("skipped"):
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    ax = axes[0]
    y_true = exp3_results["all_y_true"]
    y_prob = exp3_results["all_y_prob"]
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC={exp3_results['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Code Correctness Prediction (TDA Features)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Feature importance
    ax = axes[1]
    feat_imp = exp3_results.get("feature_importance", {})
    if feat_imp:
        names = list(feat_imp.keys())[:10]
        vals = [feat_imp[n] for n in names]
        ax.barh(range(len(names)), vals[::-1], color="tab:blue", alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_xlabel("|Coefficient|")
        ax.set_title("Feature Importance")
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Exp 3: Correctness Prediction from Topology", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "exp3_correctness_prediction.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_overview(exp1_results, exp2_results, exp3_results, out_dir):
    """4-panel overview figure."""
    label_map = {1: "easy", 2: "medium", 3: "hard"}
    per_level = exp1_results["per_level"]
    levels_sorted = sorted(per_level.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: H1 entropy by difficulty
    ax = axes[0, 0]
    h1_e = [per_level[lv]["h1_entropy"] for lv in levels_sorted]
    ax.plot(
        [label_map[lv] for lv in levels_sorted],
        h1_e,
        "-o",
        color="tab:red",
        linewidth=2,
        markersize=10,
    )
    ax.set_ylabel("H1 Persistence Entropy")
    ax.set_title(f"H1 Entropy by Difficulty (p={exp1_results['p_value']:.3f})")
    ax.grid(True, alpha=0.3)
    nm_text = "Non-monotonic" if exp1_results["non_monotonic"] else "Monotonic"
    ax.annotate(nm_text, xy=(0.05, 0.95), xycoords="axes fraction", fontsize=11, fontweight="bold")

    # Panel 2: Cross-domain comparison
    ax = axes[0, 1]
    if exp2_results is not None:
        comps = exp2_results["comparisons"]
        keys_show = [
            "math_easy_vs_code_easy",
            "math_hard_vs_code_hard",
            "math_easy_vs_code_hard",
            "code_easy_vs_code_hard",
        ]
        short_labels = ["M-easy\nvs\nC-easy", "M-hard\nvs\nC-hard", "M-easy\nvs\nC-hard", "C-easy\nvs\nC-hard"]
        vals = [comps.get(k, 0) for k in keys_show]
        colors = ["tab:blue", "tab:blue", "tab:red", "tab:green"]
        ax.bar(range(len(vals)), vals, color=colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_ylabel("Wasserstein-1 Distance")
        ax.set_title("Cross-Domain Topology")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No MATH data\navailable", ha="center", va="center", fontsize=14)
        ax.set_title("Cross-Domain Topology (skipped)")

    # Panel 3: Permutation null
    ax = axes[1, 0]
    null_dists = exp1_results["null_dists"]
    observed = exp1_results["observed_mean_wasserstein"]
    ax.hist(null_dists, bins=25, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(observed, color="red", linewidth=2, label=f"Observed={observed:.4f}")
    ax.set_xlabel("Mean Wasserstein")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test (z={exp1_results['z_score']:.2f})")
    ax.legend()

    # Panel 4: ROC curve
    ax = axes[1, 1]
    if not exp3_results.get("skipped"):
        y_true = exp3_results["all_y_true"]
        y_prob = exp3_results["all_y_prob"]
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax.plot(fpr, tpr, "b-", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title(f"Code AUROC={exp3_results['auroc']:.3f}")
    else:
        ax.text(0.5, 0.5, "Skipped\n(class imbalance)", ha="center", va="center", fontsize=14)
        ax.set_title("Correctness Prediction (skipped)")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Branch 5: Code Comprehension Topology (HumanEval / Qwen2.5-1.5B)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "overview.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Branch 5: Code comprehension topology")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip GPU extraction")
    parser.add_argument("--n-perms", type=int, default=N_PERMS)
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    t_start = time.time()
    npz_path = os.path.join(DATA_DIR, "code_hidden_states.npz")

    # ---- Phase 1: Extraction ----
    if args.skip_extraction and os.path.exists(npz_path):
        print(f"Skipping extraction, loading {npz_path}")
    else:
        print("=" * 60)
        print("PHASE 1: Extract hidden states + evaluate correctness")
        print("=" * 60)
        problems = load_humaneval_with_difficulty()
        npz_path = extract_and_evaluate(problems, seed=args.seed)

    # ---- Load extracted data ----
    data = np.load(npz_path, allow_pickle=True)
    last_hidden = data["last_hidden_states"]
    levels = data["difficulty_levels"]
    correct = data["correct"]
    n_total = len(levels)

    label_map = {1: "easy", 2: "medium", 3: "hard"}
    print(f"\nLoaded: {n_total} problems, dim={last_hidden.shape[1]}")
    for lv in sorted(np.unique(levels)):
        mask = levels == lv
        n_lv = mask.sum()
        n_corr = correct[mask].sum()
        print(f"  {label_map.get(lv, lv)}: {n_lv} problems, {n_corr}/{n_lv} correct ({n_corr/n_lv:.1%})")

    # ---- Phase 2: Experiments ----
    exp1_results = run_exp1(last_hidden, levels, n_perms=args.n_perms, seed=args.seed)
    exp2_results = run_exp2(last_hidden, levels, seed=args.seed)
    exp3_results = run_exp3(npz_path, seed=args.seed)

    # ---- Figures ----
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    plot_exp1(exp1_results, FIGURES_DIR)
    plot_exp2(exp2_results, FIGURES_DIR)
    plot_exp3(exp3_results, FIGURES_DIR)
    plot_overview(exp1_results, exp2_results, exp3_results, FIGURES_DIR)

    # ---- Save results JSON ----
    runtime = time.time() - t_start

    # Prepare serializable results
    exp1_serial = {k: v for k, v in exp1_results.items() if k not in ("null_dists", "analyzers")}
    exp2_serial = None
    if exp2_results is not None:
        exp2_serial = {k: v for k, v in exp2_results.items() if k != "group_analyzers"}
    exp3_serial = {k: v for k, v in exp3_results.items() if k not in ("all_y_true", "all_y_prob")}

    results = {
        "branch": "experiment/tda-code",
        "dataset": "openai/openai_humaneval",
        "n_problems": int(n_total),
        "difficulty_bins": {
            label_map[lv]: int((levels == lv).sum()) for lv in sorted(np.unique(levels))
        },
        "correctness": {
            "overall": float(correct.mean()),
            "per_difficulty": {
                label_map[lv]: float(correct[levels == lv].mean())
                for lv in sorted(np.unique(levels))
            },
        },
        "exp1_h1_entropy_by_difficulty": exp1_results["h1_entropies"],
        "exp1_level1_minimum": exp1_results["level1_minimum"],
        "exp1_non_monotonic": exp1_results["non_monotonic"],
        "exp1_permutation_p": exp1_results["p_value"],
        "exp1_z_score": exp1_results["z_score"],
        "exp2_math_easy_code_easy_wasserstein": (
            exp2_serial["comparisons"]["math_easy_vs_code_easy"] if exp2_serial else None
        ),
        "exp2_math_hard_code_hard_wasserstein": (
            exp2_serial["comparisons"]["math_hard_vs_code_hard"] if exp2_serial else None
        ),
        "exp2_math_easy_code_hard_wasserstein": (
            exp2_serial["comparisons"]["math_easy_vs_code_hard"] if exp2_serial else None
        ),
        "exp3_correctness_auroc": exp3_results["auroc"],
        "exp3_top_feature": exp3_results["top_feature"],
        "runtime_seconds": round(runtime, 1),
    }

    results_path = os.path.join(DATA_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset: HumanEval ({n_total} problems)")
    print(f"Model: {MODEL_ID}")
    print(f"Correctness: {correct.sum()}/{n_total} = {correct.mean():.1%}")
    print(f"\nExp 1 — Difficulty topology:")
    for lv in sorted(exp1_results["h1_entropies"]):
        print(f"  {lv}: H1 entropy = {exp1_results['h1_entropies'][lv]:.3f}")
    print(f"  Non-monotonic: {exp1_results['non_monotonic']}")
    print(f"  Level-1 minimum: {exp1_results['level1_minimum']}")
    print(f"  Permutation z={exp1_results['z_score']:.2f}, p={exp1_results['p_value']:.4f}")
    if exp2_serial:
        print(f"\nExp 2 — Cross-domain Wasserstein:")
        for k, v in exp2_serial["comparisons"].items():
            print(f"  {k}: {v:.4f}")
    print(f"\nExp 3 — Correctness AUROC: {exp3_results['auroc']:.3f}")
    print(f"  Top feature: {exp3_results['top_feature']}")
    print(f"\nRuntime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
