"""Branch B: Simplicial Complex Comparison — VR vs Directed Flag vs Dowker.

Head-to-head comparison of three simplicial complex constructions on
attention matrices from Qwen2.5-1.5B on MATH-500. Tests whether preserving
attention asymmetry produces more discriminative topological features.

Usage:
    python scripts/branches/complex_compare.py
    python scripts/branches/complex_compare.py --skip-extraction  # if raw matrices already saved
"""

import argparse
import json
import os
import sys
import time
import warnings
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "complex_compare")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "complex_compare")

PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_math_subset(n_per_level: int = 10, seed: int = 42):
    """Load MATH-500 and sample n_per_level per difficulty."""
    from datasets import load_dataset

    ds = None
    for name, loader in [
        ("HuggingFaceH4/MATH-500", lambda: load_dataset("HuggingFaceH4/MATH-500", split="test")),
        ("hendrycks/competition_math", lambda: load_dataset("hendrycks/competition_math", split="test")),
        ("lighteval/MATH", lambda: load_dataset("lighteval/MATH", split="test")),
    ]:
        try:
            ds = loader()
            print(f"Loaded {len(ds)} problems from {name}")
            break
        except Exception as e:
            print(f"Could not load {name}: {e}")

    if ds is None:
        print("ERROR: Could not load MATH dataset.")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    problems_by_level: dict[int, list] = {k: [] for k in range(1, 6)}

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

    sampled = []
    for level in range(1, 6):
        pool = problems_by_level[level]
        n_sample = min(n_per_level, len(pool))
        indices = rng.choice(len(pool), size=n_sample, replace=False)
        sampled.extend([pool[i] for i in indices])
        print(f"Level {level}: sampled {n_sample}/{len(pool)}")

    print(f"Total: {len(sampled)} problems")
    return sampled


def extract_attention_matrices(n_per_level=10, seed=42, max_length=512):
    """Extract raw attention matrices from Qwen2.5-1.5B terminal layer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    problems = load_math_subset(n_per_level=n_per_level, seed=seed)

    print("\nLoading Qwen2.5-1.5B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="auto",
        output_attentions=True,
        output_hidden_states=False,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    terminal_layer = n_layers - 1  # layer 27
    print(f"Total layers: {n_layers}, extracting from layer {terminal_layer}")

    all_matrices = []
    all_levels = []
    all_n_tokens = []
    skipped = []

    t0 = time.time()
    for idx, problem in enumerate(problems):
        if idx % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(problems)}] {elapsed:.0f}s elapsed")

        prompt = PROMPT_TEMPLATE.format(problem=problem["problem"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)

        try:
            with torch.no_grad():
                outputs = model(**inputs)

            # Head-averaged attention: (n_heads, T, T) → (T, T)
            attn = outputs.attentions[terminal_layer][0].mean(dim=0).cpu().float().numpy()
            all_matrices.append(attn)
            all_levels.append(problem["level"])
            all_n_tokens.append(attn.shape[0])

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on problem {idx}, skipping")
            skipped.append(idx)
            torch.cuda.empty_cache()
            continue

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"Extracted {len(all_matrices)}/{len(problems)} in {elapsed:.1f}s")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    out_path = os.path.join(DATA_DIR, "raw_attention_matrices.npz")
    np.savez_compressed(
        out_path,
        attention_matrices=np.array(all_matrices, dtype=object),
        difficulty_levels=np.array(all_levels),
        n_tokens=np.array(all_n_tokens),
    )
    print(f"Saved: {out_path}")
    return all_matrices, all_levels


def load_attention_matrices():
    """Load pre-extracted attention matrices."""
    path = os.path.join(DATA_DIR, "raw_attention_matrices.npz")
    d = np.load(path, allow_pickle=True)
    return list(d["attention_matrices"]), list(d["difficulty_levels"].astype(int))


# ─── Construction 1: Symmetric VR ───────────────────────────────────────────

def symmetric_vr_ph(attn_matrix, max_dim=1, subsample=64):
    """Standard VR PH on symmetrized attention distance matrix."""
    import ripser

    sym = (attn_matrix + attn_matrix.T) / 2.0
    dist = 1.0 - sym
    np.clip(dist, 0.0, 1.0, out=dist)
    np.fill_diagonal(dist, 0.0)

    n = dist.shape[0]
    if n > subsample:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, subsample, replace=False))
        dist = dist[np.ix_(idx, idx)]

    result = ripser.ripser(dist, maxdim=max_dim, distance_matrix=True)
    return result["dgms"]


# ─── Construction 2: Directed Flag Complex ──────────────────────────────────

def directed_flag_ph(attn_matrix, max_dim=1, threshold_k=8, subsample=64):
    """Directed flag complex PH via GUDHI SimplexTree.

    Builds directed flag complex: edges come from the directed attention
    graph (top-k per query token). Triangles require directed 3-cliques
    (i→j, j→k, i→k). This preserves asymmetry: a triangle {i,j,k} only
    exists when directed paths connect all three vertices.
    """
    import gudhi

    n = attn_matrix.shape[0]

    if n > subsample:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, subsample, replace=False))
        attn_matrix = attn_matrix[np.ix_(idx, idx)]
        n = subsample

    # Build sparse directed adjacency: top-k outgoing edges per token
    sparse_adj = np.zeros_like(attn_matrix)
    for i in range(n):
        row = attn_matrix[i].copy()
        row[i] = 0
        k = min(threshold_k, n - 1)
        top_k_idx = np.argsort(row)[-k:]
        sparse_adj[i, top_k_idx] = row[top_k_idx]

    # Distance matrix: strong attention → low distance
    dist = np.where(sparse_adj > 0, 1.0 - sparse_adj, np.inf)

    st = gudhi.SimplexTree()

    # Add vertices
    for i in range(n):
        st.insert([i], filtration=0.0)

    # Add edges from directed graph (undirected simplex with min filtration)
    for i in range(n):
        for j in range(n):
            if sparse_adj[i, j] > 0:
                filt = float(dist[i, j])
                if st.find([i, j]):
                    current = st.filtration([i, j])
                    st.insert([i, j], filtration=min(filt, current))
                else:
                    st.insert([i, j], filtration=filt)

    # Add triangles only for directed 3-cliques: i→j, j→k, i→k
    for i in range(n):
        out_i = set(np.where(sparse_adj[i] > 0)[0]) - {i}
        for j in out_i:
            out_j = set(np.where(sparse_adj[j] > 0)[0]) - {i, j}
            common = out_i & out_j
            for k in common:
                filt = max(float(dist[i, j]), float(dist[j, k]), float(dist[i, k]))
                st.insert([i, j, k], filtration=filt)

    st.make_filtration_non_decreasing()
    st.compute_persistence()

    diagrams = []
    for d in range(max_dim + 1):
        pairs = st.persistence_intervals_in_dimension(d)
        if len(pairs) > 0:
            finite = pairs[np.isfinite(pairs[:, 1])]
            diagrams.append(finite)
        else:
            diagrams.append(np.empty((0, 2)))
    return diagrams


# ─── Construction 3: Dowker Complex (manual via GUDHI) ──────────────────────

def dowker_ph(attn_matrix, max_dim=1, threshold=0.05, subsample=64):
    """Dowker complex PH via GUDHI SimplexTree."""
    import gudhi

    n = attn_matrix.shape[0]

    if n > subsample:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, subsample, replace=False))
        attn_matrix = attn_matrix[np.ix_(idx, idx)]
        n = subsample

    st = gudhi.SimplexTree()

    # Add vertices
    for i in range(n):
        st.insert([i], filtration=0.0)

    # For each key token k, find query tokens that attend to it above threshold
    for k in range(n):
        attendees = np.where(attn_matrix[:, k] > threshold)[0]
        if len(attendees) < 2:
            continue

        # Add edges
        for i_idx in range(len(attendees)):
            for j_idx in range(i_idx + 1, len(attendees)):
                i, j = int(attendees[i_idx]), int(attendees[j_idx])
                filt = max(1.0 - float(attn_matrix[i, k]), 1.0 - float(attn_matrix[j, k]))
                st.insert([i, j], filtration=filt)

        # Add triangles — needed even for max_dim=1, because H1 computation
        # requires 2-simplices to determine when 1-cycles die. Without them,
        # all H1 features have infinite death and get filtered out.
        if len(attendees) >= 3:
            # Cap attendees to prevent combinatorial explosion
            att_capped = attendees[:min(len(attendees), 40)]
            for triple in combinations(att_capped, 3):
                filt = max(1.0 - float(attn_matrix[int(t), k]) for t in triple)
                st.insert([int(t) for t in triple], filtration=filt)

    st.make_filtration_non_decreasing()
    st.compute_persistence()

    diagrams = []
    for d in range(max_dim + 1):
        pairs = st.persistence_intervals_in_dimension(d)
        if len(pairs) > 0:
            finite = pairs[np.isfinite(pairs[:, 1])]
            diagrams.append(finite)
        else:
            diagrams.append(np.empty((0, 2)))
    return diagrams


# ─── Feature Extraction ─────────────────────────────────────────────────────

def persistence_entropy(dgm):
    """Shannon entropy of persistence diagram lifetimes."""
    if len(dgm) == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
    if len(lifetimes) == 0:
        return 0.0
    total = lifetimes.sum()
    if total == 0 or not np.isfinite(total):
        return 0.0
    probs = lifetimes / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def extract_features(diagrams):
    """Extract feature vector from list of persistence diagrams [H0, H1]."""
    features = {}
    for dim in range(min(2, len(diagrams))):
        dgm = diagram_to_array(diagrams[dim])
        # Filter to finite features only
        if len(dgm) > 0:
            finite_mask = np.isfinite(dgm[:, 1])
            dgm = dgm[finite_mask]
        prefix = f"h{dim}"
        features[f"{prefix}_count"] = len(dgm)
        features[f"{prefix}_entropy"] = persistence_entropy(dgm)
        if len(dgm) > 0:
            lifetimes = dgm[:, 1] - dgm[:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes)]
            if len(lifetimes) > 0:
                features[f"{prefix}_max_persistence"] = float(np.max(lifetimes))
                features[f"{prefix}_total_persistence"] = float(np.sum(lifetimes))
                features[f"{prefix}_mean_persistence"] = float(np.mean(lifetimes))
            else:
                features[f"{prefix}_max_persistence"] = 0.0
                features[f"{prefix}_total_persistence"] = 0.0
                features[f"{prefix}_mean_persistence"] = 0.0
        else:
            features[f"{prefix}_max_persistence"] = 0.0
            features[f"{prefix}_total_persistence"] = 0.0
            features[f"{prefix}_mean_persistence"] = 0.0
    # Fill missing dims
    for dim in range(len(diagrams), 2):
        prefix = f"h{dim}"
        features[f"{prefix}_count"] = 0
        features[f"{prefix}_entropy"] = 0.0
        features[f"{prefix}_max_persistence"] = 0.0
        features[f"{prefix}_total_persistence"] = 0.0
        features[f"{prefix}_mean_persistence"] = 0.0
    return features


def diagram_to_array(dgm):
    """Ensure diagram is a numpy array of shape (n, 2)."""
    if isinstance(dgm, np.ndarray):
        if dgm.ndim == 2 and dgm.shape[1] >= 2:
            return dgm[:, :2].astype(float)
        elif dgm.ndim == 1 and len(dgm) == 0:
            return np.empty((0, 2))
    if isinstance(dgm, list):
        if len(dgm) == 0:
            return np.empty((0, 2))
        return np.array(dgm, dtype=float)[:, :2]
    return np.empty((0, 2))


# ─── Wasserstein Distance ───────────────────────────────────────────────────

def wasserstein_distance_diagrams(dgm1, dgm2):
    """Approximate Wasserstein-1 distance between two persistence diagrams."""
    from scipy.spatial.distance import cdist

    d1 = diagram_to_array(dgm1)
    d2 = diagram_to_array(dgm2)

    # Add diagonal projections
    if len(d1) == 0 and len(d2) == 0:
        return 0.0

    if len(d1) == 0:
        proj = np.column_stack([d2.mean(axis=1), d2.mean(axis=1)])
        return float(np.sum(np.abs(d2[:, 1] - d2[:, 0])) / 2)
    if len(d2) == 0:
        return float(np.sum(np.abs(d1[:, 1] - d1[:, 0])) / 2)

    # Use bottleneck-style approximation via lifetimes
    l1 = np.sort(d1[:, 1] - d1[:, 0])[::-1]
    l2 = np.sort(d2[:, 1] - d2[:, 0])[::-1]

    max_len = max(len(l1), len(l2))
    l1_padded = np.zeros(max_len)
    l2_padded = np.zeros(max_len)
    l1_padded[:len(l1)] = l1
    l2_padded[:len(l2)] = l2

    return float(np.sum(np.abs(l1_padded - l2_padded)))


# ─── Experiments ─────────────────────────────────────────────────────────────

def run_all_constructions(matrices, levels, threshold_k=8):
    """Run all three constructions on all matrices."""
    n = len(matrices)
    results = {"symmetric_vr": [], "directed_flag": [], "dowker": []}

    for i in range(n):
        if i % 10 == 0:
            print(f"  Computing constructions for problem {i}/{n}...")

        attn = matrices[i]

        # Symmetric VR
        dgms = symmetric_vr_ph(attn)
        results["symmetric_vr"].append(dgms)

        # Directed Flag
        dgms = directed_flag_ph(attn, threshold_k=threshold_k)
        if dgms is None:
            # Fallback: empty diagrams
            dgms = [np.empty((0, 2)), np.empty((0, 2))]
        results["directed_flag"].append(dgms)

        # Dowker
        dgms = dowker_ph(attn)
        results["dowker"].append(dgms)

    return results


def exp1_feature_comparison(construction_results, levels):
    """Experiment 1: Feature count and entropy comparison by difficulty."""
    print("\n=== Experiment 1: Feature Count & Entropy ===")
    levels = np.array(levels)
    summary = {}

    for name, diagrams_list in construction_results.items():
        summary[name] = {}
        for lvl in range(1, 6):
            mask = levels == lvl
            h0_counts, h1_counts, h0_ents, h1_ents = [], [], [], []

            for i in np.where(mask)[0]:
                dgms = diagrams_list[i]
                feats = extract_features(dgms)
                h0_counts.append(feats["h0_count"])
                h1_counts.append(feats["h1_count"])
                h0_ents.append(feats["h0_entropy"])
                h1_ents.append(feats["h1_entropy"])

            summary[name][f"level_{lvl}"] = {
                "h0_count": f"{np.mean(h0_counts):.1f} ± {np.std(h0_counts):.1f}",
                "h1_count": f"{np.mean(h1_counts):.1f} ± {np.std(h1_counts):.1f}",
                "h0_entropy": f"{np.mean(h0_ents):.3f} ± {np.std(h0_ents):.3f}",
                "h1_entropy": f"{np.mean(h1_ents):.3f} ± {np.std(h1_ents):.3f}",
                "h0_count_mean": float(np.mean(h0_counts)),
                "h1_count_mean": float(np.mean(h1_counts)),
                "h0_entropy_mean": float(np.mean(h0_ents)),
                "h1_entropy_mean": float(np.mean(h1_ents)),
                "h1_entropy_std": float(np.std(h1_ents)),
            }

        # Print table
        print(f"\n{name}:")
        print(f"  {'Level':<8} {'H0 count':<16} {'H1 count':<16} {'H0 entropy':<20} {'H1 entropy':<20}")
        for lvl in range(1, 6):
            d = summary[name][f"level_{lvl}"]
            print(f"  {lvl:<8} {d['h0_count']:<16} {d['h1_count']:<16} {d['h0_entropy']:<20} {d['h1_entropy']:<20}")

    return summary


def exp2_permutation_test(construction_results, levels, n_perms=200, seed=42):
    """Experiment 2: Wasserstein permutation test for discriminative power."""
    print("\n=== Experiment 2: Permutation Test ===")
    rng = np.random.default_rng(seed)
    levels = np.array(levels)
    results = {}

    for name, diagrams_list in construction_results.items():
        # Extract H1 entropy per problem
        entropies = []
        for dgms in diagrams_list:
            feats = extract_features(dgms)
            entropies.append(feats["h1_entropy"])
        entropies = np.array(entropies)

        # Observed statistic: variance of group means
        observed = np.var([entropies[levels == lvl].mean() for lvl in range(1, 6)])

        # Permutation distribution
        perm_stats = []
        for _ in range(n_perms):
            perm_levels = rng.permutation(levels)
            stat = np.var([entropies[perm_levels == lvl].mean() for lvl in range(1, 6)])
            perm_stats.append(stat)
        perm_stats = np.array(perm_stats)

        z_score = (observed - perm_stats.mean()) / (perm_stats.std() + 1e-10)
        p_value = float(np.mean(perm_stats >= observed))

        results[name] = {"z_score": float(z_score), "p_value": p_value}
        print(f"  {name}: z={z_score:.2f}, p={p_value:.4f}")

    winner = max(results, key=lambda k: results[k]["z_score"])
    results["winner"] = winner
    print(f"  Winner (highest z): {winner}")
    return results


def exp3_correctness_prediction(construction_results, levels):
    """Experiment 3: Logistic regression AUROC for correctness prediction."""
    print("\n=== Experiment 3: Correctness Prediction ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    levels = np.array(levels)
    # Binary target: levels 4-5 = "hard" (proxy for incorrect)
    y = (levels >= 4).astype(int)

    results = {}
    all_feature_matrices = {}

    for name, diagrams_list in construction_results.items():
        # Extract features
        feature_list = []
        for dgms in diagrams_list:
            feats = extract_features(dgms)
            feature_list.append([
                feats["h0_count"], feats["h1_count"],
                feats["h0_entropy"], feats["h1_entropy"],
                feats["h0_max_persistence"], feats["h1_max_persistence"],
                feats["h0_total_persistence"], feats["h1_total_persistence"],
            ])
        X = np.array(feature_list)
        # Sanitize: replace NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        all_feature_matrices[name] = X

        # Standardize + logistic regression with 5-fold CV
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="roc_auc")
        auroc = float(scores.mean())
        results[name] = auroc
        print(f"  {name}: AUROC = {auroc:.3f} (±{scores.std():.3f})")

    # Combined features
    X_combined = np.hstack(list(all_feature_matrices.values()))
    X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="roc_auc")
    results["combined"] = float(scores.mean())
    print(f"  combined: AUROC = {results['combined']:.3f} (±{scores.std():.3f})")

    max_individual = max(results[k] for k in ["symmetric_vr", "directed_flag", "dowker"])
    results["asymmetry_helps"] = results["combined"] - max_individual > 0.03
    print(f"  Asymmetry helps (combined > max_individual + 0.03): {results['asymmetry_helps']}")

    return results


def exp4_threshold_sensitivity(matrices, levels, thresholds=(3, 5, 8, 12, 20)):
    """Experiment 4: Directed flag threshold sensitivity."""
    print("\n=== Experiment 4: Threshold Sensitivity ===")
    rng = np.random.default_rng(42)
    levels = np.array(levels)
    results = {}

    for k in thresholds:
        print(f"  threshold_k={k}...")
        h1_counts = []
        h1_entropies = []

        for i, attn in enumerate(matrices):
            dgms = directed_flag_ph(attn, threshold_k=k)
            if dgms is None:
                h1_counts.append(0)
                h1_entropies.append(0.0)
            else:
                feats = extract_features(dgms)
                h1_counts.append(feats["h1_count"])
                h1_entropies.append(feats["h1_entropy"])

        h1_counts = np.array(h1_counts)
        h1_entropies = np.array(h1_entropies)

        # Permutation z-score
        observed = np.var([h1_entropies[levels == lvl].mean() for lvl in range(1, 6)])
        perm_stats = []
        for _ in range(200):
            perm_levels = rng.permutation(levels)
            stat = np.var([h1_entropies[perm_levels == lvl].mean() for lvl in range(1, 6)])
            perm_stats.append(stat)
        perm_stats = np.array(perm_stats)
        z_score = (observed - perm_stats.mean()) / (perm_stats.std() + 1e-10)

        results[f"k{k}"] = {
            "h1_count": float(h1_counts.mean()),
            "z_score": float(z_score),
        }
        print(f"    H1 count={h1_counts.mean():.1f}, z={z_score:.2f}")

    # Determine stable range
    z_scores = [results[f"k{k}"]["z_score"] for k in thresholds]
    z_range = max(z_scores) - min(z_scores)
    if z_range < 1.0:
        stable_range = f"k{thresholds[0]}-k{thresholds[-1]} (all stable, range={z_range:.2f})"
    else:
        # Find contiguous range where z-scores are within 1.0 of each other
        best_start = 0
        best_len = 1
        for start in range(len(thresholds)):
            for end in range(start + 1, len(thresholds)):
                if max(z_scores[start:end+1]) - min(z_scores[start:end+1]) < 1.0:
                    if end - start + 1 > best_len:
                        best_start = start
                        best_len = end - start + 1
        stable_range = f"k{thresholds[best_start]}-k{thresholds[best_start + best_len - 1]}"

    results["stable_range"] = stable_range
    print(f"  Stable range: {stable_range}")
    return results


def exp5_diagram_comparison(construction_results, levels, matrices):
    """Experiment 5: Inter-construction Wasserstein distances."""
    print("\n=== Experiment 5: Diagram Structure Comparison ===")
    levels = np.array(levels)

    # Pick one representative per difficulty level
    representative_idx = []
    for lvl in range(1, 6):
        candidates = np.where(levels == lvl)[0]
        representative_idx.append(candidates[0])
    print(f"  Representatives: {representative_idx}")

    # Compute pairwise Wasserstein distances (H1) averaged across all problems
    names = ["symmetric_vr", "directed_flag", "dowker"]
    pair_dists = {}

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if j <= i:
                continue
            dists = []
            for k in range(len(levels)):
                d1 = diagram_to_array(construction_results[name1][k][1]) if len(construction_results[name1][k]) > 1 else np.empty((0, 2))
                d2 = diagram_to_array(construction_results[name2][k][1]) if len(construction_results[name2][k]) > 1 else np.empty((0, 2))
                dists.append(wasserstein_distance_diagrams(d1, d2))
            pair_key = f"{name1.split('_')[0]}_{name2.split('_')[0]}" if "flag" not in name1 else f"{'vr' if 'vr' in name1 else 'flag'}_{name2.split('_')[0]}"
            pair_dists[f"{name1}_{name2}"] = float(np.mean(dists))
            print(f"  d({name1}, {name2}) = {np.mean(dists):.4f}")

    return {
        "vr_flag": pair_dists.get("symmetric_vr_directed_flag", 0),
        "vr_dowker": pair_dists.get("symmetric_vr_dowker", 0),
        "flag_dowker": pair_dists.get("directed_flag_dowker", 0),
    }, representative_idx


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_entropy_comparison(exp1_summary):
    """Figure 1: H1 entropy by difficulty level for each construction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    names = ["symmetric_vr", "directed_flag", "dowker"]
    titles = ["Symmetric VR (Baseline)", "Directed Flag Complex", "Dowker Complex"]

    for ax, name, title in zip(axes, names, titles):
        means = [exp1_summary[name][f"level_{lvl}"]["h1_entropy_mean"] for lvl in range(1, 6)]
        stds = [exp1_summary[name][f"level_{lvl}"]["h1_entropy_std"] for lvl in range(1, 6)]
        ax.bar(range(1, 6), means, yerr=stds, capsize=5,
               color=["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"],
               alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Difficulty Level")
        ax.set_title(title)
        ax.set_xticks(range(1, 6))
    axes[0].set_ylabel("H1 Persistence Entropy")

    plt.suptitle("H1 Entropy by Difficulty Level — Three Simplicial Complex Constructions", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "entropy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved entropy_comparison.png")


def plot_auroc_comparison(exp3_results):
    """Figure 2: AUROC bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["symmetric_vr", "directed_flag", "dowker", "combined"]
    labels = ["Symmetric VR", "Directed Flag", "Dowker", "Combined"]
    values = [exp3_results[n] for n in names]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_title("Correctness Prediction AUROC — Three Constructions")
    ax.set_ylim(0.3, 1.0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "auroc_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved auroc_comparison.png")


def plot_threshold_stability(exp4_results):
    """Figure 3: Directed flag threshold stability curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [3, 5, 8, 12, 20]
    h1_counts = [exp4_results[f"k{k}"]["h1_count"] for k in thresholds]
    z_scores = [exp4_results[f"k{k}"]["z_score"] for k in thresholds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(thresholds, h1_counts, "o-", color="#FF9800", linewidth=2, markersize=8)
    ax1.set_xlabel("threshold_k (edges per token)")
    ax1.set_ylabel("Mean H1 Count")
    ax1.set_title("H1 Feature Count vs Threshold")
    ax1.set_xticks(thresholds)

    ax2.plot(thresholds, z_scores, "s-", color="#F44336", linewidth=2, markersize=8)
    ax2.set_xlabel("threshold_k (edges per token)")
    ax2.set_ylabel("Permutation Z-score")
    ax2.set_title("Discriminative Power vs Threshold")
    ax2.axhline(y=1.96, color="gray", linestyle="--", alpha=0.5, label="p=0.05")
    ax2.set_xticks(thresholds)
    ax2.legend()

    plt.suptitle(f"Directed Flag Complex — Threshold Sensitivity\nStable range: {exp4_results['stable_range']}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "threshold_stability.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved threshold_stability.png")


def plot_diagram_comparison(construction_results, levels, representative_idx):
    """Figure 4: 5×3 grid of persistence diagrams."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["symmetric_vr", "directed_flag", "dowker"]
    titles = ["Symmetric VR", "Directed Flag", "Dowker"]
    levels = np.array(levels)

    fig, axes = plt.subplots(5, 3, figsize=(15, 20))

    for row, idx in enumerate(representative_idx):
        lvl = levels[idx]
        for col, (name, title) in enumerate(zip(names, titles)):
            ax = axes[row, col]
            dgms = construction_results[name][idx]

            # Plot H0 and H1
            for dim, color, marker in [(0, "#2196F3", "o"), (1, "#F44336", "^")]:
                if dim < len(dgms):
                    dgm = diagram_to_array(dgms[dim])
                    if len(dgm) > 0:
                        ax.scatter(dgm[:, 0], dgm[:, 1], c=color, marker=marker,
                                   alpha=0.6, s=30, label=f"H{dim}", edgecolors="black", linewidth=0.3)

            # Diagonal
            lim = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1.0
            ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
            ax.set_xlim(-0.02, None)
            ax.set_ylim(-0.02, None)

            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Level {lvl}\nDeath", fontsize=10)
            if row == 4:
                ax.set_xlabel("Birth")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.suptitle("Persistence Diagrams: 5 Problems × 3 Constructions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "diagram_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved diagram_comparison.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def determine_verdict(exp2_results, exp3_results, exp5_dists):
    """Determine overall verdict."""
    vr_z = exp2_results["symmetric_vr"]["z_score"]
    flag_z = exp2_results["directed_flag"]["z_score"]
    dowker_z = exp2_results["dowker"]["z_score"]
    max_asym_z = max(flag_z, dowker_z)

    vr_auroc = exp3_results["symmetric_vr"]
    combined_auroc = exp3_results["combined"]
    asymmetry_helps = exp3_results["asymmetry_helps"]

    if max_asym_z > vr_z + 1.0 and asymmetry_helps:
        return "asymmetry_substantially_helps"
    elif max_asym_z > vr_z + 0.3 or combined_auroc > vr_auroc + 0.02:
        return "asymmetry_marginally_helps"
    else:
        return "symmetrization_loses_nothing"


def main():
    parser = argparse.ArgumentParser(description="Branch B: Simplicial Complex Comparison")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip attention extraction (use saved matrices)")
    parser.add_argument("--n-per-level", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Step 1: Get raw attention matrices
    if args.skip_extraction:
        print("Loading pre-extracted attention matrices...")
        matrices, levels = load_attention_matrices()
    else:
        print("Extracting attention matrices from Qwen2.5-1.5B...")
        matrices, levels = extract_attention_matrices(
            n_per_level=args.n_per_level, seed=42
        )

    print(f"\n{len(matrices)} problems, levels: {np.unique(levels, return_counts=True)}")

    # Step 2: Run all three constructions
    print("\n--- Running three constructions ---")
    t0 = time.time()
    construction_results = run_all_constructions(matrices, levels)
    print(f"Constructions complete in {time.time() - t0:.1f}s")

    # Step 3: Experiments
    exp1 = exp1_feature_comparison(construction_results, levels)
    exp2 = exp2_permutation_test(construction_results, levels)
    exp3 = exp3_correctness_prediction(construction_results, levels)
    exp4 = exp4_threshold_sensitivity(matrices, levels)
    exp5_dists, rep_idx = exp5_diagram_comparison(construction_results, levels, matrices)

    # Step 4: Figures
    print("\n--- Generating figures ---")
    plot_entropy_comparison(exp1)
    plot_auroc_comparison(exp3)
    plot_threshold_stability(exp4)
    plot_diagram_comparison(construction_results, levels, rep_idx)

    # Step 5: Determine verdict
    verdict = determine_verdict(exp2, exp3, exp5_dists)
    print(f"\n=== VERDICT: {verdict} ===")

    # Step 6: Save results
    results = {
        "branch": "experiment/tda-complex-compare",
        "n_problems": len(matrices),
        "constructions": ["symmetric_vr", "directed_flag", "dowker"],
        "exp1_h1_entropy_by_level": {
            name: {f"level_{lvl}": exp1[name][f"level_{lvl}"]["h1_entropy"]
                   for lvl in range(1, 6)}
            for name in ["symmetric_vr", "directed_flag", "dowker"]
        },
        "exp1_h1_count_by_level": {
            name: {f"level_{lvl}": exp1[name][f"level_{lvl}"]["h1_count"]
                   for lvl in range(1, 6)}
            for name in ["symmetric_vr", "directed_flag", "dowker"]
        },
        "exp2_permutation_z": {
            name: exp2[name]["z_score"]
            for name in ["symmetric_vr", "directed_flag", "dowker"]
        },
        "exp2_permutation_p": {
            name: exp2[name]["p_value"]
            for name in ["symmetric_vr", "directed_flag", "dowker"]
        },
        "exp2_winner": exp2["winner"],
        "exp3_auroc": {
            "symmetric_vr": exp3["symmetric_vr"],
            "directed_flag": exp3["directed_flag"],
            "dowker": exp3["dowker"],
            "combined": exp3["combined"],
        },
        "exp3_asymmetry_helps": exp3["asymmetry_helps"],
        "exp4_threshold_stability": exp4,
        "exp4_stable_range": exp4["stable_range"],
        "exp5_inter_construction_distances": exp5_dists,
        "overall_verdict": verdict,
    }

    results_path = os.path.join(DATA_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
