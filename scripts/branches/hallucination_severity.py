"""Branch 1: Hallucination Severity — Topological Signatures of Confident Errors.

Classifies incorrect model answers by severity (near-miss vs hallucination) and
tests whether topological features of hidden states distinguish severity levels.

Uses pool-index matching to align correctness labels with hidden states.

Usage:
    python scripts/branches/hallucination_severity.py
"""

import argparse
import json
import os
import re
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

warnings.filterwarnings("ignore", message=".*more columns than rows.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "transformer")
OUT_DIR = os.path.join(REPO_ROOT, "data", "hallucination")
FIG_DIR = os.path.join(REPO_ROOT, "figures", "hallucination")

# MATH-500 pool sizes per level (from HuggingFaceH4/MATH-500)
MATH500_POOL_SIZES = {1: 43, 2: 90, 3: 105, 4: 128, 5: 134}


# ---------------------------------------------------------------------------
# Answer extraction & severity scoring
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Extract \\boxed{...} answer from text, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth = 0
    start = idx + 7
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return text[start:].strip()


def extract_predicted_answer(text: str) -> str:
    """Extract the actual answer from model output that may contain reasoning."""
    text = text.strip()

    # 1. Try \boxed{}
    boxed = extract_boxed(text)
    if boxed:
        return boxed

    # 2. Try 'answer is X' pattern
    m = re.search(r'(?:answer is|answer:?)\s*[\\($]?\s*(.+?)(?:\\[).\]]|\n|$)', text, re.IGNORECASE)
    if m:
        ans = m.group(1).strip().rstrip('.,;:)')
        if len(ans) < 100:
            return ans

    # 3. First line, strip LaTeX delimiters and trailing explanation
    first_line = text.split('\n')[0].strip()
    first_line = re.sub(r'\\[\])].*$', '', first_line).strip()
    first_line = re.sub(r'^\\[\[(]', '', first_line).strip()
    # Strip trailing parenthetical
    first_line = re.sub(r'\s*\(.*$', '', first_line).strip()

    if len(first_line) < 80:
        return first_line

    # 4. Try first number
    m = re.search(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    if m:
        return m.group(0)

    return first_line[:80]


def clean_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.strip()
    s = re.sub(r'\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}', r'\1', s)
    s = s.replace('$', '')
    s = s.rstrip('.,;:')
    # Normalize \frac14 → \frac{1}{4}
    s = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', s)
    # Normalize whitespace and remove spaces around commas in tuples
    s = re.sub(r'\s*,\s*', ',', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def try_numeric(s: str):
    """Try to parse a string as a number. Returns float or None."""
    s = clean_answer(s)
    s = s.replace(',', '')
    # Handle \frac{a}{b}
    frac_match = re.match(r'^\\frac\{([^}]+)\}\{([^}]+)\}$', s)
    if frac_match:
        try:
            return float(frac_match.group(1)) / float(frac_match.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    # Handle mixed numbers like 137 \frac{1}{2}
    mixed_match = re.match(r'^(-?\d+)\s*\\frac\{(\d+)\}\{(\d+)\}$', s)
    if mixed_match:
        try:
            whole = float(mixed_match.group(1))
            num = float(mixed_match.group(2))
            den = float(mixed_match.group(3))
            return whole + num / den if whole >= 0 else whole - num / den
        except (ValueError, ZeroDivisionError):
            pass
    # Handle a/b
    slash_match = re.match(r'^(-?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$', s)
    if slash_match:
        try:
            return float(slash_match.group(1)) / float(slash_match.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    # Strip degree symbols
    s = re.sub(r'[°^]\\circ', '', s).strip()
    s = re.sub(r'\\circ', '', s).strip()
    try:
        return float(s)
    except ValueError:
        return None


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Levenshtein edit distance normalized by max length."""
    if not s1 and not s2:
        return 0.0
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 1.0
    # Standard DP
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / max(n, m)


def classify_severity(pred_raw: str, gt_raw: str) -> tuple[str, float]:
    """Classify error severity.

    Returns (severity_label, distance_score) where:
    - 'near_miss': numerically within 25% or edit_dist < 0.3
    - 'moderate': within 100% or 0.3 <= edit_dist < 0.7
    - 'hallucination': > 100% relative error, edit_dist >= 0.7, or type mismatch
    """
    pred = clean_answer(extract_predicted_answer(pred_raw))
    gt = clean_answer(gt_raw)

    # Try numeric comparison first
    pred_num = try_numeric(pred)
    gt_num = try_numeric(gt)

    if pred_num is not None and gt_num is not None:
        if gt_num == 0:
            if pred_num == 0:
                return 'near_miss', 0.0
            rel_err = min(abs(pred_num), 10.0)
        else:
            rel_err = abs(pred_num - gt_num) / abs(gt_num)

        if rel_err <= 0.25:
            return 'near_miss', rel_err
        elif rel_err <= 1.0:
            return 'moderate', min(rel_err, 1.0)
        else:
            return 'hallucination', 1.0

    # One is numeric, other isn't → type mismatch → hallucination
    if (pred_num is not None) != (gt_num is not None):
        return 'hallucination', 1.0

    # String comparison
    ed = normalized_edit_distance(pred.lower(), gt.lower())
    if ed < 0.3:
        return 'near_miss', ed
    elif ed < 0.7:
        return 'moderate', ed
    else:
        return 'hallucination', ed


# ---------------------------------------------------------------------------
# Pool-index matching (align correctness to hidden states)
# ---------------------------------------------------------------------------

def build_alignment_map(seed: int = 42) -> dict[int, int]:
    """Build mapping: correctness_index -> hidden_state_index.

    Reconstructs the rng.choice sampling from evaluate_correctness.py to
    determine which hidden state (in dataset order) corresponds to each
    correctness label (in rng.choice order).
    """
    rng = np.random.default_rng(seed)
    c_to_h = {}
    h_offset = 0
    c_offset = 0

    for level in range(1, 6):
        pool_size = MATH500_POOL_SIZES[level]
        n_sample = min(100, pool_size)
        indices = rng.choice(pool_size, size=n_sample, replace=False)

        for j in range(n_sample):
            c_to_h[c_offset + j] = h_offset + int(indices[j])

        h_offset += pool_size
        c_offset += n_sample

    return c_to_h


# ---------------------------------------------------------------------------
# PH computation helpers
# ---------------------------------------------------------------------------

def compute_persistence(cloud, max_dim=1, max_edge=None):
    """Compute persistence diagrams with Ripser."""
    from ripser import ripser
    result = ripser(cloud, maxdim=max_dim, thresh=max_edge or np.inf)
    return result['dgms']


def persistence_entropy(dgm):
    """Compute persistence entropy of a diagram."""
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    total = lifetimes.sum()
    if total == 0:
        return 0.0
    p = lifetimes / total
    return float(-np.sum(p * np.log2(p + 1e-30)))


def total_persistence(dgm):
    """Total persistence = sum of finite lifetimes."""
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    return float(lifetimes.sum())


def max_lifetime(dgm):
    """Maximum finite lifetime."""
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if len(lifetimes) == 0:
        return 0.0
    return float(lifetimes.max())


def n_features(dgm):
    """Number of finite features."""
    lifetimes = dgm[:, 1] - dgm[:, 0]
    return int(np.isfinite(lifetimes).sum())


def extract_topo_features(dgms):
    """Extract per-problem topological feature vector from persistence diagrams."""
    features = []
    for dim in range(len(dgms)):
        dgm = dgms[dim]
        features.extend([
            persistence_entropy(dgm),
            total_persistence(dgm),
            max_lifetime(dgm),
            n_features(dgm),
        ])
        # Birth statistics for finite features
        lifetimes = dgm[:, 1] - dgm[:, 0]
        finite_mask = np.isfinite(lifetimes)
        births = dgm[finite_mask, 0]
        if len(births) > 0:
            features.extend([np.mean(births), np.std(births)])
        else:
            features.extend([0.0, 0.0])
    return np.array(features, dtype=np.float64)


FEATURE_NAMES = []
for dim in [0, 1]:
    prefix = f"H{dim}"
    FEATURE_NAMES.extend([
        f"{prefix}_entropy", f"{prefix}_total_persistence",
        f"{prefix}_max_lifetime", f"{prefix}_n_features",
        f"{prefix}_mean_birth", f"{prefix}_std_birth",
    ])


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------

def wasserstein_distance(dgm1, dgm2, order=1):
    """Compute Wasserstein distance between persistence diagrams."""
    try:
        from gudhi.wasserstein import wasserstein_distance as gudhi_wd
        d1 = dgm1[np.isfinite(dgm1[:, 1])] if len(dgm1) > 0 else np.empty((0, 2))
        d2 = dgm2[np.isfinite(dgm2[:, 1])] if len(dgm2) > 0 else np.empty((0, 2))
        if len(d1) == 0 and len(d2) == 0:
            return 0.0
        return float(gudhi_wd(d1, d2, order=order))
    except ImportError:
        # Fallback: bottleneck-style approximation
        l1 = dgm1[:, 1] - dgm1[:, 0]
        l2 = dgm2[:, 1] - dgm2[:, 0]
        l1 = np.sort(l1[np.isfinite(l1)])[::-1]
        l2 = np.sort(l2[np.isfinite(l2)])[::-1]
        n = max(len(l1), len(l2))
        l1_pad = np.pad(l1, (0, n - len(l1)))
        l2_pad = np.pad(l2, (0, n - len(l2)))
        return float(np.sum(np.abs(l1_pad - l2_pad)))


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hallucination severity topological analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-perms", type=int, default=200)
    parser.add_argument("--subsample", type=int, default=200)
    parser.add_argument("--n-pca", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------------------------
    # Load data & compute severity
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    hidden = np.load(os.path.join(DATA_DIR, "math500_hidden_states_aligned.npz"), allow_pickle=True)
    correct = np.load(os.path.join(DATA_DIR, "math500_correctness.npz"), allow_pickle=True)

    hidden_states = hidden["last_hidden_states"]  # (500, 1536)
    h_levels = hidden["difficulty_levels"]         # (500,)
    layer_states = hidden["layer_hidden_states"]   # (500, 29, 1536)

    c_correct = correct["correct"]                 # (433,)
    c_levels = correct["difficulty_levels"]         # (433,)
    c_predicted = correct["predicted_answers"]      # (433,) object
    c_ground_truth = correct["ground_truth"]        # (433,) object

    print(f"Hidden states: {hidden_states.shape}")
    print(f"Layer states: {layer_states.shape}")
    print(f"Correctness: {len(c_correct)} ({c_correct.sum()} correct, {(~c_correct).sum()} incorrect)")

    # Build alignment map
    c_to_h = build_alignment_map(seed=args.seed)
    print(f"Alignment map: {len(c_to_h)} entries")

    # Verify alignment: check level consistency
    n_level_match = sum(1 for c_idx, h_idx in c_to_h.items()
                        if c_levels[c_idx] == h_levels[h_idx])
    print(f"Level consistency check: {n_level_match}/{len(c_to_h)} match")
    assert n_level_match == len(c_to_h), "Level mismatch in alignment!"

    # Classify severity for incorrect problems
    print("\nClassifying error severity...")
    severity_labels = []
    severity_scores = []
    severity_h_indices = []  # hidden state indices for incorrect problems
    correct_h_indices = []   # hidden state indices for correct problems

    for c_idx in range(len(c_correct)):
        h_idx = c_to_h[c_idx]
        if c_correct[c_idx]:
            correct_h_indices.append(h_idx)
            continue

        sev, score = classify_severity(
            str(c_predicted[c_idx]),
            str(c_ground_truth[c_idx])
        )
        severity_labels.append(sev)
        severity_scores.append(score)
        severity_h_indices.append(h_idx)

    severity_labels = np.array(severity_labels)
    severity_scores = np.array(severity_scores)
    severity_h_indices = np.array(severity_h_indices)
    correct_h_indices = np.array(correct_h_indices)

    n_correct = len(correct_h_indices)
    n_near_miss = int((severity_labels == 'near_miss').sum())
    n_moderate = int((severity_labels == 'moderate').sum())
    n_hallucination = int((severity_labels == 'hallucination').sum())

    print(f"\nSeverity distribution:")
    print(f"  Correct:       {n_correct}")
    print(f"  Near-miss:     {n_near_miss}")
    print(f"  Moderate:      {n_moderate}")
    print(f"  Hallucination: {n_hallucination}")

    # Print some examples per category
    for sev in ['near_miss', 'moderate', 'hallucination']:
        mask = severity_labels == sev
        idxs = np.where(mask)[0][:3]
        print(f"\n  --- {sev} examples ---")
        for idx in idxs:
            # Find the original correctness index
            c_idx_list = [c for c in range(len(c_correct)) if not c_correct[c]]
            c_idx = c_idx_list[idx]
            pred = str(c_predicted[c_idx])[:60]
            gt = str(c_ground_truth[c_idx])[:40]
            print(f"    pred={pred!r}")
            print(f"      gt={gt!r}  score={severity_scores[idx]:.3f}")

    # Get hidden state indices per severity bin
    near_miss_h = severity_h_indices[severity_labels == 'near_miss']
    moderate_h = severity_h_indices[severity_labels == 'moderate']
    halluc_h = severity_h_indices[severity_labels == 'hallucination']

    severity_method = "string_distance"

    # Check if we have enough data in each bin
    min_bin = min(n_near_miss, n_moderate, n_hallucination)
    if min_bin < 5:
        print(f"\nWARNING: Smallest bin has only {min_bin} problems.")
        if n_near_miss < 5 or n_hallucination < 5:
            print("Falling back to difficulty-proxy severity...")
            severity_method = "difficulty_proxy"
            # L1-L2 wrong → near_miss, L3 → moderate, L4-L5 → hallucination
            severity_labels = np.array([
                'near_miss' if h_levels[h] <= 2 else
                'moderate' if h_levels[h] == 3 else
                'hallucination'
                for h in severity_h_indices
            ])
            near_miss_h = severity_h_indices[severity_labels == 'near_miss']
            moderate_h = severity_h_indices[severity_labels == 'moderate']
            halluc_h = severity_h_indices[severity_labels == 'hallucination']
            n_near_miss = len(near_miss_h)
            n_moderate = len(moderate_h)
            n_hallucination = len(halluc_h)
            print(f"  Proxy distribution: near_miss={n_near_miss}, moderate={n_moderate}, hallucination={n_hallucination}")

    # -----------------------------------------------------------------------
    # Experiment 1: Topology by error severity
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TOPOLOGY BY ERROR SEVERITY")
    print("=" * 70)

    from sklearn.decomposition import PCA

    def get_cloud(h_indices, layer=-1, n_pca=args.n_pca, sub=args.subsample):
        """Get PCA-reduced point cloud for a set of problem indices."""
        if layer == -1:
            cloud = hidden_states[h_indices]
        else:
            cloud = layer_states[h_indices, layer, :]

        if cloud.shape[0] > sub:
            idx = rng.choice(cloud.shape[0], sub, replace=False)
            cloud = cloud[idx]
        # Cap PCA components to min(n_pca, n_samples - 1, n_features)
        n_comp = min(n_pca, cloud.shape[0] - 1, cloud.shape[1])
        if n_comp > 0 and cloud.shape[1] > n_comp:
            pca = PCA(n_components=n_comp, random_state=args.seed)
            cloud = pca.fit_transform(cloud)
        return cloud

    bins = {
        'correct': correct_h_indices,
        'near_miss': near_miss_h,
        'moderate': moderate_h,
        'hallucination': halluc_h,
    }

    # Compute PH for each bin at terminal layer
    print("\nComputing persistence for each severity bin (terminal layer)...")
    bin_dgms = {}
    bin_entropy = {}
    for name, h_idxs in bins.items():
        if len(h_idxs) < 3:
            print(f"  {name}: skipped (n={len(h_idxs)})")
            continue
        cloud = get_cloud(h_idxs, layer=-1)
        dgms = compute_persistence(cloud, max_dim=1)
        bin_dgms[name] = dgms
        h0_ent = persistence_entropy(dgms[0])
        h1_ent = persistence_entropy(dgms[1]) if len(dgms) > 1 else 0.0
        bin_entropy[name] = {'H0': h0_ent, 'H1': h1_ent}
        print(f"  {name} (n={len(h_idxs)}): H0_entropy={h0_ent:.3f}, H1_entropy={h1_ent:.3f}")

    # Wasserstein distance between near_miss and hallucination
    print("\nWasserstein distances between severity bins (H1):")
    wd_pairs = {}
    for a, b in [('near_miss', 'hallucination'), ('near_miss', 'moderate'),
                  ('moderate', 'hallucination'), ('correct', 'hallucination')]:
        if a in bin_dgms and b in bin_dgms:
            wd = wasserstein_distance(bin_dgms[a][1], bin_dgms[b][1])
            wd_pairs[f"{a}_vs_{b}"] = wd
            print(f"  {a} vs {b}: W1={wd:.4f}")

    # Permutation test: shuffle severity labels among incorrect problems
    print(f"\nPermutation test (near_miss vs hallucination, {args.n_perms} perms)...")
    if 'near_miss' in bin_dgms and 'hallucination' in bin_dgms:
        observed_wd = wd_pairs.get('near_miss_vs_hallucination', 0.0)
        all_incorrect_h = np.concatenate([near_miss_h, moderate_h, halluc_h])

        # We test near_miss vs hallucination specifically
        nm_size = len(near_miss_h)
        hal_size = len(halluc_h)
        perm_wds = []

        for p in range(args.n_perms):
            if (p + 1) % 50 == 0:
                print(f"  Permutation {p + 1}/{args.n_perms}...")
            perm_idx = rng.permutation(len(all_incorrect_h))
            perm_nm = all_incorrect_h[perm_idx[:nm_size]]
            perm_hal = all_incorrect_h[perm_idx[nm_size:nm_size + hal_size]]

            cloud_nm = get_cloud(perm_nm)
            cloud_hal = get_cloud(perm_hal)

            dgm_nm = compute_persistence(cloud_nm, max_dim=1)
            dgm_hal = compute_persistence(cloud_hal, max_dim=1)

            perm_wds.append(wasserstein_distance(dgm_nm[1], dgm_hal[1]))

        perm_wds = np.array(perm_wds)
        perm_mean = perm_wds.mean()
        perm_std = perm_wds.std()
        z_score = (observed_wd - perm_mean) / (perm_std + 1e-10)
        p_value = float(np.mean(perm_wds >= observed_wd))

        print(f"  Observed W1: {observed_wd:.4f}")
        print(f"  Null: {perm_mean:.4f} +/- {perm_std:.4f}")
        print(f"  Z-score: {z_score:.3f}, p-value: {p_value:.4f}")
    else:
        z_score, p_value, observed_wd = 0.0, 1.0, 0.0
        perm_wds = np.array([0.0])

    # -----------------------------------------------------------------------
    # Experiment 2: Severity prediction from topology
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SEVERITY PREDICTION FROM TOPOLOGY")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    # Extract per-problem topological features for all aligned problems
    print("Extracting per-problem topological features...")

    all_h_indices = list(c_to_h.values())
    all_labels = []  # 'correct', 'near_miss', 'moderate', 'hallucination'

    incorrect_idx = 0
    for c_idx in range(len(c_correct)):
        h_idx = c_to_h[c_idx]
        if c_correct[c_idx]:
            all_labels.append('correct')
        else:
            all_labels.append(severity_labels[incorrect_idx])
            incorrect_idx += 1

    all_labels = np.array(all_labels)
    all_h_indices = np.array(all_h_indices)

    # Compute token trajectory PH features per problem
    # Use terminal-layer hidden state + PCA for per-problem point cloud
    # Actually, per-problem we have a single point (1536-d vector), not a point cloud.
    # For per-problem features, use the token trajectories across layers.
    token_trajs = hidden["token_trajectories"]

    X_features = []
    valid_mask = []
    for i, h_idx in enumerate(all_h_indices):
        traj = token_trajs[h_idx]
        if traj is None or (isinstance(traj, np.ndarray) and traj.size == 0):
            X_features.append(np.zeros(len(FEATURE_NAMES)))
            valid_mask.append(False)
            continue
        traj = np.array(traj, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] < 3:
            X_features.append(np.zeros(len(FEATURE_NAMES)))
            valid_mask.append(False)
            continue

        # PCA reduce if needed
        n_comp = min(args.n_pca, traj.shape[0] - 1, traj.shape[1])
        if n_comp > 0 and traj.shape[1] > n_comp:
            pca = PCA(n_components=n_comp, random_state=args.seed)
            traj = pca.fit_transform(traj)
        # Subsample tokens if too many
        if traj.shape[0] > args.subsample:
            idx = rng.choice(traj.shape[0], args.subsample, replace=False)
            traj = traj[idx]

        try:
            dgms = compute_persistence(traj, max_dim=1)
            X_features.append(extract_topo_features(dgms))
            valid_mask.append(True)
        except Exception:
            X_features.append(np.zeros(len(FEATURE_NAMES)))
            valid_mask.append(False)

    X = np.array(X_features, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = np.array(valid_mask)

    print(f"Feature matrix: {X.shape}, {valid_mask.sum()} valid problems")

    # --- 3-class: correct vs near_miss vs hallucination ---
    # Merge moderate into hallucination for cleaner signal if moderate is small
    y_3class = np.array([
        0 if l == 'correct' else
        1 if l == 'near_miss' else
        2  # moderate + hallucination
        for l in all_labels
    ])
    class_names = ['correct', 'near_miss', 'halluc+moderate']

    mask_valid = valid_mask
    X_v = X[mask_valid]
    y_3c_v = y_3class[mask_valid]

    print(f"\n3-class distribution: {dict(zip(*np.unique(y_3c_v, return_counts=True)))}")

    # Stratified 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    f1_scores = []
    acc_scores = []

    for train_idx, test_idx in skf.split(X_v, y_3c_v):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_v[train_idx])
        X_te = scaler.transform(X_v[test_idx])

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed,
                                 )
        clf.fit(X_tr, y_3c_v[train_idx])
        y_pred = clf.predict(X_te)
        f1_scores.append(f1_score(y_3c_v[test_idx], y_pred, average='macro'))
        acc_scores.append(accuracy_score(y_3c_v[test_idx], y_pred))

    f1_3class = float(np.mean(f1_scores))
    acc_3class = float(np.mean(acc_scores))
    print(f"3-class macro F1: {f1_3class:.3f} (+/- {np.std(f1_scores):.3f})")
    print(f"3-class accuracy: {acc_3class:.3f} (+/- {np.std(acc_scores):.3f})")

    # --- Binary: correct vs hallucination+moderate ---
    mask_ch = np.isin(y_3c_v, [0, 2])
    X_ch = X_v[mask_ch]
    y_ch = (y_3c_v[mask_ch] == 2).astype(int)

    auroc_ch = 0.5
    if len(np.unique(y_ch)) == 2 and np.sum(y_ch == 0) >= 5 and np.sum(y_ch == 1) >= 5:
        aurocs_ch = []
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for train_idx, test_idx in skf2.split(X_ch, y_ch):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_ch[train_idx])
            X_te = scaler.transform(X_ch[test_idx])
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
            clf.fit(X_tr, y_ch[train_idx])
            y_prob = clf.predict_proba(X_te)[:, 1]
            if len(np.unique(y_ch[test_idx])) > 1:
                aurocs_ch.append(roc_auc_score(y_ch[test_idx], y_prob))
        auroc_ch = float(np.mean(aurocs_ch)) if aurocs_ch else 0.5
    print(f"Binary AUROC (correct vs halluc+moderate): {auroc_ch:.3f}")

    # --- Binary: near_miss vs hallucination+moderate ---
    mask_nh = np.isin(y_3c_v, [1, 2])
    X_nh = X_v[mask_nh]
    y_nh = (y_3c_v[mask_nh] == 2).astype(int)

    auroc_nh = 0.5
    if len(np.unique(y_nh)) == 2 and np.sum(y_nh == 0) >= 5 and np.sum(y_nh == 1) >= 5:
        aurocs_nh = []
        skf3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for train_idx, test_idx in skf3.split(X_nh, y_nh):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_nh[train_idx])
            X_te = scaler.transform(X_nh[test_idx])
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed)
            clf.fit(X_tr, y_nh[train_idx])
            y_prob = clf.predict_proba(X_te)[:, 1]
            if len(np.unique(y_nh[test_idx])) > 1:
                aurocs_nh.append(roc_auc_score(y_nh[test_idx], y_prob))
        auroc_nh = float(np.mean(aurocs_nh)) if aurocs_nh else 0.5
    print(f"Binary AUROC (near_miss vs halluc+moderate): {auroc_nh:.3f}")

    # Feature importance (full model, 3-class)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_v)
    clf_full = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed,
                                  )
    clf_full.fit(X_s, y_3c_v)
    importance = np.mean(np.abs(clf_full.coef_), axis=0)
    top_idx = np.argsort(importance)[::-1]
    top_features = [(FEATURE_NAMES[i], float(importance[i])) for i in top_idx[:10]]
    print("\nTop features (3-class importance):")
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")

    # -----------------------------------------------------------------------
    # Experiment 3: Layer-wise severity signal
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: LAYER-WISE SEVERITY SIGNAL")
    print("=" * 70)

    n_layers = int(hidden["num_layers"])
    sev_bins_exp3 = {
        'near_miss': near_miss_h,
        'hallucination': halluc_h,
    }

    if len(near_miss_h) < 3 or len(halluc_h) < 3:
        print("Skipping: insufficient data in bins")
        peak_layer, peak_z = -1, 0.0
        layer_z_scores = []
    else:
        print(f"Computing per-layer z-scores ({n_layers} layers, {args.n_perms} perms each)...")
        print(f"  near_miss: n={len(near_miss_h)}, hallucination: n={len(halluc_h)}")

        # For each layer: compute Wasserstein between near_miss and hallucination PH
        # then permutation test
        layer_z_scores = []
        all_sev_h = np.concatenate([near_miss_h, halluc_h])
        nm_n = len(near_miss_h)

        for layer in range(n_layers):
            if (layer + 1) % 5 == 0 or layer == 0:
                print(f"  Layer {layer + 1}/{n_layers}...")

            # Observed
            cloud_nm = get_cloud(near_miss_h, layer=layer)
            cloud_hal = get_cloud(halluc_h, layer=layer)
            dgm_nm = compute_persistence(cloud_nm, max_dim=1)
            dgm_hal = compute_persistence(cloud_hal, max_dim=1)
            obs_wd = wasserstein_distance(dgm_nm[1], dgm_hal[1])

            # Permutation null
            null_wds = []
            for _ in range(min(args.n_perms, 50)):  # Cap at 50 per layer for speed
                perm = rng.permutation(len(all_sev_h))
                perm_nm = all_sev_h[perm[:nm_n]]
                perm_hal = all_sev_h[perm[nm_n:]]

                c_nm = get_cloud(perm_nm, layer=layer)
                c_hal = get_cloud(perm_hal, layer=layer)
                d_nm = compute_persistence(c_nm, max_dim=1)
                d_hal = compute_persistence(c_hal, max_dim=1)
                null_wds.append(wasserstein_distance(d_nm[1], d_hal[1]))

            null_wds = np.array(null_wds)
            z = (obs_wd - null_wds.mean()) / (null_wds.std() + 1e-10)
            layer_z_scores.append(float(z))

        layer_z_scores_arr = np.array(layer_z_scores)
        peak_layer = int(np.argmax(layer_z_scores_arr))
        peak_z = float(layer_z_scores_arr[peak_layer])
        print(f"\nPeak severity signal: layer {peak_layer + 1}/{n_layers} (z={peak_z:.3f})")
        print(f"Terminal layer z: {layer_z_scores_arr[-1]:.3f}")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # Fig 1: Severity distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ['Correct', 'Near-miss', 'Moderate', 'Hallucination']
    counts = [n_correct, n_near_miss, n_moderate, n_hallucination]
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    ax.bar(cats, counts, color=colors, edgecolor='black', linewidth=0.5)
    for i, (c, cnt) in enumerate(zip(cats, counts)):
        ax.text(i, cnt + 2, str(cnt), ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel("Number of problems")
    ax.set_title(f"Error Severity Distribution (method: {severity_method})")
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(os.path.join(FIG_DIR, "severity_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 2: H0/H1 entropy by severity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, dim_name in enumerate(['H0', 'H1']):
        ax = axes[ax_idx]
        bin_names = [k for k in ['correct', 'near_miss', 'moderate', 'hallucination'] if k in bin_entropy]
        entropies = [bin_entropy[k][dim_name] for k in bin_names]
        ax.bar(bin_names, entropies, color=[colors[['correct', 'near_miss', 'moderate', 'hallucination'].index(k)]
                                            for k in bin_names],
               edgecolor='black', linewidth=0.5)
        ax.set_ylabel(f"{dim_name} Persistence Entropy")
        ax.set_title(f"{dim_name} Entropy by Severity")
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle("Topological Complexity by Error Severity", fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "entropy_by_severity.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 3: Permutation test null distribution
    if len(perm_wds) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(perm_wds, bins=30, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5,
                label='Null distribution')
        ax.axvline(observed_wd, color='red', linewidth=2, linestyle='--',
                   label=f'Observed W1={observed_wd:.4f}')
        ax.set_xlabel("Wasserstein-1 Distance (H1)")
        ax.set_ylabel("Count")
        ax.set_title(f"Permutation Test: Near-miss vs Hallucination (z={z_score:.2f}, p={p_value:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(FIG_DIR, "permutation_test.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Fig 4: Layer-wise z-score profile
    if layer_z_scores:
        fig, ax = plt.subplots(figsize=(10, 5))
        layers = np.arange(1, n_layers + 1)
        ax.plot(layers, layer_z_scores, 'o-', color='darkred', linewidth=2, markersize=4)
        ax.axhline(1.96, color='gray', linestyle='--', alpha=0.5, label='z=1.96')
        ax.axvline(peak_layer + 1, color='red', linestyle=':', alpha=0.5,
                   label=f'Peak: L{peak_layer + 1} (z={peak_z:.2f})')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Z-score (near-miss vs hallucination)")
        ax.set_title("Layer-wise Severity Discrimination Signal")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(FIG_DIR, "layerwise_severity_zscore.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Fig 5: Feature importance
    fig, ax = plt.subplots(figsize=(10, 5))
    top_n = min(len(top_features), 12)
    names_plot = [f[0] for f in top_features[:top_n]]
    vals_plot = [f[1] for f in top_features[:top_n]]
    ax.barh(range(top_n), vals_plot[::-1], color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_plot[::-1], fontsize=9)
    ax.set_xlabel("|Mean Coefficient|")
    ax.set_title("Feature Importance for Severity Classification (3-class)")
    ax.grid(True, alpha=0.3, axis='x')
    fig.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Figures saved to", FIG_DIR)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "branch": "experiment/tda-hallucination",
        "n_correct": n_correct,
        "n_near_miss": n_near_miss,
        "n_moderate": n_moderate,
        "n_hallucination": n_hallucination,
        "severity_method": severity_method,
        "exp1_h0_entropy_by_severity": {k: v['H0'] for k, v in bin_entropy.items()},
        "exp1_h1_entropy_by_severity": {k: v['H1'] for k, v in bin_entropy.items()},
        "exp1_wasserstein_pairs": {k: round(v, 6) for k, v in wd_pairs.items()},
        "exp1_wasserstein_z": round(z_score, 4),
        "exp1_wasserstein_p": round(p_value, 4),
        "exp1_n_perms": args.n_perms,
        "exp2_3class_f1": round(f1_3class, 4),
        "exp2_3class_accuracy": round(acc_3class, 4),
        "exp2_correct_vs_hallucination_auroc": round(auroc_ch, 4),
        "exp2_nearmiss_vs_hallucination_auroc": round(auroc_nh, 4),
        "exp2_top_features": top_features,
        "exp3_layer_z_scores": [round(z, 4) for z in layer_z_scores],
        "exp3_peak_severity_layer": peak_layer + 1 if peak_layer >= 0 else None,
        "exp3_peak_severity_zscore": round(peak_z, 4) if peak_layer >= 0 else None,
        "exp3_terminal_layer_zscore": round(layer_z_scores[-1], 4) if layer_z_scores else None,
        "exp3_n_perms_per_layer": min(args.n_perms, 50),
        "config": {
            "seed": args.seed,
            "n_perms": args.n_perms,
            "subsample": args.subsample,
            "n_pca": args.n_pca,
        },
    }

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Severity method: {severity_method}")
    print(f"Distribution: {n_correct} correct, {n_near_miss} near-miss, {n_moderate} moderate, {n_hallucination} hallucination")
    print(f"Exp1 — Wasserstein z={z_score:.3f}, p={p_value:.4f}")
    print(f"Exp2 — 3-class F1={f1_3class:.3f}, correct-vs-halluc AUROC={auroc_ch:.3f}, nm-vs-halluc AUROC={auroc_nh:.3f}")
    if layer_z_scores:
        print(f"Exp3 — Peak layer={peak_layer + 1}/{n_layers}, z={peak_z:.3f}, terminal z={layer_z_scores[-1]:.3f}")


if __name__ == "__main__":
    main()
