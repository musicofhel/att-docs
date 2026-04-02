"""Direction 2: Predict problem-solving correctness from topological features.

Trains logistic regression on persistence-derived features to predict whether
a model will solve a math problem correctly. Reports AUROC, accuracy, and
feature importance.

Usage:
    python scripts/run_correctness_prediction.py
    python scripts/run_correctness_prediction.py --correctness data/transformer/math500_correctness.npz
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from att.config import set_seed
from att.llm import HiddenStateLoader, TopologicalFeatureExtractor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "transformer")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "llm")


def main():
    parser = argparse.ArgumentParser(description="Correctness prediction from TDA features")
    parser.add_argument(
        "--hidden-states",
        default=os.path.join(DATA_DIR, "math500_hidden_states.npz"),
    )
    parser.add_argument(
        "--correctness",
        default=os.path.join(DATA_DIR, "math500_correctness.npz"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-set", choices=["summary", "image"], default="summary")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load data
    loader = HiddenStateLoader(args.hidden_states)
    print(loader)

    correctness_data = np.load(args.correctness, allow_pickle=True)
    correct_all = correctness_data["correct"]
    correct_levels = correctness_data["difficulty_levels"]
    print(f"Correctness labels: {correct_all.sum()}/{len(correct_all)} correct ({correct_all.mean():.1%})")

    # Extract features per problem (token trajectories)
    print(f"\nExtracting {args.feature_set} features per problem...")
    tfe = TopologicalFeatureExtractor(
        max_dim=1,
        feature_set=args.feature_set,
        n_pca_components=30,
        subsample=100,
        seed=args.seed,
    )
    X, levels = tfe.extract_per_problem(loader)
    feature_names = tfe.feature_names

    # Align: use min(hidden, correctness) per level
    # Both use seed=42 level-sorted sampling, so first N match per level
    hidden_levels = loader.levels
    aligned_hidden_idx = []
    aligned_correct_idx = []
    for lv in range(1, 6):
        h_idx = np.where(hidden_levels == lv)[0]
        c_idx = np.where(correct_levels == lv)[0]
        n = min(len(h_idx), len(c_idx))
        aligned_hidden_idx.extend(h_idx[:n].tolist())
        aligned_correct_idx.extend(c_idx[:n].tolist())

    aligned_hidden_idx = np.array(aligned_hidden_idx)
    aligned_correct_idx = np.array(aligned_correct_idx)

    X = X[aligned_hidden_idx]
    levels = levels[aligned_hidden_idx]
    y = correct_all[aligned_correct_idx]

    print(f"Feature matrix: {X.shape} (aligned from {len(hidden_levels)} hidden, {len(correct_all)} correctness)")
    print(f"Labels: {y.sum()} correct, {(~y).sum()} incorrect")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

    # --- Overall prediction ---
    print("\n--- Overall Correctness Prediction (5-fold CV) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    aurocs, accs = [], []
    all_y_true, all_y_prob = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            C=1.0, penalty="l2", max_iter=1000, random_state=args.seed
        )
        clf.fit(X_train_s, y_train)

        y_prob = clf.predict_proba(X_test_s)[:, 1]
        y_pred = clf.predict(X_test_s)

        if len(np.unique(y_test)) > 1:
            aurocs.append(roc_auc_score(y_test, y_prob))
        accs.append(accuracy_score(y_test, y_pred))
        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

    mean_auroc = np.mean(aurocs) if aurocs else 0.5
    mean_acc = np.mean(accs)
    print(f"AUROC: {mean_auroc:.3f} (±{np.std(aurocs):.3f})")
    print(f"Accuracy: {mean_acc:.3f} (±{np.std(accs):.3f})")

    # --- Per-difficulty prediction ---
    print("\n--- Per-Difficulty Prediction ---")
    per_level_results = {}
    for level in sorted(np.unique(levels)):
        mask = levels == level
        X_level = X[mask]
        y_level = y[mask]
        n_correct = y_level.sum()
        n_total = len(y_level)

        if n_correct < 2 or (n_total - n_correct) < 2:
            print(f"Level {level}: skipped (too few of one class: {n_correct}/{n_total})")
            per_level_results[int(level)] = {"auroc": None, "n": int(n_total)}
            continue

        skf_level = StratifiedKFold(n_splits=min(5, min(n_correct, n_total - n_correct)),
                                     shuffle=True, random_state=args.seed)
        level_aurocs = []
        for train_idx, test_idx in skf_level.split(X_level, y_level):
            X_tr, X_te = X_level[train_idx], X_level[test_idx]
            y_tr, y_te = y_level[train_idx], y_level[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=args.seed)
            clf.fit(X_tr_s, y_tr)

            if len(np.unique(y_te)) > 1:
                y_prob_l = clf.predict_proba(X_te_s)[:, 1]
                level_aurocs.append(roc_auc_score(y_te, y_prob_l))

        level_auroc = np.mean(level_aurocs) if level_aurocs else None
        auroc_str = f"{level_auroc:.3f}" if level_auroc is not None else "N/A"
        print(f"Level {level}: AUROC={auroc_str} ({n_correct}/{n_total} correct)")
        per_level_results[int(level)] = {
            "auroc": float(level_auroc) if level_auroc else None,
            "n": int(n_total),
            "n_correct": int(n_correct),
        }

    # --- Feature importance ---
    print("\n--- Feature Importance (full model) ---")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf_full = LogisticRegression(C=1.0, penalty="l2", max_iter=1000, random_state=args.seed)
    clf_full.fit(X_s, y)
    importance = np.abs(clf_full.coef_[0])
    top_idx = np.argsort(importance)[::-1][:10]
    print("Top 10 features:")
    for i in top_idx:
        print(f"  {feature_names[i]}: {importance[i]:.4f}")

    # --- ROC plot ---
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    if len(np.unique(all_y_true)) > 1:
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"Overall AUROC={mean_auroc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Correctness Prediction from Topological Features")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(FIGURES_DIR, "correctness_roc.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nSaved: {path}")

    # --- Feature importance bar plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    top_n = min(15, len(feature_names), len(top_idx))
    top_importance = importance[top_idx[:top_n]]
    top_names = [feature_names[i] for i in top_idx[:top_n]]
    ax.barh(range(top_n), top_importance[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("|Coefficient|")
    ax.set_title("Feature Importance for Correctness Prediction")
    ax.grid(True, alpha=0.3, axis="x")
    path = os.path.join(FIGURES_DIR, "correctness_feature_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # --- Save results ---
    results = {
        "overall_auroc": float(mean_auroc),
        "overall_accuracy": float(mean_acc),
        "per_level": per_level_results,
        "top_features": {feature_names[i]: float(importance[i]) for i in top_idx[:10]},
        "feature_set": args.feature_set,
        "n_features": len(feature_names),
        "n_problems": int(len(y)),
        "config": {"seed": args.seed, "feature_set": args.feature_set},
    }
    out_path = os.path.join(DATA_DIR, "correctness_prediction_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
