#!/usr/bin/env python3
"""
Manifold Shape Characterization: What Geometry Predicts Difficulty?

Computes 70+ manifold descriptors per problem, finds correlations with
difficulty and correctness, and identifies which manifold types map
onto specific difficulty levels.

Branch: experiment/tda-manifold-shapes
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ripser
from persim import wasserstein as wasserstein_distance
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Data loading ──────────────────────────────────────────────────────────────

print("Loading data...")
data = np.load("data/transformer/math500_hidden_states_aligned.npz", allow_pickle=True)
token_trajs = data["token_trajectories"]
levels = data["difficulty_levels"]

corr_data = np.load("data/transformer/math500_correctness.npz", allow_pickle=True)
correct = corr_data["correct"]

print(f"  {len(token_trajs)} problems, {len(levels)} levels, {len(correct)} correctness labels")


# ── Helper ────────────────────────────────────────────────────────────────────

def persistence_entropy(lifetimes):
    """Shannon entropy of the persistence lifetime distribution."""
    if len(lifetimes) == 0 or np.sum(lifetimes) == 0:
        return 0.0
    p = lifetimes / np.sum(lifetimes)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ── Group A: Persistence-Based ────────────────────────────────────────────────

def persistence_descriptors(cloud, max_dim=2, subsample=100):
    """Full persistence descriptor battery."""
    n = cloud.shape[0]
    if n > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, subsample, replace=False)
        cloud = cloud[idx]

    result = ripser.ripser(cloud, maxdim=max_dim)
    dgms = result["dgms"]

    features = {}
    for d in range(max_dim + 1):
        dgm = dgms[d]
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            lifetimes = np.array([0.0])
        else:
            lifetimes = finite[:, 1] - finite[:, 0]

        features[f"H{d}_count"] = len(finite)
        features[f"H{d}_total_persistence"] = float(np.sum(lifetimes))
        features[f"H{d}_max_lifetime"] = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0
        features[f"H{d}_mean_lifetime"] = float(np.mean(lifetimes))
        features[f"H{d}_std_lifetime"] = float(np.std(lifetimes))
        features[f"H{d}_entropy"] = float(persistence_entropy(lifetimes))

        features[f"H{d}_lifetime_skew"] = float(skew(lifetimes)) if len(lifetimes) > 2 else 0.0
        features[f"H{d}_lifetime_kurtosis"] = float(kurtosis(lifetimes)) if len(lifetimes) > 3 else 0.0
        features[f"H{d}_lifetime_q25"] = float(np.percentile(lifetimes, 25)) if len(lifetimes) > 0 else 0.0
        features[f"H{d}_lifetime_q75"] = float(np.percentile(lifetimes, 75)) if len(lifetimes) > 0 else 0.0
        features[f"H{d}_lifetime_iqr"] = features[f"H{d}_lifetime_q75"] - features[f"H{d}_lifetime_q25"]

        if len(finite) > 0:
            features[f"H{d}_mean_birth"] = float(np.mean(finite[:, 0]))
            features[f"H{d}_mean_death"] = float(np.mean(finite[:, 1]))
            features[f"H{d}_birth_spread"] = float(np.std(finite[:, 0]))
            features[f"H{d}_death_spread"] = float(np.std(finite[:, 1]))
        else:
            features[f"H{d}_mean_birth"] = 0.0
            features[f"H{d}_mean_death"] = 0.0
            features[f"H{d}_birth_spread"] = 0.0
            features[f"H{d}_death_spread"] = 0.0

    # Cross-dimensional ratios
    if features["H0_count"] > 0:
        features["H1_H0_ratio"] = features["H1_count"] / features["H0_count"]
    else:
        features["H1_H0_ratio"] = 0.0
    if features["H1_count"] > 0:
        features["H2_H1_ratio"] = features["H2_count"] / features["H1_count"]
    else:
        features["H2_H1_ratio"] = 0.0

    # Euler characteristic approximation
    features["euler_approx"] = features["H0_count"] - features["H1_count"] + features["H2_count"]

    return features


# ── Group B: Intrinsic Geometry ───────────────────────────────────────────────

def geometry_descriptors(cloud, k=10):
    """Intrinsic geometry without PH."""
    n, d = cloud.shape
    features = {}

    # 1. Intrinsic dimension (TwoNN)
    nn = NearestNeighbors(n_neighbors=min(3, n)).fit(cloud)
    dists, _ = nn.kneighbors(cloud)
    if dists.shape[1] >= 3:
        mu = dists[:, 2] / (dists[:, 1] + 1e-15)
        mu = mu[mu > 1]
        features["intrinsic_dim_twonn"] = float(1.0 / np.mean(np.log(mu))) if len(mu) > 0 else 0.0
    else:
        features["intrinsic_dim_twonn"] = 0.0

    # 2. Intrinsic dimension (MLE, Levina-Bickel)
    k_actual = min(k + 1, n)
    nn_k = NearestNeighbors(n_neighbors=k_actual).fit(cloud)
    dists_k, _ = nn_k.kneighbors(cloud)
    dists_k = dists_k[:, 1:]  # exclude self
    if dists_k.shape[1] >= 2:
        log_ratios = np.log(dists_k[:, -1:] / (dists_k[:, :-1] + 1e-15) + 1e-15)
        features["intrinsic_dim_mle"] = float(1.0 / np.mean(log_ratios)) if np.mean(log_ratios) > 0 else 0.0
    else:
        features["intrinsic_dim_mle"] = 0.0

    # 3. Local dimension variance
    local_dims = []
    for i in range(n):
        d_i = dists_k[i]
        if len(d_i) >= 2 and d_i[-1] > 0:
            lr = np.log(d_i[-1] / (d_i[:-1] + 1e-15) + 1e-15)
            if np.mean(lr) > 0:
                local_dims.append(1.0 / np.mean(lr))
    features["dim_variance"] = float(np.var(local_dims)) if local_dims else 0.0
    features["dim_cv"] = float(np.std(local_dims) / np.mean(local_dims)) if local_dims and np.mean(local_dims) > 0 else 0.0

    # 4. Isotropy
    centered = cloud - cloud.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    sv = sv[sv > 0]
    if len(sv) > 0:
        features["isotropy"] = float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0
        sv_norm = sv / sv.sum()
        features["effective_rank_sv"] = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-15))))
        features["sv_entropy"] = float(-np.sum(sv_norm * np.log(sv_norm + 1e-15)))
        features["condition_number"] = float(sv[0] / sv[-1]) if sv[-1] > 0 else 1e10
    else:
        features["isotropy"] = 0.0
        features["effective_rank_sv"] = 0.0
        features["sv_entropy"] = 0.0
        features["condition_number"] = 1e10

    # 5. Volume estimate
    k_eff = min(max(int(features.get("intrinsic_dim_twonn", 5)), 1), len(sv))
    features["volume_proxy"] = float(np.prod(sv[:k_eff]))

    return features


# ── Group C: Clustering and Connectivity ──────────────────────────────────────

def clustering_descriptors(cloud, max_k=8):
    """Cluster structure analysis."""
    n = cloud.shape[0]
    features = {}

    # 1. Optimal k via silhouette
    best_sil = -1
    best_k = 1
    for k in range(2, min(max_k + 1, n)):
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=42)
            labels = km.fit_predict(cloud)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(cloud, labels)
            features[f"silhouette_k{k}"] = float(sil)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        except Exception:
            features[f"silhouette_k{k}"] = 0.0
    features["optimal_k"] = best_k
    features["best_silhouette"] = float(best_sil) if best_sil > -1 else 0.0

    # 2. DBSCAN
    cloud_scaled = StandardScaler().fit_transform(cloud)
    for eps in [0.5, 1.0, 2.0]:
        db = DBSCAN(eps=eps, min_samples=3).fit(cloud_scaled)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        noise_frac = float(np.mean(db.labels_ == -1))
        features[f"dbscan_eps{eps}_clusters"] = n_clusters
        features[f"dbscan_eps{eps}_noise"] = noise_frac

    # 3. Nearest-neighbor graph statistics
    nn = NearestNeighbors(n_neighbors=min(6, n)).fit(cloud)
    dists, indices = nn.kneighbors(cloud)
    if dists.shape[1] >= 2:
        features["mean_nn_dist"] = float(np.mean(dists[:, 1]))
        features["std_nn_dist"] = float(np.std(dists[:, 1]))
        features["nn_dist_cv"] = float(np.std(dists[:, 1]) / np.mean(dists[:, 1])) if np.mean(dists[:, 1]) > 0 else 0.0
    else:
        features["mean_nn_dist"] = 0.0
        features["std_nn_dist"] = 0.0
        features["nn_dist_cv"] = 0.0

    # 4. Hub score
    if dists.shape[1] >= 2:
        in_degree = np.bincount(indices[:, 1], minlength=n)
        features["max_hub_degree"] = int(np.max(in_degree))
        in_norm = in_degree / (in_degree.sum() + 1e-15)
        features["hub_entropy"] = float(-np.sum(in_norm * np.log(in_norm + 1e-15)))
    else:
        features["max_hub_degree"] = 0
        features["hub_entropy"] = 0.0

    return features


# ── Group D: Curvature ────────────────────────────────────────────────────────

def curvature_descriptors(cloud, k=10):
    """Discrete curvature estimates on k-NN graph."""
    n = cloud.shape[0]
    k_actual = min(k + 1, n)

    nn = NearestNeighbors(n_neighbors=k_actual).fit(cloud)
    dists, indices = nn.kneighbors(cloud)

    features = {}

    # Ollivier-Ricci curvature approximation
    curvatures = []
    for i in range(min(n, 200)):
        if dists.shape[1] < 2:
            break
        j = indices[i, 1]
        d_ij = dists[i, 1]
        if d_ij < 1e-15:
            continue
        ni = set(indices[i, 1:].tolist()) - {j}
        nj = set(indices[j, 1:].tolist()) - {i}
        if not ni or not nj:
            continue
        ni_pts = cloud[list(ni)]
        nj_pts = cloud[list(nj)]
        cross_dists = np.linalg.norm(ni_pts[:, None] - nj_pts[None, :], axis=2)
        w1_approx = float(np.mean(np.min(cross_dists, axis=1)))
        curv = 1.0 - w1_approx / d_ij
        curvatures.append(curv)

    if curvatures:
        features["mean_curvature"] = float(np.mean(curvatures))
        features["std_curvature"] = float(np.std(curvatures))
        features["min_curvature"] = float(np.min(curvatures))
        features["max_curvature"] = float(np.max(curvatures))
        features["negative_curvature_frac"] = float(np.mean(np.array(curvatures) < 0))
    else:
        features["mean_curvature"] = 0.0
        features["std_curvature"] = 0.0
        features["min_curvature"] = 0.0
        features["max_curvature"] = 0.0
        features["negative_curvature_frac"] = 0.0

    return features


# ── Group E: Shape Comparison to Templates ────────────────────────────────────

def template_distances(cloud, subsample=100):
    """Distance from the point cloud to known manifold templates."""
    n = cloud.shape[0]
    d = cloud.shape[1]
    rng = np.random.default_rng(42)
    if n > subsample:
        cloud = cloud[rng.choice(n, subsample, replace=False)]
        n = subsample

    # Compute PH on the actual cloud
    result = ripser.ripser(cloud, maxdim=1)
    dgm = result["dgms"][1]
    dgm_finite = dgm[np.isfinite(dgm[:, 1])]

    features = {}
    cloud_scale = np.mean(np.linalg.norm(cloud - cloud.mean(0), axis=1))
    cloud_std = np.std(cloud, axis=0)
    cloud_mean = np.mean(cloud, axis=0)

    def compute_template_dist(template_pts, name):
        r = ripser.ripser(template_pts, maxdim=1)
        dgm_t = r["dgms"][1]
        dgm_t_f = dgm_t[np.isfinite(dgm_t[:, 1])]
        if len(dgm_finite) > 0 and len(dgm_t_f) > 0:
            features[f"dist_to_{name}"] = float(wasserstein_distance(dgm_finite, dgm_t_f))
        elif len(dgm_finite) == 0 and len(dgm_t_f) == 0:
            features[f"dist_to_{name}"] = 0.0
        else:
            # One has features, other doesn't — use total persistence as proxy
            total_a = float(np.sum(dgm_finite[:, 1] - dgm_finite[:, 0])) if len(dgm_finite) > 0 else 0.0
            total_b = float(np.sum(dgm_t_f[:, 1] - dgm_t_f[:, 0])) if len(dgm_t_f) > 0 else 0.0
            features[f"dist_to_{name}"] = abs(total_a - total_b)

    # Template 1: Sphere
    sphere_pts = rng.standard_normal((subsample, d))
    sphere_pts /= np.linalg.norm(sphere_pts, axis=1, keepdims=True)
    sphere_pts *= cloud_scale
    compute_template_dist(sphere_pts, "sphere")

    # Template 2: Gaussian blob
    blob_pts = rng.standard_normal((subsample, d))
    blob_pts *= cloud_std
    blob_pts += cloud_mean
    compute_template_dist(blob_pts, "blob")

    # Template 3: Two clusters
    half = subsample // 2
    two_cluster = np.vstack([
        rng.standard_normal((half, d)) * 0.5 + cloud_mean - cloud_std,
        rng.standard_normal((subsample - half, d)) * 0.5 + cloud_mean + cloud_std,
    ])
    compute_template_dist(two_cluster, "two_clusters")

    # Template 4: Noisy circle
    theta = rng.uniform(0, 2 * np.pi, subsample)
    circle_pts = np.zeros((subsample, d))
    circle_pts[:, 0] = np.cos(theta) * cloud_scale
    circle_pts[:, 1] = np.sin(theta) * cloud_scale
    circle_pts += rng.standard_normal((subsample, d)) * 0.1 * np.mean(cloud_std)
    compute_template_dist(circle_pts, "circle")

    # Template 5: Noisy torus
    theta1 = rng.uniform(0, 2 * np.pi, subsample)
    theta2 = rng.uniform(0, 2 * np.pi, subsample)
    R, r_t = 2.0, 1.0
    torus_pts = np.zeros((subsample, d))
    torus_pts[:, 0] = (R + r_t * np.cos(theta2)) * np.cos(theta1) * cloud_scale / 3.0
    torus_pts[:, 1] = (R + r_t * np.cos(theta2)) * np.sin(theta1) * cloud_scale / 3.0
    torus_pts[:, 2] = r_t * np.sin(theta2) * cloud_scale / 3.0
    torus_pts += rng.standard_normal((subsample, d)) * 0.1 * np.mean(cloud_std)
    compute_template_dist(torus_pts, "torus")

    # Which template is closest?
    template_dists = {
        "sphere": features["dist_to_sphere"],
        "blob": features["dist_to_blob"],
        "two_clusters": features["dist_to_two_clusters"],
        "circle": features["dist_to_circle"],
        "torus": features["dist_to_torus"],
    }
    features["nearest_template"] = min(template_dists, key=template_dists.get)
    features["nearest_template_dist"] = float(min(template_dists.values()))

    return features


# ── Part 2: Compute Everything ────────────────────────────────────────────────

print("\nComputing manifold descriptors for all problems...")
all_features = []
all_levels = []
all_correct = []
skipped = 0

for idx in range(len(token_trajs)):
    traj = token_trajs[idx]
    if traj.shape[0] < 20:
        skipped += 1
        continue

    n_comp = min(30, traj.shape[0] - 1, traj.shape[1])
    if n_comp < 3:
        skipped += 1
        continue

    pca = PCA(n_components=n_comp)
    cloud = pca.fit_transform(traj)

    feats = {}
    feats.update(persistence_descriptors(cloud))
    feats.update(geometry_descriptors(cloud))
    feats.update(clustering_descriptors(cloud))
    feats.update(curvature_descriptors(cloud))
    feats.update(template_distances(cloud))

    all_features.append(feats)
    all_levels.append(int(levels[idx]))
    all_correct.append(bool(correct[idx]) if idx < len(correct) else False)

    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(token_trajs)} problems ({skipped} skipped)...")

print(f"  Done: {len(all_features)} problems processed, {skipped} skipped")

# Convert to DataFrame
df = pd.DataFrame(all_features)
df["level"] = all_levels
df["correct"] = all_correct

feature_cols = [c for c in df.columns if c not in ("level", "correct", "nearest_template")]
n_features = len(feature_cols)
print(f"  {n_features} numeric features computed")


# ── Part 3a: Spearman correlation with difficulty ─────────────────────────────

print("\n=== Correlations with Difficulty Level ===")
diff_corrs = {}
for col in feature_cols:
    vals = df[col].values.astype(float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    r, p = stats.spearmanr(df["level"], vals)
    diff_corrs[col] = {"r": float(r) if np.isfinite(r) else 0.0,
                       "p": float(p) if np.isfinite(p) else 1.0}

sorted_diff = sorted(diff_corrs.items(), key=lambda x: abs(x[1]["r"]), reverse=True)

print("\nTop 20 features correlated with difficulty:")
for feat, vals in sorted_diff[:20]:
    sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else "ns"
    print(f"  {feat:>35s}: r={vals['r']:+.3f}  p={vals['p']:.4f} {sig}")


# ── Part 3b: Correlations with correctness ────────────────────────────────────

print("\n=== Correlations with Correctness ===")
corr_corrs = {}
y_int = df["correct"].astype(int).values
for col in feature_cols:
    vals = df[col].values.astype(float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    r, p = stats.pointbiserialr(y_int, vals)
    corr_corrs[col] = {"r": float(r) if np.isfinite(r) else 0.0,
                       "p": float(p) if np.isfinite(p) else 1.0}

sorted_corr = sorted(corr_corrs.items(), key=lambda x: abs(x[1]["r"]), reverse=True)

print("\nTop 20 features correlated with correctness:")
for feat, vals in sorted_corr[:20]:
    sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else "ns"
    print(f"  {feat:>35s}: r={vals['r']:+.3f}  p={vals['p']:.4f} {sig}")


# ── Part 3c: Template frequency by difficulty ─────────────────────────────────

print("\n=== Template Distribution by Difficulty Level ===")
template_dist = {}
for level in sorted(df["level"].unique()):
    mask = df["level"] == level
    templates = df.loc[mask, "nearest_template"].value_counts(normalize=True)
    template_dist[str(level)] = {k: round(float(v), 3) for k, v in templates.items()}
    print(f"Level {level}: {dict(templates.round(3))}")


# ── Part 3d: Multivariate prediction ─────────────────────────────────────────

print("\n=== Multivariate AUROC (5-fold CV) ===")
X_all = df[feature_cols].values.astype(float)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
y = df["correct"].values.astype(int)

pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, C=0.1))])

auroc_results = {}

scores = cross_val_score(pipe, X_all, y, cv=5, scoring="roc_auc")
auroc_results["full_battery"] = float(scores.mean())
print(f"Full battery AUROC: {scores.mean():.3f} +/- {scores.std():.3f}")

# Persistence-only
pers_cols = [c for c in feature_cols if c.startswith("H0_") or c.startswith("H1_") or c.startswith("H2_") or c.startswith("euler") or c.startswith("H1_H0") or c.startswith("H2_H1")]
X_pers = np.nan_to_num(df[pers_cols].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
scores_pers = cross_val_score(pipe, X_pers, y, cv=5, scoring="roc_auc")
auroc_results["persistence"] = float(scores_pers.mean())
print(f"Persistence-only AUROC: {scores_pers.mean():.3f} +/- {scores_pers.std():.3f}")

# Geometry-only
geom_cols = [c for c in feature_cols if any(k in c for k in ["dim", "isotropy", "curvature", "volume", "condition", "sv_", "effective_rank"])]
X_geom = np.nan_to_num(df[geom_cols].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
scores_geom = cross_val_score(pipe, X_geom, y, cv=5, scoring="roc_auc")
auroc_results["geometry"] = float(scores_geom.mean())
print(f"Geometry-only AUROC: {scores_geom.mean():.3f} +/- {scores_geom.std():.3f}")

# Clustering-only
clust_cols = [c for c in feature_cols if any(k in c for k in ["silhouette", "dbscan", "hub", "nn_dist", "optimal_k"])]
X_clust = np.nan_to_num(df[clust_cols].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
scores_clust = cross_val_score(pipe, X_clust, y, cv=5, scoring="roc_auc")
auroc_results["clustering"] = float(scores_clust.mean())
print(f"Clustering-only AUROC: {scores_clust.mean():.3f} +/- {scores_clust.std():.3f}")

# Curvature-only
curv_cols = [c for c in feature_cols if "curvature" in c]
X_curv = np.nan_to_num(df[curv_cols].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
scores_curv = cross_val_score(pipe, X_curv, y, cv=5, scoring="roc_auc")
auroc_results["curvature"] = float(scores_curv.mean())
print(f"Curvature-only AUROC: {scores_curv.mean():.3f} +/- {scores_curv.std():.3f}")

# Template-only
tmpl_cols = [c for c in feature_cols if "dist_to_" in c or "nearest_template_dist" in c]
X_tmpl = np.nan_to_num(df[tmpl_cols].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
scores_tmpl = cross_val_score(pipe, X_tmpl, y, cv=5, scoring="roc_auc")
auroc_results["template"] = float(scores_tmpl.mean())
print(f"Template-only AUROC: {scores_tmpl.mean():.3f} +/- {scores_tmpl.std():.3f}")


# ── Part 4: Figures ───────────────────────────────────────────────────────────

print("\nGenerating figures...")

# Curvature by level for results
curv_by_level = {}
for level in sorted(df["level"].unique()):
    mask = df["level"] == level
    curv_by_level[str(level)] = float(df.loc[mask, "mean_curvature"].mean())


# ── Figure 1: Correlation heatmap ─────────────────────────────────────────────

top30_feats = [f for f, _ in sorted_diff[:30]]
corr_matrix = np.zeros((len(top30_feats), 2))
for i, feat in enumerate(top30_feats):
    corr_matrix[i, 0] = diff_corrs[feat]["r"]
    corr_matrix[i, 1] = corr_corrs[feat]["r"]

fig, ax = plt.subplots(figsize=(6, 12))
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Difficulty", "Correctness"], fontsize=11)
ax.set_yticks(range(len(top30_feats)))
ax.set_yticklabels(top30_feats, fontsize=8)

# Annotate cells
for i in range(len(top30_feats)):
    for j in range(2):
        val = corr_matrix[i, j]
        color = "white" if abs(val) > 0.3 else "black"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color=color)

plt.colorbar(im, ax=ax, label="Spearman r", shrink=0.6)
ax.set_title("Top 30 Feature Correlations", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/manifold_shapes/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved correlation_heatmap.png")


# ── Figure 2: Template distribution ──────────────────────────────────────────

template_names = ["blob", "sphere", "circle", "torus", "two_clusters"]
template_colors = {"blob": "#4e79a7", "sphere": "#f28e2b", "circle": "#e15759",
                   "torus": "#76b7b2", "two_clusters": "#59a14f"}

fig, ax = plt.subplots(figsize=(10, 6))
levels_sorted = sorted(df["level"].unique())
bottoms = np.zeros(len(levels_sorted))

for tmpl in template_names:
    fracs = []
    for level in levels_sorted:
        fracs.append(template_dist.get(str(level), {}).get(tmpl, 0.0))
    ax.bar(range(len(levels_sorted)), fracs, bottom=bottoms, label=tmpl,
           color=template_colors[tmpl], edgecolor="white", linewidth=0.5)
    bottoms += np.array(fracs)

ax.set_xticks(range(len(levels_sorted)))
ax.set_xticklabels([f"Level {l}" for l in levels_sorted])
ax.set_ylabel("Fraction of Problems")
ax.set_title("Nearest Manifold Template by Difficulty Level", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", framealpha=0.9)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("figures/manifold_shapes/template_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved template_distribution.png")


# ── Figure 3: Scatter matrix of top 4 features ───────────────────────────────

top4_feats = [f for f, _ in sorted_diff[:4]]
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
level_colors = {1: "#4e79a7", 2: "#59a14f", 3: "#f28e2b", 4: "#e15759", 5: "#b07aa1"}

for i, fi in enumerate(top4_feats):
    for j, fj in enumerate(top4_feats):
        ax = axes[i][j]
        if i == j:
            # Diagonal: histogram by level
            for level in sorted(df["level"].unique()):
                mask = df["level"] == level
                vals = df.loc[mask, fi].values.astype(float)
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    ax.hist(vals, bins=15, alpha=0.5, color=level_colors.get(level, "gray"),
                            label=f"L{level}", density=True)
        else:
            for level in sorted(df["level"].unique()):
                mask = df["level"] == level
                x_vals = df.loc[mask, fj].values.astype(float)
                y_vals = df.loc[mask, fi].values.astype(float)
                valid = np.isfinite(x_vals) & np.isfinite(y_vals)
                ax.scatter(x_vals[valid], y_vals[valid], s=8, alpha=0.4,
                           color=level_colors.get(level, "gray"), label=f"L{level}")

        if i == 3:
            ax.set_xlabel(fj, fontsize=7)
        if j == 0:
            ax.set_ylabel(fi, fontsize=7)
        ax.tick_params(labelsize=6)

# Legend
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=level_colors[l],
                       markersize=8, label=f"Level {l}") for l in sorted(level_colors.keys())]
fig.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)
fig.suptitle("Scatter Matrix: Top 4 Difficulty-Correlated Features", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/manifold_shapes/scatter_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved scatter_matrix.png")


# ── Figure 4: AUROC by feature group ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
groups = list(auroc_results.keys())
aurocs = [auroc_results[g] for g in groups]
colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2", "#b07aa1"]
bars = ax.bar(range(len(groups)), aurocs, color=colors[:len(groups)], edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g.replace("_", "\n") for g in groups], fontsize=10)
ax.set_ylabel("AUROC (5-fold CV)")
ax.set_title("Correctness Prediction: AUROC by Feature Group", fontsize=13, fontweight="bold")
ax.set_ylim(0.45, max(aurocs) + 0.05)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
ax.legend()
plt.tight_layout()
plt.savefig("figures/manifold_shapes/auroc_by_group.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved auroc_by_group.png")


# ── Figure 5: Curvature violin plots ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
curv_data = []
curv_labels = []
for level in sorted(df["level"].unique()):
    mask = df["level"] == level
    vals = df.loc[mask, "mean_curvature"].values.astype(float)
    vals = vals[np.isfinite(vals)]
    curv_data.append(vals)
    curv_labels.append(f"Level {level}")

parts = ax.violinplot(curv_data, positions=range(len(curv_labels)), showmeans=True, showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(list(level_colors.values())[i % len(level_colors)])
    pc.set_alpha(0.7)

ax.set_xticks(range(len(curv_labels)))
ax.set_xticklabels(curv_labels)
ax.set_ylabel("Mean Ollivier-Ricci Curvature")
ax.set_title("Curvature Distribution by Difficulty Level", fontsize=13, fontweight="bold")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/manifold_shapes/curvature_violins.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved curvature_violins.png")


# ── Output JSON ───────────────────────────────────────────────────────────────

print("\nSaving results...")

best_diff_feat = sorted_diff[0]
best_corr_feat = sorted_corr[0]

results = {
    "branch": "experiment/tda-manifold-shapes",
    "n_problems": len(all_features),
    "n_features": n_features,
    "top_10_difficulty_correlations": [
        {"feature": f, "spearman_r": round(v["r"], 4), "p": round(v["p"], 6)}
        for f, v in sorted_diff[:10]
    ],
    "top_10_correctness_correlations": [
        {"feature": f, "spearman_r": round(v["r"], 4), "p": round(v["p"], 6)}
        for f, v in sorted_corr[:10]
    ],
    "template_distribution_by_level": template_dist,
    "auroc_by_group": {k: round(v, 4) for k, v in auroc_results.items()},
    "best_single_feature_difficulty": {
        "feature": best_diff_feat[0],
        "r": round(best_diff_feat[1]["r"], 4)
    },
    "best_single_feature_correctness": {
        "feature": best_corr_feat[0],
        "r": round(best_corr_feat[1]["r"], 4)
    },
    "curvature_by_level": {str(k): round(v, 4) for k, v in curv_by_level.items()},
    "finding": ""  # filled below
}

# Generate summary finding
finding_parts = []
finding_parts.append(f"{n_features} manifold descriptors computed across {len(all_features)} problems.")
finding_parts.append(f"Best difficulty predictor: {best_diff_feat[0]} (r={best_diff_feat[1]['r']:+.3f}).")
finding_parts.append(f"Best correctness predictor: {best_corr_feat[0]} (r={best_corr_feat[1]['r']:+.3f}).")
finding_parts.append(f"Full battery AUROC: {auroc_results['full_battery']:.3f}.")

# Template summary
level1_tmpl = max(template_dist.get("1", {}), key=template_dist.get("1", {}).get, default="?")
level5_tmpl = max(template_dist.get("5", {}), key=template_dist.get("5", {}).get, default="?")
finding_parts.append(f"Level 1 most resembles {level1_tmpl}; Level 5 most resembles {level5_tmpl}.")

# Curvature trend
curv_vals = [curv_by_level.get(str(l), 0) for l in sorted(df["level"].unique())]
if len(curv_vals) >= 2:
    trend = "increasingly negative" if curv_vals[-1] < curv_vals[0] else "increasingly positive"
    finding_parts.append(f"Curvature trends {trend} with difficulty (L1={curv_vals[0]:.3f}, L5={curv_vals[-1]:.3f}).")

results["finding"] = " ".join(finding_parts)
print(f"\nFinding: {results['finding']}")

with open("data/manifold_shapes/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved data/manifold_shapes/results.json")

# Also save the full DataFrame
df.to_csv("data/manifold_shapes/all_descriptors.csv", index=False)
print("  Saved data/manifold_shapes/all_descriptors.csv")

print("\nDone!")
