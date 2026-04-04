#!/usr/bin/env python3
"""
Attractor Hypothesis: Do LLM Layer Trajectories Trace Chaotic Attractors?

Checkpoint-based: each stage saves to disk, so we survive session drops.
Run repeatedly — it skips completed stages automatically.
"""

import json, warnings, sys, time
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import ripser
from persim import wasserstein as wasserstein_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "attractor_hypothesis"
FIG_DIR = ROOT / "figures" / "attractor_hypothesis"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = DATA_DIR / "checkpoint.json"

def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"completed": []}

def save_checkpoint(ck):
    CHECKPOINT.write_text(json.dumps(ck, indent=2))

def done(stage):
    return stage in load_checkpoint()["completed"]

def mark_done(stage):
    ck = load_checkpoint()
    if stage not in ck["completed"]:
        ck["completed"].append(stage)
    save_checkpoint(ck)

# =============================================================================
# ODE integrators using scipy (vectorized C, ~100x faster than Python loops)
# =============================================================================

def gen_aizawa(n_pts=5000, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    def rhs(t, s):
        x, y, z = s
        return [(z-b)*x - d*y, d*x + (z-b)*y,
                c + a*z - z**3/3 - (x**2+y**2)*(1+e*z) + f*z*x**3]
    sol = solve_ivp(rhs, [0, 600], [0.1, 0.0, 0.0], max_step=0.01, dense_output=True)
    t_sample = np.linspace(100, 600, n_pts)
    return sol.sol(t_sample).T

def gen_lorenz(n_pts=5000, sigma=10, rho=28, beta=8/3):
    def rhs(t, s):
        x, y, z = s
        return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]
    sol = solve_ivp(rhs, [0, 100], [1.0, 1.0, 1.0], max_step=0.01, dense_output=True)
    t_sample = np.linspace(20, 100, n_pts)
    return sol.sol(t_sample).T

def gen_rossler(n_pts=5000, a=0.2, b=0.2, c=5.7):
    def rhs(t, s):
        x, y, z = s
        return [-y-z, x+a*y, b+z*(x-c)]
    sol = solve_ivp(rhs, [0, 500], [1.0, 1.0, 1.0], max_step=0.01, dense_output=True)
    t_sample = np.linspace(50, 500, n_pts)
    return sol.sol(t_sample).T

def gen_thomas(n_pts=5000, b=0.208186):
    def rhs(t, s):
        x, y, z = s
        return [np.sin(y)-b*x, np.sin(z)-b*y, np.sin(x)-b*z]
    sol = solve_ivp(rhs, [0, 2500], [1.0, 0.0, 0.0], max_step=0.05, dense_output=True)
    t_sample = np.linspace(500, 2500, n_pts)
    return sol.sol(t_sample).T

def gen_halvorsen(n_pts=5000, a=1.89):
    def rhs(t, s):
        x, y, z = s
        return [-a*x-4*y-4*z-y**2, -a*y-4*z-4*x-z**2, -a*z-4*x-4*y-x**2]
    sol = solve_ivp(rhs, [0, 100], [-1.48, -1.51, 2.04], max_step=0.005, dense_output=True)
    t_sample = np.linspace(10, 100, n_pts)
    return sol.sol(t_sample).T

# =============================================================================
# Topology helpers
# =============================================================================

def persistence_entropy(dgm):
    lt = dgm[:, 1] - dgm[:, 0]
    lt = lt[lt > 0]
    if len(lt) == 0: return 0.0
    p = lt / lt.sum()
    return float(-np.sum(p * np.log(p + 1e-15)))

def persistence_image(dgm, res=50, sigma=0.1, br=None, pr=None):
    if len(dgm) == 0: return np.zeros((res, res))
    births = dgm[:, 0]; pers = dgm[:, 1] - dgm[:, 0]
    m = np.isfinite(pers) & (pers > 0)
    births, pers = births[m], pers[m]
    if len(births) == 0: return np.zeros((res, res))
    if br is None: br = (births.min()-0.1, births.max()+0.1)
    if pr is None: pr = (0, pers.max()+0.1)
    X, Y = np.meshgrid(np.linspace(*br, res), np.linspace(*pr, res))
    img = np.zeros_like(X)
    for b, p in zip(births, pers):
        img += p * np.exp(-((X-b)**2 + (Y-p)**2) / (2*sigma**2))
    return img

def betti_curve(dgm, n_pts=200, fr=None):
    if len(dgm) == 0:
        f = np.linspace(*(fr if fr else (0,1)), n_pts)
        return f, np.zeros(n_pts)
    b, d = dgm[:, 0], dgm[:, 1]
    m = np.isfinite(d); b, d = b[m], d[m]
    if len(b) == 0:
        f = np.linspace(*(fr if fr else (0,1)), n_pts)
        return f, np.zeros(n_pts)
    if fr is None: fr = (b.min(), d.max())
    f = np.linspace(*fr, n_pts)
    # Vectorized: for each filtration value, count features alive
    bc = np.array([np.sum((b <= fi) & (d > fi)) for fi in f])
    return f, bc

def topo_sig(pts, label="", maxdim=2):
    print(f"  PH: {label} ({pts.shape[0]}pts, {pts.shape[1]}D)...", end="", flush=True)
    t0 = time.time()
    r = ripser.ripser(pts, maxdim=maxdim)
    dgms = r["dgms"]
    sig = {"label": label}
    for dim in range(maxdim + 1):
        dgm = dgms[dim]
        fin = dgm[np.isfinite(dgm[:, 1])]
        lt = fin[:, 1] - fin[:, 0] if len(fin) > 0 else np.array([])
        sig[f"H{dim}_count"] = len(fin)
        sig[f"H{dim}_entropy"] = persistence_entropy(fin) if len(fin) > 0 else 0.0
        sig[f"H{dim}_max_lifetime"] = float(lt.max()) if len(lt) > 0 else 0.0
        sig[f"H{dim}_total_persistence"] = float(lt.sum()) if len(lt) > 0 else 0.0
    parts = [f"H{d}={sig[f'H{d}_count']}" for d in range(maxdim+1)]
    print(f" {time.time()-t0:.1f}s  {' '.join(parts)}")
    return sig, dgms


# =============================================================================
# STAGE 1: Reference attractors
# =============================================================================

def stage1_reference_attractors():
    if done("stage1"):
        print("Stage 1: already done, loading...")
        d = np.load(DATA_DIR / "reference_attractors.npz", allow_pickle=True)
        sigs = json.loads((DATA_DIR / "reference_sigs.json").read_text())
        # Reload diagrams
        dgms = {}
        for name in ["aizawa", "lorenz", "rossler", "thomas", "halvorsen"]:
            dgms[name] = [d[f"{name}_dgm{i}"] for i in range(3)]
        return {n: d[n] for n in ["aizawa","lorenz","rossler","thomas","halvorsen"]}, sigs, dgms

    print("=" * 60)
    print("STAGE 1: Reference Attractors")
    print("=" * 60)

    gens = {"aizawa": gen_aizawa, "lorenz": gen_lorenz, "rossler": gen_rossler,
            "thomas": gen_thomas, "halvorsen": gen_halvorsen}

    attractors = {}
    sigs = {}
    dgms = {}
    save_dict = {}

    for name, fn in gens.items():
        print(f"\n{name}...")
        traj = fn(n_pts=5000)
        print(f"  shape: {traj.shape}, range: [{traj.min():.2f}, {traj.max():.2f}]")
        attractors[name] = traj
        save_dict[name] = traj

        # Subsample 1000 for PH
        idx = np.linspace(0, len(traj)-1, 1000, dtype=int)
        sig, dg = topo_sig(traj[idx], label=name)
        sigs[name] = sig
        dgms[name] = dg
        for i, d in enumerate(dg):
            save_dict[f"{name}_dgm{i}"] = d

    np.savez_compressed(DATA_DIR / "reference_attractors.npz", **save_dict)
    with open(DATA_DIR / "reference_sigs.json", "w") as f:
        json.dump(sigs, f, indent=2)

    # Plot
    fig = plt.figure(figsize=(20, 4))
    for i, (name, traj) in enumerate(attractors.items()):
        ax = fig.add_subplot(1, 5, i+1, projection="3d")
        idx = np.linspace(0, len(traj)-1, 3000, dtype=int)
        sub = traj[idx]
        ax.scatter(sub[:,0], sub[:,1], sub[:,2], c=cm.viridis(np.linspace(0,1,len(sub))),
                   s=0.5, alpha=0.5)
        ax.set_title(name.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    plt.suptitle("Reference Chaotic Attractors", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "reference_attractors_3d.png", dpi=200, bbox_inches="tight")
    plt.close()

    mark_done("stage1")
    return attractors, sigs, dgms


# =============================================================================
# STAGE 2: LLM trajectories + 3D visualizations
# =============================================================================

def stage2_llm_trajectories():
    if done("stage2"):
        print("Stage 2: already done, loading...")
        d = np.load(ROOT / "data" / "transformer" / "math500_hidden_states_aligned.npz", allow_pickle=True)
        layer_states = d["layer_hidden_states"]
        levels = d["difficulty_levels"]
        li = {}
        for lv in sorted(np.unique(levels)):
            li[int(lv)] = np.where(levels == lv)[0]
        return layer_states, levels, li

    print("\n" + "=" * 60)
    print("STAGE 2: LLM Trajectories + 3D Plots")
    print("=" * 60)

    d = np.load(ROOT / "data" / "transformer" / "math500_hidden_states_aligned.npz", allow_pickle=True)
    layer_states = d["layer_hidden_states"]  # (500, 29, 1536)
    levels = d["difficulty_levels"]
    li = {}
    for lv in sorted(np.unique(levels)):
        li[int(lv)] = np.where(levels == lv)[0]
        print(f"  Level {lv}: {len(li[int(lv)])} problems")

    # 3D trajectories — one representative per level
    fig = plt.figure(figsize=(20, 4))
    for i, (lv, idxs) in enumerate(sorted(li.items())):
        traj = layer_states[idxs[0]]  # (29, 1536)
        pca = PCA(n_components=3)
        t3 = pca.fit_transform(traj)
        ax = fig.add_subplot(1, 5, i+1, projection="3d")
        n = t3.shape[0]
        cols = cm.coolwarm(np.linspace(0, 1, n))
        for j in range(n-1):
            ax.plot(t3[j:j+2,0], t3[j:j+2,1], t3[j:j+2,2], color=cols[j], linewidth=1.5)
        ax.scatter(t3[:,0], t3[:,1], t3[:,2], c=cols, s=30, zorder=5, edgecolors="k", linewidth=0.3)
        ax.set_title(f"Level {lv}", fontsize=11, fontweight="bold")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    plt.suptitle("LLM Layer Trajectories (PCA→3D, blue=early, red=late)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "3d_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Per-level aggregate clouds
    fig = plt.figure(figsize=(20, 4))
    for i, (lv, idxs) in enumerate(sorted(li.items())):
        cloud = layer_states[idxs].reshape(-1, layer_states.shape[2])
        pca = PCA(n_components=3)
        c3 = pca.fit_transform(cloud)
        lc = np.tile(np.arange(layer_states.shape[1]), len(idxs))
        ax = fig.add_subplot(1, 5, i+1, projection="3d")
        ax.scatter(c3[:,0], c3[:,1], c3[:,2], c=lc, cmap="coolwarm", s=1, alpha=0.3)
        ax.set_title(f"Level {lv} ({len(idxs)} probs)", fontsize=11, fontweight="bold")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    plt.suptitle("Per-Level Aggregate Clouds (PCA→3D)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "level_clouds_3d.png", dpi=200, bbox_inches="tight")
    plt.close()

    mark_done("stage2")
    return layer_states, levels, li


# =============================================================================
# STAGE 3: Topological matching (Wasserstein, PI correlation, Betti curves)
# =============================================================================

def stage3_topology(layer_states, level_indices, ref_sigs, ref_dgms):
    if done("stage3"):
        print("Stage 3: already done, loading...")
        r = json.loads((DATA_DIR / "stage3_results.json").read_text())
        # Reload level dgms
        d = np.load(DATA_DIR / "level_dgms.npz", allow_pickle=True)
        level_dgms = {}
        for lv in level_indices:
            level_dgms[lv] = [d[f"lv{lv}_dgm{i}"] for i in range(2)]
        return r["wass"], r["nearest"], r["pi_corr"], r["pi_names"], r["level_sigs"], level_dgms

    print("\n" + "=" * 60)
    print("STAGE 3: Topological Matching")
    print("=" * 60)

    # Compute PH per level
    level_dgms = {}
    level_sigs = {}
    save_dict = {}

    for lv, idxs in sorted(level_indices.items()):
        cloud = layer_states[idxs].reshape(-1, layer_states.shape[2])
        pca = PCA(n_components=min(10, cloud.shape[0], cloud.shape[1]))
        cpca = pca.fit_transform(cloud)
        if cpca.shape[0] > 500:
            idx = np.random.default_rng(42).choice(cpca.shape[0], 500, replace=False)
            cpca = cpca[idx]
        sig, dg = topo_sig(cpca, label=f"Level {lv}", maxdim=1)
        level_sigs[lv] = sig
        level_dgms[lv] = dg
        for i, d in enumerate(dg):
            save_dict[f"lv{lv}_dgm{i}"] = d

    np.savez_compressed(DATA_DIR / "level_dgms.npz", **save_dict)

    # 4a: Wasserstein
    print("\nWasserstein distances...")
    wass = {}
    for lv, ldg in sorted(level_dgms.items()):
        wass[lv] = {}
        lf = ldg[1]; lf = lf[np.isfinite(lf[:,1])]
        for name, rdg in ref_dgms.items():
            rf = rdg[1]; rf = rf[np.isfinite(rf[:,1])]
            if len(lf) == 0 or len(rf) == 0:
                wass[lv][name] = float("inf")
            else:
                try:
                    wass[lv][name] = float(wasserstein_dist(lf, rf))
                except:
                    wass[lv][name] = float("inf")
        nearest = min(wass[lv], key=wass[lv].get)
        print(f"  Level {lv}: nearest={nearest} (d={wass[lv][nearest]:.4f})")

    nearest_by_level = {str(lv): min(d, key=d.get) for lv, d in wass.items()}

    # Heatmap
    levels = sorted(wass.keys())
    att_names = list(wass[levels[0]].keys())
    mat = np.array([[wass[lv][a] for a in att_names] for lv in levels])
    mat = np.where(np.isinf(mat), np.nanmax(mat[np.isfinite(mat)])*1.5, mat)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(att_names))); ax.set_xticklabels([a.capitalize() for a in att_names], rotation=45)
    ax.set_yticks(range(len(levels))); ax.set_yticklabels([f"Level {l}" for l in levels])
    for i in range(len(levels)):
        for j in range(len(att_names)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i,j] > mat.max()*0.6 else "black")
        j_min = np.argmin(mat[i])
        ax.add_patch(plt.Rectangle((j_min-0.5, i-0.5), 1, 1, fill=False, edgecolor="blue", linewidth=2.5))
    plt.colorbar(im, ax=ax, label="Wasserstein-1 (H1)")
    ax.set_title("Wasserstein Distance: LLM Levels → Reference Attractors", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wasserstein_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 4b: PI correlation
    print("\nPersistence image correlations...")
    all_d = []
    for dgms in list(level_dgms.values()) + list(ref_dgms.values()):
        f = dgms[1]; f = f[np.isfinite(f[:,1])]
        if len(f) > 0: all_d.append(f)
    if all_d:
        ac = np.vstack(all_d)
        br = (ac[:,0].min()-0.1, ac[:,0].max()+0.1)
        pr = (0, (ac[:,1]-ac[:,0]).max()+0.1)
    else:
        br, pr = (0, 1), (0, 1)

    pi_names = [f"L{lv}" for lv in sorted(level_dgms)] + [n.capitalize() for n in ref_dgms]
    pis = []
    for lv in sorted(level_dgms):
        f = level_dgms[lv][1]; f = f[np.isfinite(f[:,1])]
        pis.append(persistence_image(f, br=br, pr=pr).flatten())
    for name in ref_dgms:
        f = ref_dgms[name][1]; f = f[np.isfinite(f[:,1])]
        pis.append(persistence_image(f, br=br, pr=pr).flatten())
    pis = np.array(pis)

    n = len(pis)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if np.std(pis[i]) < 1e-15 or np.std(pis[j]) < 1e-15:
                corr[i,j] = 0.0
            else:
                corr[i,j] = np.corrcoef(pis[i], pis[j])[0, 1]

    # 4c: Betti curves
    print("Betti curves...")
    all_max = 0
    for dgms in list(level_dgms.values()) + list(ref_dgms.values()):
        f = dgms[1]; f = f[np.isfinite(f[:,1])]
        if len(f) > 0: all_max = max(all_max, f[:,1].max())
    fr = (0, all_max * 1.1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    clv = cm.viridis(np.linspace(0.2, 0.9, len(level_dgms)))
    for i, lv in enumerate(sorted(level_dgms)):
        f = level_dgms[lv][1]; f = f[np.isfinite(f[:,1])]
        fs, bc = betti_curve(f, fr=fr)
        fn = (fs - fr[0]) / (fr[1] - fr[0])
        axes[0].plot(fn, bc, label=f"Level {lv}", color=clv[i], linewidth=2)
    axes[0].set_xlabel("Normalized Filtration"); axes[0].set_ylabel("β₁")
    axes[0].set_title("LLM Level Betti Curves (H1)", fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    cref = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i, (name, dgms) in enumerate(ref_dgms.items()):
        f = dgms[1]; f = f[np.isfinite(f[:,1])]
        fs, bc = betti_curve(f, fr=fr)
        fn = (fs - fr[0]) / (fr[1] - fr[0])
        axes[1].plot(fn, bc, label=name.capitalize(), color=cref[i], linewidth=2)
    axes[1].set_xlabel("Normalized Filtration"); axes[1].set_ylabel("β₁")
    axes[1].set_title("Reference Attractor Betti Curves (H1)", fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "betti_curves_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save
    r = {
        "wass": {str(lv): {n: round(v, 4) for n, v in d.items()} for lv, d in wass.items()},
        "nearest": nearest_by_level,
        "pi_corr": corr.tolist(),
        "pi_names": pi_names,
        "level_sigs": {str(lv): s for lv, s in level_sigs.items()},
    }
    with open(DATA_DIR / "stage3_results.json", "w") as f:
        json.dump(r, f, indent=2)

    mark_done("stage3")
    return wass, nearest_by_level, corr, pi_names, level_sigs, level_dgms


# =============================================================================
# STAGE 4: Dynamical systems analysis
# =============================================================================

def stage4_dynamics(layer_states, level_indices):
    if done("stage4"):
        print("Stage 4: already done, loading...")
        return json.loads((DATA_DIR / "stage4_results.json").read_text())

    print("\n" + "=" * 60)
    print("STAGE 4: Dynamical Systems Analysis")
    print("=" * 60)

    results = {"lyapunov": {}, "rqa": {}, "corr_dim": {}, "corr_dim_log": {}}

    # 5a: Lyapunov
    print("\n--- Lyapunov Exponents ---")
    for lv, idxs in sorted(level_indices.items()):
        trajs = layer_states[idxs]  # (n, 29, 1536)
        n_probs, n_layers = trajs.shape[0], trajs.shape[1]
        flat = trajs.reshape(-1, trajs.shape[2])
        pca = PCA(n_components=min(50, flat.shape[0], flat.shape[1]))
        flat_pca = pca.fit_transform(flat)
        traj_pca = flat_pca.reshape(n_probs, n_layers, -1)

        x0 = traj_pca[:, 0, :]
        nn = NearestNeighbors(n_neighbors=min(6, len(x0))).fit(x0)
        dists, indices = nn.kneighbors(x0)

        log_div = np.zeros(n_layers - 1)
        count = np.zeros(n_layers - 1)

        for i in range(n_probs):
            for j_idx in range(1, indices.shape[1]):
                j = indices[i, j_idx]
                d0 = dists[i, j_idx]
                if d0 < 1e-15: continue
                for l in range(1, n_layers):
                    dl = np.linalg.norm(traj_pca[i, l] - traj_pca[j, l])
                    if dl > 1e-15:
                        log_div[l-1] += np.log(dl / d0)
                        count[l-1] += 1

        m = count > 0
        log_div[m] /= count[m]
        layers = np.arange(1, n_layers)
        if m.sum() >= 3:
            slope, _, r, _, _ = stats.linregress(layers[m], log_div[m])
        else:
            slope, r = 0.0, 0.0

        results["lyapunov"][str(lv)] = {"lambda": round(float(slope), 4),
                                         "r2": round(float(r**2), 4),
                                         "log_div": log_div.tolist()}
        print(f"  Level {lv}: λ={slope:.4f} R²={r**2:.3f} {'POS' if slope > 0 else 'neg'}")

    # Plot Lyapunov
    lvs = sorted(results["lyapunov"].keys(), key=int)
    lyaps = [results["lyapunov"][l]["lambda"] for l in lvs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2ecc71" if l > 0 else "#3498db" for l in lyaps]
    axes[0].bar([f"L{l}" for l in lvs], lyaps, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Maximal Lyapunov Exponent (λ)")
    axes[0].set_title("Lyapunov by Difficulty", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    clv = cm.viridis(np.linspace(0.2, 0.9, len(lvs)))
    for i, l in enumerate(lvs):
        ld = results["lyapunov"][l]["log_div"]
        axes[1].plot(range(1, len(ld)+1), ld, "o-", color=clv[i], label=f"Level {l}", markersize=3)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Mean log(d_l/d_0)")
    axes[1].set_title("Divergence of Nearby Trajectories", fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lyapunov_by_level.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 5b: Recurrence
    print("\n--- Recurrence Analysis ---")
    rec_mats = {}
    for lv, idxs in sorted(level_indices.items()):
        metrics_list = []
        for pi in idxs[:50]:
            traj = layer_states[pi]
            pca = PCA(n_components=min(10, traj.shape[0], traj.shape[1]))
            tp = pca.fit_transform(traj)
            D = cdist(tp, tp)
            thr = 0.1 * np.max(D)
            R = (D < thr).astype(float)
            n = R.shape[0]

            # DET
            diag_l = []
            for k in range(-n+1, n):
                dg = np.diag(R, k); ln = 0
                for v in dg:
                    if v > 0.5: ln += 1
                    else:
                        if ln >= 2: diag_l.append(ln)
                        ln = 0
                if ln >= 2: diag_l.append(ln)
            rp = np.sum(R)
            det = sum(diag_l) / rp if rp > 0 else 0
            mean_l = np.mean(diag_l) if diag_l else 0

            # LAM
            vert_l = []
            for col in range(n):
                ln = 0
                for row in range(n):
                    if R[row, col] > 0.5: ln += 1
                    else:
                        if ln >= 2: vert_l.append(ln)
                        ln = 0
                if ln >= 2: vert_l.append(ln)
            lam = sum(vert_l) / rp if rp > 0 else 0

            metrics_list.append({"RR": np.sum(R)/(n*n), "DET": det, "L": mean_l, "LAM": lam})

        avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
        results["rqa"][str(lv)] = {k: round(v, 4) for k, v in avg.items()}
        print(f"  Level {lv}: DET={avg['DET']:.3f} LAM={avg['LAM']:.3f} L={avg['L']:.2f}")

        # Store one recurrence matrix for plotting
        traj = layer_states[idxs[0]]
        pca = PCA(n_components=min(10, traj.shape[0], traj.shape[1]))
        tp = pca.fit_transform(traj)
        D = cdist(tp, tp)
        rec_mats[lv] = (D < 0.1 * np.max(D)).astype(float)

    # Plot recurrence
    plot_lvs = [lv for lv in [1, 3, 5] if lv in rec_mats]
    fig, axes = plt.subplots(1, len(plot_lvs), figsize=(5*len(plot_lvs), 4.5))
    if len(plot_lvs) == 1: axes = [axes]
    for i, lv in enumerate(plot_lvs):
        axes[i].imshow(rec_mats[lv], cmap="binary", origin="lower")
        axes[i].set_title(f"Level {lv} Recurrence", fontweight="bold")
        axes[i].set_xlabel("Layer"); axes[i].set_ylabel("Layer")
    plt.suptitle("Recurrence Plots", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "recurrence_plots.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 5c: Correlation dimension
    print("\n--- Correlation Dimension ---")
    for lv, idxs in sorted(level_indices.items()):
        cloud = layer_states[idxs].reshape(-1, layer_states.shape[2])
        pca = PCA(n_components=min(10, cloud.shape[0], cloud.shape[1]))
        cpca = pca.fit_transform(cloud)
        if cpca.shape[0] > 1000:
            idx = np.random.default_rng(42).choice(cpca.shape[0], 1000, replace=False)
            cpca = cpca[idx]

        dists = pdist(cpca)
        r_vals = np.logspace(np.log10(np.percentile(dists, 1)),
                             np.log10(np.percentile(dists, 50)), 20)
        C_r = np.array([np.mean(dists < r) for r in r_vals])
        m = C_r > 0
        if m.sum() >= 5:
            lr, lc = np.log(r_vals[m]), np.log(C_r[m])
            slope, _, r, _, _ = stats.linregress(lr, lc)
            results["corr_dim"][str(lv)] = {"D2": round(float(slope), 3), "R2": round(float(r**2), 3)}
            results["corr_dim_log"][str(lv)] = {"log_r": lr.tolist(), "log_C": lc.tolist()}
        else:
            results["corr_dim"][str(lv)] = {"D2": 0.0, "R2": 0.0}
            results["corr_dim_log"][str(lv)] = {"log_r": [], "log_C": []}
        print(f"  Level {lv}: D₂={results['corr_dim'][str(lv)]['D2']:.3f}")

    # Plot correlation dimension
    fig, ax = plt.subplots(figsize=(8, 5))
    clv = cm.viridis(np.linspace(0.2, 0.9, len(level_indices)))
    for i, lv in enumerate(sorted(level_indices)):
        d = results["corr_dim_log"][str(lv)]
        if d["log_r"]:
            ax.plot(d["log_r"], d["log_C"], "o-", color=clv[i], label=f"Level {lv}", markersize=4)
    ax.set_xlabel("log(r)"); ax.set_ylabel("log(C(r))")
    ax.set_title("Correlation Integral (Grassberger-Procaccia D₂)", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_dimension.png", dpi=200, bbox_inches="tight")
    plt.close()

    with open(DATA_DIR / "stage4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    mark_done("stage4")
    return results


# =============================================================================
# STAGE 5: Aizawa parameter fitting
# =============================================================================

def stage5_aizawa_fit(level_dgms, level_sigs):
    if done("stage5"):
        print("Stage 5: already done, loading...")
        return json.loads((DATA_DIR / "stage5_results.json").read_text())

    print("\n" + "=" * 60)
    print("STAGE 5: Aizawa Parameter Fitting")
    print("=" * 60)

    h1c = {lv: s["H1_count"] for lv, s in level_sigs.items()}
    target = max(h1c, key=h1c.get)
    print(f"Target: Level {target} (H1={h1c[target]})")

    tdg = level_dgms[target][1]
    tf = tdg[np.isfinite(tdg[:,1])]

    best_dist = float("inf")
    best_params = None
    n_tried = n_fail = 0

    for a in np.linspace(0.5, 1.5, 5):
        for b in np.linspace(0.3, 1.0, 5):
            for c in np.linspace(0.3, 1.0, 5):
                n_tried += 1
                try:
                    traj = gen_aizawa(n_pts=1000, a=a, b=b, c=c)
                    if np.any(np.abs(traj) > 1e6) or np.any(np.isnan(traj)):
                        n_fail += 1; continue
                    idx = np.linspace(0, len(traj)-1, 300, dtype=int)
                    r = ripser.ripser(traj[idx], maxdim=1)
                    dgm = r["dgms"][1]
                    fin = dgm[np.isfinite(dgm[:,1])]
                    if len(fin) == 0: n_fail += 1; continue
                    d = wasserstein_dist(tf, fin)
                    if d < best_dist:
                        best_dist = d; best_params = {"a": float(a), "b": float(b), "c": float(c)}
                except:
                    n_fail += 1

    print(f"Tried {n_tried}, failed {n_fail}")
    if best_params:
        print(f"Best: a={best_params['a']:.3f} b={best_params['b']:.3f} c={best_params['c']:.3f} Wass={best_dist:.4f}")
    else:
        best_params = {"a": 0.95, "b": 0.7, "c": 0.6}
        best_dist = float("inf")

    # Plot
    traj = gen_aizawa(n_pts=3000, a=best_params["a"], b=best_params["b"], c=best_params["c"])
    idx = np.linspace(0, len(traj)-1, 1000, dtype=int)
    sub = traj[idx]
    fr = ripser.ripser(sub, maxdim=1)
    fit_dgm = fr["dgms"][1]; fit_f = fit_dgm[np.isfinite(fit_dgm[:,1])]

    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(sub[:,0], sub[:,1], sub[:,2], c=cm.viridis(np.linspace(0,1,len(sub))), s=1, alpha=0.5)
    ax.set_title(f"Best-Fit Aizawa\na={best_params['a']:.2f}, b={best_params['b']:.2f}, c={best_params['c']:.2f}",
                 fontweight="bold")
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    ax = fig.add_subplot(1, 3, 2)
    if len(fit_f) > 0: ax.scatter(fit_f[:,0], fit_f[:,1], alpha=0.6, s=20, label="Best-fit Aizawa")
    if len(tf) > 0: ax.scatter(tf[:,0], tf[:,1], alpha=0.6, s=20, marker="^", label=f"Level {target}")
    all_pts = np.concatenate([p for p in [fit_f, tf] if len(p) > 0])
    lim = [all_pts.min()-0.1, all_pts.max()+0.1]
    ax.plot(lim, lim, "k--", alpha=0.3)
    ax.set_xlabel("Birth"); ax.set_ylabel("Death")
    ax.set_title("H1 Persistence Diagrams", fontweight="bold")
    ax.legend(); ax.set_aspect("equal")

    ax = fig.add_subplot(1, 3, 3)
    afr = (min(all_pts[:,0].min(), 0), all_pts[:,1].max()*1.1)
    fa, ba = betti_curve(fit_f, fr=afr)
    ft, bt = betti_curve(tf, fr=afr)
    ax.plot(fa, ba, label="Best-fit Aizawa", linewidth=2)
    ax.plot(ft, bt, label=f"Level {target}", linewidth=2, linestyle="--")
    ax.set_xlabel("Filtration"); ax.set_ylabel("β₁")
    ax.set_title("H1 Betti Curve Comparison", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "aizawa_fit.png", dpi=200, bbox_inches="tight")
    plt.close()

    result = {"target_level": target, "best_params": best_params,
              "best_distance": round(best_dist, 4), "n_tried": n_tried, "n_failed": n_fail}
    with open(DATA_DIR / "stage5_results.json", "w") as f:
        json.dump(result, f, indent=2)
    mark_done("stage5")
    return result


# =============================================================================
# STAGE 6: Final results JSON
# =============================================================================

def stage6_compile(ref_sigs, wass, nearest, pi_corr, pi_names, level_sigs,
                   dyn_results, aizawa_result):
    print("\n" + "=" * 60)
    print("STAGE 6: Compile Final Results")
    print("=" * 60)

    lyap = dyn_results["lyapunov"]
    rqa = dyn_results["rqa"]
    cd = dyn_results["corr_dim"]

    lyap_pos = [int(lv) for lv, r in lyap.items() if r["lambda"] > 0]
    avg_d2 = np.mean([cd[lv]["D2"] for lv in cd])
    avg_det = np.mean([rqa[lv]["DET"] for lv in rqa])
    min_w = min(min(d.values()) for d in wass.values())

    ev = 0
    if len(lyap_pos) >= 3: ev += 1
    if 2.0 <= avg_d2 <= 8.0: ev += 1
    if avg_det > 0.3: ev += 1
    if min_w < 5.0: ev += 1

    if ev >= 4 and min_w < 2.0: verdict = "attractor_confirmed"
    elif ev >= 2: verdict = "attractor_partial"
    else: verdict = "no_attractor"

    vis = "; ".join(f"Level {lv} nearest to {nearest[str(lv)]}" for lv in sorted(int(k) for k in nearest))

    results = {
        "branch": "experiment/tda-attractor-hypothesis",
        "reference_attractors": {
            name: {"H1_count": s["H1_count"],
                   "H1_entropy": round(s["H1_entropy"], 4),
                   "corr_dim": "N/A (3D native)"}
            for name, s in ref_sigs.items()
        },
        "wasserstein_to_references": {str(lv): {n: round(v, 4) for n, v in d.items()} for lv, d in wass.items()},
        "nearest_attractor_by_level": nearest,
        "pi_correlation_matrix": [[round(float(x), 4) for x in row] for row in pi_corr.tolist()],
        "lyapunov_exponents": {lv: r["lambda"] for lv, r in lyap.items()},
        "lyapunov_positive_levels": lyap_pos,
        "recurrence_det_by_level": {lv: r["DET"] for lv, r in rqa.items()},
        "recurrence_lam_by_level": {lv: r["LAM"] for lv, r in rqa.items()},
        "correlation_dimension_by_level": {lv: r["D2"] for lv, r in cd.items()},
        "aizawa_fit_params": aizawa_result["best_params"],
        "aizawa_fit_distance": aizawa_result["best_distance"],
        "visual_resemblance": vis,
        "overall_verdict": verdict,
        "evidence_summary": {
            "positive_lyapunov_levels": len(lyap_pos),
            "avg_correlation_dimension": round(avg_d2, 3),
            "avg_determinism": round(avg_det, 3),
            "min_wasserstein_to_any_attractor": round(min_w, 4),
        },
    }

    with open(DATA_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print("\nWasserstein Distance to Reference Attractors (H1):")
    hdr = f"{'':>10} | {'Aizawa':>8} | {'Lorenz':>8} | {'Rössler':>8} | {'Thomas':>8} | {'Halvorsen':>10}"
    print(hdr); print("-"*len(hdr))
    for lv in sorted(wass.keys()):
        row = f"Level {lv:>4} |"
        for n in ["aizawa","lorenz","rossler","thomas","halvorsen"]:
            d = wass[lv][n]
            mk = " *" if n == nearest[str(lv)] else "  "
            row += f" {d:>6.3f}{mk}|"
        print(row)

    print(f"\nLyapunov: {', '.join(f'L{l}={lyap[l]['lambda']:.4f}' for l in sorted(lyap))}")
    print(f"Positive: {lyap_pos}")
    print(f"Corr Dim: {', '.join(f'L{l}={cd[l]['D2']:.2f}' for l in sorted(cd))}")
    print(f"DET: {', '.join(f'L{l}={rqa[l]['DET']:.3f}' for l in sorted(rqa))}")
    print(f"Aizawa fit: {aizawa_result['best_params']}, Wass={aizawa_result['best_distance']:.4f}")
    print(f"\nVerdict: {verdict}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(42)
    t0 = time.time()

    attractors, ref_sigs, ref_dgms = stage1_reference_attractors()
    layer_states, levels, level_indices = stage2_llm_trajectories()
    wass, nearest, pi_corr, pi_names, level_sigs, level_dgms = stage3_topology(
        layer_states, level_indices, ref_sigs, ref_dgms)
    dyn_results = stage4_dynamics(layer_states, level_indices)
    aizawa_result = stage5_aizawa_fit(level_dgms, level_sigs)
    results = stage6_compile(ref_sigs, wass, nearest, pi_corr, pi_names, level_sigs,
                             dyn_results, aizawa_result)

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Results: {DATA_DIR / 'results.json'}")
    print(f"Figures: {FIG_DIR}/")


if __name__ == "__main__":
    main()
