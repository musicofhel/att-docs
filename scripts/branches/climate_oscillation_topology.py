#!/usr/bin/env python3
"""Branch 9: Climate Oscillations — Topology of El Niño and Other Cycles.

Applies Takens embedding + persistent homology to climate indices (ENSO, NAO)
to test whether topological analysis reveals structure that spectral analysis
misses — particularly during transition years between El Niño and La Niña.

Four experiments:
  1. ENSO attractor topology (Takens embedding + PH)
  2. Sliding-window topology across decades (TransitionDetector)
  3. El Niño vs La Niña attractor comparison (Wasserstein distance)
  4. ENSO–NAO topological coupling (BindingDetector)
"""

import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from att.binding.detector import BindingDetector
from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer
from att.transitions.detector import TransitionDetector

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_noaa_corr(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse NOAA correlation-style data (year + 12 monthly values per row).

    Returns (years, months_flat, values_flat) — missing = -99.99 replaced with NaN.
    """
    years, vals = [], []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            yr = int(float(parts[0]))
        except ValueError:
            continue
        if yr < 1900 or yr > 2100:
            continue
        row = []
        for v in parts[1:13]:
            f = float(v)
            row.append(np.nan if f < -90 else f)
        years.append(yr)
        vals.append(row)

    years = np.array(years)
    vals = np.array(vals)  # (n_years, 12)

    # Build flat time axis (fractional year)
    t_flat = []
    v_flat = []
    for i, yr in enumerate(years):
        for m in range(12):
            t_flat.append(yr + (m + 0.5) / 12.0)
            v_flat.append(vals[i, m])
    return years, np.array(t_flat), np.array(v_flat)


def download_index(url: str, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Download a NOAA climate index. Returns (time, values) or None on failure."""
    print(f"  Downloading {name} from {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ATT-Research/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        _, t, v = parse_noaa_corr(raw)
        # Drop NaN tails
        mask = ~np.isnan(v)
        if mask.sum() < 120:
            print(f"    Too few valid values ({mask.sum()}) — falling back to synthetic")
            return None
        # Trim leading/trailing NaN
        first, last = np.where(mask)[0][[0, -1]]
        t, v = t[first : last + 1], v[first : last + 1]
        # Interpolate interior NaNs
        interior_nan = np.isnan(v)
        if interior_nan.any():
            v[interior_nan] = np.interp(t[interior_nan], t[~interior_nan], v[~interior_nan])
        print(f"    Got {len(v)} months ({t[0]:.1f}–{t[-1]:.1f})")
        return t, v
    except Exception as e:
        print(f"    Download failed: {e}")
        return None


def synthetic_enso(n_years: int = 70, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-periodic oscillation with intermittent regime changes."""
    rng = np.random.default_rng(seed)
    n = n_years * 12
    t = np.arange(n) / 12.0 + 1954.0  # start at 1954
    # Irregular 2–7 year cycle via random walk on frequency
    freq = 0.3 + 0.15 * rng.standard_normal(n)
    phase = 2 * np.pi * np.cumsum(freq) / 12.0
    enso = 1.5 * np.sin(phase) + 0.5 * rng.standard_normal(n)

    # Inject known "events": stronger amplitude around 1977, 1998, 2016
    for event_year, amp in [(1977, 2.0), (1998, 2.5), (2016, 2.0)]:
        idx = int((event_year - t[0]) * 12)
        if 0 <= idx < n - 24:
            enso[idx : idx + 24] *= amp / 1.5  # amplify 2-year window

    return t, enso


def synthetic_nao(n_months: int, seed: int = 43) -> np.ndarray:
    """AR(1) process resembling NAO with weak ENSO-like modulation."""
    rng = np.random.default_rng(seed)
    nao = np.zeros(n_months)
    for i in range(1, n_months):
        nao[i] = 0.3 * nao[i - 1] + rng.standard_normal()
    return nao


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def classify_enso_months(
    values: np.ndarray, threshold: float = 0.5, min_consecutive: int = 5
) -> np.ndarray:
    """Classify months as El Niño (+1), La Niña (-1), or Neutral (0).

    Uses ±threshold with min_consecutive months rule (standard ONI definition).
    """
    n = len(values)
    labels = np.zeros(n, dtype=int)

    # First pass: raw threshold
    raw = np.zeros(n, dtype=int)
    raw[values > threshold] = 1
    raw[values < -threshold] = -1

    # Second pass: require min_consecutive consecutive months
    i = 0
    while i < n:
        if raw[i] != 0:
            state = raw[i]
            j = i
            while j < n and raw[j] == state:
                j += 1
            if j - i >= min_consecutive:
                labels[i:j] = state
            i = j
        else:
            i += 1

    return labels


def embed_and_ph(
    signal: np.ndarray,
    delay: int | str = "auto",
    dimension: int | str = "auto",
    max_dim: int = 2,
    subsample: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Takens embed + PH. Returns (cloud, ph_results)."""
    emb = TakensEmbedder(delay=delay, dimension=dimension)
    cloud = emb.fit_transform(signal)
    pa = PersistenceAnalyzer(max_dim=max_dim)
    res = pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return cloud, res, emb


def count_features(diagrams: list, dim: int) -> int:
    """Count finite-lifetime features in dimension dim."""
    if dim >= len(diagrams):
        return 0
    dgm = diagrams[dim]
    if len(dgm) == 0:
        return 0
    return int(np.sum(np.isfinite(dgm[:, 1])))


def wasserstein_1d(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Approximate Wasserstein-1 via lifetime distribution comparison."""
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    lt1 = dgm1[:, 1] - dgm1[:, 0] if len(dgm1) > 0 else np.array([0.0])
    lt2 = dgm2[:, 1] - dgm2[:, 0] if len(dgm2) > 0 else np.array([0.0])
    lt1 = lt1[np.isfinite(lt1)]
    lt2 = lt2[np.isfinite(lt2)]
    if len(lt1) == 0:
        lt1 = np.array([0.0])
    if len(lt2) == 0:
        lt2 = np.array([0.0])
    # Use scipy Wasserstein for 1D distributions
    return float(stats.wasserstein_distance(lt1, lt2))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment1_enso_attractor(enso: np.ndarray, t: np.ndarray, subsample: int,
                                seed: int, fig_dir: Path) -> dict:
    """Exp 1: ENSO attractor topology via Takens + PH."""
    print("\n=== Experiment 1: ENSO Attractor Topology ===")

    # Embed ENSO
    cloud, ph, emb = embed_and_ph(enso, max_dim=2, subsample=subsample, seed=seed)
    delay = emb.delay_ if hasattr(emb, "delay_") else "unknown"
    dim = emb.dimension_ if hasattr(emb, "dimension_") else "unknown"
    print(f"  Takens: delay={delay}, dimension={dim}")
    print(f"  Cloud shape: {cloud.shape}")

    h0 = count_features(ph["diagrams"], 0)
    h1 = count_features(ph["diagrams"], 1)
    h2 = count_features(ph["diagrams"], 2)
    ent = ph["persistence_entropy"]
    print(f"  H0={h0}, H1={h1}, H2={h2}")
    print(f"  Entropy: H0={ent[0]:.3f}, H1={ent[1]:.3f}, H2={ent[2]:.3f}")

    # Reference: Rössler attractor (known single H1 loop)
    from att.synthetic.generators import rossler_system
    ross = rossler_system(n_steps=5000, seed=seed)
    ross_sig = ross[:, 0]  # x-component
    ross_cloud, ross_ph, _ = embed_and_ph(ross_sig, max_dim=2, subsample=subsample, seed=seed)
    ross_h1 = count_features(ross_ph["diagrams"], 1)
    ross_h2 = count_features(ross_ph["diagrams"], 2)
    print(f"  Rössler reference: H1={ross_h1}, H2={ross_h2}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Exp 1: ENSO Attractor Topology", fontsize=14, fontweight="bold")

    # Panel 1: Time series
    ax = axes[0, 0]
    ax.plot(t, enso, "b-", linewidth=0.5, alpha=0.7)
    ax.axhline(0.5, color="r", linestyle="--", alpha=0.5, label="El Niño threshold")
    ax.axhline(-0.5, color="b", linestyle="--", alpha=0.5, label="La Niña threshold")
    ax.fill_between(t, enso, 0, where=enso > 0.5, alpha=0.3, color="red")
    ax.fill_between(t, enso, 0, where=enso < -0.5, alpha=0.3, color="blue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Niño 3.4 Anomaly (°C)")
    ax.set_title("ENSO Time Series")
    ax.legend(fontsize=7)

    # Panel 2: 2D projection of attractor
    ax = axes[0, 1]
    if cloud.shape[1] >= 2:
        sc = ax.scatter(cloud[:, 0], cloud[:, 1], c=np.arange(len(cloud)), cmap="viridis",
                        s=1, alpha=0.5)
        ax.set_xlabel("Embed dim 1")
        ax.set_ylabel("Embed dim 2")
    ax.set_title(f"Takens Attractor (τ={delay}, d={dim})")

    # Panel 3: 3D projection
    ax = axes[0, 2]
    ax.remove()
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    if cloud.shape[1] >= 3:
        ax3.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=np.arange(len(cloud)),
                    cmap="viridis", s=1, alpha=0.3)
        ax3.set_xlabel("d1")
        ax3.set_ylabel("d2")
        ax3.set_zlabel("d3")
    ax3.set_title("3D Projection")

    # Panel 4: Persistence diagrams H1
    ax = axes[1, 0]
    dgm1 = ph["diagrams"][1]
    if len(dgm1) > 0:
        finite = dgm1[np.isfinite(dgm1[:, 1])]
        if len(finite) > 0:
            ax.scatter(finite[:, 0], finite[:, 1], s=10, alpha=0.6, c="red", label="H1")
    dgm2 = ph["diagrams"][2]
    if len(dgm2) > 0:
        finite2 = dgm2[np.isfinite(dgm2[:, 1])]
        if len(finite2) > 0:
            ax.scatter(finite2[:, 0], finite2[:, 1], s=15, alpha=0.8, c="purple",
                       marker="^", label="H2")
    # Diagonal
    all_pts = []
    for d in [1, 2]:
        if d < len(ph["diagrams"]) and len(ph["diagrams"][d]) > 0:
            fin = ph["diagrams"][d][np.isfinite(ph["diagrams"][d][:, 1])]
            if len(fin) > 0:
                all_pts.extend(fin.flatten())
    if all_pts:
        lo, hi = min(all_pts), max(all_pts)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("Persistence Diagram (H1, H2)")
    ax.legend(fontsize=8)

    # Panel 5: Betti comparison ENSO vs Rössler
    ax = axes[1, 1]
    x_pos = [0, 1, 2]
    enso_betti = [h0, h1, h2]
    ross_betti = [count_features(ross_ph["diagrams"], 0), ross_h1, ross_h2]
    w = 0.35
    ax.bar([p - w / 2 for p in x_pos], enso_betti, w, label="ENSO", color="steelblue")
    ax.bar([p + w / 2 for p in x_pos], ross_betti, w, label="Rössler", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["H0", "H1", "H2"])
    ax.set_ylabel("Feature Count")
    ax.set_title("Betti Numbers: ENSO vs Rössler")
    ax.legend()

    # Panel 6: Entropy comparison
    ax = axes[1, 2]
    ross_ent = ross_ph["persistence_entropy"]
    ax.bar([p - w / 2 for p in x_pos], ent, w, label="ENSO", color="steelblue")
    ax.bar([p + w / 2 for p in x_pos], ross_ent, w, label="Rössler", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["H0", "H1", "H2"])
    ax.set_ylabel("Persistence Entropy")
    ax.set_title("Persistence Entropy: ENSO vs Rössler")
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_dir / "exp1_enso_attractor.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp1_enso_attractor.png")

    return {
        "exp1_takens_delay": int(delay) if isinstance(delay, (int, np.integer)) else str(delay),
        "exp1_takens_dim": int(dim) if isinstance(dim, (int, np.integer)) else str(dim),
        "exp1_cloud_shape": list(cloud.shape),
        "exp1_enso_h0": h0,
        "exp1_enso_h1_features": h1,
        "exp1_enso_h1_entropy": round(float(ent[1]), 4),
        "exp1_enso_h2_features": h2,
        "exp1_enso_h2_entropy": round(float(ent[2]), 4),
        "exp1_rossler_h1": ross_h1,
        "exp1_rossler_h2": ross_h2,
    }


def experiment2_sliding_window(enso: np.ndarray, t: np.ndarray, subsample: int,
                                seed: int, fig_dir: Path) -> dict:
    """Exp 2: Sliding-window topology across decades."""
    print("\n=== Experiment 2: Sliding-Window Topology ===")

    # First get embedding params from Takens on full series
    emb_probe = TakensEmbedder(delay="auto", dimension="auto")
    emb_probe.fit_transform(enso)
    e_delay = emb_probe.delay_ if hasattr(emb_probe, "delay_") else 9
    e_dim = emb_probe.dimension_ if hasattr(emb_probe, "dimension_") else 5
    print(f"  Embedding params: delay={e_delay}, dim={e_dim}")

    # TransitionDetector: 10-year window, 1-year step
    td = TransitionDetector(
        window_size=120,  # 10 years
        step_size=12,     # 1 year
        max_dim=1,
        subsample=subsample,
    )
    result = td.fit_transform(enso, seed=seed, embedding_dim=e_dim, embedding_delay=e_delay)

    scores = result["transition_scores"]
    centers = result["window_centers"]
    # Convert center indices to years
    all_center_years = t[0] + centers / 12.0
    # transition_scores are distances between consecutive windows → one fewer element
    # Use midpoints between consecutive window centers
    if len(scores) < len(all_center_years):
        center_years = (all_center_years[:-1] + all_center_years[1:]) / 2.0
    else:
        center_years = all_center_years

    print(f"  Windows: {len(all_center_years)}, Transition scores: {len(scores)}")
    print(f"  Score range: {scores.min():.2f} – {scores.max():.2f}")

    # Detect changepoints (indices into transition_scores array)
    changepoints = td.detect_changepoints(method="cusum")
    cp_years = [float(center_years[cp]) for cp in changepoints if cp < len(center_years)]
    print(f"  CUSUM changepoints: {[f'{y:.1f}' for y in cp_years]}")

    # Known events
    known_events = {
        "1976-77 Pacific shift": 1977.0,
        "1997-98 super El Niño": 1998.0,
        "2015-16 super El Niño": 2016.0,
    }

    detected_shifts = []
    for name, event_yr in known_events.items():
        # Check if any score peak is within 3 years of event
        near = np.abs(center_years - event_yr) < 3.0
        if near.any():
            local_max = scores[near].max()
            global_thresh = np.percentile(scores, 75)
            detected = bool(local_max > global_thresh)
        else:
            detected = False
        detected_shifts.append({"event": name, "year": event_yr, "detected": detected})
        status = "DETECTED" if detected else "not detected"
        print(f"  {name}: {status}")

    # Also check if CUSUM changepoints fall near known events
    for name, event_yr in known_events.items():
        near_cp = [y for y in cp_years if abs(y - event_yr) < 3.0]
        if near_cp:
            print(f"    CUSUM near {name}: {[f'{y:.1f}' for y in near_cp]}")

    # --- Figure ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Exp 2: Sliding-Window Topological Analysis", fontsize=14, fontweight="bold")

    # Panel 1: ENSO time series
    ax = axes[0]
    ax.plot(t, enso, "b-", linewidth=0.5, alpha=0.7)
    ax.fill_between(t, enso, 0, where=enso > 0.5, alpha=0.3, color="red")
    ax.fill_between(t, enso, 0, where=enso < -0.5, alpha=0.3, color="blue")
    for name, yr in known_events.items():
        ax.axvline(yr, color="green", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_ylabel("Niño 3.4 (°C)")
    ax.set_title("ENSO Index with Known Climate Shifts")

    # Panel 2: Transition scores
    ax = axes[1]
    ax.plot(center_years, scores, "k-", linewidth=1)
    ax.fill_between(center_years, scores, alpha=0.3, color="orange")
    for name, yr in known_events.items():
        ax.axvline(yr, color="green", linestyle="--", alpha=0.7, linewidth=1.5)
    for cp_yr in cp_years:
        ax.axvline(cp_yr, color="red", linestyle=":", alpha=0.8, linewidth=1.5)
    ax.set_ylabel("Topology Score\n(image distance)")
    ax.set_title("Topological Transition Scores (10-yr sliding window)")

    # Panel 3: Bottleneck distances
    if "distances" in result and len(result["distances"]) > 0:
        dists = result["distances"]
        # distances has one fewer element than scores
        dist_years = center_years[: len(dists)]
        ax = axes[2]
        ax.plot(dist_years, dists, "m-", linewidth=1)
        ax.fill_between(dist_years, dists, alpha=0.3, color="purple")
        for name, yr in known_events.items():
            ax.axvline(yr, color="green", linestyle="--", alpha=0.7, linewidth=1.5)
        ax.set_ylabel("Bottleneck Distance")
        ax.set_title("Consecutive Window Bottleneck Distances")
    ax.set_xlabel("Year")

    plt.tight_layout()
    fig.savefig(fig_dir / "exp2_sliding_window.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp2_sliding_window.png")

    return {
        "exp2_n_windows": len(scores),
        "exp2_score_range": [round(float(scores.min()), 2), round(float(scores.max()), 2)],
        "exp2_cusum_changepoints": [round(y, 1) for y in cp_years],
        "exp2_detected_shifts": detected_shifts,
        "exp2_1977_detected": detected_shifts[0]["detected"],
        "exp2_1998_detected": detected_shifts[1]["detected"],
        "exp2_2016_detected": detected_shifts[2]["detected"],
    }


def experiment3_elnino_vs_lanina(enso: np.ndarray, t: np.ndarray, subsample: int,
                                  seed: int, n_perms: int, fig_dir: Path) -> dict:
    """Exp 3: El Niño vs La Niña attractor comparison."""
    print("\n=== Experiment 3: El Niño vs La Niña Attractor Comparison ===")

    labels = classify_enso_months(enso)
    n_nino = (labels == 1).sum()
    n_nina = (labels == -1).sum()
    n_neutral = (labels == 0).sum()
    print(f"  El Niño months: {n_nino}, La Niña months: {n_nina}, Neutral: {n_neutral}")

    # Extract contiguous segments for each regime, concatenate
    nino_vals = enso[labels == 1]
    nina_vals = enso[labels == -1]

    results = {}

    # Embed each regime separately
    for name, vals in [("elnino", nino_vals), ("lanina", nina_vals)]:
        if len(vals) < 50:
            print(f"  {name}: too few months ({len(vals)}), using fallback delay/dim")
            cloud, ph, emb = embed_and_ph(vals, delay=3, dimension=3, max_dim=1,
                                           subsample=min(subsample, len(vals) - 10),
                                           seed=seed)
        else:
            cloud, ph, emb = embed_and_ph(vals, max_dim=1, subsample=subsample, seed=seed)

        h1 = count_features(ph["diagrams"], 1)
        ent = ph["persistence_entropy"][1] if len(ph["persistence_entropy"]) > 1 else 0.0
        delay = emb.delay_ if hasattr(emb, "delay_") else "?"
        dim = emb.dimension_ if hasattr(emb, "dimension_") else "?"
        print(f"  {name}: cloud={cloud.shape}, τ={delay}, d={dim}, H1={h1}, ent={ent:.3f}")

        results[f"exp3_{name}_n_months"] = int(len(vals))
        results[f"exp3_{name}_h1_features"] = h1
        results[f"exp3_{name}_h1_entropy"] = round(float(ent), 4)
        results[f"_{name}_diagrams"] = ph["diagrams"]  # temp, for Wasserstein

    # Wasserstein distance between H1 diagrams
    dgm_nino = results.pop("_elnino_diagrams")
    dgm_nina = results.pop("_lanina_diagrams")

    # Use H1 diagrams (finite lifetimes only)
    nino_h1 = dgm_nino[1] if len(dgm_nino) > 1 else np.array([]).reshape(0, 2)
    nina_h1 = dgm_nina[1] if len(dgm_nina) > 1 else np.array([]).reshape(0, 2)
    nino_h1_fin = nino_h1[np.isfinite(nino_h1[:, 1])] if len(nino_h1) > 0 else nino_h1
    nina_h1_fin = nina_h1[np.isfinite(nina_h1[:, 1])] if len(nina_h1) > 0 else nina_h1

    observed_w = wasserstein_1d(nino_h1_fin, nina_h1_fin)
    print(f"  Wasserstein(H1 lifetimes): {observed_w:.4f}")

    # Permutation test: shuffle El Niño/La Niña labels
    rng = np.random.default_rng(seed)
    combined = np.concatenate([nino_vals, nina_vals])
    null_w = []
    for _ in range(n_perms):
        perm = rng.permutation(len(combined))
        perm_a = combined[perm[: len(nino_vals)]]
        perm_b = combined[perm[len(nino_vals) :]]
        # Quick PH for permuted
        try:
            _, ph_a, _ = embed_and_ph(perm_a, delay=3, dimension=3, max_dim=1,
                                       subsample=min(subsample, len(perm_a) - 10),
                                       seed=seed)
            _, ph_b, _ = embed_and_ph(perm_b, delay=3, dimension=3, max_dim=1,
                                       subsample=min(subsample, len(perm_b) - 10),
                                       seed=seed)
            da = ph_a["diagrams"][1] if len(ph_a["diagrams"]) > 1 else np.array([]).reshape(0, 2)
            db = ph_b["diagrams"][1] if len(ph_b["diagrams"]) > 1 else np.array([]).reshape(0, 2)
            da = da[np.isfinite(da[:, 1])] if len(da) > 0 else da
            db = db[np.isfinite(db[:, 1])] if len(db) > 0 else db
            null_w.append(wasserstein_1d(da, db))
        except Exception:
            continue

    null_w = np.array(null_w) if null_w else np.array([0.0])
    p_val = float((null_w >= observed_w).sum() + 1) / (len(null_w) + 1)
    z_score = (observed_w - null_w.mean()) / (null_w.std() + 1e-10)
    print(f"  Permutation test: p={p_val:.4f}, z={z_score:.2f} ({len(null_w)} perms)")

    results["exp3_wasserstein"] = round(observed_w, 4)
    results["exp3_wasserstein_p"] = round(p_val, 4)
    results["exp3_wasserstein_z"] = round(float(z_score), 2)
    results["exp3_n_permutations"] = len(null_w)

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Exp 3: El Niño vs La Niña Topology", fontsize=14, fontweight="bold")

    # Panel 1: Time series colored by regime
    ax = axes[0, 0]
    ax.plot(t, enso, "gray", linewidth=0.3, alpha=0.5)
    nino_mask = labels == 1
    nina_mask = labels == -1
    ax.scatter(t[nino_mask], enso[nino_mask], c="red", s=2, alpha=0.7, label="El Niño")
    ax.scatter(t[nina_mask], enso[nina_mask], c="blue", s=2, alpha=0.7, label="La Niña")
    ax.set_xlabel("Year")
    ax.set_ylabel("Niño 3.4 (°C)")
    ax.set_title(f"ENSO Regimes (El Niño: {n_nino}mo, La Niña: {n_nina}mo)")
    ax.legend(fontsize=8)

    # Panel 2: H1 persistence diagrams for each regime
    ax = axes[0, 1]
    if len(nino_h1_fin) > 0:
        ax.scatter(nino_h1_fin[:, 0], nino_h1_fin[:, 1], c="red", s=12, alpha=0.6,
                   label=f"El Niño ({len(nino_h1_fin)})")
    if len(nina_h1_fin) > 0:
        ax.scatter(nina_h1_fin[:, 0], nina_h1_fin[:, 1], c="blue", s=12, alpha=0.6,
                   label=f"La Niña ({len(nina_h1_fin)})")
    all_pts = np.concatenate([p for p in [nino_h1_fin, nina_h1_fin] if len(p) > 0])
    if len(all_pts) > 0:
        lo, hi = all_pts.min(), all_pts.max()
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("H1 Persistence Diagrams by Regime")
    ax.legend(fontsize=8)

    # Panel 3: Lifetime distributions
    ax = axes[1, 0]
    nino_lt = (nino_h1_fin[:, 1] - nino_h1_fin[:, 0]) if len(nino_h1_fin) > 0 else np.array([])
    nina_lt = (nina_h1_fin[:, 1] - nina_h1_fin[:, 0]) if len(nina_h1_fin) > 0 else np.array([])
    if len(nino_lt) > 0:
        ax.hist(nino_lt, bins=20, alpha=0.5, color="red", label="El Niño", density=True)
    if len(nina_lt) > 0:
        ax.hist(nina_lt, bins=20, alpha=0.5, color="blue", label="La Niña", density=True)
    ax.axvline(observed_w, color="green", linestyle="--", linewidth=2,
               label=f"W1 dist={observed_w:.2f}")
    ax.set_xlabel("H1 Lifetime")
    ax.set_ylabel("Density")
    ax.set_title("H1 Lifetime Distributions")
    ax.legend(fontsize=8)

    # Panel 4: Permutation null distribution
    ax = axes[1, 1]
    ax.hist(null_w, bins=25, alpha=0.7, color="gray", label="Null")
    ax.axvline(observed_w, color="red", linestyle="--", linewidth=2,
               label=f"Observed (p={p_val:.3f})")
    ax.set_xlabel("Wasserstein Distance")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test (z={z_score:.2f})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / "exp3_elnino_vs_lanina.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp3_elnino_vs_lanina.png")

    return results


def experiment4_enso_nao_binding(enso: np.ndarray, nao: np.ndarray,
                                  subsample: int, n_surrogates: int,
                                  seed: int, fig_dir: Path) -> dict:
    """Exp 4: ENSO–NAO topological coupling via BindingDetector."""
    print("\n=== Experiment 4: ENSO–NAO Topological Coupling ===")

    # Align lengths
    n = min(len(enso), len(nao))
    enso_aligned = enso[:n]
    nao_aligned = nao[:n]
    print(f"  Aligned length: {n} months")

    # Cross-correlation as baseline
    cc = np.corrcoef(enso_aligned, nao_aligned)[0, 1]
    print(f"  Pearson correlation: {cc:.3f}")

    # Binding
    det = BindingDetector(max_dim=1, method="persistence_image", baseline="max")
    det.fit(enso_aligned, nao_aligned, subsample=subsample, seed=seed)
    score = det.binding_score()
    print(f"  Binding score: {score:.2f}")

    sig = det.test_significance(
        n_surrogates=n_surrogates, method="phase_randomize", seed=seed,
        subsample=subsample,
    )
    print(f"  p={sig['p_value']:.4f}, z={sig['z_score']:.2f}, significant={sig['significant']}")

    # Also compute binding for ENSO with itself (lagged) as reference
    lag_months = 6
    enso_lag = enso_aligned[lag_months:]
    enso_lead = enso_aligned[:-lag_months]
    det_self = BindingDetector(max_dim=1, method="persistence_image", baseline="max")
    det_self.fit(enso_lead, enso_lag, subsample=subsample, seed=seed)
    self_score = det_self.binding_score()
    self_sig = det_self.test_significance(
        n_surrogates=n_surrogates, method="phase_randomize", seed=seed,
        subsample=subsample,
    )
    print(f"  ENSO self-binding (6mo lag): score={self_score:.2f}, "
          f"p={self_sig['p_value']:.4f}, z={self_sig['z_score']:.2f}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Exp 4: ENSO–NAO Topological Coupling", fontsize=14, fontweight="bold")

    # Panel 1: Both time series
    ax = axes[0, 0]
    ax.plot(enso_aligned[:240], "r-", alpha=0.7, label="ENSO", linewidth=0.8)
    ax.plot(nao_aligned[:240], "b-", alpha=0.7, label="NAO", linewidth=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Index Value")
    ax.set_title(f"ENSO & NAO (first 20 years, r={cc:.3f})")
    ax.legend()

    # Panel 2: Binding scores comparison
    ax = axes[0, 1]
    bars = ax.bar(["ENSO–NAO", "ENSO self\n(6mo lag)"], [score, self_score],
                  color=["steelblue", "coral"])
    ax.set_ylabel("Binding Score")
    ax.set_title("Topological Binding")
    # Add p-value labels
    for bar, p in zip(bars, [sig["p_value"], self_sig["p_value"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"p={p:.3f}", ha="center", fontsize=10)

    # Panel 3: Surrogate null distribution (ENSO–NAO)
    ax = axes[1, 0]
    surr = sig["surrogate_scores"]
    ax.hist(surr, bins=25, alpha=0.7, color="gray", label="Surrogates")
    ax.axvline(score, color="red", linestyle="--", linewidth=2,
               label=f"Observed (z={sig['z_score']:.2f})")
    ax.set_xlabel("Binding Score")
    ax.set_ylabel("Count")
    ax.set_title(f"ENSO–NAO Null Distribution (p={sig['p_value']:.3f})")
    ax.legend(fontsize=8)

    # Panel 4: Surrogate null distribution (ENSO self)
    ax = axes[1, 1]
    surr_self = self_sig["surrogate_scores"]
    ax.hist(surr_self, bins=25, alpha=0.7, color="gray", label="Surrogates")
    ax.axvline(self_score, color="red", linestyle="--", linewidth=2,
               label=f"Observed (z={self_sig['z_score']:.2f})")
    ax.set_xlabel("Binding Score")
    ax.set_ylabel("Count")
    ax.set_title(f"ENSO Self-Binding Null Distribution (p={self_sig['p_value']:.3f})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / "exp4_enso_nao_binding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp4_enso_nao_binding.png")

    return {
        "exp4_n_months": n,
        "exp4_pearson_correlation": round(float(cc), 4),
        "exp4_enso_nao_binding": round(float(score), 2),
        "exp4_enso_nao_p": round(float(sig["p_value"]), 4),
        "exp4_enso_nao_z": round(float(sig["z_score"]), 2),
        "exp4_enso_nao_significant": sig["significant"],
        "exp4_enso_self_binding": round(float(self_score), 2),
        "exp4_enso_self_p": round(float(self_sig["p_value"]), 4),
        "exp4_enso_self_z": round(float(self_sig["z_score"]), 2),
    }


# ---------------------------------------------------------------------------
# Overview figure
# ---------------------------------------------------------------------------

def make_overview(all_results: dict, fig_dir: Path):
    """4-panel overview figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Branch 9: Climate Oscillation Topology — Overview",
                 fontsize=14, fontweight="bold")

    # Panel 1: ENSO Betti numbers vs Rössler
    ax = axes[0, 0]
    dims = ["H0", "H1", "H2"]
    enso_b = [all_results.get("exp1_enso_h0", 0),
              all_results.get("exp1_enso_h1_features", 0),
              all_results.get("exp1_enso_h2_features", 0)]
    ross_b = [0, all_results.get("exp1_rossler_h1", 0), all_results.get("exp1_rossler_h2", 0)]
    w = 0.35
    ax.bar([i - w / 2 for i in range(3)], enso_b, w, color="steelblue", label="ENSO")
    ax.bar([i + w / 2 for i in range(3)], ross_b, w, color="coral", label="Rössler")
    ax.set_xticks(range(3))
    ax.set_xticklabels(dims)
    ax.set_ylabel("Feature Count")
    ax.set_title("Exp 1: Attractor Topology")
    ax.legend()

    # Panel 2: Transition scores summary
    ax = axes[0, 1]
    shifts = all_results.get("exp2_detected_shifts", [])
    if shifts:
        names = [s["event"].split()[0] for s in shifts]
        detected = [1 if s["detected"] else 0 for s in shifts]
        colors = ["green" if d else "red" for d in detected]
        ax.barh(names, detected, color=colors)
        ax.set_xlim(-0.1, 1.5)
        ax.set_xlabel("Detected (1=yes)")
        ax.set_title("Exp 2: Known Event Detection")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # Panel 3: El Niño vs La Niña
    ax = axes[1, 0]
    nino_ent = all_results.get("exp3_elnino_h1_entropy", 0)
    nina_ent = all_results.get("exp3_lanina_h1_entropy", 0)
    p_val = all_results.get("exp3_wasserstein_p", 1.0)
    ax.bar(["El Niño", "La Niña"], [nino_ent, nina_ent], color=["red", "blue"])
    ax.set_ylabel("H1 Persistence Entropy")
    ax.set_title(f"Exp 3: Regime Comparison (W1 p={p_val:.3f})")

    # Panel 4: ENSO–NAO binding
    ax = axes[1, 1]
    enso_nao_b = all_results.get("exp4_enso_nao_binding", 0)
    enso_self_b = all_results.get("exp4_enso_self_binding", 0)
    enso_nao_p = all_results.get("exp4_enso_nao_p", 1.0)
    ax.bar(["ENSO–NAO", "ENSO self"], [enso_nao_b, enso_self_b],
           color=["steelblue", "coral"])
    ax.set_ylabel("Binding Score")
    ax.set_title(f"Exp 4: Topological Coupling (p={enso_nao_p:.3f})")

    plt.tight_layout()
    fig.savefig(fig_dir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved overview.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Branch 9: Climate Oscillation Topology")
    parser.add_argument("--subsample", type=int, default=200)
    parser.add_argument("--n_surrogates", type=int, default=100)
    parser.add_argument("--n_perms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-synthetic", action="store_true",
                        help="Skip NOAA download, use synthetic data")
    args = parser.parse_args()

    t0 = time.time()
    data_dir = Path("data/climate")
    fig_dir = Path("figures/climate")
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Branch 9: Climate Oscillations — Topological Binding")
    print("=" * 70)

    # --- Load data ---
    data_source = "synthetic"
    enso_result = None
    nao_result = None

    if not args.force_synthetic:
        enso_result = download_index(
            "https://psl.noaa.gov/data/correlation/nina34.anom.data", "Niño 3.4"
        )
        nao_result = download_index(
            "https://psl.noaa.gov/data/correlation/nao.data", "NAO"
        )

    if enso_result is not None:
        t_enso, enso = enso_result
        data_source = "NOAA"
    else:
        print("  Using synthetic ENSO")
        t_enso, enso = synthetic_enso(n_years=70, seed=args.seed)

    if nao_result is not None:
        t_nao, nao = nao_result
        # Align to ENSO time range
        mask = (t_nao >= t_enso[0]) & (t_nao <= t_enso[-1])
        if mask.sum() > 120:
            nao = nao[mask]
        else:
            print("  NAO alignment too short, using synthetic")
            nao = synthetic_nao(len(enso), seed=args.seed + 1)
    else:
        print("  Using synthetic NAO")
        nao = synthetic_nao(len(enso), seed=args.seed + 1)

    print(f"\n  Data source: {data_source}")
    print(f"  ENSO: {len(enso)} months ({t_enso[0]:.1f}–{t_enso[-1]:.1f})")
    print(f"  NAO: {len(nao)} months")

    # --- Run experiments ---
    all_results = {
        "branch": "experiment/tda-climate",
        "data_source": data_source,
        "n_months": len(enso),
    }

    r1 = experiment1_enso_attractor(enso, t_enso, args.subsample, args.seed, fig_dir)
    all_results.update(r1)

    r2 = experiment2_sliding_window(enso, t_enso, args.subsample, args.seed, fig_dir)
    all_results.update(r2)

    r3 = experiment3_elnino_vs_lanina(enso, t_enso, args.subsample, args.seed,
                                      args.n_perms, fig_dir)
    all_results.update(r3)

    r4 = experiment4_enso_nao_binding(enso, nao, args.subsample, args.n_surrogates,
                                      args.seed, fig_dir)
    all_results.update(r4)

    # --- Overview figure ---
    make_overview(all_results, fig_dir)

    # --- Save results ---
    runtime = time.time() - t0
    all_results["runtime_seconds"] = round(runtime, 1)

    # Remove non-serializable items
    clean = {}
    for k, v in all_results.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (np.bool_,)):
            clean[k] = bool(v)
        else:
            clean[k] = v

    with open(data_dir / "results.json", "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nSaved results.json")
    print(f"Total runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
