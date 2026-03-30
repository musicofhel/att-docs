#!/usr/bin/env python3
"""Screen financial regime detection: attractor structure + cross-asset binding.

Tests whether ATT's topological tools detect meaningful structure in
financial time series and whether they can identify regime transitions
(market crises) using VIX > 30 as ground truth.

Three parts:
  Part 1: Attractor structure detection (embedding + persistence + surrogate test)
  Part 2: Regime detection via 4-method changepoint benchmark on SPY
  Part 3: Cross-asset binding (SPY-TLT, SPY-GLD)

Usage:
    python scripts/screen_financial_regimes.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch
from tqdm import tqdm

from att.binding.detector import BindingDetector
from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS = ["SPY", "TLT", "GLD"]  # VIX separate (ground truth)
VIX_TICKER = "^VIX"
YEARS = 5
WINDOW_SIZE = 500
STEP_SIZE = 50
MAX_DIM = 1
SUBSAMPLE = 200
PI_RES = 20
N_SEEDS = 3
N_SURROGATES = 15
VIX_CRISIS_THRESHOLD = 30
TOLERANCE_DAYS = 20
N_JOBS = min(8, mp.cpu_count())

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "financial"


# ---------------------------------------------------------------------------
# Data loading: yfinance with disk cache
# ---------------------------------------------------------------------------

def fetch_data() -> dict[str, "pd.DataFrame"]:
    """Fetch daily OHLCV for all tickers, cache to data/financial/."""
    import pandas as pd

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_tickers = TICKERS + [VIX_TICKER]
    data = {}

    # Check cache first
    all_cached = True
    for ticker in all_tickers:
        safe_name = ticker.replace("^", "").replace("/", "_")
        cache_path = DATA_DIR / f"{safe_name}_daily.parquet"
        if not cache_path.exists():
            all_cached = False
            break

    if all_cached:
        print("  Loading cached data from data/financial/")
        for ticker in all_tickers:
            safe_name = ticker.replace("^", "").replace("/", "_")
            cache_path = DATA_DIR / f"{safe_name}_daily.parquet"
            data[ticker] = pd.read_parquet(cache_path)
            print(f"    {ticker}: {len(data[ticker])} rows "
                  f"({data[ticker].index[0].date()} to "
                  f"{data[ticker].index[-1].date()})")
        return data

    # Fetch via yfinance
    print("  Fetching data via yfinance...")
    try:
        import yfinance as yf

        for ticker in all_tickers:
            safe_name = ticker.replace("^", "").replace("/", "_")
            cache_path = DATA_DIR / f"{safe_name}_daily.parquet"

            t = yf.Ticker(ticker)
            df = t.history(period=f"{YEARS}y", interval="1d")
            if df.empty:
                raise RuntimeError(f"yfinance returned no data for {ticker}")

            # Save to parquet
            df.to_parquet(cache_path)
            data[ticker] = df
            print(f"    {ticker}: {len(df)} rows "
                  f"({df.index[0].date()} to {df.index[-1].date()})")

        return data

    except Exception as e:
        print(f"  yfinance failed: {e}")
        print("  Falling back to synthetic regime-switching data")
        return _generate_synthetic_fallback()


def _generate_synthetic_fallback() -> dict[str, "pd.DataFrame"]:
    """Generate synthetic Hamilton regime-switching data as last resort."""
    import pandas as pd

    print("  WARNING: Using synthetic regime-switching data (two-state Hamilton model)")
    print("  Results are illustrative only -- not real market data!")

    rng = np.random.default_rng(42)
    n_days = YEARS * 252  # ~5 years of trading days
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n_days)

    # Hamilton two-state model parameters
    # State 0: low vol (normal), State 1: high vol (crisis)
    mu = [0.0005, -0.001]     # daily drift per state
    sigma = [0.01, 0.03]       # daily vol per state
    P = [[0.98, 0.02],         # transition matrix
         [0.05, 0.95]]

    data = {}
    for ticker in TICKERS:
        states = np.zeros(n_days, dtype=int)
        state = 0
        for i in range(1, n_days):
            if rng.random() < P[state][1 - state]:
                state = 1 - state
            states[i] = state

        returns = np.array([
            rng.normal(mu[s], sigma[s]) for s in states
        ])
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "Open": prices * (1 + rng.normal(0, 0.001, n_days)),
            "High": prices * (1 + np.abs(rng.normal(0, 0.005, n_days))),
            "Low": prices * (1 - np.abs(rng.normal(0, 0.005, n_days))),
            "Close": prices,
            "Volume": rng.integers(1e6, 1e8, n_days),
        }, index=dates)
        data[ticker] = df

    # Synthetic VIX: high when SPY in crisis state
    spy_returns = np.log(data["SPY"]["Close"]).diff().fillna(0).values
    rolling_vol = pd.Series(spy_returns).rolling(20).std().fillna(0.01).values
    vix = 15 + 100 * rolling_vol + rng.normal(0, 1, n_days)
    vix = np.clip(vix, 9, 80)

    data[VIX_TICKER] = pd.DataFrame({
        "Open": vix,
        "High": vix * 1.02,
        "Low": vix * 0.98,
        "Close": vix,
        "Volume": rng.integers(1e5, 1e7, n_days),
    }, index=dates)

    # Cache synthetic data
    for ticker, df in data.items():
        safe_name = ticker.replace("^", "").replace("/", "_")
        cache_path = DATA_DIR / f"{safe_name}_daily.parquet"
        df.to_parquet(cache_path)
        print(f"    {ticker} (synthetic): {len(df)} rows")

    return data


def compute_log_returns(df: "pd.DataFrame") -> np.ndarray:
    """Compute log returns from Close prices, drop NaN."""
    import pandas as pd
    close = df["Close"].values.astype(float)
    returns = np.diff(np.log(close))
    # Drop any NaN/inf
    mask = np.isfinite(returns)
    return returns[mask]


def get_vix_crisis_dates(vix_df: "pd.DataFrame") -> list[int]:
    """Find indices where VIX crosses above threshold (crisis onsets).

    Returns indices into trading day sequence (0-based).
    """
    vix = vix_df["Close"].values.astype(float)
    crisis_onsets = []
    in_crisis = False
    for i in range(len(vix)):
        if vix[i] >= VIX_CRISIS_THRESHOLD and not in_crisis:
            crisis_onsets.append(i)
            in_crisis = True
        elif vix[i] < VIX_CRISIS_THRESHOLD:
            in_crisis = False
    return crisis_onsets


# ---------------------------------------------------------------------------
# Part 1: Attractor structure detection
# ---------------------------------------------------------------------------

def phase_randomize_returns(returns: np.ndarray, seed: int) -> np.ndarray:
    """Phase-randomize a time series (preserve amplitude spectrum)."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    ft = np.fft.rfft(returns)
    phases = rng.uniform(0, 2 * np.pi, len(ft))
    # Keep DC and Nyquist real
    phases[0] = 0
    if n % 2 == 0:
        phases[-1] = 0
    ft_rand = ft * np.exp(1j * phases)
    return np.fft.irfft(ft_rand, n=n)


def run_part1(data: dict) -> tuple[dict, bool]:
    """Part 1: Attractor structure detection.

    Returns (results_dict, any_significant).
    """
    import pandas as pd

    print(f"\n{'#' * 72}")
    print("PART 1: ATTRACTOR STRUCTURE DETECTION")
    print(f"{'#' * 72}")

    results = {}
    any_significant = False

    for ticker in TICKERS:
        print(f"\n  --- {ticker} ---")
        returns = compute_log_returns(data[ticker])
        n = len(returns)
        print(f"  {n} daily log returns")

        # Step 1: Takens embedding
        print(f"  [1] Takens embedding (auto delay, auto dim)...")
        embedder = TakensEmbedder("auto", "auto")
        try:
            embedder.fit(returns)
            delay = embedder.delay_
            dim = embedder.dimension_
            cloud = embedder.transform(returns)
            auto_ok = True
            print(f"      delay={delay}, dim={dim}, cloud={cloud.shape}")

            # Log condition number
            if cloud.shape[0] > cloud.shape[1]:
                cond = np.linalg.cond(cloud[:min(1000, cloud.shape[0])])
                print(f"      condition number: {cond:.1f}")
                if cond > 1e6:
                    print(f"      WARNING: high condition number -- embedding may be degenerate")
            else:
                cond = float("nan")

        except Exception as e:
            print(f"      auto embedding failed: {e}")
            print(f"      trying manual: delay=1, dim=5")
            auto_ok = False
            embedder = TakensEmbedder(delay=1, dimension=5)
            embedder.fit(returns)
            cloud = embedder.transform(returns)
            delay = 1
            dim = 5
            cond = np.linalg.cond(cloud[:min(1000, cloud.shape[0])])
            print(f"      manual cloud={cloud.shape}, cond={cond:.1f}")

        # Step 2: Persistence analysis
        print(f"  [2] Persistence analysis (max_dim={MAX_DIM}, subsample=400)...")
        pa = PersistenceAnalyzer(max_dim=MAX_DIM, backend="ripser")
        pa.fit_transform(cloud, subsample=400, seed=42)

        betti = []
        total_pers = []
        for d in range(MAX_DIM + 1):
            dgm = pa.diagrams_[d] if d < len(pa.diagrams_) else np.empty((0, 2))
            n_features = len(dgm)
            betti.append(n_features)
            if n_features > 0:
                pers = dgm[:, 1] - dgm[:, 0]
                pers = pers[pers > 1e-10]
                tp = float(np.sum(pers))
                total_pers.append(tp)
            else:
                total_pers.append(0.0)

        # Persistence entropy
        all_pers = []
        for dgm in pa.diagrams_:
            if len(dgm) > 0:
                p = dgm[:, 1] - dgm[:, 0]
                p = p[p > 1e-10]
                all_pers.extend(p.tolist())
        if all_pers:
            all_pers = np.array(all_pers)
            probs = all_pers / all_pers.sum()
            entropy = -float(np.sum(probs * np.log(probs + 1e-30)))
        else:
            entropy = 0.0

        print(f"      Betti numbers: {betti}")
        print(f"      Total persistence: {[f'{t:.4f}' for t in total_pers]}")
        print(f"      Persistence entropy: {entropy:.4f}")

        # Step 3: Surrogate test
        print(f"  [3] Surrogate test ({N_SURROGATES} surrogates x {N_SEEDS} seeds)...")
        z_scores = []
        for seed_idx in range(N_SEEDS):
            seed = 42 + seed_idx * 100
            # Observed total persistence
            obs_pers = sum(total_pers)

            surr_pers_values = []
            for s in range(N_SURROGATES):
                surr_seed = seed + s + 1
                surr_returns = phase_randomize_returns(returns, surr_seed)

                # Embed surrogate
                try:
                    surr_embedder = TakensEmbedder(delay=delay, dimension=dim)
                    surr_embedder.fit(surr_returns)
                    surr_cloud = surr_embedder.transform(surr_returns)
                except Exception:
                    continue

                # Compute persistence
                surr_pa = PersistenceAnalyzer(max_dim=MAX_DIM, backend="ripser")
                surr_pa.fit_transform(surr_cloud, subsample=400, seed=surr_seed)

                surr_tp = 0.0
                for dgm in surr_pa.diagrams_:
                    if len(dgm) > 0:
                        p = dgm[:, 1] - dgm[:, 0]
                        p = p[p > 1e-10]
                        surr_tp += float(np.sum(p))
                surr_pers_values.append(surr_tp)

            if len(surr_pers_values) >= 5:
                surr_mean = np.mean(surr_pers_values)
                surr_std = np.std(surr_pers_values, ddof=1)
                z = (obs_pers - surr_mean) / surr_std if surr_std > 1e-10 else 0.0
                z_scores.append(z)
                print(f"      seed={seed}: obs={obs_pers:.4f}, "
                      f"surr_mean={surr_mean:.4f}+/-{surr_std:.4f}, z={z:.2f}")
            else:
                print(f"      seed={seed}: too few valid surrogates")

        mean_z = float(np.mean(z_scores)) if z_scores else 0.0
        significant = mean_z > 0

        results[ticker] = {
            "auto_ok": auto_ok,
            "delay": delay,
            "dim": dim,
            "condition": cond,
            "betti": betti,
            "total_persistence": total_pers,
            "entropy": entropy,
            "z_scores": z_scores,
            "mean_z": mean_z,
            "significant": significant,
        }

        if significant:
            any_significant = True
            print(f"      PASS: mean z={mean_z:.2f} > 0 -- "
                  f"attractor structure exceeds phase-randomized null")
        else:
            print(f"      FAIL: mean z={mean_z:.2f} <= 0 -- "
                  f"no excess structure beyond linear correlations")

    # Summary table
    print(f"\n{'=' * 72}")
    print("PART 1 SUMMARY: ATTRACTOR STRUCTURE")
    print(f"{'=' * 72}")
    hdr = (f"{'Ticker':<8} {'Auto':>5} {'Delay':>6} {'Dim':>4} "
           f"{'Cond':>10} {'B0':>5} {'B1':>5} {'Entropy':>8} {'Mean Z':>8} {'Sig':>5}")
    print(hdr)
    print(f"{'-' * 72}")
    for ticker in TICKERS:
        r = results[ticker]
        auto_s = "OK" if r["auto_ok"] else "FAIL"
        cond_s = f"{r['condition']:.0f}" if not np.isnan(r["condition"]) else "--"
        sig_s = "YES" if r["significant"] else "NO"
        print(f"{ticker:<8} {auto_s:>5} {r['delay']:>6} {r['dim']:>4} "
              f"{cond_s:>10} {r['betti'][0]:>5} {r['betti'][1]:>5} "
              f"{r['entropy']:>8.4f} {r['mean_z']:>8.2f} {sig_s:>5}")

    if not any_significant:
        print(f"\n  KILL CRITERION: z <= 0 on ALL series.")
        print(f"  No attractor structure detected beyond linear correlations.")
        print(f"  Parts 2 and 3 SKIPPED.")

    return results, any_significant


# ---------------------------------------------------------------------------
# Parallel PH infrastructure (copied from benchmark_changepoint_methods.py)
# ---------------------------------------------------------------------------

def _ph_worker(args):
    """Compute PH on one window cloud."""
    cloud, max_dim, subsample, seed = args
    pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser")
    pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return pa.diagrams_


def parallel_windowed_ph(cloud, label="PH"):
    """Sliding-window PH -> persistence image L2 distances."""
    n_points = len(cloud)
    starts = list(range(0, n_points - WINDOW_SIZE + 1, STEP_SIZE))
    centers = np.array([s + WINDOW_SIZE // 2 for s in starts])
    windows = [cloud[s : s + WINDOW_SIZE] for s in starts]

    args = [(w, MAX_DIM, SUBSAMPLE, 42) for w in windows]
    with mp.Pool(N_JOBS) as pool:
        all_diagrams = list(tqdm(
            pool.imap(_ph_worker, args),
            total=len(args), desc=f"  {label}",
        ))

    # Shared birth/persistence ranges
    birth_min, birth_max, pers_max = float("inf"), float("-inf"), 0.0
    has_data = False
    for dgms in all_diagrams:
        for dgm in dgms:
            if len(dgm) > 0:
                has_data = True
                birth_min = min(birth_min, float(dgm[:, 0].min()))
                birth_max = max(birth_max, float(dgm[:, 0].max()))
                p = dgm[:, 1] - dgm[:, 0]
                p = p[p > 1e-10]
                if len(p) > 0:
                    pers_max = max(pers_max, float(p.max()))

    br = (birth_min, birth_max) if has_data else (0, 1)
    pr = (0.0, pers_max) if has_data and pers_max > 0 else (0, 1)

    # Persistence images on shared grid
    images = []
    for dgms in all_diagrams:
        pa = PersistenceAnalyzer(max_dim=MAX_DIM)
        pa.diagrams_ = dgms
        imgs = pa.to_image(resolution=PI_RES, birth_range=br, persistence_range=pr)
        images.append(imgs)

    # Consecutive L2 distances
    dists = []
    for i in range(len(images) - 1):
        d = sum(
            float(np.sqrt(np.sum((images[i][k] - images[i + 1][k]) ** 2)))
            for k in range(MAX_DIM + 1)
        )
        dists.append(d)

    return centers, np.array(dists)


# ---------------------------------------------------------------------------
# CUSUM changepoint detection
# ---------------------------------------------------------------------------

def cusum_changepoints(scores):
    """Forward CUSUM, threshold = mean + 2*std. Returns indices into scores."""
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    threshold = mu + 2 * sigma
    cusum = 0.0
    cps = []
    for i, s in enumerate(scores):
        cusum = max(0.0, cusum + (s - mu))
        if cusum > threshold:
            cps.append(i)
            cusum = 0.0
    return cps


def scores_to_samples(cps, centers):
    """Convert score-array indices to sample positions."""
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    return np.array([int(midpoints[i]) for i in cps if i < len(midpoints)])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(true_s, det_s, tolerance):
    """Precision, recall, F1, mean detection lag."""
    true_s = np.asarray(true_s, dtype=float)
    det_s = np.asarray(det_s, dtype=float)
    n_true = len(true_s)
    n_det = len(det_s)

    if n_det == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_lag": float("nan"), "n_detected": 0, "n_true": n_true}
    if n_true == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_lag": float("nan"), "n_detected": n_det, "n_true": 0}

    # True positives + lag
    tp, lags = 0, []
    for d in det_s:
        offsets = d - true_s
        best = int(np.argmin(np.abs(offsets)))
        if abs(offsets[best]) <= tolerance:
            tp += 1
            lags.append(int(offsets[best]))

    # Recall: fraction of true transitions with >=1 detection nearby
    recalled = sum(1 for t in true_s if np.any(np.abs(det_s - t) <= tolerance))

    prec = tp / n_det if n_det > 0 else 0.0
    rec = recalled / n_true if n_true > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_lag = float(np.mean(lags)) if lags else float("nan")

    return {"precision": prec, "recall": rec, "f1": f1,
            "mean_lag": mean_lag, "n_detected": n_det, "n_true": n_true}


# ---------------------------------------------------------------------------
# Method 1: Topological (Takens -> sliding-window PH -> PI L2 -> CUSUM)
# ---------------------------------------------------------------------------

def method_topological(signal, label="topo"):
    """Topological changepoint detection."""
    t_embed = time.time()
    MAX_FIT_SAMPLES = 20000
    fit_signal = signal[:MAX_FIT_SAMPLES] if len(signal) > MAX_FIT_SAMPLES else signal

    embedder = TakensEmbedder("auto", "auto")
    try:
        embedder.fit(fit_signal)
    except Exception:
        embedder = TakensEmbedder(delay=1, dimension=5)
        embedder.fit(fit_signal)
    cloud = embedder.transform(signal)
    print(f"    Takens: delay={embedder.delay_}, dim={embedder.dimension_}, "
          f"cloud={cloud.shape} ({time.time()-t_embed:.1f}s)", flush=True)

    centers, dists = parallel_windowed_ph(cloud, label=label)
    cps = cusum_changepoints(dists)
    return scores_to_samples(cps, centers)


# ---------------------------------------------------------------------------
# Method 2: Window-based spectral (Welch PSD L2 -> CUSUM)
# ---------------------------------------------------------------------------

def method_spectral(signal):
    """Spectral changepoint detection using Welch PSD distances."""
    n = len(signal)
    starts = list(range(0, n - WINDOW_SIZE + 1, STEP_SIZE))
    centers = np.array([s + WINDOW_SIZE // 2 for s in starts])

    psds = []
    for s in starts:
        _, psd = welch(signal[s : s + WINDOW_SIZE], fs=1.0,
                       nperseg=min(256, WINDOW_SIZE))
        psds.append(psd)

    dists = np.array([
        float(np.sqrt(np.sum((psds[i] - psds[i + 1]) ** 2)))
        for i in range(len(psds) - 1)
    ])

    cps = cusum_changepoints(dists)
    return scores_to_samples(cps, centers)


# ---------------------------------------------------------------------------
# Method 3: Window-based variance (abs diff -> CUSUM)
# ---------------------------------------------------------------------------

def method_variance(signal):
    """Variance changepoint detection."""
    n = len(signal)
    starts = list(range(0, n - WINDOW_SIZE + 1, STEP_SIZE))
    centers = np.array([s + WINDOW_SIZE // 2 for s in starts])

    variances = np.array([np.var(signal[s : s + WINDOW_SIZE]) for s in starts])
    diffs = np.abs(np.diff(variances))

    cps = cusum_changepoints(diffs)
    return scores_to_samples(cps, centers)


# ---------------------------------------------------------------------------
# Method 4: BOCPD (Bayesian Online Changepoint Detection)
# ---------------------------------------------------------------------------

def _simple_bocpd(signal, hazard_rate=1.0 / 500):
    """Gaussian BOCPD (Adams & MacKay 2007), truncated run length."""
    max_len = 5000
    if len(signal) > max_len:
        step = len(signal) // max_len
        sig = signal[::step]
        scale = step
    else:
        sig = signal
        scale = 1

    T = len(sig)
    R = min(500, T)  # max run length

    rl_probs = np.zeros(R)
    rl_probs[0] = 1.0

    mu0 = float(np.mean(sig[: min(100, T)]))
    var0 = float(np.var(sig[: min(100, T)])) + 1e-6

    sums = np.zeros(R)
    sq_sums = np.zeros(R)
    counts = np.zeros(R)
    map_rl = np.zeros(T)

    for t in range(T):
        x = sig[t]

        # Predictive probability per run length (Gaussian)
        n = counts + 1e-10
        means = np.where(counts > 1, sums / n, mu0)
        vrs = np.where(counts > 1,
                       np.maximum(sq_sums / n - means ** 2, 1e-6), var0)

        log_p = -0.5 * np.log(2 * np.pi * vrs) - 0.5 * (x - means) ** 2 / vrs
        pred = np.exp(log_p - np.max(log_p))

        growth = rl_probs * pred * (1 - hazard_rate)
        cp_mass = float(np.sum(rl_probs * pred * hazard_rate))

        new_rl = np.zeros(R)
        new_rl[0] = cp_mass
        new_rl[1:] = growth[:-1]

        total = new_rl.sum()
        if total > 0:
            new_rl /= total
        rl_probs = new_rl
        map_rl[t] = np.argmax(rl_probs)

        # Update sufficient statistics
        sums[1:] = sums[:-1] + x
        sums[0] = 0.0
        sq_sums[1:] = sq_sums[:-1] + x * x
        sq_sums[0] = 0.0
        counts[1:] = counts[:-1] + 1
        counts[0] = 0

    # Changepoints: where MAP run length drops sharply
    cps = np.where(np.diff(map_rl) < -10)[0]
    return (cps * scale).astype(int)


def method_bocpd(signal):
    """BOCPD using simple Gaussian implementation."""
    return _simple_bocpd(signal)


# ---------------------------------------------------------------------------
# Part 2: Regime detection
# ---------------------------------------------------------------------------

def run_part2(data: dict) -> dict:
    """Part 2: 4-method changepoint benchmark on SPY with VIX ground truth."""
    print(f"\n{'#' * 72}")
    print("PART 2: REGIME DETECTION (SPY vs VIX GROUND TRUTH)")
    print(f"{'#' * 72}")

    spy_returns = compute_log_returns(data["SPY"])
    vix_crisis_indices = get_vix_crisis_dates(data[VIX_TICKER])

    # VIX indices are into the price series; returns are 1 shorter
    # Adjust: crisis index i in prices -> index i-1 in returns (approximately)
    crisis_in_returns = [max(0, c - 1) for c in vix_crisis_indices]
    n_crises = len(crisis_in_returns)

    print(f"  SPY: {len(spy_returns)} daily log returns")
    print(f"  VIX crisis onsets (>{VIX_CRISIS_THRESHOLD}): {n_crises} events")
    if n_crises > 0:
        print(f"  Crisis indices: {crisis_in_returns}")
    print(f"  Tolerance: +/-{TOLERANCE_DAYS} trading days")

    if n_crises == 0:
        print(f"  WARNING: No VIX crisis events found. Evaluation will show 0 recall.")

    results = {}

    # Method 1: Topological
    print(f"\n  [1/4] Topological")
    t1 = time.time()
    det = method_topological(spy_returns, label="topo-SPY")
    r = evaluate(crisis_in_returns, det, TOLERANCE_DAYS)
    results["Topological"] = r
    print(f"    -> {r['n_detected']} det, F1={r['f1']:.2f} ({time.time()-t1:.1f}s)")

    # Method 2: Spectral
    print(f"\n  [2/4] Spectral")
    t1 = time.time()
    det = method_spectral(spy_returns)
    r = evaluate(crisis_in_returns, det, TOLERANCE_DAYS)
    results["Spectral"] = r
    print(f"    -> {r['n_detected']} det, F1={r['f1']:.2f} ({time.time()-t1:.1f}s)")

    # Method 3: Variance
    print(f"\n  [3/4] Variance")
    t1 = time.time()
    det = method_variance(spy_returns)
    r = evaluate(crisis_in_returns, det, TOLERANCE_DAYS)
    results["Variance"] = r
    print(f"    -> {r['n_detected']} det, F1={r['f1']:.2f} ({time.time()-t1:.1f}s)")

    # Method 4: BOCPD
    print(f"\n  [4/4] BOCPD")
    t1 = time.time()
    det = method_bocpd(spy_returns)
    r = evaluate(crisis_in_returns, det, TOLERANCE_DAYS)
    results["BOCPD"] = r
    print(f"    -> {r['n_detected']} det, F1={r['f1']:.2f} ({time.time()-t1:.1f}s)")

    # Print table
    print(f"\n{'=' * 72}")
    print(f"PART 2 RESULTS: SPY REGIME DETECTION ({n_crises} VIX crisis onsets)")
    print(f"{'=' * 72}")
    hdr = (f"{'Method':<16} {'Precision':>9} {'Recall':>9} {'F1':>9} "
           f"{'Mean Lag':>9} {'Detections':>10}")
    print(hdr)
    print(f"{'-' * 72}")
    for method_name in ["Topological", "Spectral", "Variance", "BOCPD"]:
        r = results[method_name]
        lag = (f"{r['mean_lag']:+.0f}"
               if not np.isnan(r["mean_lag"]) else "--")
        print(f"{method_name:<16} {r['precision']:>9.2f} {r['recall']:>9.2f} "
              f"{r['f1']:>9.2f} {lag:>9} {r['n_detected']:>10}")

    return results


# ---------------------------------------------------------------------------
# Part 3: Cross-asset binding
# ---------------------------------------------------------------------------

def run_part3(data: dict) -> dict:
    """Part 3: Cross-asset binding (SPY-TLT, SPY-GLD)."""
    print(f"\n{'#' * 72}")
    print("PART 3: CROSS-ASSET BINDING")
    print(f"{'#' * 72}")

    pairs = [("SPY", "TLT"), ("SPY", "GLD")]
    results = {}

    for ticker_a, ticker_b in pairs:
        print(f"\n  --- {ticker_a} vs {ticker_b} ---")

        returns_a = compute_log_returns(data[ticker_a])
        returns_b = compute_log_returns(data[ticker_b])

        # Align lengths
        min_len = min(len(returns_a), len(returns_b))
        returns_a = returns_a[:min_len]
        returns_b = returns_b[:min_len]
        print(f"  {min_len} aligned daily returns")

        pair_key = f"{ticker_a}-{ticker_b}"
        seed_results = []

        for seed_idx in range(N_SEEDS):
            seed = 42 + seed_idx * 100
            print(f"  seed={seed}:")

            try:
                bd = BindingDetector(max_dim=MAX_DIM)
                bd.fit(returns_a, returns_b, subsample=400, seed=seed)

                sig_result = bd.test_significance(
                    n_surrogates=N_SURROGATES, seed=seed, subsample=400,
                )

                print(f"    observed={sig_result['observed_score']:.4f}, "
                      f"z={sig_result['z_score']:.2f}, "
                      f"p={sig_result['p_value']:.3f}, "
                      f"sig={sig_result['significant']}")

                seed_results.append({
                    "seed": seed,
                    "observed_score": sig_result["observed_score"],
                    "z_score": sig_result["z_score"],
                    "p_value": sig_result["p_value"],
                    "significant": sig_result["significant"],
                })

            except Exception as e:
                print(f"    ERROR: {e}")
                seed_results.append({
                    "seed": seed,
                    "observed_score": float("nan"),
                    "z_score": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                    "error": str(e),
                })

        results[pair_key] = seed_results

    # Summary table
    print(f"\n{'=' * 72}")
    print("PART 3 SUMMARY: CROSS-ASSET BINDING")
    print(f"{'=' * 72}")
    hdr = (f"{'Pair':<12} {'Seed':>6} {'Observed':>10} {'Z-score':>9} "
           f"{'P-value':>9} {'Significant':>12}")
    print(hdr)
    print(f"{'-' * 72}")
    for pair_key, seed_results in results.items():
        for sr in seed_results:
            obs_s = f"{sr['observed_score']:.4f}" if not np.isnan(sr["observed_score"]) else "ERROR"
            z_s = f"{sr['z_score']:.2f}" if not np.isnan(sr["z_score"]) else "--"
            p_s = f"{sr['p_value']:.3f}" if not np.isnan(sr["p_value"]) else "--"
            sig_s = "YES" if sr["significant"] else "NO"
            print(f"{pair_key:<12} {sr['seed']:>6} {obs_s:>10} {z_s:>9} "
                  f"{p_s:>9} {sig_s:>12}")

    return results


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def print_verdict(part1_results: dict, part1_pass: bool,
                  part2_results: dict | None, part3_results: dict | None):
    """Print final verdict."""
    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")

    # Part 1
    n_sig = sum(1 for r in part1_results.values() if r["significant"])
    print(f"\n1. ATTRACTOR STRUCTURE: {n_sig}/{len(TICKERS)} series show "
          f"structure beyond phase-randomized null")
    for ticker in TICKERS:
        r = part1_results[ticker]
        status = "PASS" if r["significant"] else "FAIL"
        print(f"   {ticker}: z={r['mean_z']:.2f} [{status}]")

    if not part1_pass:
        print(f"\n   EARLY STOP: No attractor structure detected in any series.")
        print(f"   Financial returns at daily frequency do not exhibit")
        print(f"   topological attractor structure beyond linear correlations.")
        print(f"   Parts 2 and 3 were skipped.")
        print(f"\n{'=' * 72}")
        return

    # Part 2
    if part2_results is not None:
        print(f"\n2. REGIME DETECTION (SPY vs VIX crises):")
        methods = ["Topological", "Spectral", "Variance", "BOCPD"]
        f1s = {}
        for m in methods:
            r = part2_results.get(m)
            if r is not None:
                f1s[m] = r["f1"]
                print(f"   {m}: F1={r['f1']:.2f}, "
                      f"P={r['precision']:.2f}, R={r['recall']:.2f}")

        if f1s:
            best = max(f1s, key=f1s.get)
            topo_f1 = f1s.get("Topological", 0.0)
            print(f"\n   Best method: {best} (F1={f1s[best]:.2f})")

            if topo_f1 >= max(f1s.values()) - 0.02:
                print(f"   Topology is COMPETITIVE for financial regime detection")
            elif topo_f1 > 0:
                print(f"   Topology DETECTS regimes but is not the best method")
            else:
                print(f"   Topology FAILS to detect financial regime transitions")

    # Part 3
    if part3_results is not None:
        print(f"\n3. CROSS-ASSET BINDING:")
        for pair_key, seed_results in part3_results.items():
            n_sig_pair = sum(1 for sr in seed_results if sr["significant"])
            mean_z = np.mean([sr["z_score"] for sr in seed_results
                              if not np.isnan(sr["z_score"])])
            status = "DETECTED" if n_sig_pair >= 2 else "NOT DETECTED"
            print(f"   {pair_key}: {n_sig_pair}/{N_SEEDS} seeds significant, "
                  f"mean z={mean_z:.2f} [{status}]")

        any_binding = any(
            sum(1 for sr in srs if sr["significant"]) >= 2
            for srs in part3_results.values()
        )
        if any_binding:
            print(f"\n   Cross-asset topological binding IS detectable")
        else:
            print(f"\n   Cross-asset topological binding is NOT reliably detectable")

    # Overall
    print(f"\n{'=' * 40}")
    print(f"OVERALL ASSESSMENT")
    print(f"{'=' * 40}")

    positives = []
    if n_sig > 0:
        positives.append(f"{n_sig}/{len(TICKERS)} series have attractor structure")
    if part2_results:
        topo_f1 = part2_results.get("Topological", {}).get("f1", 0)
        if topo_f1 > 0.1:
            positives.append(f"regime detection F1={topo_f1:.2f}")
    if part3_results:
        any_binding = any(
            sum(1 for sr in srs if sr["significant"]) >= 2
            for srs in part3_results.values()
        )
        if any_binding:
            positives.append("cross-asset binding detected")

    if positives:
        print(f"\n  ATT FINDS SIGNAL IN FINANCIAL DATA:")
        for p in positives:
            print(f"    + {p}")
    else:
        print(f"\n  ATT DOES NOT FIND MEANINGFUL SIGNAL IN FINANCIAL DATA")
        print(f"  Daily financial returns lack sufficient deterministic")
        print(f"  structure for topological methods to add value.")

    print(f"\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("Screen: financial regime detection via ATT")
    print(f"Tickers: {TICKERS}, VIX ground truth: {VIX_TICKER}")
    print(f"Window={WINDOW_SIZE}, step={STEP_SIZE}, subsample={SUBSAMPLE}, "
          f"PI_res={PI_RES}")
    print(f"Seeds={N_SEEDS}, surrogates={N_SURROGATES}, "
          f"VIX threshold={VIX_CRISIS_THRESHOLD}")
    print(f"Workers: {N_JOBS}")
    print()

    # Fetch data
    print(f"{'#' * 72}")
    print("DATA LOADING")
    print(f"{'#' * 72}")
    data = fetch_data()

    # Part 1: Attractor structure
    part1_results, part1_pass = run_part1(data)

    # Parts 2 & 3: only if Part 1 passes
    part2_results = None
    part3_results = None
    if part1_pass:
        part2_results = run_part2(data)
        part3_results = run_part3(data)
    else:
        print(f"\n  Parts 2 and 3 SKIPPED (kill criterion met in Part 1)")

    # Verdict
    print_verdict(part1_results, part1_pass, part2_results, part3_results)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
