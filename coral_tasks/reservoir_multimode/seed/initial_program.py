"""Reservoir multimode — seed program (Crocker image-CUSUM baseline).

Runs the Crocker stack image-CUSUM detector across 3 modes × 3 seeds (9 evals)
on a degrading Echo State Network. Reports mean F1.

Background
==========
Two prior tracer-bullet experiments tested topology-based reservoir monitoring
on the `sr_drift` mode only:
  - Crocker stack image-CUSUM (Branch 3): F1 = 0.988
  - Dense binding (Branch 2 control arm A): F1 = 0.929

Neither was tested on `noise` or `connection_death`. The graduation requirement
is multi-mode replication: mean F1 ≥ 0.80 across all 3 modes × 3 seeds.

SURPRISE: this seed reimplements the Crocker image-CUSUM detector and achieves
mean F1 = 0.986 across all 9 (mode, seed) pairs. Per-mode means:
  - sr_drift          = 0.988
  - noise             = 0.988
  - connection_death  = 0.984

The detector generalizes across modes with effectively zero degradation. The
graduation target is already met. Your job pivots to validation and theory
(see task.yaml — surrogate ablation, inverted binding, extra seeds).

YOUR JOB
========
Edit `initial_program.py` to improve mean F1 across the 9 (mode, seed) pairs.
The grader requires all 9 pairs to be present.

Strategies to consider:
  - Combine Crocker stack (H1) with H0 (component merging)
  - Replace image-CUSUM with column L1 to baseline
  - Use dense binding (joint vs marginal persistence) per window
  - Two-stage: cheap variance/SR-estimate gate + topology only on triggers
  - Per-mode specialized detectors with a mode-classification frontend
  - **INVERTED BINDING** — the Künneth interpretation of binding score is that
    binding measures topological *independence*, not coupling. As reservoirs
    drift toward synchronization (sr_drift in particular), the binding score
    should DROP, not spike. Try alarming on binding decreases relative to a
    rolling baseline. If this works, set `uses_inverted_binding: true` in your
    output payload — the grader records it for posterity. This is the only
    place in the project where the Künneth prediction can be tested directly.
  - Tune SIGMA_ALARM threshold per mode (currently 2.0)
  - Add longer baseline windows for noise mode (where the drift is gradual)

Constraints
===========
  - Don't modify reservoir_degradation_monitor.simulate_degradation. The
    grader doesn't import the simulator but it relies on N_HEALTHY=5000 as the
    ground-truth boundary, baked into per_eval F1 calculations.
  - Output format (LAST stdout line MUST be a single JSON object):

      {
        "score": 0.62,
        "per_eval": [
            {"mode": "sr_drift", "seed": 0, "f1": 0.99,
             "precision": 1.0, "recall": 0.98},
            ... (all 9 pairs required)
        ],
        "strategy": "crocker_image_cusum",
        "uses_inverted_binding": false
      }

References (read-only, vendored under reference/):
  - reference/run_crocker_reservoir.py — Branch 3 (PASS, F1=0.988 on sr_drift)
  - reference/run_vineyard_reservoir.py — Branch 2 (3 arms, A=0.929 dense binding)
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(__file__))

import reservoir_degradation_monitor as rdm  # noqa: E402
from att.embedding.takens import TakensEmbedder  # noqa: E402
from att.topology.persistence import PersistenceAnalyzer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW = 2000
STRIDE = 200
N_HEALTHY = rdm.N_HEALTHY  # 5000
N_TOTAL = rdm.N_TOTAL  # 15000
N_FILTRATION_STEPS = 100
SUBSAMPLE = 250
MAX_DIM = 1
SIGMA_ALARM = 2.0
N_BASELINE_WINDOWS = 25

MODES = ["sr_drift", "noise", "connection_death"]
SEEDS = [0, 1, 2]

# ---------------------------------------------------------------------------
# Per-window persistence and Crocker stack
# ---------------------------------------------------------------------------


def pc1(states: np.ndarray) -> np.ndarray:
    Xc = states - states.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[0]


def window_diagrams(window: np.ndarray, seed: int) -> list[np.ndarray]:
    if np.std(window) < 1e-12:
        return [np.empty((0, 2)) for _ in range(MAX_DIM + 1)]
    try:
        emb = TakensEmbedder(delay="auto", dimension="auto")
        emb.fit(window)
    except Exception:
        emb = TakensEmbedder(delay=10, dimension=3)
        emb.fit(window)
    cloud = emb.transform(window)
    pa = PersistenceAnalyzer(max_dim=MAX_DIM, backend="ripser")
    pa.fit_transform(cloud, subsample=SUBSAMPLE, seed=seed)
    return [
        pa.diagrams_[d] if d < len(pa.diagrams_) else np.empty((0, 2))
        for d in range(MAX_DIM + 1)
    ]


def build_crocker(
    diagrams_list: list[list[np.ndarray]], n_steps: int = N_FILTRATION_STEPS,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n_windows = len(diagrams_list)
    matrices: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    for d in range(MAX_DIM + 1):
        all_b: list[float] = []
        all_dth: list[float] = []
        for ws in diagrams_list:
            dgm = ws[d]
            if len(dgm):
                all_b.extend(dgm[:, 0].tolist())
                all_dth.extend(dgm[:, 1].tolist())
        if not all_b:
            grids.append(np.linspace(0.0, 1.0, n_steps))
            matrices.append(np.zeros((n_steps, n_windows)))
            continue
        eps_min = float(min(all_b))
        eps_max = float(max(all_dth))
        if eps_max <= eps_min:
            eps_max = eps_min + 1.0
        grid = np.linspace(eps_min, eps_max, n_steps)
        mat = np.zeros((n_steps, n_windows))
        for w_idx, ws in enumerate(diagrams_list):
            dgm = ws[d]
            if len(dgm) == 0:
                continue
            for b, dth in dgm:
                mat[:, w_idx] += (grid >= b) & (grid < dth)
        grids.append(grid)
        matrices.append(mat)
    return matrices, grids


def arm_image_cusum(crocker: np.ndarray, n_baseline: int) -> np.ndarray:
    baseline_col = crocker[:, :n_baseline].mean(axis=1)
    diff = np.abs(crocker - baseline_col[:, None]).sum(axis=0)
    return np.cumsum(diff)


def alarms_from_signal(
    signal: np.ndarray, n_baseline: int, k: float = SIGMA_ALARM,
) -> np.ndarray:
    base = signal[:n_baseline]
    mu = float(base.mean())
    sd = max(float(base.std()), 1e-15)
    return signal > mu + k * sd


def prf1(alarms: np.ndarray, ground_truth: np.ndarray) -> dict:
    tp = int(((alarms == 1) & (ground_truth == 1)).sum())
    fp = int(((alarms == 1) & (ground_truth == 0)).sum())
    fn = int(((alarms == 0) & (ground_truth == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": float(f1), "precision": float(prec), "recall": float(rec)}


# ---------------------------------------------------------------------------
# Per-(mode, seed) evaluation
# ---------------------------------------------------------------------------


def evaluate_one(mode: str, seed: int) -> dict:
    t0 = time.time()
    states, _, _ = rdm.simulate_degradation(mode, seed)
    signal = pc1(states)

    starts = list(range(0, N_TOTAL - WINDOW + 1, STRIDE))
    diagrams_list: list[list[np.ndarray]] = []
    for w_idx, s in enumerate(starts):
        diagrams_list.append(window_diagrams(signal[s : s + WINDOW], seed=seed + w_idx))

    matrices, _ = build_crocker(diagrams_list)
    crocker_h1 = matrices[1]

    ground_truth = np.array([1 if s >= N_HEALTHY else 0 for s in starts], dtype=int)

    sig = arm_image_cusum(crocker_h1, N_BASELINE_WINDOWS)
    alarms = alarms_from_signal(sig, N_BASELINE_WINDOWS, SIGMA_ALARM)
    scores = prf1(alarms.astype(int), ground_truth)
    scores["mode"] = mode
    scores["seed"] = seed
    scores["runtime_s"] = round(time.time() - t0, 1)
    return scores


def main():
    results: list[dict] = []
    for mode in MODES:
        for seed in SEEDS:
            r = evaluate_one(mode, seed)
            print(
                f"  mode={mode} seed={seed} F1={r['f1']:.3f} "
                f"P={r['precision']:.3f} R={r['recall']:.3f} "
                f"({r['runtime_s']:.1f}s)",
                file=sys.stderr,
            )
            results.append(r)

    f1s = [r["f1"] for r in results]
    mean_f1 = float(sum(f1s) / len(f1s))

    payload = {
        "score": mean_f1,
        "per_eval": results,
        "strategy": "crocker_image_cusum",
        "uses_inverted_binding": False,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
