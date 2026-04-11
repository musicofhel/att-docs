"""Cardiac feature search — seed program (baseline).

Loads MIT-BIH records {200, 201, 207, 210, 217}, slides 10s windows,
Takens-embeds each window, computes a single topological feature
(persistence entropy of H1), and emits the per-window features as JSON
on the LAST stdout line.

The grader (held outside this directory) does:
  1. Loads the same windows (using the window_size / step_size from config)
  2. Computes ground-truth window labels from the .atr annotations
  3. Runs leave-one-record-out logistic regression
  4. Returns mean AUROC across 5 folds

YOUR JOB
========
This baseline is intentionally weak (a single feature). You should:
  - Add more topological features (max_h1_lifetime, total_h1_persistence,
    Betti curve moments, persistence landscape stats, ...)
  - Try non-topological features (variance, RMS, dominant frequency, ...)
    if they help — tag them as "signal" not "topo" so the leaderboard
    records the mix.
  - Try different window/step sizes (beat-scale ~360 samples, multi-scale,
    overlapping ...)
  - Try different Takens embedding parameters
  - Stack many features and let the LR figure out the weights

You may NOT look at the .atr annotation files — your features must be
computed from the ECG signal alone. The grader does the train/test split.

Output format (last stdout line MUST be a single JSON object):
  {
    "features": [
        {"record": "200", "window_idx": 0, "values": [f1, f2, ...]},
        ...
    ],
    "feature_names": ["persistence_entropy_h1"],
    "feature_tags":  ["topo"],
    "config": {"window_size": 3600, "step_size": 1800}
  }
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import wfdb

warnings.filterwarnings("ignore")

from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer

RECORDS = ["200", "201", "207", "210", "217"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "mitdb")

# Window config — 360 Hz × 10s = 3600 samples per window
WINDOW_SIZE = 3600
STEP_SIZE = 1800
SUBSAMPLE = 200

FEATURE_NAMES = ["persistence_entropy_h1"]
FEATURE_TAGS = ["topo"]


def takens_embed(window: np.ndarray) -> np.ndarray:
    try:
        emb = TakensEmbedder(delay="auto", dimension="auto")
        emb.fit(window)
    except Exception:
        emb = TakensEmbedder(delay=8, dimension=4)
        emb.fit(window)
    return emb.transform(window)


def persistence_entropy_h1(window: np.ndarray, seed: int) -> float:
    cloud = takens_embed(window)
    pa = PersistenceAnalyzer(max_dim=1, backend="ripser")
    pa.fit_transform(cloud, subsample=SUBSAMPLE, seed=seed)
    if len(pa.diagrams_) < 2:
        return 0.0
    dgm1 = pa.diagrams_[1]
    if len(dgm1) == 0:
        return 0.0
    lifetimes = dgm1[:, 1] - dgm1[:, 0]
    lifetimes = lifetimes[lifetimes > 1e-10]
    if len(lifetimes) == 0:
        return 0.0
    total = float(lifetimes.sum())
    if total <= 0:
        return 0.0
    p = lifetimes / total
    return float(-np.sum(p * np.log(p + 1e-15)))


def extract_features(record: str) -> list[dict]:
    rec_path = os.path.join(DATA_DIR, record)
    sig = wfdb.rdrecord(rec_path).p_signal[:, 0]
    out = []
    n = len(sig)
    for w_idx, start in enumerate(range(0, n - WINDOW_SIZE + 1, STEP_SIZE)):
        window = sig[start:start + WINDOW_SIZE]
        if np.std(window) < 1e-10:
            continue
        try:
            f = persistence_entropy_h1(window, seed=w_idx + 1)
        except Exception as e:
            print(f"WARN record={record} window={w_idx} skipped: {e}", file=sys.stderr)
            continue
        out.append({
            "record": record,
            "window_idx": w_idx,
            "values": [f],
        })
    return out


def main():
    all_features: list[dict] = []
    for rec in RECORDS:
        feats = extract_features(rec)
        print(f"record={rec} n_windows={len(feats)}", file=sys.stderr)
        all_features.extend(feats)

    payload = {
        "features": all_features,
        "feature_names": FEATURE_NAMES,
        "feature_tags": FEATURE_TAGS,
        "config": {
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
        },
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
