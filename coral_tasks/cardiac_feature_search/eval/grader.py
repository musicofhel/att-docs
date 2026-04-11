"""Grader for the cardiac_feature_search CORAL task.

Strategy
--------
The agent's program writes a JSON payload to stdout. The payload contains:

  {
    "features": [
       {"record": "200", "window_idx": 0, "values": [f1, f2, ...], "tags": ["topo", "topo", ...]},
       ...
    ],
    "feature_names": ["max_h1_lifetime", "persistence_entropy_h1", ...],
    "feature_tags":  ["topo", "topo", ...],
    "config": {...optional knobs the agent picked, e.g. window/step/embedding...},
    "score_self_estimate": 0.6   # optional, ignored by grader; for the agent's own logging
  }

The grader holds the ground-truth beat-class labels for each window. It uses
the same window/step the agent used (encoded in the per-window 'window_idx'
together with the agent's config block) to compute window-level binary labels
from the MIT-BIH annotations, then runs a leave-one-record-out logistic
regression and reports the mean AUROC across the 5 folds.

Records: {200, 201, 207, 210, 217} (held in seed/data/mitdb/).
Tolerance: a window is "abnormal" if at least 50% of beats in the window are
non-normal (any AAMI symbol other than N, L, R, e, j).

Anti-cheat: agents do NOT receive the labels. Their job is to extract per-window
features. The grader does the train/test split.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

from coral.grader import TaskGrader

RECORDS = ["200", "201", "207", "210", "217"]
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
DATA_RELPATH = "data/mitdb"


class Grader(TaskGrader):
    def evaluate(self):
        program = self.args.get("program_file", "initial_program.py")
        program_path = os.path.join(self.codebase_path, program)
        if not os.path.exists(program_path):
            return self.fail(f"missing {program}")

        try:
            result = self.run_program(program)
        except (TimeoutError, FileNotFoundError) as e:
            return self.fail(f"run failed: {e}")

        if result.returncode != 0:
            tail = (result.stderr or "")[-800:]
            return self.fail(f"crash rc={result.returncode}: {tail}")

        payload = self._parse_last_json(result.stdout)
        if payload is None:
            tail = (result.stdout or "")[-500:]
            return self.fail(f"no JSON payload in stdout (last 500: {tail!r})")

        features = payload.get("features")
        if not isinstance(features, list) or not features:
            return self.fail("payload missing non-empty 'features' list")

        n_records_seen = len({f.get("record") for f in features})
        if n_records_seen < len(RECORDS):
            return self.fail(
                f"features cover only {n_records_seen} records, expected {len(RECORDS)}"
            )

        # Load ground-truth labels using the agent's window/step config
        cfg = payload.get("config") or {}
        try:
            window = int(cfg.get("window_size", 3600))
            step = int(cfg.get("step_size", 1800))
        except Exception as e:
            return self.fail(f"bad config in payload: {e}")

        try:
            labels_by_record = self._load_labels(window, step)
        except Exception as e:
            return self.fail(f"label load failed: {e}")

        # Group features by record
        by_record: dict[str, list[dict]] = defaultdict(list)
        for f in features:
            rec = f.get("record")
            if rec is not None:
                by_record[rec].append(f)
        for rec in RECORDS:
            if rec not in by_record:
                return self.fail(f"no features for record {rec}")

        # Match features to labels by window_idx
        X_by_record: dict[str, list[list[float]]] = {}
        y_by_record: dict[str, list[int]] = {}
        for rec in RECORDS:
            rec_feats = sorted(by_record[rec], key=lambda d: d.get("window_idx", -1))
            label_dict = labels_by_record[rec]
            xs = []
            ys = []
            for f in rec_feats:
                wi = f.get("window_idx")
                if wi not in label_dict:
                    continue
                vals = f.get("values")
                if not isinstance(vals, list) or not vals:
                    continue
                xs.append([float(v) for v in vals])
                ys.append(int(label_dict[wi]))
            if not xs:
                return self.fail(f"no usable feature/label pairs for record {rec}")
            X_by_record[rec] = xs
            y_by_record[rec] = ys

        # Verify class balance — every record must have both classes for AUROC
        for rec in RECORDS:
            ys = y_by_record[rec]
            if len(set(ys)) < 2:
                # Use a degenerate fold AUROC of 0.5 for that record (no signal)
                pass

        # LOO over records
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score
            import numpy as np
        except ImportError as e:
            return self.fail(f"sklearn import failed: {e}")

        feature_lengths = {len(x) for rec in RECORDS for x in X_by_record[rec]}
        if len(feature_lengths) != 1:
            return self.fail(
                f"inconsistent feature vector lengths across records: {feature_lengths}"
            )
        n_features = feature_lengths.pop()

        fold_aurocs: list[float] = []
        for held in RECORDS:
            X_train = []
            y_train = []
            for rec in RECORDS:
                if rec == held:
                    continue
                X_train.extend(X_by_record[rec])
                y_train.extend(y_by_record[rec])
            X_train = np.array(X_train, dtype=float)
            y_train = np.array(y_train, dtype=int)
            X_test = np.array(X_by_record[held], dtype=float)
            y_test = np.array(y_by_record[held], dtype=int)

            if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                fold_aurocs.append(0.5)
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            try:
                clf = LogisticRegression(
                    max_iter=2000, random_state=0, solver="lbfgs",
                )
                clf.fit(X_train_s, y_train)
                proba = clf.predict_proba(X_test_s)[:, 1]
                fold_aurocs.append(float(roc_auc_score(y_test, proba)))
            except Exception as e:
                return self.fail(f"LR/AUROC failed on held={held}: {e}")

        mean_auroc = float(sum(fold_aurocs) / len(fold_aurocs))

        feature_names = payload.get("feature_names") or [
            f"f{i}" for i in range(n_features)
        ]
        feature_tags = payload.get("feature_tags") or ["unknown"] * n_features
        topo_count = sum(1 for t in feature_tags if t == "topo")

        explanation = json.dumps({
            "mean_auroc": round(mean_auroc, 4),
            "fold_aurocs": [round(a, 4) for a in fold_aurocs],
            "n_features": n_features,
            "n_topo_features": topo_count,
            "feature_names": feature_names[:20],
            "window_size": window,
            "step_size": step,
        })
        return self.score(mean_auroc, explanation=explanation)

    # ------------------------------------------------------------------ helpers

    def _parse_last_json(self, stdout: str) -> dict | None:
        if not stdout:
            return None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None

    def _load_labels(self, window: int, step: int) -> dict[str, dict[int, int]]:
        """Load MIT-BIH annotations and produce per-window binary labels.

        Runs in a subprocess via the agent's Python interpreter (which has
        wfdb installed) so that CORAL's own env does not need wfdb.

        A window is labeled 1 (abnormal) if >50% of its beats are non-normal.
        """
        data_dir = os.path.join(self.codebase_path, DATA_RELPATH)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"data dir not found: {data_dir}")

        normal_set = "{" + ",".join(repr(s) for s in NORMAL_SYMBOLS) + "}"
        records_repr = repr(RECORDS)

        script = f"""
import json, os
import wfdb

DATA_DIR = {data_dir!r}
RECORDS = {records_repr}
NORMAL = {normal_set}
WINDOW = {window}
STEP = {step}

out = {{}}
for rec in RECORDS:
    rp = os.path.join(DATA_DIR, rec)
    ann = wfdb.rdann(rp, 'atr')
    sig_len = wfdb.rdrecord(rp).p_signal.shape[0]
    samples = ann.sample
    symbols = ann.symbol
    rec_out = {{}}
    n_windows = (sig_len - WINDOW) // STEP + 1
    for w in range(n_windows):
        s = w * STEP
        e = s + WINDOW
        idxs = [i for i, smp in enumerate(samples) if s <= smp < e]
        if not idxs:
            continue
        n_abnormal = sum(1 for i in idxs if symbols[i] not in NORMAL)
        rec_out[str(w)] = 1 if n_abnormal / len(idxs) >= 0.5 else 0
    out[rec] = rec_out
print(json.dumps(out))
"""
        raw = self.run_script_json(script, timeout=300)
        return {rec: {int(k): v for k, v in d.items()} for rec, d in raw.items()}
