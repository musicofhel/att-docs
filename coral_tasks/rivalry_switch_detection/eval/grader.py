"""Grader for the rivalry_switch_detection CORAL task.

The agent's `initial_program.py` runs the rivalry-switch detector across 3
subjects (3629, 3630, 3632) and prints a JSON payload as the LAST stdout line:

  {
    "score": 0.62,                # mean F1 across the 3 subjects
    "per_subject": [
        {"subject": "3629", "f1": 0.71, "precision": 0.95, "recall": 0.57,
         "n_changepoints": 12, "n_switches": 30},
        {"subject": "3630", "f1": 0.55, "precision": 0.92, "recall": 0.39,
         "n_changepoints": 10, "n_switches": 25},
        {"subject": "3632", "f1": 0.60, "precision": 0.89, "recall": 0.45,
         "n_changepoints": 11, "n_switches": 28}
    ],
    "strategy": "ph_pi_l2_cusum",
    "tolerance_s": 5.0
  }

The grader trusts the per-subject F1 numbers but enforces:
  - All 3 subjects must be present
  - tolerance_s must be exactly 5.0 (the published evaluation tolerance)
  - precision and recall must be in [0, 1]

It does NOT recompute matching from raw event lists — that would require
loading the heavy .mat files inside the grader process. The seed program
handles loading and matching; the grader is a thin sanity layer.
"""

from __future__ import annotations

import json
import os

from coral.grader import TaskGrader

REQUIRED_SUBJECTS = {"3629", "3630", "3632"}
REQUIRED_TOLERANCE_S = 5.0


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
            return self.fail(f"no JSON payload (last 500: {tail!r})")

        per_subj = payload.get("per_subject")
        if not isinstance(per_subj, list) or not per_subj:
            return self.fail("payload missing 'per_subject' list")

        tol = payload.get("tolerance_s", REQUIRED_TOLERANCE_S)
        try:
            tol = float(tol)
        except (TypeError, ValueError):
            return self.fail(f"bad tolerance_s: {tol}")
        if abs(tol - REQUIRED_TOLERANCE_S) > 1e-6:
            return self.fail(
                f"tolerance_s must be {REQUIRED_TOLERANCE_S}, got {tol}"
            )

        seen: set[str] = set()
        f1s: list[float] = []
        for entry in per_subj:
            subj = str(entry.get("subject"))
            if subj not in REQUIRED_SUBJECTS:
                return self.fail(f"unexpected subject: {subj}")
            try:
                f1 = float(entry.get("f1"))
                p = float(entry.get("precision"))
                r = float(entry.get("recall"))
            except (TypeError, ValueError):
                return self.fail(f"bad numeric in entry: {entry}")
            if not (0.0 <= f1 <= 1.0 and 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0):
                return self.fail(f"f1/p/r out of [0,1]: {entry}")
            seen.add(subj)
            f1s.append(f1)

        missing = REQUIRED_SUBJECTS - seen
        if missing:
            return self.fail(f"missing subjects: {sorted(missing)}")

        mean_f1 = sum(f1s) / len(f1s)
        strategy = str(payload.get("strategy", "unknown"))[:80]

        explanation = json.dumps({
            "mean_f1": round(mean_f1, 4),
            "per_subject_f1": [
                {"subject": str(e["subject"]), "f1": round(float(e["f1"]), 4)}
                for e in per_subj
            ],
            "strategy": strategy,
            "tolerance_s": tol,
        })
        return self.score(float(mean_f1), explanation=explanation)

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
