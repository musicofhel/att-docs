"""Grader for the vineyard_rescue CORAL task.

The agent's initial_program.py runs a tracking-based binding detector on
sr_drift (seed=42) and prints a JSON payload as the LAST stdout line:

  {
    "score": 0.94,
    "f1": 0.94,
    "precision": 0.96,
    "recall": 0.92,
    "strategy": "union_find_greedy",
    "uses_tracking": true,
    "n_windows": 261,
    "onset_window": 100,
    "first_alarm_window": 101,
    "arm_a_f1": 0.929,
    "tracking_lift": 0.011
  }

The grader validates the payload and returns the F1 score.
"""

from __future__ import annotations

import json
import os

from coral.grader import TaskGrader


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

        # Extract and validate F1
        try:
            f1 = float(payload.get("f1", payload.get("score")))
        except (TypeError, ValueError):
            return self.fail("payload missing 'f1' or 'score'")

        if not (0.0 <= f1 <= 1.0):
            return self.fail(f"f1 out of [0,1]: {f1}")

        strategy = str(payload.get("strategy", "unknown"))[:80]
        uses_tracking = bool(payload.get("uses_tracking", False))
        arm_a_f1 = payload.get("arm_a_f1")
        tracking_lift = payload.get("tracking_lift")

        explanation = json.dumps({
            "f1": round(f1, 4),
            "precision": payload.get("precision"),
            "recall": payload.get("recall"),
            "strategy": strategy,
            "uses_tracking": uses_tracking,
            "arm_a_f1": arm_a_f1,
            "tracking_lift": round(float(tracking_lift), 4) if tracking_lift is not None else None,
            "n_windows": payload.get("n_windows"),
            "first_alarm_window": payload.get("first_alarm_window"),
        })
        return self.score(float(f1), explanation=explanation)

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
