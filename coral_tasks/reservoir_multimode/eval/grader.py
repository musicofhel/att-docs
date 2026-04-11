"""Grader for the reservoir_multimode CORAL task.

The agent's `initial_program.py` runs the full 3 modes × 3 seeds grid (9 evals)
and prints a JSON payload as the LAST stdout line:

  {
    "score": 0.62,                # mean F1 across the 9 (mode, seed) pairs
    "per_eval": [
        {"mode": "sr_drift", "seed": 0, "f1": 0.99, "precision": 1.0, "recall": 0.98},
        ...
    ],
    "strategy": "crocker_image_cusum",   # name the agent picked
    "uses_inverted_binding": false       # OPTIONAL — set true if alarming on binding drops
  }

The grader is intentionally thin: it trusts the per_eval F1 numbers but
recomputes the mean from them as a sanity check, and it requires all 9
(mode, seed) pairs to be present.

Anti-cheat: the seed program imports reservoir_degradation_monitor (vendored
under seed/) which holds the canonical N_HEALTHY=5000 boundary. Agents
*should* compute F1 against the boundary as ground truth — they cannot
rewrite the simulator without the grader catching the missing modes/seeds.
"""

from __future__ import annotations

import json
import os

from coral.grader import TaskGrader

REQUIRED_MODES = {"sr_drift", "noise", "connection_death"}
REQUIRED_SEEDS = {0, 1, 2}
N_REQUIRED_EVALS = len(REQUIRED_MODES) * len(REQUIRED_SEEDS)  # 9


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

        per_eval = payload.get("per_eval")
        if not isinstance(per_eval, list) or not per_eval:
            return self.fail("payload missing 'per_eval' list")

        # Verify all 9 (mode, seed) pairs covered
        seen: set[tuple[str, int]] = set()
        f1s: list[float] = []
        per_mode: dict[str, list[float]] = {}
        for entry in per_eval:
            mode = entry.get("mode")
            try:
                seed = int(entry.get("seed"))
            except (TypeError, ValueError):
                return self.fail(f"per_eval entry has bad seed: {entry}")
            if mode not in REQUIRED_MODES:
                return self.fail(f"unexpected mode: {mode}")
            if seed not in REQUIRED_SEEDS:
                return self.fail(f"unexpected seed: {seed}")
            try:
                f1 = float(entry.get("f1"))
            except (TypeError, ValueError):
                return self.fail(f"per_eval entry has bad f1: {entry}")
            if not (0.0 <= f1 <= 1.0):
                return self.fail(f"f1 out of [0,1]: {f1}")
            seen.add((mode, seed))
            f1s.append(f1)
            per_mode.setdefault(mode, []).append(f1)

        missing = {(m, s) for m in REQUIRED_MODES for s in REQUIRED_SEEDS} - seen
        if missing:
            return self.fail(f"missing (mode,seed) pairs: {sorted(missing)}")

        mean_f1 = sum(f1s) / len(f1s)
        per_mode_mean = {m: round(sum(v) / len(v), 4) for m, v in per_mode.items()}

        strategy = str(payload.get("strategy", "unknown"))[:80]
        uses_inverted = bool(payload.get("uses_inverted_binding", False))

        explanation = json.dumps({
            "mean_f1": round(mean_f1, 4),
            "per_mode_mean_f1": per_mode_mean,
            "n_evals": len(f1s),
            "strategy": strategy,
            "uses_inverted_binding": uses_inverted,
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
