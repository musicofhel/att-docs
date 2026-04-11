# CORAL follow-on experiments — summary

**Date**: 2026-04-12
**Prior round**: `2026-04-11-hybrid-tracer-bullets-summary.md`
**Plan**: `~/.claude/plans/hazy-wobbling-penguin.md`
**CORAL repo**: `~/CORAL` (v0.2.0)

## What was built

Three ATT branches, each containing a single `coral_tasks/<task_name>/`
directory with `task.yaml`, `eval/grader.py`, `seed/` (program + vendored
dependencies + pyproject.toml). Each task was validated end-to-end using
`uv run coral validate` before commit.

| Branch | Task dir | Seed score | Graduation target | Status |
|---|---|---:|---:|---|
| `experiment/coral-cardiac-feature-search` | `cardiac_feature_search/` | **0.5712 AUROC** (mean LOO) | ≥ 0.65 | validated, not launched |
| `experiment/coral-reservoir-multimode` | `reservoir_multimode/` | **0.9863 F1** (9 evals) | ≥ 0.80 | validated, seed already passes |
| `experiment/coral-rivalry-switch-detection` | `rivalry_switch_detection/` | **0.5577 F1** (3 subjects) | ≥ 0.70 | validated, not launched |

The original Branch D (vineyard rescue) is deferred — it is optional in the
plan and scheduled only if all three of A/B/C succeed.

## Surprise finding (Branch B)

The Crocker stack image-CUSUM detector, which was only tested on `sr_drift`
in the prior tracer round, **generalizes across all 3 reservoir failure
modes** with effectively zero F1 degradation:

```
sr_drift          F1 = 0.988  (matches prior tracer, F1=0.988)
noise             F1 = 0.988  (NEW — first multi-mode test)
connection_death  F1 = 0.984  (NEW — first multi-mode test)
mean              F1 = 0.986
```

This is the most important scientific result in the CORAL task-building
exercise. The graduation criterion was met before any agent touched the
task. Branch B's `task.yaml` has been rewritten to direct agents toward
higher-value goals:
- Extra-seed robustness (seeds {3,4,5,6,7})
- Inverted-binding Künneth confirmation
- Phase-randomized surrogate ablation
- Stretch F1 > 0.99

## Plan correction

The plan referenced "Branch C: sleep transition" but the
`experiment/tda-sleep` branch is actually about **binocular rivalry**
(Nie/Katyal/Engel 2023), not sleep PSG. Branch C was renamed to
`experiment/coral-rivalry-switch-detection`. The data (~12 GB) is already
on disk at `~/att-docs/data/eeg/rivalry_ssvep/`. The seed matches the
published baseline (mean F1 ≈ 0.57).

## How to launch

From `~/CORAL`:

```bash
# Cardiac feature search (~2-3 hr with 4 agents)
uv run coral start -c ~/att-docs/coral_tasks/cardiac_feature_search/task.yaml

# Reservoir multimode (~3-5 hr with 4 agents; seed already passes)
uv run coral start -c ~/att-docs/coral_tasks/reservoir_multimode/task.yaml

# Rivalry switch detection (~4-6 hr with 4 agents)
uv run coral start -c ~/att-docs/coral_tasks/rivalry_switch_detection/task.yaml
```

Do NOT run multiple tasks concurrently — CORAL parallelizes across N agents
within a single task, and running tasks in parallel saturates compute and
confuses leaderboards.

## Per-branch handoffs

Each branch has its own handoff doc:

- `experiment/coral-cardiac-feature-search:.claude/handoff/2026-04-12-coral-cardiac-feature-search.md`
- `experiment/coral-reservoir-multimode:.claude/handoff/2026-04-12-coral-reservoir-multimode.md`
- `experiment/coral-rivalry-switch-detection:.claude/handoff/2026-04-12-coral-rivalry-switch-detection.md`

## Pattern notes (lessons for future CORAL tasks)

1. **The grader runs in CORAL's Python environment**, not the agent's.
   Any third-party deps the grader needs (e.g., `wfdb`) must either be
   installed into CORAL's venv, or offloaded to the agent's interpreter
   via `self.run_script_json(script, timeout=N)`. The cardiac grader uses
   the subprocess approach — cleaner, no CORAL env pollution.

2. **`seed/pyproject.toml` with `[tool.uv.sources]` is the right pattern**
   for editable ATT install. It causes `TaskGrader.run_program()` to use
   `uv run --project <codebase>`, which picks up the agent's venv. Without
   it, `run_program` falls back to `sys.executable` (CORAL's interpreter)
   which lacks the deps.

3. **The `data/` dir in each seed is git-tracked on its branch** for small
   datasets (MIT-BIH, 9.4 MB). For large datasets (rivalry EEG, 12 GB),
   the seed program uses absolute paths into `~/att-docs/data/` which is
   already gitignored on master.

4. **Grader timeouts matter** — one attempt at max subsample can eat the
   entire 1800s budget. Branch C uses 3600s. Agents need to know about
   this and are told in task.yaml tips to drop subsample during iteration.

5. **Anti-cheat via asymmetric access**: grader holds labels, agents see
   signal only. Grader enforces config constraints (window_size, tolerance,
   required subjects) and validates payload schema.

## What's next

1. **Launch Branch A first** (cleanest open question, moderate compute).
2. **Launch Branch B second** — but note that the "real" goal is now
   Künneth/surrogate validation, not F1. The 0.986 seed is already
   publishable.
3. **Launch Branch C third** — most expensive per attempt, may need
   per-subject thresholding to move the needle.
4. **Skip Branch D** unless A/B/C all improve on their seeds.

Post-launch, follow the per-branch graduation paths in each handoff doc.

## Git state

```
master                                                 (untouched — no commits)
experiment/coral-cardiac-feature-search                (2 commits ahead)
experiment/coral-reservoir-multimode                   (2 commits ahead)
experiment/coral-rivalry-switch-detection              (2 commits ahead)
```

None pushed. Each branch has one commit adding the coral_tasks/ directory
and one commit adding the per-branch handoff doc. This summary doc is the
only thing committed to master (via this commit).
