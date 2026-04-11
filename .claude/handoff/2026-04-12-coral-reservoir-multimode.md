# CORAL: reservoir multimode — Handoff

**Date**: 2026-04-12
**Branch**: `experiment/coral-reservoir-multimode`
**Task dir**: `coral_tasks/reservoir_multimode/`

## Goal

Multi-mode graduation of the Crocker stack reservoir-degradation detector.
The prior tracer round (Branch 3, `experiment/crocker-stack-reservoir`)
showed F1=0.988 on `sr_drift` alone. The graduation requirement is
replication across `noise` and `connection_death`.

## Score

Mean F1 across 3 modes × 3 seeds = 9 (mode, seed) pairs:
- modes ∈ {sr_drift, noise, connection_death}
- seeds ∈ {0, 1, 2}

Per-window ground truth: window start_idx ≥ N_HEALTHY (5000) → drift.

The grader trusts the per-eval F1 values the program returns but enforces
full coverage of the 9 pairs.

## Seed result (verified by `coral validate`)

**mean F1 = 0.9863** across all 9 pairs (strategy: `crocker_image_cusum`)

Per-mode means:
- `sr_drift`         = 0.988
- `noise`            = 0.988
- `connection_death` = 0.984

**This is Branch B's main scientific finding by itself**: Crocker stack
image-CUSUM generalizes across reservoir failure modes with effectively zero
degradation. It is a universal reservoir-degradation early-warning, not a
sr-drift-specific trick.

Validation runtime: ~8.5 minutes wallclock.

## Task framing pivot

Because the seed already crushes the 0.80 graduation criterion, the task
description directs agents toward four higher-value goals:

1. **Extra-seed robustness**: re-run the same detector on seeds {3,4,5,6,7}
   and confirm F1 stays high. Report as `extra_seed_f1` in the payload.
2. **Inverted binding (Künneth prediction)**: alarm on binding score DROPS
   (not spikes). The Künneth cross-term interpretation says binding ~
   topological *independence*; synchronization → low binding. A single mode
   achieving F1 ≥ 0.60 with inverted binding counts as a theoretical
   confirmation. Set `uses_inverted_binding: true` in the payload.
3. **Phase-randomized surrogate ablation**: replace the drift phase with a
   phase-randomized version of the healthy phase. Detector should NOT
   alarm. Report `surrogate_f1` per eval.
4. **Stretch F1 above 0.99**.

## Vendored files (read-only references)

```
seed/
├── reservoir_degradation_monitor.py       ← experiment/neuromorphic-snn
├── reference/
│   ├── run_crocker_reservoir.py           ← experiment/crocker-stack-reservoir
│   └── run_vineyard_reservoir.py          ← experiment/vineyard-binding-reservoir
```

## File layout

```
coral_tasks/reservoir_multimode/
├── task.yaml                    # 4 agents, sonnet, 80 turns, research=true
├── eval/grader.py              # mean F1 across 9 required (mode,seed) pairs
├── .gitignore                  # results/ .coral/
└── seed/
    ├── initial_program.py      # Crocker image-CUSUM, 9-eval grid
    ├── reservoir_degradation_monitor.py  # simulate_degradation + MinimalESN
    ├── reference/              # prior tracer scripts, read-only
    ├── pyproject.toml          # editable att-toolkit + deps
    └── .gitignore              # .venv/ __pycache__/
```

## Launch

```bash
cd ~/CORAL && uv run coral start \
    -c ~/att-docs/coral_tasks/reservoir_multimode/task.yaml
```

## Post-CORAL graduation

Because the seed already passes, the high-value post-run actions are:

1. If an agent confirmed inverted binding: lift the strategy to a dedicated
   section in the preprint connecting the Künneth finding to reservoir
   monitoring. This is the strongest theoretical bridge in the project.
2. If an agent ran surrogate ablation and the detector did NOT alarm: add
   the null result as a robustness section.
3. If extra-seed F1 holds: lift `build_crocker()` to `att/topology/crocker.py`
   as a production-ready process-control primitive.
4. Update `RESEARCH.md` with the cross-mode generalization finding.
