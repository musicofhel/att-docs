# CORAL: rivalry switch detection — Handoff

**Date**: 2026-04-12
**Branch**: `experiment/coral-rivalry-switch-detection`
**Task dir**: `coral_tasks/rivalry_switch_detection/`

## Goal

Improve the recall of EEG-based perceptual switch detection during binocular
rivalry, while keeping the already-publication-grade precision. This replaces
"vineyard rescue" as Branch C in the plan, after discovering that the
`experiment/tda-sleep` branch is actually about binocular rivalry (not sleep
PSG, despite the branch name).

## Background

Dataset: Nie/Katyal/Engel (2023), 84 subjects, 34-channel EEG at 360 Hz,
preprocessed (CSD reference, ICA, artifact rejection). Button-press events
mark subjective perceptual switches; ground truth = switch times in seconds.

Published baseline (`scripts/batch_eeg.py` on `experiment/tda-sleep`, mean
across N=80 subjects, ±5s tolerance):
- precision = 0.941
- recall    = 0.406
- F1        ≈ 0.57

The recall gap is the dominant problem. Precision is already
publication-grade; the detector misses most behavioral switches.

## Score

Mean F1 across 3 subjects {3629, 3630, 3632} at ±5s tolerance. The grader
trusts the per-subject F1 values returned by the program (loading the .mat
files inside the grader process is prohibitively slow) but enforces:
- All 3 subjects present
- tolerance_s = 5.0 (matches published evaluation)
- precision / recall / F1 ∈ [0, 1]

## Seed result (verified by `coral validate`)

**mean F1 = 0.5577** (strategy: `ph_pi_l2_cusum`)

Per-subject F1:
- 3629 = 0.7576
- 3630 = 0.3750
- 3632 = 0.5405

The mean matches the published 0.57 baseline almost exactly. The large
inter-subject spread (0.38 – 0.76) hints at adaptive-per-subject strategies
being potentially high-value.

Validation runtime: ~27 minutes wallclock per attempt with default
SUBSAMPLE=300.

## Targets

- 0.57 = published baseline (matched by seed)
- 0.65 = improvement
- 0.70 = strong improvement (Branch C success)
- 0.80 = approaching the precision ceiling

## Cost management

Per-attempt wallclock is ~27 min with the default settings. The grader
timeout is set to 3600s to give headroom, but agents should drop
`TD_SUBSAMPLE` from 300 to 150–200 during exploration (runs drop to ~10–15
min) and restore it for their final attempt. This is documented in the
task.yaml tips block.

## Strategies suggested

- Bayesian online changepoint detection (BOCPD)
- Page-Hinkley test
- Adaptive per-subject CUSUM threshold
- Multi-channel input (Pz, P3, P4, O1, O2 in addition to Oz)
- Cross-channel binding (BindingDetector on Oz/Pz)
- Per-band PH (alpha-only, beta-only, theta-only)
- Hybrid spectral-topological: alpha power gate → PH on triggered windows
- Persistence landscape L1 instead of PI L2

## Data location

The rivalry EEG data (~12 GB across 85 subjects) lives at
`~/att-docs/data/eeg/rivalry_ssvep/` — gitignored. Three subject dirs used:
- `Sucharit - 012516_3629/`
- `Sucharit - 012716_3630/`
- `Sucharit - 012816_3632/`

Each has `Epochs/` (6 preprocessed .mat files) and `Behavior/` (button-press
.mat files). Path is hardcoded in `seed/initial_program.py`.

## File layout

```
coral_tasks/rivalry_switch_detection/
├── task.yaml                    # 4 agents, sonnet, 80 turns, timeout 3600
├── eval/grader.py              # mean F1 across 3 subjects, tolerance=5s
├── .gitignore                  # results/ .coral/
└── seed/
    ├── initial_program.py      # PI-L2 + CUSUM baseline, 3 subjects
    ├── pyproject.toml          # editable att-toolkit + deps
    └── .gitignore              # .venv/ __pycache__/
```

## Launch

```bash
cd ~/CORAL && uv run coral start \
    -c ~/att-docs/coral_tasks/rivalry_switch_detection/task.yaml
```

## Post-CORAL graduation

If an agent achieves mean F1 ≥ 0.70 across 3 subjects:

1. Lift the winning detector to a new method in
   `att/transitions/detector.py` (e.g., `TransitionDetector(mode="bocpd")`).
2. Re-run on the full 80-subject dataset using `scripts/batch_eeg.py` with
   the new method. Publish the improved precision/recall numbers.
3. Update the preprint's Experiment 7 section with the new results.
4. If the winning strategy is hybrid spectral-topological: document as
   empirical validation of architecture #5 from the hybrid document.
