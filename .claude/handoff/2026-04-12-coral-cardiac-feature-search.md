# CORAL: cardiac feature search — Handoff

**Date**: 2026-04-12
**Branch**: `experiment/coral-cardiac-feature-search`
**Task dir**: `coral_tasks/cardiac_feature_search/`

## Goal

Search for topological (or signal-level) features that discriminate abnormal
vs normal cardiac windows in MIT-BIH ECG data, recovering the architecture
killed in the prior tracer round (Branch 1, "tiered sentinel cardiac",
F1=0.20, killed because persistence entropy at 10s windows is rhythm-class
invariant).

## Score

Mean AUROC across leave-one-record-out CV over MIT-BIH records
{200, 201, 207, 210, 217}. The grader holds the labels (computed from .atr
beat annotations: window labeled abnormal if ≥50% of beats are non-AAMI-N).
Each fold trains a fixed-seed logistic regression on 4 records, tests on the
5th. Score = mean AUROC across 5 folds.

## Anti-cheat

Agents only see the ECG signal, never the annotations. They must tag features
as `topo` or `signal` so the leaderboard records the mix of the winning
solution.

## Seed result (verified by `coral validate`)

  mean AUROC = **0.5712**
  fold AUROCs = [0.7496, 0.5677, 0.4728, 0.3616, 0.7046]
  feature: `persistence_entropy_h1` (1 feature, tagged `topo`)
  window/step: 3600/1800 samples (10s/5s @ 360Hz)

The wide fold variance (0.36 → 0.75) suggests there is signal in some records
but not others — perfect setup for CORAL agents to find a more discriminating
feature.

## Targets

- 0.50 = random / baseline
- 0.65 = architecture rescued (graduation)
- 0.75 = publishable

## File layout

```
coral_tasks/cardiac_feature_search/
├── task.yaml                    # CORAL config (4 agents, sonnet, 80 turns)
├── eval/grader.py              # LOO-CV AUROC, label loading via subprocess
├── .gitignore                  # results/ .coral/
└── seed/
    ├── initial_program.py      # baseline: persistence_entropy_h1 only
    ├── pyproject.toml          # editable att-toolkit + ripser/persim/wfdb
    ├── .gitignore              # .venv/ __pycache__/
    └── data/mitdb/             # 5 records (200, 201, 207, 210, 217), 9.4 MB
```

## Launch

```bash
cd ~/CORAL && uv run coral start \
    -c ~/att-docs/coral_tasks/cardiac_feature_search/task.yaml
```

## Post-CORAL graduation

If a winning attempt achieves AUROC ≥ 0.65:
1. Inspect `~/CORAL/results/cardiac_feature_search/latest/leaderboard.json`
2. Locate the winning agent's `initial_program.py`
3. Tag each feature as `topo` or `signal` — the composition matters
4. Lift the winning extractor to `att/neuro/cardiac.py`
5. Run on full MIT-BIH (48 records) with LOO-CV for confidence intervals
6. Add to preprint as a cardiac application section

If the winning solution contains zero PH features, that's an informative
final kill for cardiac topology at this window scale — document it as a
negative result and close the direction.
