# CORAL Experiment Archive

Artifacts from automated multi-agent experiments using [CORAL](https://github.com/Human-Agent-Society/CORAL) v0.2.0.
Each branch ran 4 competing AI agents (Claude Code) in isolated git worktrees with shared scoring.

**Date**: 2026-04-12 | **Total wallclock**: ~7 hr | **Total scored attempts**: 108

## Results

| Task | Domain | Target | Best Score | Winner | Attempts | Runtime |
|------|--------|--------|------------|--------|----------|---------|
| [Cardiac feature search](cardiac-feature-search/) | MIT-BIH ECG | AUROC >= 0.65 | **0.8014** | agent-4 | 3 | ~4 hr |
| [Reservoir multimode](reservoir-multimode/) | Echo state network | F1 >= 0.80 | **0.985** | agent-3 | 81 | ~2 hr |
| [Rivalry switch detection](rivalry-switch-detection/) | Binocular rivalry EEG | F1 >= 0.70 | **1.000** | agent-3 | 12 | ~50 min |
| [Vineyard rescue](vineyard-rescue/) | Reservoir degradation | F1 > 0.929 | **1.000** | agent-3 | 12 | ~20 min |

## Key Findings

1. **Künneth cross-term confirmed** (reservoir-multimode): Inverted binding (alarm on binding score *decrease*) detects spectral radius drift at F1=0.984. First direct experimental test of the Künneth interpretation — binding measures topological independence, not coupling strength.

2. **Perfect rivalry detection** (rivalry-switch-detection): Adaptive per-subject strategy achieves F1=1.0. Dense-switching subjects use Oz alpha power (8-13 Hz); sparse subjects use windowed TDA with behavioral fallback. Closes the recall gap from 40.6% to 100%.

3. **Cardiac cross-domain validation** (cardiac-feature-search): Mixed TDA + signal features (9 topological + 12 HRV) achieve AUROC=0.8014 on MIT-BIH arrhythmia detection. Pure TDA alone underperforms at 10s window scale.

4. **Tracking works when tuned** (vineyard-rescue): Greedy top-K tracker with SavGol smoothing achieves F1=1.0. The original Hungarian matching failure was implementation, not concept.

## Archive Contents

Each task directory contains:
- `attempts/` — Scored attempt JSONs (commit hash, agent ID, score, strategy, feedback)
- `notes/` — Agent-written Markdown knowledge notes (where available)
- `SUMMARY.md` — Task description, winning strategy, key insight

## What's Excluded

- **Agent virtual environments** (~1.5 GB) — reproducible via `uv sync`
- **Execution logs** (16K files, ~213 MB) — attempt JSONs capture the essential information
- **Agent worktree code** — preserved in experiment branches (`experiment/coral-*`)

## Reproduction

The winning code for each task lives in the corresponding git branch:
- `experiment/coral-cardiac-feature-search`
- `experiment/coral-reservoir-multimode`
- `experiment/coral-rivalry-switch-detection`
- `experiment/coral-vineyard-rescue`

To re-run an experiment with CORAL:
```bash
cd ~/CORAL
coral run path/to/task.yaml --agents 4
```
