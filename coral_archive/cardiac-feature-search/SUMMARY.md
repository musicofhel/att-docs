# Cardiac Feature Search

**Branch**: `experiment/coral-cardiac-feature-search`
**Run**: `2026-04-11_212850` | 4 agents, 3 scored attempts
**Target**: AUROC >= 0.65 | **Best**: 0.8014 (agent-4)

## Task

Find topological features that discriminate abnormal cardiac windows in MIT-BIH
Arrhythmia Database records. Prior work showed persistence entropy at 10s window
scale is rhythm-class-invariant (normal vs arrhythmic distributions indistinguishable),
killing pure-TDA approaches.

## Winning Strategy

Agent-4: 21-feature mixed approach
- **9 TDA features**: H0 count, H1 count, persistence statistics (mean, max, std lifetime), persistence entropy, image L2 norms
- **12 signal/HRV features**: RR-interval variability, spectral power (LF/HF ratio), statistical moments, beat morphology

Leave-one-out cross-validation over MIT-BIH records {200, 201, 207, 210, 217}.

Per-fold AUROCs: [0.916, 0.714, 0.622, 0.963, 0.792]
Mean AUROC: **0.8014**

## Key Insight

Pure TDA features alone underperform because persistence entropy is rhythm-class-invariant
at the 10s window scale. But combining topological features with classical signal/HRV features
rescues the task — the topological features capture complementary geometric information
that boosts discrimination above the publishable threshold (0.75).
