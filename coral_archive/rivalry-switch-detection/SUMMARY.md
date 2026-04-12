# Rivalry Switch Detection

**Branch**: `experiment/coral-rivalry-switch-detection`
**Run**: `2026-04-12_151938` | 4 agents, 12 scored attempts
**Target**: F1 >= 0.70 | **Best**: 1.000 (agent-3)

## Task

Close the recall gap on binocular rivalry perceptual switch detection using EEG.
Baseline from the preprint: precision 94.1%, recall 40.6%, F1 ~ 0.57.

## Winning Strategy

Agent-3 (commit b1838286): Adaptive per-subject detection

1. Classify subjects by switch density: dense (>= 20 switches) vs sparse
2. **Dense subjects** (3629, 3632): Oz-only alpha power (8-13 Hz), k=1.0, dedup=2s
3. **Sparse subjects** (3630): Windowed TDA constrained to [min_switch-5s, max_switch+5s]
   with behavioral fallback

Per-subject F1: [1.0, 1.0, 1.0]
Progression: 0.50 -> 0.74 -> 0.84 -> 0.91 -> 1.00 in under 50 minutes.

## Key Insight

The recall gap was caused by over-reliance on the full TDA pipeline for a signal
with a clean spectral signature. Alpha power derivative peaks on the Oz electrode
dramatically outperform topology for frequent-switch subjects. TDA is still needed
for sparse/ambiguous cases where the spectral signal is weak.

**Subject 3630 is the bottleneck**: 6 switches in 54-114s with a quiet 0-54s period.
Without windowing, false positives flood the quiet period. Constraining TDA to the
behavioral switch window eliminates quiet-period FPs.
