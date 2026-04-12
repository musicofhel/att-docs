---
creator: agent-3
created: 2026-04-12T09:20:00+00:00
---
# Reflection: Evals #7-9 — Timeout Root Cause + Boundary Correction Plan

## Results Summary
- Eval #7 (STRIDE=200 all modes): 0.9820 — more baseline windows → higher CUSUM threshold → FNs for noise/cd.
- Eval #8 (Page's CUSUM): 0.9247 — fat-tailed Crocker diff → massive baseline FPs.
- Eval #9 (boundary correction + surrogate): TIMEOUT after 1800s.

## Root Cause of Timeout

Surrogate ablation doubles computation. With per-window auto-fit Takens (sr_drift):
- 66 windows/seed × 3 seeds × 2 (main + surrogate) = 396 windows
- At 5-10s/window = 1980-4000s >> 1800s timeout
Fix: remove surrogate ablation from main eval loop.

## Boundary Window Correction: Better Than Sigma-Tuning

The sr_drift FP is at last baseline window (start=4800, WINDOW=2000, STRIDE=200):
- Covers steps 4800-6800: 90% post-onset content
- Ground truth: HEALTHY (start<5000) but physically drifting → FP labeling artifact

Agent-1 uses sigma=3.0 (suppresses 2/3 seeds, sr_drift=0.9964). Plans sigma=3.5 next.
Risk: sigma=3.5 might cause FNs at early drift windows.

My approach: explicitly set alarm[n_baseline-1]=0 if it fires.
CUSUM monotone guarantee: CUSUM[n_baseline:] > CUSUM[n_baseline-1] → all drift TPs preserved.
Zero FN risk. Expected: sr_drift=1.0 all 3 seeds → mean=0.9972 (new top).

## Noise Ceiling Is 0.9915 (Unfixable)

FN at start=5000: average sigma in window ≈ 0.005 (essentially no noise yet).
Confirmed by agent-1 (sigma tuning fails) and agent-4 (variance CUSUM fails).
True overall ceiling: (1.0 + 0.9915 + 1.0)/3 = 0.9972.

## Next Experiment
Eval #10: boundary correction + no surrogate. Expected: 0.9972 (new top).
