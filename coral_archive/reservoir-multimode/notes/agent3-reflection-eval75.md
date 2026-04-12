---
creator: agent-3
created: 2026-04-12T10:00:00+00:00
---
# Reflection: Eval 75 - Noise FN Analysis

Score: 0.9972 (sr_drift=1.0, noise=0.9915, cd=1.0) — boundary correction for all modes had no effect.

## Key finding
noise=0.9915 is caused by FNs (missed drift alarms), NOT FPs.
Extending boundary correction to noise/cd was a no-op → confirms no FP at boundary for noise.

## Analysis
3 noise seeds. If one has F1=0.9746 (TP=19, FN=1), two have F1=1.0:
avg = (1.0+1.0+0.9746)/3 = 0.9915 matches observed.
One seed misses one early drift window.

## Next
Try sigma=1.5 for noise specifically. This lowers the alarm threshold for noise mode,
catching early-onset drift windows that sigma=2.0 misses.
Risk: could introduce FPs in noise baseline for other seeds.
