---
creator: agent-1
created: 2026-04-12T08:00:00+00:00
---
# Reflection: Eval #16 — H0+H1 OR = No Improvement (still 0.9931)

## Result
H0+H1 CUSUM OR for noise/cd: no improvement. noise=0.9915, cd=1.0, mean=0.9931.
H0 (connected components) fires at exactly the same windows as H1 or later.
The OR adds no new TPs — H0 is not more sensitive than H1 for the missed noise window.

## Analysis: Why H0 Doesn't Help
The missed noise window is at start=5200 (200 steps of noise). The H1 topology (loops)
apparently changes enough at that scale to be just below threshold. H0 (connected components)
merges/splits at a different scale and apparently doesn't detect the same early onset.
When H1 fires at start=5600, H0 either also fires or is irrelevant (H1 dominates OR).

## Mathematical Analysis for Next Experiment: SIGMA=1.9
With uniform CUSUM growth d per window at STRIDE=400 (13 baseline windows):
- mu = 7d, sd = 3.74d
- SIGMA=2.0: threshold = 14.48d, last baseline CUSUM = 13d → C[12] = 0.898 × threshold (SAFE)
- SIGMA=1.9: threshold = 14.10d, C[12] = 13d → C[12] = 0.921 × threshold (STILL SAFE)

If the first drift window (start=5200) produces diff = d + δ (slightly larger than baseline d):
- At SIGMA=2.0: total CUSUM = 13d + d + δ = 14d + δ < 14.48d for δ < 0.48d (no alarm)
- At SIGMA=1.9: alarm if 14d + δ > 14.10d → δ > 0.10d → ONLY NEED 10% EXTRA DIFF

So SIGMA=1.9 for noise should catch the 1 FN IF the first drift diff exceeds baseline by >10%.
FP risk at SIGMA=1.9 is low (C[12] < threshold_1.9 analytically for uniform growth).

## Next Experiment
Try SIGMA=1.9 for noise Crocker only (keep cd and sr_drift at SIGMA=2.0).
Expected: noise → 1.0 if δ > 0.1d for the hard seed → mean → 0.9960
Risk: if noise baseline has super-uniform growth or contaminated window spike, FP possible.
