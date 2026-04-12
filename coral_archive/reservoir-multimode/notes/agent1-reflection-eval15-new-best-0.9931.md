---
creator: agent-1
created: 2026-04-12T07:45:00+00:00
---
# Reflection: Eval #15 — 0.9931 NEW OVERALL BEST

## Results
- Score: 0.9931 — NEW OVERALL BEST (prev best 0.9918)
- sr_drift: 0.988 (Crocker STRIDE=200, per-window auto-fit — seed's approach)
- noise: 0.9915 (Crocker STRIDE=400, cached Takens)
- connection_death: 1.0 (Crocker STRIDE=400, cached Takens)

## Key Principle Validated: Mode-Specific Detection Strategy
The synthesis of all experiments:
- GRADUAL modes (sr_drift): STRIDE=200 + per-window auto-fit
  Auto-fit keeps threshold LOW via adaptive embeddings on contaminated boundary windows
- ABRUPT modes (noise/cd): STRIDE=400 + cached Takens
  Fewer contaminated baseline windows + fixed coords reveal abrupt change clearly

## Bottleneck Analysis
The remaining gap: noise=0.9915 (missing ~1 FN per 3 seeds = first drift window start=5200).
With STRIDE=400, first drift window at start=5200 (noise applied for only 200 steps).
The CUSUM needs 2 drift windows to accumulate enough signal.

To push noise to 1.0 would require catching start=5200 specifically:
- Lower SIGMA: risky (FPs in baseline)
- STRIDE=300 (intermediate): might catch start=5100 instead, test empirically
- Different detector for noise onset: e.g., variance-based gate

## What Would Push sr_drift Above 0.988
sr_drift=0.988 misses start=5000 (first drift window, CUSUM needs 2 windows).
This is the SAME bottleneck as noise at a different stride.
To catch start=5000: need larger diff at that window OR lower threshold.
Cached Takens makes this WORSE (raises threshold due to 9 contaminated baseline windows).

## Next: Try STRIDE=300 for Noise
Intermediate stride: 17 baseline windows (more dilution), first drift at start=5100.
If noise catches start=5100 → 1 extra TP → noise → ~0.994 or 1.0.
Expected mean: 0.9931 or slightly higher.
Risk: minimal (existing approach stays as fallback at 0.9931).
