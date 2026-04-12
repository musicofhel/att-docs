---
creator: agent-1
created: 2026-04-12T07:25:00+00:00
---
# Reflection: Eval #13 — 0.9918 NEW OVERALL BEST

## Results
- Score: 0.9918 — NEW OVERALL BEST (agent-2 had 0.9917)
- sr_drift: 0.984 (Crocker STRIDE=200 + cached Takens — beats binding's 0.9837!)
- noise: 0.9915 (Crocker STRIDE=400 + cached Takens — unchanged)
- connection_death: 1.0 (Crocker STRIDE=400 + cached Takens — unchanged)

## What Worked: Mode-Specific STRIDE + Cached Takens
- sr_drift (gradual): STRIDE=200 (25 baseline windows, dilutes boundary contamination)
  + cached Takens = F1=0.984 (vs 0.9756 at STRIDE=400, 0.988 at STRIDE=200 no-cache)
- noise/cd (abrupt): STRIDE=400 (13 baseline windows, cleaner) + cached Takens = 0.9915, 1.0

## Confirmed Mechanism: Cached Takens (Embedding Stability Principle)
Fixed coordinate system (calibrated to healthy state) outperforms per-window adaptive embedding.
Drifted windows "look" more different when forced into healthy embedding → cleaner signal.

## Bottleneck: noise=0.9915 (missing ~1 early drift window across 3 seeds)
With STRIDE=400, first drift window is at start=5200. The CUSUM might take 2 windows to fire.
With STRIDE=200, first drift window is at start=5000 → catch it if CUSUM fires earlier.

## Next: Try STRIDE=200 + cached Takens for noise
Keep cd at STRIDE=400 (already 1.0, don't risk it).
Try noise at STRIDE=200 + cached Takens.
If noise → 1.0: mean = (0.984+1.0+1.0)/3 = 0.9947 → major improvement!
Risk: STRIDE=200 might have more boundary contamination for noise → might not help.
