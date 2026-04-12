---
creator: agent-1
created: 2026-04-12T06:20:00+00:00
---
# Reflection: Eval #6 — Why PC1+PC2 Inverted Binding Got F1=0.65 Not 0.984

## Summary
Eval #6 attempted hybrid: PC1+PC2 inverted binding for sr_drift (expected 0.984) + Crocker for noise/cd.
- Actual sr_drift F1: 0.6506 (not 0.984!)
- noise F1: 0.9876 (correct, Crocker works)
- connection_death F1: 0.9835 (correct, Crocker works)
- Mean F1: 0.8739

## Root Cause: STRIDE Mismatch
The KEY difference between my implementation and agent-2's ebe78d4 (sr_drift F1=0.9837):
- My code: STRIDE=200 → 46 windows, 25 baseline windows
- Agent-2: STRIDE=400 → 33 windows, 13 baseline windows

With STRIDE=200, the binding scores between overlapping windows (90% overlap) are noisier
in a way that inflates baseline variance → the drift signal can't clear the sigma threshold.
With STRIDE=400 (80% overlap), the binding signal transitions more cleanly from baseline.

This is surprising: finer stride HURTS inverted binding, but doesn't hurt Crocker (which
is monotone CUSUM → immune to per-window variance issues).

## Surprises
- PC1+PC2 with direct threshold gets 0.984 at STRIDE=400 but only 0.65 at STRIDE=200
- Stride matters fundamentally differently for BindingDetector vs Crocker CUSUM
- The monotone CUSUM accumulates signal over many windows → stride doesn't matter much
- BindingDetector per-window threshold → sensitive to baseline variance → stride matters a lot

## Confidence Assessment
95% confident that eval 7 (STRIDE=400 for binding, STRIDE=200 for Crocker) will give ~0.985.
Evidence: agent-2's ebe78d4 gets EXACTLY this split: 0.984 for sr_drift with STRIDE=400.

## Next After Eval 7
If hybrid achieves ~0.985, pivot to:
1. Can we push sr_drift binding F1 from 0.984 to 0.99? (tune SIGMA_ALARM down slightly?)
2. Can we add the one-sided CUSUM of binding drops for sr_drift while maintaining 0.984?
3. Final synthesis: write comprehensive skill capturing the full detector design
