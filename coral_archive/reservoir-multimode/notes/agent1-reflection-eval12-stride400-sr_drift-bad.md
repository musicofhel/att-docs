---
creator: agent-1
created: 2026-04-12T07:15:00+00:00
---
# Reflection: Eval #12 — Crocker STRIDE=400 cached for ALL modes: sr_drift HURTS

## Results
- sr_drift: 0.9756 (WORSE — was 0.9837 with inverted binding)
- noise: 0.9915 (same)
- connection_death: 1.0 (same)
- mean: 0.9890 (vs 0.9917 with hybrid)

## Root Cause: STRIDE=400 + Cached Takens BAD for sr_drift
With cached Takens from window 0 (healthy), the boundary window (start=4800, 90% drift)
shows a LARGE diff (healthy embedding sees drift topology clearly). This large diff inflates
the CUSUM baseline significantly. With only 13 baseline windows, the threshold is set high.

For noise/cd (abrupt degradation): drift-phase CUSUM grows RAPIDLY past the raised threshold.
For sr_drift (gradual degradation): drift-phase CUSUM grows SLOWLY → alarm fires later → FNs.

## Mode-Specific STRIDE Principle
- Abrupt failure modes (noise, cd): STRIDE=400 + cached Takens works GREAT
  - Abrupt topology change → fast CUSUM growth → easily crosses raised threshold
- Gradual failure modes (sr_drift): STRIDE=200 needed (25 baseline windows)
  - Gradual change → need more windows to accumulate signal → dilute boundary contamination

## Next: Crocker STRIDE=200 + cached Takens for sr_drift
Hypothesis: with 25 baseline windows, boundary contamination is diluted.
Cached Takens still makes drift appear more different than per-window auto-fit.
Expected sr_drift: 0.988-0.99 (beating current 0.9837 from binding)

If sr_drift = 0.988: mean = (0.988+0.9915+1.0)/3 = 0.9932 → NEW RECORD!

Timing: sr_drift STRIDE=200 (46 windows×3 seeds) ≈ 210s + noise/cd STRIDE=400 ≈ 90s → 300s total. Safe.
