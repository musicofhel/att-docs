---
creator: agent-4
created: 2026-04-12T02:19:25.314628
---
# Reflection: Eval #4 — Hybrid 0.9764 (inverted binding STRIDE=200 → wrong)

## Concrete Results
- Score: 0.9764 (uses_inverted_binding=true)
- sr_drift: F1=0.9671 (inverted binding, STRIDE=200)
- noise: F1=0.9746 (Crocker CUSUM)
- connection_death: F1=0.9876 (Crocker CUSUM)

## Key Surprise: STRIDE=200 HURTS binding (same finding as agent-1 Eval #6)
Agent-1 independently confirmed: STRIDE=200 for binding gives sr_drift F1=0.65-0.97,
while STRIDE=400 gives F1=0.9837. My STRIDE=200 got 0.9671 — same mechanism:

With 90% overlap (STRIDE=200), binding scores between neighboring windows are highly
autocorrelated. This inflates the ESTIMATED baseline std (consecutive correlated samples
look like smooth trend, not independent draws). Result: sigma underestimated relative to
true signal variance → threshold too tight → false positives OR threshold too loose → FNs.

With 80% overlap (STRIDE=400, 13 baseline windows), binding scores are more independent.

## Why Crocker is STRIDE-Insensitive
Crocker uses one-sided CUSUM: cumsum of absolute differences from baseline column.
CUSUM is monotonically non-decreasing → immune to per-window FPs.
Binding uses direct threshold → sensitive to baseline variance estimation.

## Analysis of Noise Regression
Noise dropped from 0.988 → 0.9746. Likely cause: surrogate ablation adds ~30s per pair
for noise, pushing some pairs to the timing boundary. Or there may be RNG interaction.
(Surrogate uses same PH seed=42 as Crocker → no conflict, but baseline/drift split timing matters.)

## Plan for Eval #5: OR(binding_CUSUM, Crocker) for sr_drift + STRIDE=400
1. Fix BIND_STRIDE = 400 (match agent-1 and agent-2 findings)
2. Add CUSUM on binding drops: monotone → no FPs in baseline
3. For sr_drift: alarm if binding_CUSUM OR Crocker_CUSUM fires
   - Both are monotone → union can only add TPs, not FPs
   - Expected: sr_drift F1 → 0.985+
4. Keep Crocker for noise/cd (already 0.988/0.984)
5. REMOVE surrogate ablation from noise/cd to save time (already confirmed, keep only sr_drift surrogate)

Expected: mean F1 ≈ (0.988 + 0.9876 + 0.9835) / 3 ≈ 0.986+ with uses_inverted_binding=true
