---
creator: agent-1
created: 2026-04-12T06:30:00+00:00
---
# Reflection: Eval #7 — Hybrid STRIDE Fix → 0.9849 (uses_inverted_binding=true)

## Results
- Score: 0.9849 (uses_inverted_binding=true)
- sr_drift: F1=0.9837 (inverted binding, STRIDE=400, PC1+PC2)
- noise: F1=0.9876 (Crocker, STRIDE=200)
- connection_death: F1=0.9835 (Crocker, STRIDE=200)

## Key Improvement Over Eval #6
STRIDE=400 (not 200) was the critical fix for inverted binding. With STRIDE=200, binding
scores in overlapping windows inflate baseline variance → drift signal can't clear 2σ → F1=0.65.
With STRIDE=400 (80% overlap), binding transitions cleanly → F1=0.9837.

## Gap Analysis: 0.9849 vs 0.9863 (baseline)
Only sr_drift is below Crocker's performance (0.9837 vs ~0.9880 for Crocker).
Noise and connection_death are identical (both use Crocker).

## Next: OR(inverted_binding, Crocker) for sr_drift
Since Crocker is monotonically non-decreasing → zero false positives in baseline.
Adding Crocker as a second arm for sr_drift (OR with binding) can only add TPs, not FPs.
If Crocker catches the 1-2 missed early drift windows that binding misses, sr_drift rises to ~0.988.
Expected mean: (0.988 + 0.9876 + 0.9835) / 3 ≈ 0.9864 — beats baseline!

## Confidence
90% confident OR(binding, Crocker) for sr_drift beats 0.9863. 
Risk: if binding adds FPs (SIGMA=2.0 should prevent this), or if Crocker and binding
catch EXACTLY the same windows (mean stays at 0.9849).
