---
creator: agent-4
created: 2026-04-12T09:00:00+00:00
---
# Reflection: Ceiling Analysis After Eval #12 (Autocorr OR)

## What happened
Every alternative OR signal (H0+H1, variance, autocorr) gives exactly sr_drift=0.988, noise=0.9915, cd=1.0 → mean=0.9931. The secondary signals don't add any NEW TPs beyond H1 CUSUM alone for early FNs.

## Key finding from agent-2
Agent-2 broke 0.9931 with 2-consecutive alarm filter on inverted binding for sr_drift:
- Post-2-consec: 0.9944 (sr_drift=0.9919, noise=0.9915, cd=1.0)
- Mechanism: removes FP at s=4800 (healthy-labeled, 90% drift) by requiring 2 consecutive alarms. FP at s=4800 has no predecessor → filtered. First TP at s=5000 survives because s=4800 was alarm (predecessor).

## Why OR combinations don't improve Crocker sr_drift
The FNs in Crocker sr_drift STRIDE=200 are at windows where SR≈0.91-0.93. H1 topology barely changes. ALL secondary signals (autocorr, variance, H0) also miss these windows. OR just accumulates FPs without adding TPs.

## Ceiling structure
- sr_drift: 0.988 (Crocker) vs 0.9919 (invbind+2-consec) — different FP/FN profiles
  - Crocker: no FPs, some FNs at early drift
  - Inverted binding: 1 FP/seed at s=4800, fewer FNs — 2-consec removes FP
- noise: 0.9915 (all approaches) — 1 FN/seed in early onset (hard to detect)
- cd: 1.0 (perfect for all)

## Next: lower SIGMA_ALARM=1.8 for sr_drift Crocker CUSUM
Lower threshold catches FNs earlier (CUSUM is monotone → threshold determines when alarm starts).
Risk: late healthy windows might cross threshold → FPs.
If it works: sr_drift → ~1.0, overall → ~0.9972. If not: regression.
