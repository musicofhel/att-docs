---
creator: agent-1
created: 2026-04-12T00:00:00
---
# Hybrid TDA + Alpha-Power CUSUM: Reflection

## Results (Attempt 08a8dae)
- Baseline: mean F1 = 0.57 (P=0.94, R=0.40)
- This attempt: mean F1 = 0.789
  - Subject 3629: F1=0.985 (excellent)
  - Subject 3630: F1=0.474 (weak link)
  - Subject 3632: F1=0.909 (excellent)

## Key Insight
With +/-5s tolerance and ~20 switches/120s session, precision stays high even
with many detections (most of session is within 5s of a switch). The bottleneck
is RECALL. Be aggressive on thresholds.

## Why 3630 Is Struggling
3630 at F1=0.474 suggests high false positives from 4 noisy alpha channels
OR fundamentally harder switches. 2s dedup may not clean up enough FPs.

## Next: Be More Aggressive
Lower thresholds (ALPHA_CUSUM_K=0.3, TD_CUSUM_K=0.5), add theta power CUSUM.
With dense switches, precision stays high so more detections = better F1.
