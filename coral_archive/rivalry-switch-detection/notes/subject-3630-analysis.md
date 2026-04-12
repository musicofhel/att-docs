---
creator: agent-1
created: 2026-04-12T00:00:00
---
# Subject 3630 — Stuck at F1~0.47

## Attempt history
- Attempt 1 (08a8dae): per-channel alpha union, k=0.8 → 3630=0.474
- Attempt 2 (3992e6f): spatial-avg alpha+theta, k=0.8 → 3630=0.471

Spatial averaging did not help 3630. The problem is NOT excessive false positives
from multiple channels — it is a fundamental signal mismatch.

## Hypothesis: Alpha SUPPRESSION
The CUSUM only accumulates POSITIVE deviations from mean (power increases).
But binocular rivalry switches can cause alpha SUPPRESSION (power decrease)
in occipital cortex — equally well-documented. If 3630 suppresses alpha at
switches rather than boosting it, the detector is blind to those events.

## Next Fix: Bidirectional CUSUM
Run two CUSUM accumulators in parallel:
  - Forward (detects increases): cusum_pos += (p - mean)
  - Reverse (detects decreases): cusum_neg += (mean - p)
Fire when EITHER exceeds threshold.

This should immediately help 3630 if its switches manifest as alpha suppression.

## Also consider
- Beta band (13-30 Hz) CUSUM — rivalry also modulates beta
- Frontal channels (Fz, F3, F4) — attention/frontoparietal rivalry signals
- The TDA detector should detect topology changes bidirectionally, but its 
  recall is limited. The spectral signals need to also be bidirectional.
