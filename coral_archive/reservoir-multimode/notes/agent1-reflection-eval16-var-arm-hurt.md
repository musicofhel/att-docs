---
creator: agent-1
created: 2026-04-12T08:45:00+00:00
---
# Eval 16 Reflection: Variance OR arm HURT noise (0.9915→0.9837)

## What happened
OR(Crocker H1 CUSUM, direct mean-neuron-std alarm) for noise caused regression.
noise: 0.9915 → 0.9837. Mean: 0.9931 → 0.9906.

## Why the variance arm added FPs
Reservoir state variance (mean neuron std per window) is NOT stable across healthy windows.
Natural variance fluctuations in the healthy regime exceeded the 2σ threshold in some windows.
The direct alarm fired on normal healthy variation → FPs → precision drop → F1 drop.

The Crocker CUSUM specifically avoids this problem because CUSUM smooths per-window noise
via accumulation. A direct (non-CUSUM) alarm on any feature is vulnerable to the high
natural variability of per-window statistics.

## Key lesson
DO NOT use direct (non-CUSUM) per-window alarms for the OR arm in noise.
Any OR arm must itself be monotone non-decreasing (like CUSUM) to be safe.

Alternative OR arms that are safe:
- CUSUM of the variance signal (not direct alarm) — accumulates; monotone
- CUSUM of any other feature (spectral, entropy, etc.)

## Current experiment: sr_drift sigma=2.5
Hypothesis: the boundary FP at start=4800 (90% drift content, labeled HEALTHY)
has CUSUM between 2.0σ and 2.5σ. Raising threshold suppresses it.
Expected: sr_drift 0.988 → 0.992+
Risk: early pure drift windows at start=5000, 5200 might also be suppressed → FNs

## Status of all noise improvement attempts
| Approach | Result |
|---|---|
| sigma=1.9 | No change (FN CUSUM below 14.10d) |
| STRIDE=200+cached | 0.9748 (contamination) |
| STRIDE=400+auto-fit | 0.9739 (adaptive hiding) |
| H0+H1 OR | No change (H0 doesn't add TPs) |
| Variance OR direct | HURT: 0.9837 (FPs in baseline) |

Conclusion: noise FN at start=5200 for the hard seed appears to be fundamentally
undetectable by Crocker topology in the first pure drift window. The CUSUM at that
point is below 14.10d (sigma=1.9 threshold). Need a genuinely different approach or
accept 0.9931 as ceiling.
