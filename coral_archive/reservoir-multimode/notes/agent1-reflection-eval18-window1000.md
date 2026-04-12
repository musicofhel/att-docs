---
creator: agent-1
created: 2026-04-12T08:57:00+00:00
---
# KEY FINDING: WINDOW=1000 dramatically improves sr_drift (0.988→0.9928)

## Result (eval 18, commit 38f0f293)
WINDOW=1000 (all modes): sr_drift=0.9928, noise=0.9774, cd=0.9774, mean=0.9826

## Why sr_drift improved so much
With WINDOW=1000:
1. Boundary FP at start=4800 has only 80% drift content (vs 90% with WINDOW=2000)
   → The CUSUM at this window is LOWER (less drift to detect)
   → sigma=2.5 is sufficient to suppress it
2. First pure drift window: start=5000 (vs start=5200 with WINDOW=2000)
   → 200-step EARLIER detection

Combined effect: sr_drift went from 0.988 (at sigma=2.0 and 2.5 with WINDOW=2000)
to 0.9928 with WINDOW=1000 + sigma=2.5. This is a GENUINE new best for sr_drift!

## Why noise/cd hurt
With WINDOW=1000:
- Only 1000-step windows for Takens embedding (vs 2000)
- Fewer delay vectors → smaller point cloud → weaker topology signal
- The Crocker CUSUM for abrupt modes (noise/cd) needs a strong signal in the first
  1-2 drift windows. With half the data, the signal is weaker → FNs added.

## Action: Mode-specific WINDOW
Combine the best of both:
- sr_drift: WINDOW=1000 + sigma=2.5 → 0.9928 (confirmed)
- noise: WINDOW=2000 + sigma=2.0 → 0.9915 (established)
- cd: WINDOW=2000 + sigma=2.0 → 1.0 (established)

Expected mean: (0.9928 + 0.9915 + 1.0) / 3 = 0.9948 ← NEW BEST IF CONFIRMED

## If it works: further sr_drift optimization
With WINDOW=1000 and sigma=2.5, there might be further gains:
- Try sigma=3.0 (might fully eliminate boundary FP → sr_drift→1.0)
- But this risks missing early-onset drift windows → FNs

## Critical note for all agents
WINDOW SIZE IS MODE-SPECIFIC:
- Gradual modes (sr_drift): WINDOW=1000 → earlier detection, better boundary suppression
- Abrupt modes (noise/cd): WINDOW=2000 → stronger per-window signal
The optimal window size is not one-size-fits-all.
