---
creator: agent-4
created: 2026-04-12T04:45:00+00:00
---
# Agent-4 Reflection: Timeout + L1+CUSUM Strategy

## Timeout Issue (Eval #1)
Running extra seeds {3-7} on top of 9 required pairs = 24 total.
At ~75s/pair, 24x75 = 1800s → hits grader timeout exactly.
RULE: Only 9 required pairs in main eval. Extra seeds timed out agent-1 too.

## Baseline (agents 2 and 3)
F1=0.9863 with H1 image-CUSUM (SIGMA_ALARM=2.0, SUBSAMPLE=250).
Per-mode: sr_drift=0.988, noise=0.988, connection_death=0.984.

## Why F1 < 1.0
Window 25 (start=5000) is first drift window. CUSUM threshold ~26*d0.
CUSUM[25] ~25*d0 + tiny drift -> often below threshold -> 1 FN per pair.
Explains F1=0.988. connection_death: 2 FNs in some seeds -> 0.984.

## Strategy: L1+CUSUM OR
- L1 = instantaneous diff (catches single-window spikes)
- CUSUM = cumsum(diff) (robust to sustained drift, lags at onset)
- OR logic: alarm if either exceeds 2-sigma
- L1 catches window 25 while CUSUM misses -> reduces FN, no added FP risk
- Different from agent-2 H0+H1 (same CUSUM type, different dim)

Expected: connection_death 0.984->0.990+, mean F1 > 0.990


## UPDATE: L1+CUSUM OR is WORSE (Eval #2 result)
Score 0.9757 vs baseline 0.9863. L1 introduces FPs in healthy windows. 
Agent-1 independently tried same approach with same result (0.9757).
LESSON: L1 signal is too noisy. CUSUM accumulation is essential for stability.

## Root cause of ~1.4% error (revised understanding)
Healthy diffs are NOT zero - they have nonzero variance due to:
  - SUBSAMPLE=250 uses different random seeds per window (seed+w_idx)
  - Different 250 points sampled per window -> stochastic PH variation
  - This noise makes baseline threshold ~26*d0 (not zero)
  - First drift window (10% avg drift) has diff[25] just barely above healthy avg
  - CUSUM[25] ~ 25*d0 + small → stays below threshold ~26*d0

## Proposed fix for Eval #3: Fixed PH subsample seed
Instead of seed=seed+w_idx (different 250 points per window), use seed=42 (same 250 points).
- Healthy windows: same 250 relative positions → more consistent PH → lower variance
- Lower variance → lower threshold → catches window 25 with same diff
- Drift windows: cloud changes shape → even same 250 points show different PH → alarm

Expected: threshold drops, F1 improves from 0.986 to 0.990+
