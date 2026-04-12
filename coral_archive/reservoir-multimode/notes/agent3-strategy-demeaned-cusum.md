---
creator: agent-3
created: 2026-04-12T04:50:00+00:00
---
# Agent-3 Strategy: Demeaned CUSUM + H0+H1 Combined

## Baseline Results (eval #1)
Score: 0.9863 (mean F1). Per-mode: sr_drift=0.988, noise=0.9876, connection_death=0.9835.
connection_death is the weakest mode.

## Why F1 < 1.0
Standard image CUSUM grows monotonically 0→V in baseline. Threshold ~1.02*V.
After onset at window 25, diff slightly increases but CUSUM needs 1-2 windows to
cross threshold → windows 25-26 often missed (FN).

## Demeaned CUSUM (eval #2)
  demeaned[i] = cumsum(diff[i] - mu_baseline)
- During baseline: oscillates near 0 (random walk, not monotone)
- After onset: grows immediately at rate (drift_diff - mu_baseline)
- Threshold is ~k*sigma_rw (much lower than 1.02*V) → earlier alarm
- Safety fallback: if FP in baseline, use standard CUSUM instead

## H0 + H1 Combined (eval #2)
- H1: loops, best for sr_drift
- H0: connected components, best for connection_death (dying connections)
- OR logic: alarm if either exceeds threshold

## CRITICAL: Extra Seeds Timeout (agent-4's finding)
Agent-4 found 24 evals at 75s/pair = 1800s timeout. My code has 15 extra.
CLAUDE.md says 30s/eval, so 720s total should be OK. Watching.

## Agent-4's L1+CUSUM OR Alternative
Instantaneous column_L1 OR image_CUSUM. L1 catches window 25 directly.
Worth trying if H0+H1 demeaned does not push above 0.99.
