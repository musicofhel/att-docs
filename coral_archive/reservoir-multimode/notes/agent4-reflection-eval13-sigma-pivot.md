---
creator: agent-4
created: 2026-04-12T09:30:00+00:00
---
# Reflection: sigma=1.8 Regressed → Pivot to WINDOW=1000

## Result: sigma=1.8 → 0.9918 (regression from 0.9931)
Lowering sigma introduced FP at s=4800 boundary window (WINDOW=2000, 90% drift).
CUSUM at s=4800 is between 1.8σ and 2.0σ. Confirmed mental model: sigma tuning
with WINDOW=2000 can only INCREASE sigma to suppress FPs, not decrease to catch FNs.

## KEY PIVOT: WINDOW=1000 (agent-1's insight confirmed at 0.9960)
Agent-1 proved WINDOW=1000 + sigma=3.0 gives sr_drift=0.9964 (vs my 0.988 at W=2000):
- Smaller window → boundary window at s=4800 has 80% drift (vs 90% at W=2000)
- Smaller window → each early-drift window (s=5000, 5200) is a 1000-step pure drift slice
- Higher sigma=3.0 suppresses boundary FP (CUSUM below 3.0σ)
- Genuine drift windows CUSUM > 3.0σ → still alarm

Agent-1 is now stuck at 0.9960 even with sigma=3.5 (same score). The ceiling for pure
H1 Crocker with W=1000 seems to be 0.9960.

## My next experiment: W=1000 + autocorr OR (running)
Adding autocorr arm to W=1000 Crocker. Both are CUSUM → safe OR.
Physical: SR↑→memory↑→autocorr↑, should be detectable even at SR=0.91.
If autocorr catches the remaining FN (1 in 1 seed): sr_drift→1.0, mean→0.997+
If not: same 0.9960 or worse (if autocorr arm adds FPs despite sigma=3.0)

## Theoretical analysis of remaining FN
With W=1000, STRIDE=200, 46 drift windows per seed:
- 2/3 seeds perfect at sigma=3.0 → F1=1.0
- 1/3 seed has 1 FN: F1≈90/91≈0.989
- Mean sr_drift = 272/273 ≈ 0.9964

The FN must be a drift window where CUSUM < 3.0σ (barely changed from baseline).
This is likely window s=5000 or 5200 (first 1-2 drift steps at SR≈0.90-0.92).
Any additional signal that's MORE sensitive at SR=0.91 could catch this FN.
