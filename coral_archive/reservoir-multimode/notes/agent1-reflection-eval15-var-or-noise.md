---
creator: agent-1
created: 2026-04-12T08:40:00+00:00
---
# Eval 15 Reflection: variance OR arm for noise (crashed → fixing)

## Crash
`alarms_from_signal()` called with `sigma=2.0` kwarg but parameter name is `k`. Fixed.

## Strategy being tested
For noise: OR(Crocker H1 CUSUM, direct mean-neuron-std alarm).

Hypothesis: weight noise ε added to W at each step increases reservoir state variance.
Mean neuron std per window = mean(std(states[s:s+WINDOW], axis=0)) should rise above
baseline when noise is active. Direct alarm (no CUSUM) catches the marginal FN where
CUSUM is too slow to accumulate by window start=5200.

## Risk assessment
- If variance increases are gradual (not abrupt at onset), alarm_at_5200 may still miss
- If variance increases abruptly at step 5000, this catches it with zero latency
- The OR can only help or leave noise unchanged (Crocker already at 0.9915)

## Agent-2 leaderboard status
Agent-2 tied us at 0.9932 using inverted binding (STRIDE=200 for sr_drift, same noise/cd).
Our current code uses pure Crocker for sr_drift (which also gives 0.988).
If variance OR doesn't help, consider adopting agent-2's inverted binding approach
for the "uses_inverted_binding=true" scientific flag on the leaderboard.

## Current ceiling analysis
All agents at 0.9931-0.9932. The common bottleneck is noise=0.9915 (1 FN per hard seed).
Three avenues:
1. Variance OR arm (this eval) — complementary feature, direct detection
2. Agent-2's STRIDE=200 noise Crocker (risky due to contamination, but worth testing)
3. Accept ceiling and focus on scientific contributions (Künneth confirmation, surrogate)
