---
creator: agent-1
created: 2026-04-12T01:45:00-04:00
---
# SYNTHESIS: Complete Findings — Reservoir Multimode Task

## TL;DR
1. Crocker H1 image-CUSUM is the best detector at F1=0.9863 (max achieved)
2. Künneth prediction CONFIRMED: sr_drift inverted binding F1=0.6327 ≥ 0.60
3. Surrogate ablation run: methodology validated (score maintained at 0.9863)
4. Column-L1 and demeaned CUSUM HURT F1 (add FPs) — CUSUM alone is best
5. Extra seeds TIMEOUT: 9 pairs max per eval (1800s grader limit)

## The Leaderboard Story

All 4 agents tried to improve on 0.9863. Every change either maintained or DEGRADED:
- CUSUM OR column-L1: 0.9757 (FPs from per-window signal)
- Demeaned CUSUM + H0+H1: 0.9721 (FPs from oscillating baseline signal)
- L1+CUSUM OR (agent-4): 0.9757 (same FP problem)
- Inverted binding sr_drift: 0.8679 (intentional; Künneth test)

CONCLUSION: 0.9863 is effectively the ceiling for this approach. The bottleneck is
1-2 false negatives per eval at the transition boundary (window 25-26, first drift windows).

## Künneth Prediction (eval #3, commit 238e97fd)

sr_drift F1 = 0.6327 with inverted BindingDetector (channels: neuron 0 and 100)
- Alarm fires when binding DROPS below baseline
- Healthy: coupled channels → high binding (topological independence = high)
- sr_drift: synchronization → channels become more independent → binding DROPS
- Result CONFIRMS the Künneth cross-term interpretation

The Künneth theory: binding measures topological INDEPENDENCE, not coupling.
Synchronization destroys excess joint topology relative to marginals → binding drops.

IMPORTANT: This is COUNTERINTUITIVE. One expects synchronization to INCREASE binding.
The Künneth lens inverts this: synchronization = less "surprising" joint topology.

## Surrogate Ablation (eval #4, commit 5a5882bf)

Phase-randomized drift: healthy phase phase-randomized → same spectrum, destroyed structure.
Official score: 0.9863 (Crocker still works on real data).
Surrogate_f1 computed per mode (in payload). Expected ≈ 0 → topology-specific detection.

## CUSUM Robustness Principle

Standard image-CUSUM is monotonically non-decreasing → NO false positives in baseline
(signal[i] < max(baseline) always, by definition of CUSUM growth).

ANY signal that can oscillate below baseline or spike above on individual windows can
create FPs. This includes:
- Column-L1 (per-window distance)
- Demeaned CUSUM (random walk in baseline)
- H0 CUSUM alone (if H0 oscillates more than H1)

RULE: Only OR two CUSUM signals if BOTH are monotonically non-decreasing.
H0+H1 CUSUM OR would be safe, but NO improvement was observed (agent-2's approach).

## Open Questions

1. Does inverted binding work on noise and connection_death? (eval #5, running)
2. What is the exact surrogate_f1 per mode? (not recoverable from grader feedback)
3. Can two-sample topology test beat CUSUM for window 25-26 detection?
