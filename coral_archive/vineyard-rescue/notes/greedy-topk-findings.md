---
creator: agent-3
created: 2026-04-12T20:15:00+00:00
---
# Greedy Top-K Tracker: Initial Findings

## What we achieved
- F1 = 0.9697 (beat arm A = 0.929 by +0.041)
- Strategy: tracked_smooth alone (NOT ensemble)
- Precision = 0.9467, Recall = 0.9938, first_alarm = w85 (onset at w100)

## Key design choices that worked
1. Top-K filtering (K=15): Keep only most persistent H1 features. Filters noise.
2. Strict max_cost (0.5x median): Much stricter than Hungarian 2.0x. Prevents phantom tracks.
3. Min track length filter (=3): Filter short-lived tracks when reconstructing diagrams.
4. NO birth/death smoothing: Smooth binding SCORES, not features.
5. Score-level SavGol (window=21, poly=3): Smooth binding scores.
6. Ensemble did NOT help: tracked_smooth alone won.

## Current bottleneck: FPs before onset
- Both agent-1 (0.9699) and agent-3 (0.9697) have ~9-10 FPs at windows 85-99
- Signal rises before official onset (w100): early topology change before ground truth onset
- To improve: find optimal SIGMA_ALARM, try CUSUM, or sustain condition

## What to try next
1. SIGMA_ALARM scan (2.0 to 4.0) to find optimal threshold
2. Sustain condition: alarm only if N consecutive windows exceed threshold
3. CUSUM detection: accumulate deviations, fire when cumulative sum > threshold
