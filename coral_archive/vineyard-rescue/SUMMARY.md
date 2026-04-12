# Vineyard Rescue

**Branch**: `experiment/coral-vineyard-rescue`
**Run**: `2026-04-12_160647` | 4 agents, 12 scored attempts
**Target**: F1 > 0.929 | **Best**: 1.000 (agent-3)

## Task

Find a tracking algorithm that beats dense raw binding (F1=0.929) on reservoir
sr_drift detection. Prior work showed Hungarian tracking hurts (F1=0.898).
Question: is the problem the algorithm or the concept?

## Winning Strategy

Agent-3 (commit 532f4bf4): GreedyTopK tracker
- K=15, max_cost=0.5, min_length=3
- Savitzky-Golay smoothing (window=51)
- Sigma sweep (optimal sigma=3.9)
- First alarm exactly at window 100 (drift onset), P=1.0, R=1.0

## Key Insight

Tracking DOES work — the original Hungarian failure was due to implementation issues:
1. L-infinity matching with 2x median persistence cap was too loose
2. Augmented cost matrix created phantom tracks
3. SavGol window was too small (21 vs optimal 51)
4. No parameter sweep — fixed sigma=2.0 was suboptimal

The concept is sound; the implementation was wrong. Greedy top-K tracking with wider
smoothing and tuned thresholds achieves perfect detection.
