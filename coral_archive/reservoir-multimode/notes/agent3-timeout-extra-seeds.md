---
creator: agent-3
created: 2026-04-12T10:20:00+00:00
---
# Eval #13 TIMEOUT: Extra Seeds Run Too Slow

## What happened
Added 5 extra seeds validation (seeds 3-7) inside the graded script.
Timed out at 1800s grader limit.

## Root cause
sr_drift with STRIDE=200 + per-window auto-fit Takens is expensive:
- 66 windows per seed × ~2-4s/window (auto-fit mutual info + TDA) = 132-264s/seed
- Required 9 pairs: 3 sr_drift seeds = 396-792s
- Extra 15 pairs: 5 more sr_drift seeds = 660-1320s additional
- Total: 1000-2100s → exceeds 1800s limit

## Fix
For extra validation seeds, use STRIDE=400 + fixed embedding for ALL modes:
- 34 windows per seed, no auto-fit overhead → ~0.5s/window × 34 = 17s/seed
- Extra 15 pairs: ~255s
- Required 9 (auto-fit sr_drift): ~600s  
- Total: ~855s < 1800s ✓

F1 for extra sr_drift seeds will be slightly lower (~0.988) but that is OK —
it is just validation, not the scored pairs.

## Current best
Score = 1.0 (commit 87fbe33) after revert. Rank #1.
