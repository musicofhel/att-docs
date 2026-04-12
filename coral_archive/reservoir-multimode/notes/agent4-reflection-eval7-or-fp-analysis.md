---
creator: agent-4
created: 2026-04-12T03:16:23.573883
---
# Reflection: Eval #7 — OR still gives sr_drift=0.9756; Root Cause Found

## Concrete Results
- Score: 0.9890 (sr_drift=0.9756, noise=0.9915, cd=1.0)
- Global PCA fix (binding) did NOT help sr_drift when using OR
- Same 0.9756 as previous OR attempts (consistent across agents 2 and 4)

## Root Cause: Window 4800 Contamination Creates FP in Crocker for sr_drift

Window at start=4800 covers steps [4800, 6800]: 200 healthy + 1800 sr_drift steps.
This window is LABELED HEALTHY (start < 5000) but contains 1800 drift steps.

With STRIDE=400 and n_baseline=13: this contaminated window is INDEX 12 (last baseline).
Crocker CUSUM: sig[12] = cumsum including large diff from contaminated window.
threshold = mean(sig[0..12]) + 2*std(sig[0..12]).

Analysis: with sig[12] >> sig[0..11] (one outlier at index 12):
  threshold ≈ 0.63 * sig[12] < sig[12] → FP at window 4800!

This creates 1 FP per seed (start=4800 labeled healthy but alarmed).
Effect on OR: precision = 20/(20+1) = 0.952 per seed, recall = 1.0, F1 = 0.976.
Mean across 3 seeds = 0.976 ≈ 0.9756. CONFIRMED!

Why doesn't cd=1.0 have this problem?
  Agent-2 uses SAME architecture and gets cd=1.0 (no FPs). 
  Hypothesis: for cd, the diff at window 4800 is RELATIVE to baseline_col which includes
  the contaminated column. Since baseline_col absorbs some of the contamination,
  diff[12] is smaller than the pure contamination signal. For sr_drift, the contamination
  at window 4800 may cause relatively larger deviation from baseline_col due to timing.
  
  Actually: for cd and noise, the dramatic sudden degradation creates consistent Crocker
  columns in drift windows. For sr_drift, the gradual drift may interact differently.

## Plan for Eval #8: Pure Binding Only (No OR) for sr_drift

Remove OR for sr_drift. Use ONLY direct inverted binding (global PCA, SUBSAMPLE=150).
Agent-1 and agent-2 both achieve 0.9917 with this approach.

Target: match 0.9917. Then investigate if there's any way to push sr_drift above 0.9837.

Notes:
- Agent-2 tried k=1.5 for binding: got 0.9839 (tiny improvement)
- The 2 FNs in sr_drift are at window 5200 (SR barely drifted after 200 steps)
- Very hard to detect SR=1.01 (barely above 0.9) with current window size
