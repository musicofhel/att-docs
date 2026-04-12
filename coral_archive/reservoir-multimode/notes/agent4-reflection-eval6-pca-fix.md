---
creator: agent-4
created: 2026-04-12T02:53:51.284045
---
# Reflection: Eval #6 — Noise/CD Fixed (0.9915/1.0), sr_drift Still 0.9679

## Results
- Score: 0.9864 (just barely above baseline 0.9863!)
- sr_drift: 0.9679 (OR with binding+Crocker — WORSE than agent-2 binding alone at 0.9837)
- noise: 0.9915 (fixed! was 0.9746 in my previous evals)
- connection_death: 1.0 (fixed! was 0.9876)

## What Fixed Noise/CD
The key changes that fixed noise/cd:
1. STRIDE=400 (first drift window=5200, skips hard start=5000)
2. Fixed Takens embedding (cached from first window, consistent state space)
3. Per-window PH seed (mode_seed + w_idx, matches reference code)

With these, noise improved from 0.9746 → 0.9915, cd from 0.9876 → 1.0.

## Why sr_drift is 0.9679 (not 0.9837)

Agent-2 gets sr_drift=0.9837 with DIRECT binding at STRIDE=400.
My OR(binding, Crocker) gets only 0.9679 — LOWER than agent-2 binding alone.
OR can only increase alarms → can only REDUCE precision if binding adds FPs.
But if OR gives less than binding alone, binding itself must be giving ~0.967.

ROOT CAUSE: Global PCA vs Healthy-only PCA for binding channels
- Agent-2: pca12(states) → global PCA over all 15000 steps
- Mine: pca2_healthy(states) → PCA fitted only on first 5000 (healthy) steps

With healthy-only PCA: PC1/PC2 capture healthy dynamics. Drift changes the state
space differently, and the healthy-fitted PCs may not capture the binding structure
of the drift regime as cleanly.

With global PCA: PC1/PC2 capture maximum variance ACROSS all states, including
drift → PCs are sensitive to the drift-induced changes → binding score drops more
dramatically at onset → cleaner detection.

Secondary: SUBSAMPLE_BIND=120 vs agent-2's 150. More samples → less stochastic 
variance in binding score estimate → better threshold calibration.

## Plan for Eval #7: Global PCA + SUBSAMPLE=150

Change pca2_healthy → pca12 (global) for binding.
Change SUBSAMPLE_BIND 120 → 150.
Keep everything else (STRIDE=400, OR, fixed Takens+per-window seed for Crocker).

Expected: binding ~0.9837 (matching agent-2), OR with Crocker → sr_drift ≥ 0.9837.
Mean: (0.9837 + 0.9915 + 1.0) / 3 = 0.9917 at minimum (matching agent-2).
If OR catches Crocker-missed windows: sr_drift → 1.0, mean → 0.997!

## Confidence
95% confident that global PCA fix will bring sr_drift to 0.9837+.
60% confident that OR with Crocker will push sr_drift above 0.984.
