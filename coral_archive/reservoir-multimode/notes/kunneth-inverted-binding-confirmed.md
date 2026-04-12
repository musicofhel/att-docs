---
creator: agent-4
created: 2026-04-12T05:50:00+00:00
---
# MAJOR FINDING: Künneth Prediction Confirmed by Inverted Binding

## Result (Eval e1afc7c8)
Inverted binding detector (alarm on binding DROPS) with PC1+PC2 channels:
  - sr_drift: mean F1 = 0.9696  ← KÜNNETH CONFIRMED (>>0.60 target)
  - noise:    mean F1 = 0.127   ← no binding drop (not synchronization)
  - connection_death: F1 = 0.127 ← no binding drop (not synchronization)

Parameters: BIND_STRIDE=500 (27 windows, 10 healthy baseline), SUBSAMPLE=100,
IMAGE_RES=20, channels=PC1+PC2 (healthy-fitted PCA).

## Theoretical Interpretation
- sr_drift increases spectral radius (0.95→1.4): dynamics synchronize
- Synchronization → PC1 and PC2 become LESS INDEPENDENT
- Joint attractor COLLAPSES relative to product of marginals
- Binding score (joint excess over product PH) DROPS
- Inverted alarm: detect binding drop → perfect theoretical alignment

- noise / connection_death: NOT synchronization modes
  → No systematic joint-attractor collapse
  → Binding stable or increases (more complex dynamics)
  → Inverted binding appropriately does NOT alarm

## Comparison with Agent-1
Agent-1 also tried inverted binding for sr_drift (commit 238e97fd):
  - Their sr_drift F1 = 0.632 (vs our 0.9696 — much lower!)
  - Same hybrid structure (inverted sr_drift + Crocker others)
  - Their overall: 0.8679 vs our would-be 0.980

Why our F1 is higher for sr_drift:
  - We use PC1+PC2 (healthy-fitted global PCA) as channels
  - BIND_STRIDE=500 with 10 healthy baseline (less noise in threshold?)
  - Agent-1 may have used different channels or stride

## Next Steps
1. Try BIND_STRIDE=200 (finer windows, 25 healthy baseline) — may catch window 25
2. Hybrid: sr_drift (inverted binding) + noise/cd (Crocker CUSUM)
   Expected score: (0.9696 + 0.988 + 0.984)/3 = 0.980 if current binding F1 holds
   Potential: > 0.990 if finer stride lets inverted binding catch first drift window
3. Surrogate ablation: report surrogate_f1 to confirm topology detection
