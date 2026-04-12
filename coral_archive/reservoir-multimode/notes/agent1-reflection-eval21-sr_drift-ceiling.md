---
creator: agent-1
created: 2026-04-12T09:12:00+00:00
---
# Eval 21: sigma=3.5 same as sigma=3.0 — sr_drift ceiling found

## Result
sigma=3.5: still 0.9964, same as sigma=3.0. Last seed's FP is above 3.5σ.

## Analysis  
The remaining FP at start=4800 for seed X has 80% drift content with WINDOW=1000.
That 80%-drift window's Crocker topology is as strong as pure drift (>3.5σ above baseline).
No threshold can distinguish it from a true drift window. This is the hard ceiling.

sr_drift ceiling = 0.9964 with WINDOW=1000, STRIDE=200, auto-fit.

The remaining gap to 1.0:
- 1/3 seeds has FP at start=4800 → F1=0.9891 per seed
- Mean = (1.0 + 1.0 + 0.9891)/3 = 0.9964

## Next: noise PC1+PC2 OR
The hard noise seed (FN at start=5200) might have PC2 topology change even when PC1 topology is stable.
Weight noise affects ALL modes of reservoir dynamics, not just PC1.
Try OR(Crocker on PC1, Crocker on PC2) for noise.
Expected: noise 0.9915 → 1.0 if PC2 catches the hard seed → mean 0.9960 → 0.9972

## Complete status
| Mode | Current F1 | Ceiling | Gap |
|---|---|---|---|
| sr_drift | 0.9964 | 0.9964 | 0 (structural FP) |
| noise | 0.9915 | 1.0? | 0.0085 (1 FN, hard seed) |
| cd | 1.0 | 1.0 | 0 (perfect) |
| Mean | 0.9960 | 0.9972? | 0.0012 if noise fixed |
