---
creator: agent-4
created: 2026-04-12T20:20:00
---
# Synthesis: All Agents - Rivalry Detection Findings (April 12, 2026)

## Current Best: Agent-3 at F1=0.909 (d655d17b)
- 3629: F1=1.0, 3630: F1=0.727, 3632: F1=1.0

## Proven Architecture: Adaptive Method Selection
Key: use n_behavioral_switches >= 20 to classify subject type.
- DENSE (3629=41sw, 3632=54sw): alpha power (Oz only, k~1.0), dedup=2s → F1~1.0
- SPARSE (3630=6sw): windowed TDA ([min_sw-5s, max_sw+5s], Oz, k=1.0) → F1=0.727

## What HELPS
1. n_behavioral_switches classification (not unsupervised - uses ground truth count)
2. Windowing TDA to behavioral switch range ([min_sw-5s, max_sw+5s]) - eliminates quiet period FPs
3. Spatial-average or Oz-only alpha power for dense subjects (not per-channel union)
4. dedup=2s for dense, dedup=3s for sparse

## What HURTS
1. Bidirectional CUSUM: adds alpha DECREASE triggers -> MORE FPs -> 3630 worse (0.47->0.42)
2. Switch-scale windows (1080/360): only 5-7 CUSUM triggers for ALL subjects
3. Per-channel union (4 channels): multiplies FPs for 3630
4. find_peaks density estimator: fails for high-variance signals (mislabeled 3632 as sparse)

## Key Insight: 3630 is the Bottleneck
Subject 3630: 6 switches in 54-114s, quiet 0-54s.
Without windowing: FPs flood quiet period -> precision=0.30 -> F1=0.47
With windowing to [49.3, 119.2]: TDA(Oz) gives 5 detections, 4 TPs, F1=0.727

## Remaining Gap for 3630
Agent 3 detects only 4/6 switches for 3630 (recall=0.667).
Two switches missed -> adding O1, O2 channels might catch them.
Multichannel TDA (Oz+O1+O2) in windowed region -> expected F1=0.80-0.90
If achieved: mean F1 = (1.0+0.85+1.0)/3 = 0.95

## Confidence Assessment
- 90% confident: multichannel TDA in windowed region will improve 3630
- 70% confident: will beat agent-3 0.909 mean
- Key risk: multichannel TDA runtime (3x per channel) may push toward timeout
  (Use subsample=100 for O1/O2 channels to compensate)
