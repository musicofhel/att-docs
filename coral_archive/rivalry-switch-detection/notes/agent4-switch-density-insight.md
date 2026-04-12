---
creator: agent-4
created: 2026-04-12T19:45:00
---
# Switch Density Analysis and Adaptive Dedup Strategy

## Subject Switch Density (critical baseline knowledge)
- Subject 3629: 41 switches, mean_gap=2.7s, 35/40 gaps < 5s (dense)
- Subject 3630: 6 switches, mean_gap=12.0s, 3/5 gaps < 5s (SPARSE)
- Subject 3632: 54 switches, mean_gap=2.2s, 50/53 gaps < 5s (very dense)

Switch ranges: 3629: 8.6-117.4s, 3630: 54.3-114.2s, 3632: 4.5-119.3s

## Metric Properties (key insight)
With +-5s tolerance, one detection covers a 10s window.
- Dense subjects (3629/3632): P(TP for random detection) ~0.98 -- just need recall
- Sparse subject (3630): P(TP for random detection) ~0.63 -- need precision

Subject 3630 has a QUIET PERIOD (0-54s with zero switches). Aggressive alpha
CUSUM fires many FPs here -- precision ~0.30 -- F1 ~0.47.

## Why Switch-Scale Windows Failed
Window=1080/step=360 -- only 116 transition scores, CUSUM fires 5-7 times only.
Dense subjects need many more triggers. Keep window=500/step=200.

## Proposed Fix: Adaptive Per-Subject Dedup
Quick alpha-power scan to estimate density L. Set dedup_s = max(2.0, 0.7/L).
Dense subjects (L > 0.15): dedup=2s (recall-focused)
Sparse subjects (L < 0.08): dedup=7-10s (precision-focused)

## Agent-1 Results (F1=0.789) - Bottleneck is Subject 3630
TDA k=1.0 + alpha CUSUM k=0.8, 4 channels, dedup=2s.
Subject 3630: F1=0.474, ~20 detections but only 6 are TPs (quiet period FPs).
With adaptive dedup (7s for 3630): expected F1=0.65+, mean=0.83+
