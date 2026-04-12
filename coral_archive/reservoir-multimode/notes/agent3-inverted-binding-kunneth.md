---
creator: agent-3
created: 2026-04-12T06:20:00+00:00
---
# MAJOR FINDING: Inverted Binding Confirms Künneth Prediction

## Key Result (eval 80cb83bb, score=0.985)
Binding score (BindingDetector on PC1+PC2) drops 200x at sr_drift onset:
- Healthy windows: binding scores 1279–3262 (mean ~2175, std ~700)
- Drift windows: binding scores 2–30 (200x DROP)
- Inverted alarm (binding < mean-2std): F1=0.984 on sr_drift — confirmed!
- Overall score: 0.985 with uses_inverted_binding=True — #1 leaderboard

## Why sr_drift F1=0.984 not 1.0
Direct threshold gives F1=0.984. Possible causes:
- FP: some healthy window's binding below threshold (2.3% chance per window,
  ~44% chance in 25 windows). Isolated, rarely consecutive.
- FN: late drift binding partially recovers (at SR=1.3+, complex dynamics
  might create more topology → binding rises above threshold).
- Test data showed start=8000 binding=2.2, but start=12000-13000 untested.

## FAILED: CUSUM OR approach (0.984→0.976)
Adding inverted CUSUM on drops (cumsum of max(0,baseline_mean-binding)) via OR:
- During baseline: CUSUM grows since EXACTLY half of baseline windows have 
  binding < baseline_mean (definitionally). This adds FP via OR.
- Caused sr_drift regression: 0.984 → 0.9762

## Current attempt: Sticky propagation
Once 2 consecutive alarms fire, propagate to end of sequence.
- If binding drops at onset (windows 25+26 both alarm → propagate from 25 to end)
- Handles late-drift FN from binding partial recovery
- Safe for isolated healthy FP (typically non-consecutive noise)

## Best Implementation (SUBSAMPLE=30, IMAGE_RES=20, delay=10, dim=3)
BindingDetector with pre-fitted embedders. ~1.5s/window healthy, ~0.7s/window drift.
Total runtime with surrogate ablation: ~650s << 1800s.

## Surrogate Ablation
Run CUSUM on phase-randomized surrogate of healthy phase substituted for drift.
Expected: surrogate_f1 ≈ 0 (detector topology-specific, not non-stationarity).
Results in payload.surrogate_ablation list.
