---
creator: agent-1
created: 2026-04-12T06:55:00+00:00
---
# Reflection: Eval #10 — STRIDE=400 Crocker noise/cd

## Results
- sr_drift: 0.9837 (binding, unchanged)
- noise: 0.9739 (WORSE! was 0.9876 at STRIDE=200)
- connection_death: 1.0 (BETTER! was 0.9835 at STRIDE=200)
- mean: 0.9859

## Why Noise Got Worse at STRIDE=400
My Crocker code fits Takens embedding INDEPENDENTLY per window (auto-fit each time).
Agent-2's code caches Takens from window 0 and uses FIXED params for all windows.

With STRIDE=400 + per-window auto-fit: each drifted window adapts its embedding to the
current (degraded) state → topology looks similar to healthy in its own "optimal" coords
→ diff is artificially small → CUSUM grows slowly → late alarm → FN → F1 drops.

With STRIDE=400 + cached Takens: all windows use the same coordinate system calibrated to
the HEALTHY state. Drifted windows appear MORE different → faster CUSUM → earlier alarm.

Agent-2's noise=0.9915 vs mine=0.9739 (same STRIDE=400): CONFIRMED: caching Takens is key.

## Fix: Cache Takens from window 0 in evaluate_crocker
Next eval: fix Takens caching + keep STRIDE=400 for noise/cd.
Expected: noise~0.99, cd~1.0, mean~0.991 — matching agent-2's 0.9917.
