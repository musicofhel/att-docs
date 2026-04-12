---
creator: agent-2
created: 2026-04-12T00:15:00+00:00
updated: 2026-04-12T00:45:00+00:00
---
# Reflection: Evals #1-2

## Eval #1 Results (baseline)
- Score: 0.9863 (mean F1 across 9 mode×seed pairs)
- Per-mode: sr_drift=0.988, noise=0.9876, connection_death=0.9835

## Eval #2 TIMEOUT
- Added H0+H1 OR CUSUM + extra seeds {3-7} (24 evals total)
- TIMED OUT at 1800s
- Lesson: actual time per eval is ~75s (not 30s as stated). 24 × 75s = 1800s → exact timeout.
- Fix: keep to 9 required pairs only. Never add extra evals to the grader run.

## Why ~75s per eval?
TakensEmbedder(delay="auto", dimension="auto") is called for EACH of 66 windows.
"auto" computes mutual information + FNN → ~1-2s per window.
66 × 1.5s = ~99s Takens fitting alone. Plus ripser + simulation = ~75-100s total.

FIX: Cache Takens params from first window, use fixed params for all subsequent windows.
Estimated speedup: 66 windows × (1.5s → 0.3s) = saves 79s per eval.
After fix: ~20s per eval → 9 × 20s = 180s total (well under 1800s limit).

## Why gap from 1.0?
Window 25 (first drift, start_idx=5000): progress≈0 → tiny topological change.
Standard image CUSUM threshold ≈ 27*d_noise; CUSUM[25] ≈ 26*d_noise → missed.
One-sided CUSUM (excess only): removes background growth → smaller threshold → earlier detection.

## Next Steps
1. Cache Takens params (speed) + one-sided CUSUM (sensitivity) + H0 OR H1 → eval #3
2. If F1 improves, continue refining
3. Inverted binding (separate eval, uses_inverted_binding=true)
4. Phase-randomized surrogate ablation (add to payload once main eval is fast)
