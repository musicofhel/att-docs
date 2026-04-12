---
creator: agent-1
created: 2026-04-12T06:40:00+00:00
---
# Reflection: Eval #8 — OR(binding, Crocker) Hurts sr_drift (0.9837→0.9756)

## Result
OR(inverted binding PC1+PC2, Crocker H1 CUSUM) at STRIDE=400 gave sr_drift=0.9756, down from 0.9837.
Mean F1=0.9822 vs 0.9849.

## Root Cause: Crocker STRIDE=400 Boundary FP
With STRIDE=400, n_baseline=13 (not 25). The last baseline window (start=4800, WINDOW=2000)
covers steps 4800-6800 — 90% DRIFT content classified as "healthy" by ground truth.
This window has a LARGE Crocker diff (topology change visible), causing the CUSUM to spike.
With only 13 baseline windows to estimate the threshold, this spike exceeds mu+2*sd,
causing a false positive that OR propagates to the binding signal.

## Key Insight: STRIDE Matters Differently for CUSUM vs Binding
- CUSUM monotone → boundary window FP contaminates baseline statistics at STRIDE=400
- Binding direct threshold → robust to individual window FPs since the HEALTHY binding
  score is stable (reservoir functions normally, PC1/PC2 dynamics are well-behaved)
- Solution: NEVER run CUSUM at STRIDE=400 for boundary-sensitive applications

## Current State
Best inverted binding: 0.9849/0.9850 (my f799778a + agent-3's 80cb83bb — essentially tied)
Best pure Crocker: 0.9863 (baseline, ranks 1-3)
Gap: 0.0013. Closure requires sr_drift binding to outperform Crocker (0.988+).

## Next: Try SIGMA_ALARM=1.8 for Binding
At 2.0σ with 13 baseline windows: empirically 0 FPs for seeds 0,1,2.
At 1.8σ: slightly lower threshold → might catch 1 more early drift window.
The binding healthy baseline is STABLE (reservoir functions normally), so low FP risk.
Expected: sr_drift ≈ 0.984-0.988, overall ~0.985-0.986.

## Agent Landscape
All 4 agents converged on 0.9849-0.9850 for uses_inverted_binding=true.
Agent-3's 80cb83bb is the peak. Any improvement would be a new milestone.
