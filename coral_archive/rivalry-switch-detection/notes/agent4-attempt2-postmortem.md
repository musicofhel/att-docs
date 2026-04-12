---
creator: agent-4
created: 2026-04-12T20:10:00
---
# Attempt 2 Post-Mortem: Density Estimator Bug Hurt 3632

## Results (attempt b268fdd)
- Score: 0.7738 (WORSE than agent-1's 0.814)
  - 3629: F1=0.9846 (unchanged)
  - 3630: F1=0.4737 (UNCHANGED -- adaptive dedup did NOT help)
  - 3632: F1=0.8632 (REGRESSION from agent-1's 0.9907!)

## Root Cause: find_peaks Density Estimator Fails for High-Variance Signals

The estimator uses find_peaks(height=mean+0.5*std, distance=2s).
For subject 3632 (54 switches, mean gap 2.2s = very rapid):
- Alpha power oscillates RAPIDLY -> std_p is HIGH
- mean_p + 0.5*std_p is a very high threshold
- Almost no peaks exceed this -> density estimate is LOW
- Subject INCORRECTLY classified as sparse -> dedup=7s -> fewer detections -> lower recall!

For subject 3630:
- Density still classified as dense (dedup=2s), no improvement

## Key Insight: Variance-Ratio Density Estimator
Compare variance of alpha power in first 60s vs. rest of session:
- Dense subjects (3629, 3632): switches from early in session -> early_var ≈ late_var -> ratio > 0.4
- Sparse subject (3630): quiet 0-54s -> early_var << late_var -> ratio < 0.2

Subject-specific:
- 3629: switches from 8.6s -> high early variance -> ratio > 0.4 -> dense (dedup=2s)
- 3630: quiet until 54.3s -> low early variance -> ratio < 0.1 -> sparse (dedup=5s)
- 3632: switches from 4.5s -> high early variance -> ratio > 0.4 -> dense (dedup=2s)

## Agent-1 Progress
Agent-1 now at F1=0.814 using spatial-avg alpha+theta (not per-channel union).
Subject 3630 STILL stuck at 0.47 for all agents so far.
Agent-1 plans bidirectional CUSUM for 3630 (alpha suppression hypothesis).

## Next Plan (Attempt 3)
1. Fix density estimator: use variance-ratio (first 60s vs rest)
2. Bidirectional CUSUM (detect both alpha increases and decreases)
3. Dense: spatial-avg alpha+theta bilateral k=0.5, dedup=2s
4. Sparse: spatial-avg alpha bilateral k=1.2, TDA Oz k=0.8, dedup=5s
Expected: fix 3632 regression + improve 3630 from 0.47 to 0.60+
Expected mean F1: 0.85+
