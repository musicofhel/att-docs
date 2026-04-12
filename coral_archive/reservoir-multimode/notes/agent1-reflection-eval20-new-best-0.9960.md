---
creator: agent-1
created: 2026-04-12T09:05:00+00:00
---
# NEW BEST: 0.9960 — sigma=3.0 for sr_drift (eval 20)

## Result (commit 947bf7564f52)
Mean F1: 0.9960 (NEW BEST, up from 0.9948 and 0.9931 prior ceiling)
- sr_drift: 0.9964 (up from 0.9928)
- noise: 0.9915 (maintained)
- cd: 1.0 (maintained)

## Trajectory 
0.9917 (baseline) → 0.9931 (mode-specific stride/embedding) → 
0.9948 (WINDOW=1000 for sr_drift) → 0.9960 (sigma=3.0 for sr_drift)

## Why it worked
sr_drift with WINDOW=1000, sigma=3.0:
- 2/3 seeds had boundary FP at start=4800 between 2.5σ and 3.0σ → NOW SUPPRESSED
- 1 seed's boundary FP above 3.0σ → still present

Per-seed analysis:
- 2 seeds: F1=1.0 (perfect)
- 1 seed: 1 FP at start=4800 → F1 = 2*(46/47)*1.0/(46/47+1.0) = 0.9891
- Mean: (1.0 + 1.0 + 0.9891)/3 = 0.9964 ✓

## Current experiment: sigma=3.5
The last seed's FP is above 3.0σ. If it's between 3.0σ and 3.5σ → sigma=3.5 suppresses it.
Expected (if FP at 3.0-3.5σ): sr_drift=1.0 → mean=0.9972

Risk: early pure drift windows may have CUSUM near 3.0-3.5σ for gradual sr_drift onset.
A regression (FNs at early drift windows) is possible. 50/50.

## Summary of what drives improvements
The CUSUM for sr_drift with WINDOW=1000 auto-fit:
- Clean baseline: CUSUM grows slowly (0-3.0σ range over 20 clean windows)
- Contaminated baseline windows (starts 4000-4800): CUSUM grows faster (3.0-4.0σ)
- First pure drift window (start=5000): CUSUM should jump strongly (>4.0σ)
- sigma=3.0 sits just above the contaminated window peak for most seeds

The last seed's contaminated window peak is higher (above 3.0σ). Either try sigma=3.5
or accept 0.9964 as sr_drift ceiling and focus on noise.
