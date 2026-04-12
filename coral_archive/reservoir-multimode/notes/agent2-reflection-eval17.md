---
creator: agent-2
created: 2026-04-12T19:30:00+00:00
---
# Reflection: Eval 17 - Timeout + Noise Fix

## Concrete results
- bb3f53b7 (0.9972): sr_drift=1.0, noise=0.9915, cd=1.0 — inverted binding k=1.5+suppress
- de1fcc7 (TIMEOUT): extra seeds {3-7} pushed runtime to ~1620s+overhead → over 1800s limit

## Timeout root cause
Extra seeds added 15 pairs x avg 60s = 900s on top of 720s = timeout.
Fix: remove extra seeds. Surrogated eval needs <1080s budget for required 9 pairs.

## Noise fix: sigma=1.5 confirmed by agents 3 and 1
noise=0.9915 = 1 FN in 1 hard seed (missed early-drift window), NOT a FP.
sigma=1.5 catches the early-drift window; no FPs at 1.5σ (abrupt noise injection).
My code uses cusum_k=1.5+suppress_healthy: CUSUM monotone arg guarantees first drift alarms.

## Current eval
noise k=1.5+suppress, sr_drift inverted binding k=1.5+suppress, cd k=2.0
Expected: mean F1=1.0, uses_inverted_binding=True (Künneth confirmation)
