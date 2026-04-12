---
creator: agent-1
created: 2026-04-12T08:52:00+00:00
---
# Eval 17 Reflection: sigma=2.5 no help → PIVOTING to WINDOW=1000

## What happened
sigma=2.5 for sr_drift: still 0.988. The boundary FP at start=4800 (90% drift content)
has CUSUM value ABOVE 2.5σ. No threshold value can suppress it without also suppressing
pure drift windows. The FP is structural — the window is 90% drift, so it looks almost
like a drift window topologically.

## Ceiling analysis: what I've exhausted
- sigma tuning for sr_drift (2.0, 2.5): no change → FP is above 2.5σ, below pure-drift level
- sigma tuning for noise (1.9, 2.0): no change → FN CUSUM below 1.9σ threshold
- Variance OR for noise: added FPs → hurt
- H0+H1 OR: same result → H0 adds no new TPs
- STRIDE=200 cached for noise: contamination → hurt

## Fundamental approach change: WINDOW=1000

With WINDOW=2000:
- First drift window: start=5200 (200 steps into drift, 200 steps detection lag)
- Boundary FP at start=4800: 90% drift content → always FP
- Contaminated baseline windows: 5/13 = 38% (starts 3200-4800)

With WINDOW=1000:
- First drift window: start=5000 (0 steps into drift — immediate detection!)
- Boundary FP at start=4800: 80% drift content → might be suppressible with sigma=2.5
- Contaminated baseline windows: 3/13 = 23% (starts 4000-4800) → LOWER threshold!

For noise specifically:
- Hard seed FN was at start=5200 with WINDOW=2000 (2nd drift window, CUSUM too slow)
- With WINDOW=1000: first drift window is start=5000 (1st drift window, more CUSUM time)
  AND the hard seed's FN window may now alarm earlier due to lower contamination threshold

Risk: smaller window → less stable Takens embedding (980 points vs 1980)
But SUBSAMPLE=250 means we're already subsampling — the ratio is similar.

## Expected outcome
- noise: 0.9915 → 1.0 (hard seed FN caught at start=5000 or 5200 with lower threshold)
- sr_drift: 0.988 → possibly higher (less contamination, earlier detection)
- cd: 1.0 → 1.0 (should maintain)
- mean: 0.9931 → 0.9944-0.9960 if noise catches up
