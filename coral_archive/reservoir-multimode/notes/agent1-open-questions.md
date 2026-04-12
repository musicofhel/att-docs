---
creator: agent-1
created: 2026-04-12T01:15:00-04:00
---
# Open Questions for Reservoir Multimode

## Unresolved

1. **Can F1 > 0.9863 be achieved at all?**
   - 4 attempts by 3 agents, all failed to beat baseline
   - Crocker H1 CUSUM seems near-optimal for this setup
   - The 0.9863 gap (window 25-26 FN) may be fundamentally hard to fix

2. **Does inverted binding work on noise and connection_death?**
   - Only tested on sr_drift (Künneth predicts this mode)
   - Noise and connection_death might also show binding changes
   - Worth testing as a secondary contribution

3. **Surrogate ablation result (pending eval #4)**
   - Does Crocker detector alarm on phase-randomized surrogate?
   - Answer pending in ~15 minutes

## Confirmed Findings

- Künneth confirmed: sr_drift inverted binding F1 = 0.6327 ≥ 0.60
- Baseline CUSUM at 0.9863 is robust across all 3 modes
- Column-L1 and demeaned CUSUM both add FPs → hurt F1
- Extra seeds timeout: only 9 pairs per eval max

## Next Experiments Worth Trying

- Two-sample topology test instead of control chart (no baseline required)
- Per-mode SIGMA_ALARM tuning (set analytically from F1 landscape)
- Fixed subsample seed=42 (agent-4's insight, not yet tested alone)
