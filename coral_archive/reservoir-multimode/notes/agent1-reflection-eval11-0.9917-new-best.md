---
creator: agent-1
created: 2026-04-12T07:05:00+00:00
---
# Reflection: Eval #11 — 0.9917 (uses_inverted_binding=true) — New Best!

## Results
- Score: 0.9917 (tied with agent-2, NEW BEST with uses_inverted_binding=true)
- sr_drift: 0.9837 (inverted binding PC1+PC2, STRIDE=400)
- noise: 0.9915 (Crocker STRIDE=400, CACHED Takens from window 0)
- connection_death: 1.0 (Crocker STRIDE=400, CACHED Takens from window 0)

## Key Insight: Cached Takens is Critical for STRIDE=400 Crocker
Root cause of why STRIDE=400 per-window auto-fit gave noise=0.9739:
- Per-window auto-fit: each drifted window adapts to its degraded state in "optimal" coords
  → topology appears similar to healthy → small diff → slow CUSUM → late alarm
- Cached Takens (from window 0, healthy): all windows use healthy coordinate system
  → drifted windows appear MORE different → large diff → fast CUSUM → early alarm

This is an EMBEDDING STABILITY principle: for topology-change detection, a FIXED reference
coordinate system (calibrated to baseline) outperforms per-window adaptive embeddings.

## Connection to Baseline Contamination at STRIDE=400
With STRIDE=400, the last baseline window (start=4800) covers 90% drift content.
- Per-window auto-fit: window adapts to mixed content → looks somewhat healthy → small spike
- Cached Takens: healthy embedding reveals the drift content → large spike

The large spike at the boundary window raises the CUSUM threshold. But for noise/cd,
the drift-phase topology changes are so dramatic that the CUSUM accelerates quickly
past the raised threshold → catch earlier drift windows.

## Why Same Fix Would Hurt sr_drift
For sr_drift (gradual synchronization onset), the drift-phase topology changes are
LESS dramatic per window. With cached Takens at STRIDE=400, the boundary spike raises
the threshold, but the drift-phase CUSUM grows slowly → alarm fires LATER → FN.
This is why we use inverted binding (not Crocker STRIDE=400) for sr_drift.

## Next: Try Crocker STRIDE=400 Cached for ALL modes (no binding)
If Crocker-cached at STRIDE=400 gives sr_drift ≥ 0.9837, we beat 0.9917.
Expected: might give sr_drift ≈ 0.96-0.98 (boundary contamination hurts gradual modes).
Risk is low since current code already supports this with a main() change.
After that: try Crocker STRIDE=200 cached for sr_drift (25 baseline windows, less contamination).
