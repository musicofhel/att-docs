---
creator: agent-1
created: 2026-04-12T07:35:00+00:00
---
# SYNTHESIS: Stride-Embedding Interaction in Crocker CUSUM Detectors

## Summary
The optimal Crocker H1 image-CUSUM configuration depends on the failure mode's temporal
character: GRADUAL degradation requires STRIDE=200 + per-window auto-fit; ABRUPT degradation
requires STRIDE=400 + cached Takens. Mixing these gives intermediate or worse results.

## Evidence Table
| Mode | STRIDE | Embedding | F1 |
|------|--------|-----------|-----|
| sr_drift (gradual) | 200 | auto-fit | 0.988 |
| sr_drift (gradual) | 200 | cached | 0.984 |
| sr_drift (gradual) | 400 | auto-fit | 0.9756 |
| sr_drift (gradual) | 400 | cached | 0.9756 |
| noise (abrupt) | 200 | auto-fit | 0.9876 |
| noise (abrupt) | 200 | cached | 0.9748 |
| noise (abrupt) | 400 | auto-fit | 0.9739 |
| noise (abrupt) | 400 | cached | **0.9915** |
| connection_death (abrupt) | 200 | auto-fit | 0.9835 |
| connection_death (abrupt) | 400 | cached | **1.0** |

## Mechanism

### Why STRIDE=400 + cached Takens is optimal for ABRUPT modes (noise, cd)
At STRIDE=400, only ONE baseline window (start=4800) contains drift content.
Cached Takens (from window 0, healthy) reveals that window's drift as a large diff.
But ABRUPT degradation means the FIRST genuine drift windows (start=5200+) have
MUCH larger diffs → CUSUM crosses threshold quickly → early alarm → high recall.
The 1 contaminated baseline window raises the threshold slightly but not enough to block.

### Why STRIDE=200 + auto-fit is optimal for GRADUAL modes (sr_drift)
At STRIDE=200, 9 baseline windows (starts 3200-4800) contain drift content.
With cached Takens: all 9 show large diffs → CUSUM grows fast in baseline → threshold
inflated massively → even genuine drift windows can't cross it → high FN count.
With per-window auto-fit: each window adapts → boundary windows "hide" their drift
content → small diff → CUSUM grows slowly → threshold stays low.
GRADUAL degradation accumulates slowly over many drift windows, each adding small
diffs → CUSUM accumulates → eventually crosses the low threshold → high recall.

## Rule of Thumb
- count_contaminated_baseline = number of baseline windows with start > N_HEALTHY - WINDOW
- If count_contaminated_baseline ≤ 1-2: USE cached Takens (safe, drift is visible)
- If count_contaminated_baseline >> 3+: USE per-window auto-fit (cached raises threshold too much)
- For abrupt degradation (noise/cd): use higher STRIDE to minimize contamination
- For gradual degradation (sr_drift): use lower STRIDE + auto-fit for dense, adaptive tracking

## Optimal Configuration (confirmed by eval 13, score 0.9918)
- sr_drift: STRIDE=200, per-window auto-fit
- noise: STRIDE=400, cached Takens
- connection_death: STRIDE=400, cached Takens
- Expected mean: ~0.9918-0.993 (next eval tests sr_drift auto-fit)
