# Branch 4 — Zigzag Temporal Reservoir (Tracer Bullet)

**Date**: 2026-04-11
**Branch**: `experiment/zigzag-reservoir`
**Architecture**: #7 — Zigzag temporal persistence
**Verdict**: **FAIL** — zigzag finds no cross-regime tracked features; the snapshot baseline already detects perfectly

## Motivation

ATT's existing reservoir continuous-ramp experiment hits F1≈0.06 at stride=200 with
per-window scalar binding scores. Hybrid architecture #7 from the 10-architecture
document proposes zigzag persistent homology over a sequence of point clouds (the
temporal analogue of the LLM `ZigzagLayerAnalyzer`'s "layers as time" axis) to find
H1 features that persist across multiple windows — features that snapshot PH on
each window independently can never see.

This branch tests whether zigzag adds value over snapshot PH on the SR drift signal.

## What we did

1. Generated `simulate_degradation(mode="sr_drift", seed=42)` from `scripts/reservoir_degradation_monitor.py`.
2. PC1 of states → 1D signal.
3. Sliced 27 windows (W=2000, stride=500) → 27 clouds covering the trajectory.
4. Per window: Takens-embed → constant-size subsample (60 points each — required for
   zigzag because vertices are vertex-indexed across all clouds).
5. PCA-reduce each cloud to ≤20 components independently.
6. Pairwise distance matrices, adaptive VR-complex threshold (we tested p30 and p70).
7. Inlined zigzag construction (mirrors `att/llm/zigzag.py:_build_and_compute_zigzag()`
   exactly — vertex always present, edges/triangles present at zigzag times where the
   distance criterion holds; uses `dionysus 2.1.8` `zigzag_homology_persistence`).
   `att/llm/zigzag.py` is on the TDA-LLM branches and not on master.
8. Detection: H1 features whose **birth** is in the post-onset half of the sequence
   AND whose **lifetime** exceeds the median healthy-region lifetime.
9. Baseline: per-window snapshot ripser PH on identical clouds, scalar = Σ (death-birth)
   over H1, 2σ alarm vs first 10 healthy clouds.

## Results

### Zigzag (two threshold trials)

| Threshold | n H0 | n H1 | H1 lifetimes | Healthy births | Drift births | Long post-onset |
|---|---|---|---|---|---|---|
| **p30** (0.37) | 194 | 2 | mean=0.50, max=0.50 | 2 | 0 | 0 |
| **p70** (5.37) | 1 | 9 | mean=0.50, max=0.50 | 0 | 9 | 0 |

In both cases, *every* H1 feature has lifetime 0.5 — the minimum non-zero zigzag lifetime
(one zigzag time step, i.e. a feature appears at one cloud or union and dies at the next).
**No H1 feature ever spans a regime change or multiple cloud transitions.** Zigzag is
not tracking anything.

### Snapshot PH baseline (same clouds, ripser)

The total H1 persistence per cloud is monotone-strong in the regime change:

```
healthy clouds 0-9:   total H1 ∈ [0.20, 0.50]   (mean ≈ 0.35)
drift clouds 10-26:   total H1 ∈ [0.64, 8.66]   (monotone-rising)
```

2σ threshold from clouds 0-9 = 0.527.
**First alarm cloud: 10 — exactly the onset.** F1 ≈ 1.0 at this stride.

### Why zigzag failed: drift expansion swamps the threshold

Diagnostic: per-cloud median pairwise distance:

```
healthy clouds (0-9):  median ≈ 0.30
drift clouds (10-26):  median rises monotonically 1.21 → 10.50
```

The chaos induced by SR > 1 expands the embedding by ~30× in pairwise scale. A *global*
adaptive threshold (the natural choice for zigzag) trades off:
- Low threshold (p30 = 0.37): healthy clouds form many edges, drift clouds form essentially none.
- High threshold (p70 = 5.37): drift clouds form edges, but the union complexes between
  healthy and drift clouds are incoherent (a healthy cloud's edges don't survive the union with
  a drift cloud because the drift cloud is essentially in a different scale regime).

In **both** cases, the union complexes between healthy and drift clouds are degenerate, so
no H1 feature can be born at one regime and die at the other — the architectural
prerequisite for zigzag-tracked features.

### Why snapshot PH wins easily

Snapshot PH measures total H1 *persistence* (not feature count) per cloud independently.
The chaos regime has fewer but much longer-lived loops, so total persistence rises
monotonically with SR drift. This is the dominant signal, and zigzag's "track features
across windows" framing is *incompatible* with a regime change that completely reorganizes
the embedding scale.

## Verdict

**FAIL** — the qualitative tracer bullet says zigzag should find a structurally
distinguishable feature snapshot PH can't see. The opposite is true here:

1. Zigzag finds no cross-regime features at any threshold we tested.
2. Snapshot PH at the same stride already detects at the exact onset cloud (perfect F1).
3. There is no architectural benefit to zigzag for this signal class.

The result is informative even if FAIL: it shows that **zigzag temporal persistence
requires regime changes that preserve the embedding scale**. Drift signals where the
state space *expands or contracts* are not in zigzag's natural regime — features cannot
be tracked across an expansion because the topology of the union complex is dominated
by the larger of the two clouds.

## What's worth keeping

- **The inlined zigzag helper** (`build_and_compute_zigzag()` in
  `scripts/branches/zigzag_reservoir/run_zigzag_reservoir.py`) is a master-branch-friendly
  copy of the LLM module's logic. Reusable for any future zigzag work that doesn't want
  to depend on `att/llm/zigzag.py` being available.
- **The diagnostic** that a global VR-complex threshold can never simultaneously serve
  healthy and chaotic clouds is the actionable insight for any future zigzag work in
  this domain. A per-cloud-normalized distance (e.g., distances divided by per-cloud
  characteristic length) might rescue this — but that's a research-level modification,
  not a tracer bullet.

## Files

- `scripts/branches/zigzag_reservoir/run_zigzag_reservoir.py` — single-file ~390 lines including the inlined zigzag construction
- `data/zigzag_reservoir/results_sr_drift_seed42.json` — full results + baseline signal + verdict
- `data/zigzag_reservoir/zigzag_barcodes_sr_drift_seed42.npz` — H0/H1 barcodes + baseline signal
- `figures/zigzag_reservoir/zigzag_barcode_sr_drift_seed42.png` — barcode + baseline trace
- Cherry-picked: `scripts/reservoir_degradation_monitor.py` from `experiment/neuromorphic-snn`

## Caveats

1. **Two threshold trials only** (p30, p70). A *per-cloud* normalized distance (or
   cloud-rank-based threshold) might rescue zigzag for this specific signal but is
   architecturally ambitious — would need to modify the zigzag construction to use
   normalized distances per cloud.
2. **n_pts = 60 per cloud** for triangle-enumeration speed (60³ = 216k vs 200³ = 8M).
   Higher subsample gives more features but doesn't change the cross-regime issue.
3. **Single mode (sr_drift)**. The other modes (`noise`, `connection_death`) might
   preserve embedding scale better and give zigzag a fairer chance — `noise` in
   particular doesn't push the system into chaos. Worth a follow-up before fully
   killing the architecture.
4. **No comparison with `experiment/neuromorphic-snn` Phase 5 zigzag work** — that
   work was on LLM hidden states with `dim` independent of regime, a very different
   structural setting where zigzag *did* find cross-layer features. The architectural
   lesson here is that zigzag needs the underlying point cloud scale to be roughly
   constant across the time axis.

## Decision

**Kill zigzag for `sr_drift`-class signals.** Suggest one quick follow-up on the
`noise` mode (where the embedding shouldn't expand) before fully killing the
architecture for reservoir monitoring. If `noise` mode also yields no cross-window
features, the kill is complete.

## Next steps

- **Move on to Branch 1** (tiered-sentinel-cardiac) per the recommended branch order.
- **Optional**: re-run this script with `--mode noise` to see if a non-expanding
  drift gives zigzag any traction. Five-minute experiment. Same script, no code change.
- **Do not promote** the inlined zigzag helper to `att/topology/` — the existing
  `att/llm/zigzag.py` already has this (on the LLM branches) and should be moved to
  `att/topology/zigzag.py` independently if anything in the codebase needs zigzag
  outside of LLM analysis.

## One-line summary

> Zigzag temporal PH finds no cross-window H1 features at any threshold on reservoir
> SR drift; snapshot PH at the same stride detects perfectly at the exact onset.
> Drift expands the embedding ~30× in scale, breaking the union-complex
> assumption that zigzag tracking depends on.
