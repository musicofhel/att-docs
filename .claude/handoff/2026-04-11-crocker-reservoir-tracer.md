# Branch 3 — Crocker Stack Reservoir (Tracer Bullet)

**Date**: 2026-04-11
**Branch**: `experiment/crocker-stack-reservoir`
**Architecture**: #10 — Crocker stack process control
**Verdict**: **PASS** (F1=0.988, baseline=0.06, target≥0.20)

## Motivation

ATT's existing reservoir continuous-ramp experiment (`scripts/reservoir_degradation_monitor.py`)
hits F1≈0.06 on spectral-radius drift detection with per-window ripser PH binding scores.
Hybrid architecture #10 from the 10-architecture document proposes treating the *full*
Betti-curve-over-time stack (a "Crocker matrix") as a 2D process-control image rather
than reducing each window to a scalar. This branch tests whether the 2D structure carries
detectable information that the per-window scalar collapses away.

## What we did

1. Generated `simulate_degradation(mode="sr_drift", seed=42)` from `scripts/reservoir_degradation_monitor.py` — 200-neuron MinimalESN, 5000 healthy + 10000 SR-drift steps (15000 total).
2. PCA-reduced the (15000, 200) state matrix to PC1 (a 1D series).
3. Slid 66 windows (W=2000, stride=200) over PC1; per window, Takens-embed (auto delay/dim), ripser → H0/H1 diagrams via `PersistenceAnalyzer`.
4. Built a shared filtration grid across all 66 windows for each homology dim (no `att.llm.crocker` dep — implemented inline since `att/llm/` doesn't exist on master). Filled `(100, 66)` Crocker matrices for H0 and H1.
5. Three detection arms on the H1 stack:
   - **A** column-wise L1 distance to baseline (mean of first 25 windows)
   - **B** image CUSUM (cumulative sum of pixel-wise deviations)
   - **C** total Betti mass per column
6. 2-sigma threshold from first 25 windows; F1 vs ground-truth (window i is drift iff `start_i >= 5000`).

## Results

| Arm | Precision | Recall | F1 | First alarm | Lag (windows) |
|---|---|---|---|---|---|
| **A** column L1 | 0.953 | 1.000 | **0.976** | w23 | -2 |
| **B** image CUSUM | 0.976 | 1.000 | **0.988** | w24 | -1 |
| **C** total betti mass | 1.000 | 0.829 | **0.907** | w32 | +7 |

- 41 drift / 25 healthy windows; onset window = 25.
- Best arm: **B (image CUSUM)** at F1=0.988 — **16.5× the baseline F1 of 0.06**.
- Runtime: 30.8s total, 0.38s/window avg.

### Mechanism

The H1 feature count collapses dramatically under SR drift:
- Healthy windows (start ≤ 4000): ~270–280 H1 features
- Drift windows (start ≥ 6000): ~20–80 H1 features

So the SR drift toward 1.4 (chaotic regime) is destroying the small-loop structure in
the Takens-embedded reservoir trajectory rather than enriching it. The Crocker stack
picks this up trivially because **every** filtration epsilon column shows the loss.
Even the dumbest arm (C, total Betti mass) gets F1=0.91; the cleverer arms (A, B)
exploit the per-epsilon shape and get near-perfect.

### Interpretation of arm winners

- **B (image CUSUM)** wins by a hair over **A**. The 2D structure is barely doing
  work beyond row-summed deviation — the action is mostly in *that windows look very
  different from baseline*, not in *which filtration scale changes most*.
- **C** lags by 7 windows because total mass is the slowest-changing feature; CUSUM/L1
  pick up shape change before total volume change.
- **The interesting verdict**: arm B ≈ arm A, so the document's "2D image-domain
  framing" is overengineered for *this specific signal*. A column-vector L1 is
  sufficient. The 2D framing might still pay off on signals where the *scale* of
  topological change carries information independent of magnitude.

### Why F1 is so high (and why that's still real)

A 2σ threshold + ground truth labelled by window-start is in-sample, but the alarm
signal is computed on independent data (the diagrams), not the labels. The slight
"pre-onset" alarms at w23/w24 are not false: window 23 starts at step 4600 and
overlaps drift by 1400/2000 = 70%, so the topology *is* already drifted — the
ground-truth labelling is the conservative one, not the alarm. If anything we're
underselling precision.

## Files

- `scripts/branches/crocker_reservoir/run_crocker_reservoir.py` — single-file implementation, ~270 lines
- `data/crocker_reservoir/results_sr_drift_seed42.json`
- `data/crocker_reservoir/crocker_h0_h1_sr_drift_seed42.npz` — full crocker matrices + grids + ground truth
- `figures/crocker_reservoir/crocker_stack_sr_drift_seed42.png` — H1 stack heatmap + 3 arm signal traces
- Pulled from sibling branch: `scripts/reservoir_degradation_monitor.py` (cherry-picked from `experiment/neuromorphic-snn`)

## Caveats

1. **Single seed**: This is a tracer bullet. The 0.988 F1 must be replicated under
   multi-seed + phase-randomized surrogates before any claim is publishable. Existing
   ATT discipline (cf. `experiment/financial-dynamics` kill criterion) requires this.
2. **Single mode**: Only `sr_drift` was tested. The `noise` and `connection_death`
   modes may show very different topological signatures. Easy follow-up: same script,
   different `--mode` flag.
3. **In-sample threshold**: 2σ from first-25-window stats is the same window region
   used for the labelling baseline. A held-out healthy region from a separate seed
   would be cleaner.
4. **No `att.llm.crocker` import**: The Crocker shared-grid logic is inlined (~25
   lines). If this branch graduates, lift `build_crocker()` into `att/topology/crocker.py`
   as a data-source-agnostic primitive — `att/llm/crocker.py` (on the TDA-LLM branches)
   has essentially the same logic and should be merged with it.
5. **Why ATT's existing F1=0.06 is so bad**: The existing reservoir monitoring uses
   per-window binding score (joint-vs-marginal PI residual), which integrates over the
   *whole* persistence image and washes out the H1 feature count collapse signal. The
   Crocker stack preserves the per-epsilon shape, which is where the signal lives.

## Next steps

- **Promote to branch-grade follow-up**: multi-seed, phase-randomized surrogates,
  ablations on window/stride/baseline-size, all 3 degradation modes. Should fit
  within 30 min runtime since per-seed is only 30s.
- **Proceed to Branch 2** (vineyard-binding-reservoir) — same data, exercises the
  binding detector instead of standalone PH. The 0.988 F1 here doesn't preclude
  Branch 2 — it answers a different question (joint vs marginal coupling).
- **Lift `build_crocker()` to `att/`** if the multi-seed follow-up confirms.

## One-line summary

> Crocker stack on PC1(reservoir state) detects SR drift with F1=0.988 vs baseline
> F1=0.06; H1 feature count collapses from ~280 to ~30 under drift; 2D image
> framing barely helps over 1D column-L1.
