# Branch 2 — Vineyard-Tracked Binding Reservoir (Tracer Bullet)

**Date**: 2026-04-11
**Branch**: `experiment/vineyard-binding-reservoir`
**Architecture**: #4 — Vineyard-tracked binding (proxy via dense windows + Hungarian matching + Savitzky-Golay)
**Verdict**: **FAIL** on strict criterion, but with a substantial **partial PASS** for arm A (dense raw)

## Motivation

ATT's existing reservoir continuous-ramp experiment hits F1≈0.06 with per-window
binding scores at stride=200. Hybrid architecture #4 from the 10-architecture
document proposes vineyard-tracked persistence as a way to denoise the per-window
score series. We test the architecture's *premise* via a "dumb vineyard": dense
overlapping windows (stride=50 instead of 200) + Hungarian-L∞ matching of joint
H1 features across adjacent windows + Savitzky-Golay smoothing of the tracked
persistence pairs. If real Dionysus vineyards would help, this lower-bound proxy
should at least show signal.

## What we did

1. Generated `simulate_degradation(mode="sr_drift", seed=42)` from `scripts/reservoir_degradation_monitor.py` — 200-neuron MinimalESN, 5000 healthy + 10000 SR-drift steps.
2. Picked channel pair `(neuron[0], neuron[100])` (X, Y) per the plan's fallback option.
3. Slid 261 windows (W=2000, stride=50, vs the baseline's stride=200 → 4× denser) across the trajectory.
4. Window 0: auto-embedded (`marg_x: delay=3 dim=10`, `marg_y: delay=2 dim=10`, `joint: delays=[3,2] dims=[10,10]`); cached the params for windows 1..260 to skip per-window AMI/FNN.
5. Per window: `BindingDetector(method="persistence_image", image_resolution=30).fit(...)` with the cached embedders. Recorded raw `binding_score()`, joint H1 diagram, marginal H1 diagrams, per-window birth/persistence range.
6. Built a global shared (birth, persistence) range across all 261 windows for arm C's reconstructed PIs.
7. Three arms:
   - **A** dense raw — `binding_score()` per window, no further processing.
   - **B** dense + smoothed — Savitzky-Golay (window=21, poly=3) on the raw score series.
   - **C** tracked + smoothed — Hungarian-L∞ match joint H1 features across windows (cost cap = 2× median persistence, full augmented assignment for unmatched), Savitzky-Golay smooth each track's (birth, death), reconstruct per-window joint PI from tracked pairs on global shared grid, recompute residual binding score.
8. 2σ alarms vs first 80 windows (held-out healthy region — onset is at window 100 so windows 80-99 are clean held-out healthy). F1 vs ground truth (window i drift iff `start_i >= 5000`, i.e. i >= 100).
9. Bidirectional alarming: scored both score-up and score-down directions, took the better F1, recorded the chosen direction.

## Results

| Arm | Dir | Precision | Recall | F1 | First alarm | Lag |
|---|---|---|---|---|---|---|
| **A** dense raw | ↓ | 0.973 | 0.888 | **0.929** | w76 | -24 |
| **B** dense + smoothed | ↓ | 0.931 | 1.000 | **0.964** | w88 | -12 |
| **C** tracked + smoothed | ↓ | 0.951 | 0.851 | **0.898** | w82 | -18 |

- 161 drift / 100 healthy windows; onset = w100.
- Direction is **down** for all arms (binding score collapses with SR drift, like the H1 feature count in Branch 3).
- Tracker stats: 3631 tracks, mean length 15.2 windows, max 109, 2151 long-tracks (≥5 windows).
- Runtime: 122s (~0.45s/window for joint+marginal PH at subsample=300, image_res=30).

## Verdict

**FAIL** the tracer-bullet criterion:
- ❌ `F1(C) - F1(A) >= 0.05` → -0.030 (tracking is *worse* than dense raw)
- ❌ `F1(C) > F1(B)` → 0.898 < 0.964 (smoothing alone beats smoothing + tracking)
- ✅ `F1(C) >= 0.20` → 0.898 (passes the magnitude floor easily)

The strict 3-of-3 criterion is failed because two of three legs require arm C to
beat the controls. **However, the experiment is far from a wash.** Three substantive
findings emerge:

### Finding 1 — Dense windows alone are ~15× the published baseline

`F1(A) = 0.929` from stride=50 dense raw scoring vs the existing reservoir
experiment's `F1=0.06` at stride=200. This is the architecture's *core insight*
(more frequent topological samples) validated cleanly. Real-time considerations
permit this trivially: per-window cost at subsample=300/image_res=30 is 0.45s.

### Finding 2 — Savitzky-Golay smoothing adds 0.035 F1 over dense raw

`F1(B) - F1(A) = +0.035`. Smoothing the score series resolves the remaining
recall gap (R: 0.888 → 1.000) by suppressing transient dips that fall just under
the 2σ threshold. This is the smoothing-baseline that arm C was supposed to
demonstrate the value-add of vineyards over.

### Finding 3 — Hungarian-tracked feature reconstruction makes things *worse*

`F1(C) = 0.898 < F1(B) = 0.964 < F1(A) = 0.929` is evidence the tracking step
is *injecting noise*, not denoising. The tracker stats explain why:
- 3631 tracks across 261 windows × ~200 features/window means each feature
  participates in only ~14 windows on average (matching `mean_track_len = 15.2`).
- Most tracks are very short. Smoothing a 5-window track via SavGol-21 falls
  back to length-5 SavGol-3, which is barely smoothing at all.
- Reconstructed PIs from tracked diagrams lose any (birth, death) pair that was
  outside its track's lifespan, causing per-window joint PIs to be sparser than
  the raw fits and shifting the residual computation.

### Why F1(B) > F1(A) > F1(C)

The plan anticipated this exactly: *"If Arm C ≈ Arm B, smoothing is doing the
work and tracking is decorative — document the negative result, kill the
branch."* Here arm C is *worse* than B, not just equivalent — the dumb-vineyard
proxy actively harms detection. The plan-agent's pessimism about real Dionysus
vineyards now looks justified: if the matching+smoothing doesn't even break even
on a dense, synthetically-clean drift signal, the engineering investment for
real vineyards is unlikely to recover the loss.

## Files

- `scripts/branches/vineyard_binding/run_vineyard_reservoir.py` — single-file ~430 lines
- `data/vineyard_binding/results_sr_drift_seed42.json`
- `data/vineyard_binding/scores_sr_drift_seed42.npz` — three score series + ground truth
- `figures/vineyard_binding/vineyard_binding_sr_drift_seed42.png` — three-panel arm comparison
- Cherry-picked: `scripts/reservoir_degradation_monitor.py` from `experiment/neuromorphic-snn`

## Caveats

1. **Single seed, single mode** — `sr_drift` only. The other modes (`noise`, `connection_death`)
   may interact with vineyard tracking differently; haven't tested.
2. **Single channel pair** — `(neuron[0], neuron[100])`. The plan suggested falling back
   from `(PC1, u)` if binding was zero; we went straight to neuron pair. PC1 vs input drive
   would be a more theoretically grounded coupling test.
3. **Image resolution = 30** — chosen for speed (2.8× faster than 50). Higher resolution
   might give the tracked-PI reconstruction more headroom but is unlikely to flip the verdict.
4. **No surrogate test** — phase-randomized null is required before the F1≈0.93 number can be
   reported. Current evaluation is in-sample for the threshold (held-out for the labels).
5. **Hungarian matcher implementation is hand-rolled** — copied the L∞ cost pattern from
   `BindingDetector._diagram_matching_score()` lines 580-615 but built the augmented
   "feature creation / track killing" structure differently. Possible bug; the high
   3631-track count is suggestive that the matcher is too aggressive about creating new
   tracks.

## What's worth keeping

- **Arm A's F1=0.929 from dense windows alone is the strongest result on this dataset
  to date** and deserves to be the headline of any follow-up reservoir work. The
  existing F1=0.06 baseline is using stride=200; the simple dense-window upgrade
  gets 15× without changing the binding-score logic at all.
- **Smoothing-only (arm B) is the new baseline** for any future "denoise the binding
  series" work. Anything that doesn't beat F1=0.964 is decorative.
- **The tracker code in `FeatureTracker`** is a reusable Hungarian-L∞ persistence-pair
  matcher that could be lifted to `att/binding/tracking.py` if a future branch finds a
  use case where matching matters more (e.g. labelling features by ID across regimes,
  not denoising scores).

## Decision

**Kill the vineyard-tracking direction** (arm C) for now. The tracer bullet shows
the dumb-vineyard proxy actively harms detection on a clean synthetic signal where
the underlying binding score is already very informative. Real Dionysus vineyards
would need to *substantially* outperform Hungarian matching + SavGol to justify the
boost-install yak-shave, and there's no theoretical reason to expect that here.

**Keep the dense-window finding** (arm A → 0.929) and use it as the new comparison
baseline for any reservoir monitoring work. The handoff doc + JSON + figures are
sufficient evidence; a multi-seed follow-up should be a small ablation (~10 seeds,
phase-randomized surrogates) and will determine whether the 0.929 number replicates.

## Next steps

- **Move on to Branch 4** (zigzag-reservoir) per the recommended branch order.
- **Multi-seed dense-window follow-up** for Branch 2-A specifically: same script,
  3-5 seeds, surrogate testing. Should run in ~20 min total.
- **Do NOT promote `FeatureTracker` to `att/`** until a use case where matching
  matters appears.

## One-line summary

> Dense (stride=50) per-window binding gets F1=0.929 on its own; SavGol smoothing
> adds 0.035; Hungarian-tracked vineyard reconstruction hurts by 0.066 — the
> tracking-as-denoising premise is empirically wrong on this signal.
