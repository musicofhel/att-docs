# Branch 1 — Tiered Sentinel Cardiac (Tracer Bullet)

**Date**: 2026-04-11
**Branch**: `experiment/tiered-sentinel-cardiac`
**Architecture**: #1 — Tiered sentinel (variance → entropy → TransitionDetector)
**Verdict**: **FAIL** with a clean architectural kill — persistence entropy structurally cannot discriminate cardiac arrhythmia at the 10-second window scale

## Motivation

ATT's existing cardiac experiment (`scripts/branches/cardiac_arrhythmia_topology.py`
on `experiment/tda-cardiac`) hits ~20% precision with NEGATIVE mean detection lag
on MIT-BIH arrhythmia onset. The negative lag means variance spikes pre-empt the
topology-based alarm. Hybrid architecture #1 from the 10-architecture document
proposes turning that bug into a feature: variance + ACF1 become the cheap *gate*
(Layer 1) for persistence entropy (Layer 2), and only entropy-confirmed windows
get the full TransitionDetector treatment (Layer 3).

This branch tests whether the tiered cascade beats either layer alone on
MIT-BIH records 200 and 207.

## What we did

1. Loaded MIT-BIH records 200 and 207 via `wfdb.rdrecord(..., pn_dir="mitdb")`. Bandpass
   0.5-40 Hz, downsample 360→128 Hz. Verbatim preprocessing from
   `scripts/branches/cardiac_arrhythmia_topology.py` (cherry-picked logic).
2. Slid 10-second windows (W=1280 samples, stride=640, 50% overlap) over each record →
   360 windows per record. Per-window ground truth: arrhythmic iff >50% non-N annotated
   beats fall in the window.
3. **Layer 1**: rolling variance + rolling ACF1 over the same 1280-sample windows.
   Z-score against the first 5 minutes (60 windows). CUSUM (k=0.5, h=5.0) on each
   z-score series. Trigger window iff either CUSUM fires.
4. **Layer 2**: persistence entropy of H1 (Takens-embed → ripser → PA) on triggered
   windows only. Use the H1 entropy of the first 60 windows as a baseline; confirm
   trigger iff `(entropy - baseline_mean) / baseline_std > 2.0`.
5. **Layer 3**: per-window PH transition score (Σ H1 lifetimes + max H1 lifetime) on
   confirmed windows only. Alarm iff score above 75th percentile of confirmed-window
   scores.
6. Three arms (apples-to-apples on the same windowing):
   - **A** topology-only — H1 entropy on EVERY window, 2σ alarm vs first-60 baseline.
   - **B** variance-only — Layer 1 alone (the CUSUM trigger output, no topology).
   - **C** tiered sentinel — full pipeline, alarm only if Layer 3 confirms.

## Results

### Record 207 (the messiest record per the existing baseline's stress test)

| Arm | Precision | Recall | F1 | Alarms |
|---|---|---|---|---|
| **A** topology only | 0.000 | 0.000 | **0.000** | 2 |
| **B** variance only | 0.882 | 0.045 | **0.085** | 17 |
| **C** tiered sentinel | 0.000 | 0.000 | **0.000** | 1 |

Key issue with record 207: **336 of 360 windows are arrhythmic** (93%). The "first 5 minutes
healthy baseline" assumption fails — the baseline region itself is mostly arrhythmic.
Z-scoring against arrhythmic statistics produces meaningless thresholds. Per the plan
("If 207 is too messy to detect anything, fall back to record 200"), repeated on 200.

### Record 200 (more balanced: 43 arrhythmic / 317 normal = 12% arrhythmic)

| Arm | Precision | Recall | F1 | Alarms |
|---|---|---|---|---|
| **A** topology only | 0.000 | 0.000 | **0.000** | 2 |
| **B** variance only | 0.310 | 0.209 | **0.250** | 29 |
| **C** tiered sentinel | 0.000 | 0.000 | **0.000** | 0 |

Arm B is *modest* — at the 10s window scale, rolling variance does pick up some
arrhythmias (precision 0.31, recall 0.21), but not most of them. **Arms A and C are zero
because Layer 2 confirms zero windows.**

### Diagnostic — why Layer 2 is structurally dead

Computed H1 persistence entropy on the first 80 baseline windows + all 43 arrhythmic
windows in record 200 separately:

```
baseline entropy (60 normal windows):  mean=3.311  std=0.363  range=[2.403, 4.108]
arrhythmic windows (43 windows):       mean=3.250  std=0.325  range=[2.403, 3.854]
```

**Arrhythmic mean is LOWER than baseline mean** (Δ = -0.061, 0.17 baseline σ). The
distributions overlap almost completely. The arrhythmic z-scores have:

```
mean = -0.17  (essentially centered on zero)
n above +2 = 0 / 43
n below -2 = 2 / 43
range: [-2.50, +1.50]
```

**Persistence entropy of an ECG window does not discriminate cardiac arrhythmia from
normal sinus rhythm at the 10-second scale.** The Takens-embedded reconstructions of
both rhythm types produce diagrams with essentially the same shape distribution.

The Layer 2 entropy gate is consequently structurally dead — it filters out *every*
candidate window the variance layer flags, because none have entropy z > 2. This is
not a tuning issue (testing inverted gates `entropy < -2` would catch only 2/43
arrhythmic windows). The topology layer simply isn't selective for this signal class
at this temporal granularity.

## Verdict

**FAIL** the tracer-bullet criterion on both records:
- ❌ `precision(C) >= 0.40` → 0.000 (zero alarms reach Layer 3 because Layer 2 kills them all)
- ❌ `f1(C) > max(f1_A, f1_B)` → 0.000 < 0.250

The failure is *architectural*, not parameter-tuning. The tiered sentinel premise is:
"variance fires false positives; topology can disambiguate." The Layer 2 diagnostic
above shows the second clause is empirically wrong on this signal — there is nothing
to disambiguate, because the topological feature distributions are statistically
indistinguishable.

## Why this matters (the negative result is informative)

The original cardiac experiment's NEGATIVE mean detection lag (variance pre-empts
topology) is fully explained by this finding:
- **Variance is doing real work** — it's the only source of the existing 20% precision number.
- **Topology converges to nothing different from variance** — because at the 10-second window
  scale, persistence entropy of normal and arrhythmic ECG are statistically equivalent.
- **The original experiment's precision is near-random** because both layers are firing on
  noise structure that happens to correlate weakly with arrhythmic windows (probably via
  amplitude transients), but neither layer has a coherent topological signature to lock onto.

The right design conclusion is: **don't gate variance with topology at the 10s
window scale**. Either use a much shorter window (1-2 second beat-by-beat, where rhythm
structure is exposed in the embedding directly), or use a non-topological feature for
the cheap second stage (e.g., RR-interval variability, which the existing experiment also
explored separately).

## Files

- `scripts/branches/tiered_sentinel/run_cardiac_tiered.py` — single-file ~430 lines
- `data/cardiac_tiered/results_record200.json` — full results, layer counts, arm scores
- `data/cardiac_tiered/results_record207.json` — record 207 results (preserved despite the baseline-corruption issue)
- `figures/cardiac_tiered/tiered_sentinel_record200.png` — 4-panel: ECG + variance z + entropy z + L3 score
- `figures/cardiac_tiered/tiered_sentinel_record207.png` — same for 207

## Caveats

1. **Two records only**, both arrhythmia-focused. The other ATT cardiac records
   (201, 210, 217) might give different results, but the structural diagnostic
   (persistence entropy of cardiac windows is rhythm-invariant) would replicate
   trivially since it's a property of how the cardiac state space is reconstructed,
   not of the specific record.
2. **10-second windows fix the temporal granularity**. A beat-by-beat windowing
   (~0.8s) might expose rhythm-class differences in the topology that wash out
   when averaging over many beats. Worth a follow-up if the architecture is to be
   fully buried.
3. **No surrogate test**. We're killing the architecture on raw F1; no
   phase-randomized null is needed because the diagnostic is at the
   feature-distribution level, not the F1 level.
4. **Layer 3 used a custom transition score**, not `att.transitions.detector.TransitionDetector`,
   because TD is built for streaming a long sequence — our Layer 3 fires on isolated
   confirmed windows, so a per-window scalar (Σ H1 + max H1) was the appropriate
   primitive.

## What's worth keeping

- **The diagnostic itself**: persistence entropy of Takens-embedded ECG is
  rhythm-class-invariant at the 10s scale. This is publishable as a cautionary note
  for any future ATT cardiac or biosignal work that wants to use entropy as a
  discriminative feature.
- **The Layer 1 + CUSUM primitives** (`rolling_variance`, `rolling_acf1`,
  `cusum_alarm`) are reusable for any future early-warning-signal scaffolding ATT
  picks up. Currently they live only in this script — promote to
  `att/transitions/early_warning.py` if a follow-up needs them.

## Decision

**Kill the tiered-sentinel direction for cardiac**. Document the entropy
diagnostic as a real architectural lesson. Suggest that any future cardiac topology
work must either:
1. Use a much shorter (beat-scale) window where the rhythm class is exposed in
   single-beat embedding shape, or
2. Use a different topological feature (max H1 lifetime, betti curve shape, RR-interval
   topology) and re-test the entropy-blindness property.

## Next steps

- **All 4 branches now complete**. Final summary handoff at
  `.claude/handoff/2026-04-11-hybrid-tracer-bullets-summary.md` is the next deliverable.
- **Optional**: re-test on record 100 (mostly normal) just to confirm Arm A behaviour
  on a clean baseline (~5 min experiment, no code change).
- **Promote `rolling_variance` / `rolling_acf1` / `cusum_alarm`** to
  `att/transitions/early_warning.py` IF a future branch needs them. Not yet.

## One-line summary

> Tiered sentinel fails on MIT-BIH cardiac arrhythmia: persistence entropy of
> 10s ECG windows is statistically identical between normal and arrhythmic
> rhythms (mean 3.31 vs 3.25, both with std ≈ 0.34), so the Layer 2 gate kills
> every Layer 1 trigger; arm B (variance only) is the only arm with non-zero F1
> (0.250) and arm A (topology only) is also dead at this window scale.
