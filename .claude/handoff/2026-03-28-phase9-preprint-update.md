# Phase 9: Preprint + Blog + Docs Update with Phase 6-8 Findings — Handoff

**Date**: 2026-03-28
**Status**: COMPLETE — preprint 21 pages, clean compile (0 errors, 0 warnings, 0 undefined refs)

## What Was Done

### 1. New Figure 9 Generated

`notebooks/fig9_zscore_calibration.py` → `figures/fig9_zscore_calibration.{pdf,png}`

- Panel (a): Z-score bar chart — Lorenz-Lorenz (all negative, red) vs Rössler-Lorenz (mostly positive, blue, 3/5 significant) at coupling=0.3
- Panel (b): Structural baseline growth — binding score at coupling=0 grows with data (45→127→164 at n=3k/10k/20k)

### 2. Preprint: Experiment 9 Added (Z-Score Calibration + Timescale Selectivity)

New experiment section with Figure 9 and Table 2 (12-cell z-score results from power_analysis.csv). Key findings:
- Lorenz-Lorenz: all 6 z-scores negative (-4.35 to -0.56), 0% detection
- Rössler-Lorenz: z-scores positive (+0.08 to +2.85), 3/5 significant at p≤0.05
- Interpretation: method is selectively sensitive to heterogeneous-timescale coupling

### 3. Preprint: Experiment 1 Strengthened (Structural Baseline)

Expanded the one-sentence baseline mention to a full paragraph explaining:
- Baseline is structural, not finite-sample (grows 45→127→164 with data size)
- Caused by dimensional mismatch between joint R^{m1+m2} and marginal spaces
- Cross-references new Experiment 9 and Figure 9b

### 4. Preprint: Twin Surrogates Added to Methods

New paragraph in Section 2.4 (Significance Testing) describing twin surrogates as an alternative to AAFT, with different null hypothesis. Citation: Thiel et al. (2006).

### 5. Preprint: Discussion Restructured

**New structure (was 3 subsections, now 4)**:
- 5.1 Novelty and Contributions — added bullet 5: timescale selectivity finding
- 5.2 Domain of Applicability (NEW) — timescale selectivity explanation, Kuramoto inverse effect (77× binding reduction), interpretation guidance
- 5.3 Limitations (EXPANDED) — added: ~30% CV + modest ensemble improvement, N-body contamination (2.5× inflation), zero same-timescale power
- 5.4 Future Directions (UPDATED) — diagram matching moved from speculative to "implemented, rho=0.2 with PI"; added adaptive z-score calibration

### 6. Preprint: Abstract + Introduction Updated

- Abstract: added (viii) z-score calibration + timescale selectivity
- Introduction contributions: added (ix) domain of applicability identification
- Experiments section: added Experiment 9 setup description

### 7. References Added

- Thiel et al. (2006) "Twin surrogates to test for complex synchronisation," Europhysics Letters
- Kuramoto (1984) "Chemical Oscillations, Waves, and Turbulence," Springer

### 8. Blog Post Updated

`blog/post.md`: Added ~500-word "Where It Works and Where It Doesn't" section after synthetic results:
- Structural baseline grows with data
- Same-timescale blind spot (zero power for Lorenz-Lorenz)
- Kuramoto inverse effect
- N-body contamination (2.5×)
- ~30% CV

Updated Limitations section: replaced generic items with specific Phase 7-8 findings.

### 9. Documentation Updated

- **API.md**: Added `n_ensemble`, `ensemble_scores`, `confidence_interval()`, z-score fields in `test_significance()`, `kuramoto_oscillators()`, interpretability warning
- **ARCHITECTURE.md**: Updated `BindingDetector.fit()` signature (n_ensemble), `test_significance()` return dict (z-score fields), added timescale selectivity note
- **README.md**: Added domain of applicability callout box after project description

## Files Created

| File | Description |
|------|-------------|
| `notebooks/fig9_zscore_calibration.py` | Figure 9 generation script |
| `figures/fig9_zscore_calibration.pdf` | Figure 9 (PDF) |
| `figures/fig9_zscore_calibration.png` | Figure 9 (PNG) |
| `.claude/handoff/2026-03-28-phase9-preprint-update.md` | This file |

## Files Modified

| File | Changes |
|------|---------|
| `paper/preprint.tex` | Experiment 9, expanded Exp 1, twin surrogates in Methods, restructured Discussion (4 subsections), updated Abstract (viii) + Intro (ix) |
| `paper/references.bib` | +2 entries (Thiel 2006, Kuramoto 1984) |
| `blog/post.md` | New "Where It Works" section (~500 words), updated Limitations |
| `API.md` | ensemble, z-score, kuramoto API entries |
| `ARCHITECTURE.md` | Updated BindingDetector contract (ensemble, z-score) |
| `README.md` | Domain of applicability callout |

## Preprint Summary

| Metric | Before | After |
|--------|--------|-------|
| Pages | 17 | 21 |
| Experiments | 8 | 9 |
| Figures | 8 | 9 |
| Tables | 1 | 2 |
| References | 17 | 19 |
| Contributions listed | 8 | 9 |
| Compile warnings | 0 | 0 |

## What's NOT in Scope

- arXiv submission (user excluded)
- PyPI upload (user excluded)
- Multi-subject EEG batch run (data not downloaded)
- New computational experiments (all results from existing Phase 7-8 result files)
