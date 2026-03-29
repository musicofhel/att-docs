# Phase 12: Final ROADMAP Gap Closure — Handoff

**Date**: 2026-03-28
**Status**: COMPLETE — binding batch N=79, tutorial notebooks, Makefile

## What Was Done

### 1. Cross-Region Binding Batch (N=79)

Ran `batch_eeg.py` WITHOUT `--skip-binding` on all 84 subjects with 10 workers.
- 81/84 processed (same 3 failures: missing epochs for 3663, 3577, 3613)
- 79 subjects with behavioral data and non-degenerate binding
- 1 subject (3647) had NaN binding (embedding issue)

**Population results**:
| Metric | Value |
|--------|-------|
| Mean binding score | 18.3 ± 16.5 (median 14.0, range 1.1–87.7) |
| Mean binding-switch rho | 0.100 ± 0.263 (median 0.136) |
| Positive rho | 48/79 (60.8%) |
| Individually significant (p<0.05) | 9/79 (11.4%) |
| **Population t-test** | **t=3.35, p=0.001** |
| **Wilcoxon signed-rank** | **W=2204, p=0.001** |
| **Binomial test** | **p=0.036** |
| Cohen's d | 0.38 (small-to-medium) |

**Key finding**: Binding-switch correlation is a **population-level phenomenon** (p=0.001), not just a single-subject observation. Effect size is small but consistent.

### 2. Tutorial Notebooks

**`notebooks/tutorial_lorenz_walkthrough.ipynb`** (ROADMAP 1.13):
- 14 code/markdown cells, 28 total cells
- Covers: Lorenz generation → Takens embedding → PH → persistence images → binding detection → significance testing → uncoupled baseline
- All 14 cells pass nbval-lax (6 min runtime)

**`notebooks/tutorial_heterogeneous_timescales.ipynb`** (ROADMAP 1.14):
- 8 code/markdown cells, 15 total cells
- Covers: Rössler-Lorenz coupling → shared vs per-channel delays → condition numbers → binding comparison → quality gate → coupling sweep
- All 8 cells pass nbval-lax (35s runtime)

### 3. Makefile

Root-level `Makefile` with 9 targets: `help`, `test`, `test-all`, `test-slow`, `lint`, `lint-fix`, `docs`, `demo`, `notebook-test`, `clean`.

### 4. Preprint Updated

- Abstract: added item (viii) for population binding (N=79, p=0.001)
- Experiment 8 methods: now describes multi-subject scope
- Experiment 8 results: new "Multi-subject validation" paragraph with full statistics
- Limitations: updated from "remains single-subject" to population evidence with caveats
- Future directions: "multi-subject binding" → "multi-electrode connectivity"
- 22 pages, clean compile

## Git Log

```
[new commit] Phase 12: binding batch N=79, tutorial notebooks, Makefile
```

## Files Created

| File | Description |
|------|-------------|
| `notebooks/tutorial_lorenz_walkthrough.ipynb` | Lorenz end-to-end tutorial |
| `notebooks/tutorial_heterogeneous_timescales.ipynb` | Heterogeneous timescale tutorial |
| `Makefile` | Root Makefile with test/lint/docs/demo targets |
| `results/batch_eeg_binding/` | Full binding batch results (81 JSONs + summary) |
| `.claude/handoff/2026-03-28-phase12-final-gaps.md` | This handoff |

## Files Modified

| File | Changes |
|------|---------|
| `paper/preprint.tex` | N=79 binding results, abstract, limitations, future directions |
| `paper/preprint.pdf` | Recompiled (22 pages) |
| `results/batch_eeg/batch_eeg_summary.csv` | Replaced with binding-populated version |

## ROADMAP Gaps Closed

| Gap | ROADMAP ref | Status |
|-----|-------------|--------|
| Tutorial notebooks | 1.13, 1.14 | CLOSED |
| Makefile | 1.15 criteria | CLOSED |
| Cross-region binding batch | Phase 5.1 | CLOSED (N=79) |

## Remaining ROADMAP Gaps (User-Constrained or Cosmetic)

- Blog external publication (4.8) — written, needs manual publish to dev.to
- README GIF (4.9) — static PNG is fine
- arXiv submission (3.11) — user constraint
- PyPI upload (4.3) — user constraint
- Cross-barcode integration (5.3) — separate research direction
