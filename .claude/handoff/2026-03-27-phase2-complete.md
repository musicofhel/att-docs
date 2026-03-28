# ATT Phase 2 Complete — Handoff

**Date**: 2026-03-27
**Repo**: `~/att-docs`
**Session**: Phase 2 finalization (notebooks + preprint draft)

## What Was Done

Phase 2 is now fully complete: all code (from prior session) + 5 paper figure notebooks + preprint draft.

### New Files (8)

| File | What It Is |
|------|-----------|
| `notebooks/fig1_coupling_sweep.ipynb` | **Paper Figure 1**: Coupled Lorenz binding score vs coupling (unimodal curve), H0/H1 breakdown |
| `notebooks/fig2_baseline_comparison.ipynb` | **Paper Figure 2**: Max vs sum baseline comparison across coupling sweep |
| `notebooks/fig3_benchmark_overlay.ipynb` | **Paper Figure 3**: All 4 methods (binding, TE, PAC, CRQA) rank-normalized overlay via CouplingBenchmark.sweep() |
| `notebooks/fig4_surrogate_null.ipynb` | **Paper Figure 4**: Surrogate null distributions for uncoupled (p>0.05) and coupled (p<0.05), 99 AAFT surrogates |
| `notebooks/fig5_heterogeneous_timescales.ipynb` | **Paper Figure 5**: Rossler-Lorenz with per-channel vs shared delays |
| `paper/preprint.tex` | LaTeX preprint draft: abstract, intro, background, method (Algorithm 1), 5 experiment descriptions, discussion, limitations |
| `paper/references.bib` | BibTeX bibliography (18 entries) |
| `figures/` | Empty directory for notebook outputs |

### Notebook Design

- All notebooks use `seed=42`, `n_steps=8000`, `subsample=500` for reproducibility
- Each saves PDF + PNG to `figures/` at 300 DPI
- Fig 3 also saves raw CSV data
- Fig 4 uses 99 surrogates (publication-standard for p < 0.01 resolution)
- Fig 5 compares auto delays vs shared delay=10 on heterogeneous system

### Preprint Structure

1. Abstract (complete)
2. Introduction — positions joint-vs-marginal PH as novel (complete)
3. Background — Takens embedding, persistent homology (complete)
4. Method — Algorithm 1 pseudocode, baseline choice analysis, quality gate, surrogates (complete)
5. Experiments 1-5 — descriptions and expected results (complete)
6. Results — placeholder sections keyed to figures (to fill after running notebooks)
7. Discussion — novelty, limitations, future directions (complete)
8. references.bib — 18 entries covering Takens, PH, surrogates, benchmarks

## What's Next

**Phase 2 is done.** Remaining work is Phase 3 (EEG + preprint submission):
- Run the 5 notebooks to generate figures
- Fill in Results sections of preprint with actual numbers
- Phase 3: EEG data, sliding-window PH, preprint finalization + arXiv submission

## Test Suite

102 tests, all green (~140s).
