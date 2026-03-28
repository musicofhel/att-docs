# ATT Figures Generated & Preprint Results Filled — Handoff

**Date**: 2026-03-27
**Repo**: `~/att-docs`
**Session**: Run all 5 notebooks, fill in preprint Results section

## What Was Done

### 1. Executed All 5 Paper Notebooks

All notebooks ran successfully via `jupyter nbconvert --execute`, generating figures to `figures/`.

| Notebook | Output Files | Key Result |
|----------|-------------|------------|
| `fig1_coupling_sweep` | PDF + PNG | Score peaks at ε≈0.1 (257.2), drops to 89.3 at ε=1.0. H0 dominates (~87%). Non-zero baseline at ε=0 (184.2). |
| `fig2_baseline_comparison` | PDF + PNG | Max and sum baselines differ by <4%. Choice is negligible for Lorenz. |
| `fig3_benchmark_overlay` | PDF + PNG + CSV | Binding score has 2.9× dynamic range. TE/PAC/CRQA essentially flat across coupling. |
| `fig4_surrogate_null` | PDF + PNG | Uncoupled p=0.250 (correct). Coupled p=0.060 (borderline, not <0.05). |
| `fig5_heterogeneous_timescales` | PDF + PNG | Auto delays: joint κ=51.5. Shared: joint κ=2671. Auto scores increase monotonically. |

### 2. Filled Preprint Results Section

**File**: `paper/preprint.tex`

- Wrote 5 Results subsections with actual numbers, figure references, and honest interpretation
- Added Table 1 (raw benchmark scores at 4 coupling values)
- Adjusted abstract: "unimodal" → "coupling-dependent" to match observed noisy-but-real response
- Updated Experiment 1 expected result to note non-zero ε=0 baseline
- Fixed undefined `\ref{sec:benchmarks}` → `\ref{sec:exp-benchmarks}`
- Clean compile: 11 pages, 0 warnings

### 3. Observations Worth Noting

- **Non-zero ε=0 baseline**: The binding score at zero coupling is 184, not near zero. This is inherent to PI subtraction — the joint PI of independent systems ≠ max of marginal PIs. The *relative* change matters.
- **Noisy mid-range**: Binding scores at ε∈[0.2, 0.8] oscillate substantially (167–246). Single-point scores need surrogate context.
- **Borderline significance**: p=0.060 at ε=0.5 with n=6000/subsample=400. Longer series or stronger coupling would likely reach significance. The false positive rate (p=0.250 at ε=0) is well controlled.
- **Benchmark result is strong**: Binding score is the *only* method with meaningful dynamic range on coupled Lorenz. TE/PAC/CRQA are flat.
- **Heterogeneous timescale result is strong**: 52× condition number difference (51 vs 2671) clearly demonstrates per-channel delay advantage.

## Files Changed/Created

| File | Action |
|------|--------|
| `figures/fig1_coupling_sweep.{pdf,png}` | Created |
| `figures/fig2_baseline_comparison.{pdf,png}` | Created |
| `figures/fig3_benchmark_overlay.{pdf,png}` | Created |
| `figures/fig3_benchmark_data.csv` | Created |
| `figures/fig4_surrogate_null.{pdf,png}` | Created |
| `figures/fig5_heterogeneous_timescales.{pdf,png}` | Created |
| `paper/preprint.tex` | Updated (Results filled, abstract adjusted, ref fixed) |
| `paper/preprint.pdf` | Generated (11 pages, clean compile) |
| `notebooks/fig1-5` | Updated (contain execution outputs) |

## Test Suite

102 tests, all green (~138s).

## What's Next

**Phase 3**: EEG data + preprint finalization + arXiv submission
- Sliding-window binding detection on EEG
- Consider: longer time series / stronger coupling for Fig 4 to get p < 0.05
- Consider: smoothing or averaging over seeds for Fig 1 to reduce mid-range noise
- Fill in Software Availability URL once published to PyPI
- arXiv submission
