# ATT Phase 4: Real EEG Validation + PyPI Packaging — Handoff

**Date**: 2026-03-28
**Repo**: `~/att-docs`

## What Was Done

### 1. Real EEG Data Download (Nie/Katyal/Engel 2023)

Downloaded Subject 1 from UMN Data Repository (DOI: 10.13020/9sy5-a716).
- **Path**: `data/eeg/rivalry_ssvep/Sucharit - 012516_3629/`
- **Format**: MATLAB .mat files (scipy.io.loadmat compatible)
- **Actual specs**: 34 channels, **360 Hz** (preprocessed, not 1024 Hz raw), CSD/ICA/artifact rejection already applied
- **Structure**: `Epochs/` (6 .mat files: rivalry + SFM + fixation + rest), `Behavior/` (button-press .mat files)
- **Per epoch**: shape (34, ~43195) float64, ~120s per session, 12 rivalry sessions total
- All 3 zip parts MD5-verified against repository checksums

### 2. Transition Detection on Real EEG (Figure 7)

Pipeline: Oz channel -> theta-alpha bandpass (4-13 Hz) -> embed_channel(auto) -> TransitionDetector -> CUSUM

| Metric | Value |
|--------|-------|
| Embedding | auto: delay=12 (33.3ms), dim=6, cond=3.56 |
| Windows | 214 (size=500, step=200, subsample=300) |
| Changepoints | 7 via CUSUM |
| Behavioral switches | 41 (29 clear, 12 mixed) |
| Precision @ 3s | **100%** (7/7, zero false alarms) |
| Recall @ 3s | 41.5% (17/41) |
| Recall @ 5s | 80.5% (33/41) |

**Key insight**: Detector captures major topological reorganizations (switch clusters), not individual alternations. H1 entropy modulation tracks switch density.

### 3. Cross-Region Binding on Real EEG (Figure 8)

Pipeline: Oz + Pz channels -> shared embedding (delay=12, dim=5) -> BindingDetector sliding windows (10s x 5s step)

| Metric | Value |
|--------|-------|
| Binding range | 5.1 -- 34.1 (6.7x variation) |
| Spearman rho (binding vs switch count) | **0.51, p=0.016** |
| High vs low activity binding | 24.5 vs 11.0 (2.2x, p=0.0014) |
| Changepoint-adjacent vs other | p=0.042 |
| Dominant homology | H0 (component merging/splitting) > H1 (loops) |

**Key insight**: Occipital-parietal binding modulates with perceptual instability. Joint topology captures coupling absent from either marginal.

### 4. Preprint Polish (paper/preprint.tex)

All issues from a 26-item review addressed:

**Critical**: AAFTR -> AAFT (paper + all code), Experiment 6 added to Experiments section, auto-with-fallback strategy defined in Method

**High**: 3 uncited refs now cited (Ripser, CRQA, PAC), 4 unused refs removed, PNG -> PDF figures, algorithm \Call macros

**Medium**: TE range corrected (1.7x not 2x), notation standardized (D for ambient dim, m for embedding dim), software URL added, bridge between Exp 1-5 and Exp 6, acronyms defined (CRQA, PAC)

**Real EEG additions**: Experiment 7 (real EEG transition detection), Figure 7 + caption, abstract updated (no "synthetic only"), limitations updated (N=1), future directions (84-subject dataset)

**Final**: 15 pages, clean compile (0 warnings), Gramfort 2013 MNE ref added

### 5. PyPI Packaging

- `pyproject.toml`: name=att-toolkit, version=0.1.0, optional extras [eeg], [gudhi], [all], CLI entry point
- `README.md`: install instructions, quick start, API overview, citation BibTeX
- `LICENSE`: MIT
- Verified: `pip install -e .`, `att --help`, `import att; att.__version__`

### 6. Codebase Cleanup

- **AAFTR -> AAFT** renamed across 8 files (5 source/test/doc + 3 handoff)
- **DATA.md** updated with actual 360 Hz preprocessed specs, corrected directory tree
- **.gitignore** created (Python defaults + data/eeg/ exclusion)

## Test Suite

115 passed (non-slow), 2 deselected (witness complex + changepoint slow tests). All green.

## Files Created

| File | Description |
|------|-------------|
| `notebooks/fig7_real_eeg.ipynb` | Real EEG transition detection pipeline |
| `notebooks/fig8_eeg_binding.ipynb` | Cross-region Oz-Pz binding analysis |
| `figures/fig7_real_eeg.{pdf,png}` | 3-panel: signal + PI distances + H1 entropy |
| `figures/fig8_eeg_binding.{pdf,png}` | 5-panel: binding timecourse + correlation + comparison |
| `README.md` | PyPI-ready README |
| `LICENSE` | MIT license |
| `.gitignore` | Python + data exclusions |
| `.claude/handoff/2026-03-28-phase4-real-eeg.md` | This file |

## Files Modified

| File | Changes |
|------|---------|
| `pyproject.toml` | att-toolkit v0.1.0, extras, CLI entry point, URLs |
| `paper/preprint.tex` | AAFT, Exp 6/7, real EEG results, Fig 7, abstract, limitations |
| `paper/references.bib` | Added Gramfort 2013, removed 4 unused entries |
| `att/surrogates/core.py` | AAFTR -> AAFT in docstring |
| `tests/test_surrogates.py` | AAFTR -> AAFT in docstring |
| `ARCHITECTURE.md` | AAFTR -> AAFT |
| `ROADMAP.md` | AAFTR -> AAFT |
| `DATA.md` | Actual 360 Hz specs, corrected directory tree, risk table |
| `.claude/handoff/2026-03-27-phase{1,2}-*.md` | AAFTR -> AAFT (3 files) |

## What's Next

1. **Multi-subject analysis**: Run pipeline on all 84 subjects, compute group statistics
2. **Add Figure 8 to preprint**: Cross-region binding results (Experiment 8)
3. **PyPI publish**: `python -m build && twine upload dist/*`
4. **arXiv submission**: Final PDF review, source tarball, submit
5. **Binding during stable vs switching epochs**: Use multiple rivalry sessions per subject for within-subject comparison
