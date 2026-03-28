# Phase 10: Commit, Fix URLs, Polish — Handoff

**Date**: 2026-03-28
**Status**: COMPLETE — CI fully green, repo private

## What Was Done

### 1. Committed Phase 7-9 Work (was uncommitted)

Two commits protecting 1,291 lines of uncommitted work:

**Commit A** (`ab9b2c9`): Phase 7-8 — method hardening + test expansion
- `att/binding/detector.py`: ensemble binding, z-score calibration, embedding param caching
- `tests/test_binding.py`: 62 new edge-case tests
- `tests/test_validation.py`: 33 validation experiments (slow)
- 3 handoff docs (Phase 6-8)

**Commit B** (`155e06d`): Phase 9 — preprint update
- `paper/preprint.tex`: Experiment 9, expanded Discussion, Abstract/Intro updates
- `paper/references.bib`: +2 citations (Thiel 2006, Kuramoto 1984)
- `paper/preprint.pdf`: compiled PDF (21 pages)
- `blog/post.md`, `API.md`, `ARCHITECTURE.md`, `README.md`: Phase 6-8 findings
- `notebooks/fig9_zscore_calibration.py` + `figures/fig9_zscore_calibration.{pdf,png}`

### 2. Fixed 11 Broken URLs

All repo references standardized to `musicofhel/att-docs` (was `attractor-topology-toolkit` or `att-toolkit`) and branch `master` (was `main`):

| File | Fixes |
|------|-------|
| `pyproject.toml` | 3 URLs (Repository, Documentation, Bug Tracker) |
| `blog/post.md` | 4 URLs + 1 display text + 1 branch ref |
| `docs/conf.py` | source_repository URL + source_branch |
| `BLOG.md` | 1 URL |
| `paper/preprint.tex` | 1 URL |

### 3. README Demo Figure

`scripts/generate_readme_figure.py` → `figures/readme_demo.png` (249KB)

4-panel static composite:
- (a) 3D Lorenz attractor
- (b) Binding image (residual PI heatmap, coupled Rössler-Lorenz ε=0.3)
- (c) Coupling sweep (binding score vs coupling strength)
- (d) Method comparison (4 methods, rank-normalized)

README TODO placeholders resolved: demo figure embedded, blog post linked, preprint linked.

### 4. CONTRIBUTING.md + CHANGELOG.md

- `CONTRIBUTING.md`: dev setup, test commands (fast/slow), code style, adding coupling methods, adding synthetic systems, PR workflow
- `CHANGELOG.md`: retroactive v0.1.0 entry covering all 9 phases

### 5. CI Verified Green + Repo Metadata

- Latest commit (`76f002a`): all CI jobs pass (lint + build + test on Python 3.10/3.11/3.12)
- All Docs workflows pass (GitHub Pages deploys)
- Homepage set: `https://musicofhel.github.io/att-docs/`
- 9 topics added: tda, persistent-homology, dynamical-systems, topological-data-analysis, coupling-detection, takens-embedding, neuroscience, eeg, python
- **Repo remains PRIVATE** (user decision)

### 6. Ruff Lint Fix

`scripts/generate_readme_figure.py` had 3 unused imports. Fixed in commit `76f002a`.

## Git Log

```
76f002a Fix ruff lint: remove unused imports in generate_readme_figure.py
89d3953 Phase 10: fix broken URLs, README demo figure, CONTRIBUTING + CHANGELOG
155e06d Phase 9: preprint update — 21 pages, 9 experiments, z-score calibration
ab9b2c9 Phase 7-8: method hardening + test expansion
e12c6ce Add 82 tests (134→216): edge cases, validation experiments, bug fix
8875a8a Phase 6: twin surrogates, Kuramoto oscillators, diagram matching, docs deployment
c062b60 Phase 5: distribution, docs, blog, demo, batch pipeline
b30f308 ATT Phases 1-4: full toolkit, real EEG validation, preprint, PyPI packaging
b3fbc28 Initial commit: add AT&T docs
```

## Files Created

| File | Description |
|------|-------------|
| `CONTRIBUTING.md` | Contributor guide |
| `CHANGELOG.md` | Retroactive v0.1.0 changelog |
| `scripts/generate_readme_figure.py` | README demo figure generator |
| `figures/readme_demo.png` | 4-panel README demo (249KB) |
| `.claude/handoff/2026-03-28-phase10-polish.md` | This file |

## Files Modified

| File | Changes |
|------|---------|
| `pyproject.toml` | 3 broken URLs fixed |
| `blog/post.md` | 4 broken URLs + branch ref fixed |
| `docs/conf.py` | source_repository + source_branch fixed |
| `BLOG.md` | 1 broken URL fixed |
| `paper/preprint.tex` | 1 broken URL fixed |
| `README.md` | Demo figure embedded, blog + preprint linked, 3 TODOs resolved |

## What's NOT Done

- Repo is **private** (user decision)
- PyPI upload excluded
- arXiv submission excluded
- Multi-subject EEG batch (data not downloaded, scripts ready at `scripts/batch_eeg.py`)
- preprint.pdf not re-committed after URL fix (cosmetic, previous version in repo)

## Possible Next Steps

1. **Make repo public** when ready for external visibility
2. **Multi-subject EEG** — download remaining 83 subjects, run `scripts/batch_eeg.py`
3. **Cross-barcode integration** (R-Cross-Barcode on VR complexes)
4. **Adaptive z-score calibration** — faster significance without full surrogate runs
5. **Blog publication** — dev.to / personal site cross-post
