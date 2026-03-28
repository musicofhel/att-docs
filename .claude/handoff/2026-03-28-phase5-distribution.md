# ATT Phase 5: Distribution + Blog + Polish — Handoff

**Date**: 2026-03-28
**Repo**: `~/att-docs` (github.com/musicofhel/att-docs, private)

## What Was Done

### 1. Figure 8 Added to Preprint (Experiment 8: Cross-Region Binding)

Added Experiment 8 to `paper/preprint.tex`: Oz–Pz cross-region binding analysis on real EEG (Nie/Katyal/Engel 2023, Subject 1).

| Metric | Value |
|--------|-------|
| Binding range | 5.1–34.1 (6.7× variation) |
| Spearman rho (binding vs switch count) | 0.51, p=0.016 |
| High vs low activity binding | 24.5 vs 11.0 (2.2×, p=0.0014) |
| Changepoint-adjacent vs other | p=0.042 |
| Dominant homology | H0 > H1 |

Preprint now 17 pages, clean compile (0 errors, 0 warnings). Abstract, introduction contributions, experiments, results, and discussion all updated.

### 2. PyPI Build (att-toolkit v0.1.0)

- `python -m build` produces `att_toolkit-0.1.0-py3-none-any.whl` (43 KB) and `att_toolkit-0.1.0.tar.gz` (47 KB)
- `twine check` passes on both artifacts
- **Not uploaded** — needs PyPI account + API token. Command: `twine upload dist/* -u __token__ -p pypi-TOKEN`

### 3. GitHub Actions CI

Created `.github/workflows/ci.yml` with 3 jobs:
- **lint**: ruff check on Python 3.12
- **test**: pytest matrix (Python 3.10/3.11/3.12), excludes slow/witness/changepoint tests
- **build**: `python -m build` verification on Python 3.12

Triggers on push to master and PRs.

### 4. Sphinx Documentation

18 files in `docs/`:
- `conf.py`: furo theme, autodoc + napoleon (NumPy-style) + intersphinx + viewcode + mathjax
- `index.rst`: landing page with feature overview and toctree
- `quickstart.rst`: two worked examples (Lorenz fingerprinting + binding detection)
- `api/`: 11 module pages covering all public API (config, synthetic, embedding, topology, binding, surrogates, benchmarks, transitions, neuro, viz, cli)
- Mock imports for heavy optional deps (mne, gudhi, ripser, etc.)

Added `[docs]` extra to `pyproject.toml`: sphinx>=7.0, furo>=2024.1, sphinx-autodoc-typehints>=2.0.

### 5. Blog Post

`blog/post.md` — "Your Brain Is a Matrix of Chaos Attractors" (2,424 words).

Covers: attractor intuition, existing method limitations, the PI subtraction method, all 8 key results with figure references, honest limitations (N=1, borderline surrogates), future directions. Target: ML engineers, neuro grad students, TDA enthusiasts.

### 6. Interactive Streamlit Demo

`demo/app.py` — 3-page app:
1. **Attractor Explorer**: system selector + 3D plot + persistence diagram
2. **Binding Detection**: coupling slider + 4-panel PI images + significance test button
3. **Coupling Sweep**: mini sweep curve + optional TE overlay

All computations cached (`@st.cache_data`), deterministic (`set_seed(42)`), fast defaults (n_steps=5000, subsample=300, n_surrogates=19).

Run: `pip install -e ".[demo]" && streamlit run demo/app.py`

### 7. Multi-Subject EEG Batch Pipeline

- `scripts/batch_eeg.py`: auto-discovers subjects, runs transition detection + binding, computes precision/recall, aggregates to summary CSV. Flags: `--n-subjects`, `--skip-binding`, `--dry-run`, `--config`.
- `scripts/download_eeg.py`: download helper for UMN repository (placeholder URLs with instructions). Flags: `--subjects`, `--extract-only`, `--verify-only`.
- `configs/batch_eeg.yaml`: all pipeline parameters (bandpass, embedding, windows, tolerances).

### 8. README Polish

- 4 shields.io badges (PyPI, Python, License, CI)
- One-liner install prominent at top
- Better structure, documentation section, citation BibTeX
- GIF placeholder for later

## Test Suite

113 passed, 4 deselected (slow/witness/changepoint). All green in 379s.

## Files Created

| File | Description |
|------|-------------|
| `.github/workflows/ci.yml` | CI: lint + test + build |
| `docs/conf.py` | Sphinx config (furo, autodoc, napoleon) |
| `docs/Makefile` | Sphinx Makefile |
| `docs/make.bat` | Windows build script |
| `docs/index.rst` | Landing page |
| `docs/quickstart.rst` | Quickstart tutorial (2 examples) |
| `docs/api/index.rst` | API reference toctree |
| `docs/api/{config,synthetic,embedding,topology,binding,surrogates,benchmarks,transitions,neuro,viz,cli}.rst` | 11 API module pages |
| `docs/_static/.gitkeep` | Static assets placeholder |
| `blog/post.md` | Blog post draft |
| `demo/app.py` | Streamlit demo (3 pages) |
| `demo/README.md` | Demo run instructions |
| `scripts/batch_eeg.py` | Multi-subject batch pipeline |
| `scripts/download_eeg.py` | EEG data download helper |
| `configs/batch_eeg.yaml` | Batch pipeline config |
| `dist/att_toolkit-0.1.0-py3-none-any.whl` | Built wheel (43 KB) |
| `dist/att_toolkit-0.1.0.tar.gz` | Built sdist (47 KB) |
| `.claude/handoff/2026-03-28-phase5-distribution.md` | This file |

## Files Modified

| File | Changes |
|------|---------|
| `paper/preprint.tex` | Experiment 8 (cross-region binding), abstract (vii), discussion |
| `pyproject.toml` | Added [docs] extra (sphinx, furo, sphinx-autodoc-typehints) |
| `README.md` | Badges, one-liner install, structure, citation, docs section |

## What's Next

1. **PyPI upload**: Create PyPI account, generate API token, `twine upload dist/*`
2. **Host docs**: `cd docs && make html`, deploy to GitHub Pages or Read the Docs
3. **Run batch EEG**: Download remaining subjects, `python scripts/batch_eeg.py data/eeg/rivalry_ssvep/ results/ --n-subjects 5` to test, then full 84
4. **Publish blog**: dev.to / personal site, update README with link
5. **arXiv submission**: Final PDF review, source tarball, submit (when ready)
6. **CI green**: Push triggers first CI run — fix any issues that surface
