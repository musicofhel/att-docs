# ATT Phase 3: Sliding-Window Transitions + EEG Pipeline — Handoff

**Date**: 2026-03-27
**Repo**: `~/att-docs`

## What Was Done

### Core Implementation (Steps 1-6)

| Component | File | Status |
|-----------|------|--------|
| `get_fallback_params()` | `att/neuro/eeg_params.py` | Done — 4 bands, sfreq scaling |
| `embed_channel()` | `att/neuro/embedding.py` | Done — auto+fallback, metadata audit trail |
| Witness complex | `att/topology/persistence.py` | Done — maxmin landmarks + GUDHI EuclideanStrongWitnessComplex |
| `TransitionDetector` | `att/transitions/detector.py` | Done — sliding-window PH, shared-grid PI, L2 distances, CUSUM |
| `plot_transition_timeline()` | `att/viz/plotting.py` | Done — 2-panel: distances+changepoints, H1 entropy |
| `EEGLoader` | `att/neuro/loader.py` | Done — MNE-based, BDF/EDF/SET/FIF/.mat support |

### EEG Data Research (Step 7)

**Primary target identified**: Nie, Katyal & Engel (2023) binocular rivalry SSVEP
- 84 subjects, 34ch, 1024 Hz, continuous raw EEG with button-press switch markers
- UMN Data Repository: https://conservancy.umn.edu/handle/11299/257166
- Same lab as Katyal 2015 paper

**Wrong dataset IDs fixed in DATA.md**:
- ds003505 = Visual Evoked Potentials (NOT rivalry)
- ds002218 = Rhythm Omission (NOT Necker cube)
- ds004019 = Obesity/Arithmetic (NOT auditory bistability)

### Figure 6 Experiment (Step 9)

Synthetic bistable EEG (alpha↔theta switching, 40s at 256 Hz):
- Auto embedding succeeded: delay=9, dim=10, condition=3.15
- 194 windows, 7 changepoints detected via CUSUM
- **2/3 ground truth transitions detected within 2s tolerance**
- Image distance range: [10.7, 84.5]

### Preprint Updates (Step 11)

- Added Experiment 6 section (sliding-window transition detection)
- Updated abstract: added point (v) about transition detection
- Updated contributions: added sliding-window transition detection
- Updated limitations: "synthetic validation" (not "synthetic only")
- Updated future directions: references pipeline readiness for real EEG
- Added Nie et al. 2023 to references.bib
- Clean compile: 12 pages, 0 warnings

## Test Suite

117 total tests (113 non-slow + 4 slow), all green.

| File | Tests |
|------|-------|
| `test_neuro.py` | 8 (fallback params ×3, embed_channel ×3, EEGLoader ×2) |
| `test_transitions.py` | 4 non-slow + 1 slow (output, distances, per-window, not-fitted, changepoints) |
| `test_topology.py` | 10 + 1 slow witness complex |
| `test_viz.py` | 10 (including transition timeline) |
| Existing | 87 (benchmarks, binding, cli, config, embedding, surrogates, synthetic) |

## Files Created/Modified

### New (8):
- `att/neuro/eeg_params.py`
- `att/neuro/embedding.py`
- `att/neuro/loader.py`
- `att/transitions/detector.py`
- `tests/test_transitions.py`
- `tests/test_neuro.py`
- `notebooks/fig6_eeg_transitions.ipynb`
- `configs/eeg_bistable.yaml`

### Modified (8):
- `att/topology/persistence.py` — witness complex added
- `att/transitions/__init__.py` — exports TransitionDetector
- `att/neuro/__init__.py` — exports all neuro public API
- `att/viz/plotting.py` — plot_transition_timeline added
- `att/viz/__init__.py` — export added
- `paper/preprint.tex` — Experiment 6, abstract, contributions, limitations, future
- `paper/references.bib` — Nie et al. 2023
- `DATA.md` — fixed wrong dataset IDs, updated primary target

### Generated:
- `figures/fig6_eeg_transitions.{pdf,png}`
- `paper/preprint.pdf` (12 pages)

## What's Next

1. **Download real EEG data**: Nie/Katyal/Engel dataset from UMN (~11.66 GB). Run pipeline on Subject 1.
2. **Step 10 (stretch)**: Cross-region binding on EEG if transition detection shows signal
3. **PyPI packaging**: Version number, README, setup for `pip install att-toolkit`
4. **arXiv submission**: Final read-through, adjust any real-data results
