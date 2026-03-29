# Phase 11: Multi-Subject EEG Validation — Handoff

**Date**: 2026-03-28
**Status**: COMPLETE — 81/84 subjects processed, preprint updated

## What Was Done

### 1. EEG Data Extraction (84 subjects)

Downloaded from `C:\Users\aaron\Downloads\EEGDATA` (UMN DRUM, Nie/Katyal/Engel 2023).
12 GB multi-part zip (eegdata.zip + .z01 + .z02) extracted with `7z` (Linux `unzip` failed on split archives — only got 28/85 subjects).
Result: 85 subject dirs, 84 with preprocessed Epochs/ (1 subject had raw CNT only).

### 2. Precision/Recall Bug Fix

`evaluate_alignment()` in `scripts/batch_eeg.py` had a bug: "precision" was computed as
`switches_matched / n_changepoints` instead of `changepoints_matched / n_changepoints`.
When multiple switches clustered near one changepoint, precision exceeded 100%.

**Fix**: Separate `true_positives` (changepoints with nearby switch) from `switches_detected`
(switches with nearby changepoint). Precision = true_positives / n_changepoints.

### 3. Parallel Processing

Added `--workers N` flag using `concurrent.futures.ProcessPoolExecutor`.
- Module-level `_worker()` function for pickling
- Path serialization (Path → str → Path) for cross-process transfer
- Sequential mode preserved for `--workers 1`
- 10 workers on 28-core machine: **77 min** (was ~8h sequential)

### 4. Multi-Subject Results (N=80)

| Metric | Value |
|--------|-------|
| Subjects processed | 81/84 (3 missing epochs) |
| Subjects with behavioral data | 80 |
| Mean changepoints/epoch | 6.3 ± 3.0 (range 3-16) |
| Mean switches/epoch | 46.9 ± 18.3 |
| **Precision @ 5s** | **94.1% ± 14.9% (median 100%)** |
| **Recall @ 5s** | **40.6% ± 13.9% (median 36.6%)** |
| Precision @ 3s | 88.4% ± 20.6% |
| Recall @ 3s | 27.0% ± 10.7% |
| Perfect precision (5s) | 64/80 subjects (80%) |
| Recall > 30% (5s) | 61/80 subjects (76%) |

**3 failures**: Subjects 3663, 3577, 3613 — missing `riv_12` epoch files.

### 5. Preprint Updated

- Abstract: N=80 results replace N=1 summary
- Experiment 7 methods: now describes full dataset scope
- Experiment 7 results: new paragraph with population statistics
- Limitations: "single subject" caveat replaced with "binding is still N=1"
- Future work: multi-subject transition detection marked complete
- 21 pages, clean compile

## Git Log

```
73231e7 Phase 11: multi-subject EEG validation (N=80) + batch parallelization
8ce0a93 Commit Phase 10 handoff and recompiled preprint PDF
```

## Files Modified

| File | Changes |
|------|---------|
| `scripts/batch_eeg.py` | Precision/recall fix, --workers flag, ProcessPoolExecutor |
| `paper/preprint.tex` | N=80 results in abstract, Exp 7, limitations, future work |
| `paper/preprint.pdf` | Recompiled (21 pages) |

## Files Created (not committed — data + results)

| File | Description |
|------|-------------|
| `data/eeg/rivalry_ssvep/Sucharit*/` | 85 subject directories (12 GB) |
| `results/batch_eeg/*.json` | 81 per-subject result JSONs |
| `results/batch_eeg/batch_eeg_summary.csv` | Aggregate summary (81 rows) |
| `results/batch_eeg/failures.json` | 3 failure records |
| `results/batch_eeg/batch_eeg.log` | Batch run log |

## Key Design Decisions

- **7z not unzip**: Linux `unzip` silently stops at volume boundaries for split archives
- **10 workers**: Conservative (28 cores available, ~300 MB/worker, 19 GB free)
- **ETA formula bug**: `remaining / n_workers` double-counts parallelism — cosmetic, doesn't affect execution
- **Transitions only**: Binding analysis (~10 min/subject additional) deferred — would need separate run
- **CUSUM threshold**: Fixed at mean + 2*std across all subjects; adaptive thresholding could improve recall

## What's NOT Done

- Cross-region binding batch (Experiment 8 still N=1) — would need `--skip-binding` removed
- Ripser++ GPU acceleration — CUDA toolkit not installed (needs nvcc)
- Tutorial notebooks (ROADMAP 1.13, 1.14)
- Makefile
- Blog external publication

## Possible Next Steps

1. **Binding batch**: Run `scripts/batch_eeg.py --workers 10` (without --skip-binding) for Oz-Pz binding on all subjects (~2.5h with parallelization)
2. **Tutorial notebooks**: Standalone Lorenz walkthrough + heterogeneous timescale demo
3. **Adaptive CUSUM threshold**: Per-subject tuning to improve recall
4. **Install CUDA toolkit**: Enable Ripser++ for faster persistence computation
