# Phase 5 Audit: Cross-File Problem Mismatch — 2026-04-02

## Branch
`experiment/neuromorphic-snn`

## Problem Found

**Root cause**: Dataset source divergence between extraction sessions.

The original `extract_hidden_states.py` loaded from the full MATH test set (~5000 problems via `hendrycks/competition_math` or `EleutherAI/hendrycks_math`), producing 100 problems per difficulty level (500 total). Later scripts (`extract_attention_weights.py`, `evaluate_correctness.py`, multi-model extraction) fell through to `HuggingFaceH4/MATH-500` (500 problems, uneven distribution: 43/90/105/128/134 per level) because the original dataset sources became unavailable on HuggingFace between sessions.

This means the Qwen hidden states contain **entirely different math problems** from the attention PH, correctness labels, and non-Qwen model hidden states.

## Impact by Direction

### CRITICAL: D10 (Attention-Hidden Binding)
- **Data**: Hidden states from full MATH (100/level), attention PH from MATH-500 (25/level)
- **Mismatch**: Different problem pools entirely — estimated 2-7 out of 25 problems overlap per level by chance
- **Shuffle test**: Within-level random pairing produces the SAME monotonic trend (Spearman r=-1.0 for both observed and shuffled). The 0.683→0.465 difficulty gradient is a **population-level signal**, not per-problem coupling.
- **Revised interpretation**: "Topological similarity between attention and hidden-state spaces decreases with difficulty" — still interesting, but NOT the same as "attention-hidden coupling decreases"

### CRITICAL: D2 (Correctness Prediction)
- **Data**: Hidden states from full MATH (100/level), correctness labels from MATH-500 (43/90/100/100/100)
- **Mismatch**: Correctness labels are for completely different problems than the features
- **AUROC=0.580** is barely above chance (0.500), consistent with spurious level-correlated signal

### MODERATE: D6 (Cross-Model)
- **Data**: Qwen from full MATH, Phi2/Pythia/StableLM from MATH-500
- **Mitigation**: H1 entropy is a per-level aggregate statistic, so cross-model comparison is likely still valid
- **Finding confirmed**: Level 1 minimum universal (4/4), non-monotonic (4/4). Phi2 nearly monotonically increasing (Spearman r=0.9, p=0.037).

### SAFE: D1, D3, D5, D7, D8, D9
These use only Qwen hidden states internally. No cross-file alignment needed.

## Additional Finding: StableLM Level 1 H1

StableLM Level 1 H1 persistence entropy ≈ 0 is **correct**: only 1 finite H1 feature (persistence=2.06). The point cloud has near-zero loop structure. Not a bug — Level 5 has 12 H1 features with entropy=2.23 for comparison.

## Fix Applied

1. **Created `scripts/extract_hidden_states_aligned.py`**: Loads exclusively from MATH-500, stores problem text hashes for cross-file verification. Supports `--subset attention` (25/level matching attention PH) or `--subset full` (all 500 MATH-500 problems).

2. **Re-extraction complete**: `math500_hidden_states_aligned.npz` — 500 MATH-500 problems (43/90/105/128/134 per level), 18 seconds, zero skips. Includes `problem_hashes` and `dataset_source` fields.

3. **D10 re-run complete**: Aligned binding shows Spearman r=-0.900, p=0.037 (significant). Level means: L1=0.740, L2=0.613, L3=0.647, L4=0.596, L5=0.462. Level 1 is HIGHER with proper matching (0.740 vs 0.683), confirming per-problem coupling adds real signal. Non-monotonic at L2/L3 — the perfect r=-1.0 in old data was coincidental.

4. **D2 re-run complete**: AUROC jumped from 0.580 to **0.787** (+0.207). The mismatch was hiding a strong predictive signal. Top predictor: H0_persistence_entropy (0.824). Per-level: L1=0.757, L2=0.691, L3=0.742, L4=0.667. Extreme class imbalance (38/433 = 8.8% correct).

## Data Files

- `data/transformer/audit_results.json` — comprehensive audit findings with old/new comparisons
- `data/transformer/math500_hidden_states_aligned.npz` — corrected hidden states (MATH-500 aligned)
- `data/transformer/attention_binding_results_aligned.json` — corrected D10 binding scores
- `data/transformer/correctness_prediction_results_aligned.json` — corrected D2 AUROC
- Original files preserved for comparison

## Lessons

1. **Never assume dataset availability is stable** — HuggingFace datasets can be removed or restructured
2. **Store problem identifiers** (hashes, indices, or text) in every extracted file for cross-file verification
3. **Fallback chains are dangerous** — different fallback sources may have different problem pools and level distributions
4. **Shuffle tests can reveal population vs per-item signals** — the D10 finding is real but misinterpreted
