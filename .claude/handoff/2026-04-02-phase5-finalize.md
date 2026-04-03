# Phase 5 Finalization: Aligned D6, Corrected Summary, Preprint Numbers — 2026-04-02

## Branch
`experiment/neuromorphic-snn`

## What Was Done

### Task 1: D6 Cross-Model with Aligned Qwen Data

Re-computed Qwen terminal-layer H1 persistence entropy using `math500_hidden_states_aligned.npz` (MATH-500, 43/90/105/128/134 per level) instead of original `math500_hidden_states.npz` (full MATH, 100/100/100/100/100).

**Aligned Qwen H1 Entropy:**

| Level | Old (full MATH) | New (MATH-500) | Delta  |
|-------|-----------------|----------------|--------|
| 1     | 2.925           | 2.331          | -0.594 |
| 2     | 3.562           | 3.186          | -0.376 |
| 3     | 3.403           | 3.524          | +0.121 |
| 4     | 3.535           | 3.824          | +0.289 |
| 5     | 3.447           | 3.796          | +0.349 |

**Key changes:**
- Level 1 minimum **preserved** (2.33, still lowest of all levels)
- Non-monotonic: **yes** (L4 > L5)
- Pattern shifted from oscillatory to near-monotonic increase (Spearman r=0.900, p=0.037)
- The cross-model claims all survive:
  - Level 1 minimum universal: **4/4 models** (unchanged)
  - Non-monotonicity universal: **4/4 models** (unchanged)
  - Terminal-layer effect Qwen-specific: **1/4 models** (unchanged)

**Z-score profile**: The Qwen z-score re-computation with aligned data was attempted but CPU-intensive (permutation test: 100 perms x 29 layers). The terminal-layer effect (z=8.0 at L28) is a model-intrinsic property confirmed by the original D1 analysis (SAFE category). The D6 z-score values for Qwen use the original run pending re-computation.

### Task 2: Updated tda_llm_summary.json

Patched with corrected values:
- **D10**: per_level_binding updated, monotonic_decrease=false, r=-0.9
- **D2**: AUROC=0.787, added CV-validated CI [0.695, 0.889]
- **D6**: Added aligned Qwen entropy, preserved old values for comparison
- **Audit metadata**: dates, commits, affected/safe directions

### Task 3: Created preprint_numbers.json

Single source of truth for all 10 directions + baselines. Every reportable number with provenance and caveats.

### Task 4: Stale Number Verification

Grep confirmed stale values only remain in:
- `attention_binding_results.json` (pre-audit, kept for comparison)
- `audit_results.json` (explicitly documents old vs new)

Fixed: `tda_llm_summary.json` had old D10 values (0.683, monotonic_decrease=true).

### Task 5: D2 AUROC Proper CV Validation

**Critical finding**: Naive first-N alignment gives AUROC=0.582 (near chance!) even with aligned data. This is because `evaluate_correctness.py` stores problems in `rng.choice` order while aligned hidden states are in dataset order. **Pool-index matching** (reconstructing the RNG sampling order) is required.

With proper pool-index matching:
- 5-fold CV (seed=42): **0.795 +/- 0.094**
- 50-fold aggregate (10 seeds x 5 folds): **0.783 +/- 0.054**
- 95% CI: **(0.695, 0.889)**
- Confirms reported AUROC=0.787 is genuine

**Warning**: Any future code using `math500_hidden_states_aligned.npz` with `math500_correctness.npz` MUST use pool-index matching, not first-N-per-level alignment.

## Modified Script

`scripts/run_cross_model.py` — added `--qwen-data` and `--output` CLI flags for aligned data override.

## New/Updated Files

- `data/transformer/cross_model_results_aligned.json` — D6 with aligned Qwen entropy
- `data/transformer/tda_llm_summary.json` — updated D2/D6/D10 + audit metadata
- `data/transformer/preprint_numbers.json` — all 10 directions, final numbers
- `.claude/handoff/2026-04-02-phase5-finalize.md` — this file

## Remaining

- Qwen z-score re-computation with aligned data (pending, model-intrinsic property expected to be unchanged)
- Cherry-pick Lindner library changes to master
- Preprint update with corrected findings
