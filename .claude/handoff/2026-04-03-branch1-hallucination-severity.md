# Branch 1: Hallucination Severity Topological Analysis — 2026-04-03

## Branch
`experiment/tda-hallucination` (from `experiment/neuromorphic-snn` @ `60e67f2`)

## What Was Done

Created `scripts/branches/hallucination_severity.py` — a standalone analysis that classifies Qwen2.5-1.5B's incorrect answers by severity and tests whether hidden-state topology distinguishes severity levels.

### Severity Classification

Used string distance + numeric comparison on `predicted_answers` vs `ground_truth` from `math500_correctness.npz`. Key challenge: predictions contain full reasoning text, not just final answers. Built `extract_predicted_answer()` to extract the actual answer (\boxed{}, first line, first number).

Thresholds: near-miss (≤25% relative error or edit_dist < 0.3), moderate (≤100% or 0.3-0.7), hallucination (>100% or type mismatch).

Distribution: 38 correct, 15 near-miss, 85 moderate, 295 hallucination.

### Pool-Index Matching

Reused the alignment approach from Phase 5 D2 fix. `math500_correctness.npz` stores 433 problems in `rng.choice(seed=42)` order; `math500_hidden_states_aligned.npz` stores 500 problems in dataset order. Reconstructed RNG sampling indices per level (MATH-500 pool sizes: 43/90/105/128/134). Verified 433/433 level consistency.

### Three Experiments

| Exp | What | Result |
|-----|------|--------|
| 1 | Point cloud PH by severity bin | H1 entropy gradient: correct(2.72)→moderate(4.81)→halluc(6.57). Permutation test z=0.78, p=0.24 (not significant — small near-miss bin) |
| 2 | Per-problem topo features → classify severity | correct-vs-halluc AUROC=**0.755**, near_miss-vs-halluc AUROC=**0.651**. Top feature: H0_entropy |
| 3 | Layer-wise z-score by severity | Peak at **layer 16/29** (z=1.74), NOT terminal (z=1.18) |

### Key Finding

Severity discrimination peaks at middle layers (L16), contrasting with difficulty signal which peaks at terminal layer (L28). Possible dissociation between difficulty encoding and error severity encoding.

### Caveats

- Near-miss bin (n=15) too small for robust PH. H1 entropy=0 is a sample-size artifact (15 points → 14 PCA dims, not enough to form loops).
- 3-class F1=0.31 reflects extreme class imbalance (38/15/380), not poor features.
- Permutation test underpowered for near_miss vs hallucination comparison.
- Some "near-miss" predictions (e.g., pred=240, gt=240) may actually be correct answers that `evaluate_correctness.py` mis-classified due to string matching issues.

## Files

- `scripts/branches/hallucination_severity.py` — full analysis script
- `data/hallucination/results.json` — all numeric results
- `figures/hallucination/` — 5 figures (severity_distribution, entropy_by_severity, permutation_test, layerwise_severity_zscore, feature_importance)

## Technical Notes

- `warnings.filterwarnings("ignore", ...)` suppresses Ripser "more columns than rows" warning (fires when PCA dims > n_points, e.g., near-miss bin)
- PCA components capped at `min(n_pca, n_samples-1, n_features)` to handle small bins
- `multi_class` kwarg removed from LogisticRegression (deprecated in current sklearn)
- Token trajectories from `math500_hidden_states_aligned.npz` used for per-problem features in Exp2 (shape: n_tokens × 1536 per problem)
- Exp3 uses 50 perms per layer (capped from 200 for speed), 29 layers total

## Remaining from TDA-LLM

- Cherry-pick Lindner library changes from `experiment/neuromorphic-snn` to `master`
- Preprint update with corrected Phase 5 findings
- Possible follow-up: re-run Branch 1 with a larger model (more near-misses expected) or merge near-miss+moderate into binary "close error" vs "hallucination"
