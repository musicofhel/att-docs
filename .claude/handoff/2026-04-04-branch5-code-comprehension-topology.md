# Branch 5: Code Comprehension — Topology of LLM Representations — 2026-04-04

## Branch
`experiment/tda-code` (from `experiment/neuromorphic-snn` @ `60e67f2`)

## What Was Done

Created `scripts/branches/code_comprehension_topology.py` — extracts hidden states from Qwen2.5-1.5B-Instruct on HumanEval (164 coding problems), bins by difficulty using canonical solution complexity, evaluates code generation correctness via execution, and runs 3 topology experiments to test whether MATH-500 findings transfer to the code domain.

### Data

- Dataset: openai/openai_humaneval (164 problems)
- Model: Qwen/Qwen2.5-1.5B-Instruct (same as MATH experiments)
- Hidden states: last token, final layer (1536-dim) + all 29 layers + token trajectories
- Difficulty bins: 3 levels (easy/medium/hard) from canonical solution complexity score (line count + control flow + nesting + function calls)
- Bin sizes: easy=54, medium=55, hard=55
- Correctness: 42/164 = 25.6% pass@1 (easy=33.3%, medium=29.1%, hard=14.5%)

### Config

Exp 1: PCA→50, PH(max_dim=2, subsample=200), 200-permutation Wasserstein test, seed=42.
Exp 2: Shared PCA on combined MATH+Code hidden states. PH(max_dim=1, subsample=200). Wasserstein-1 between group diagrams.
Exp 3: TopologicalFeatureExtractor(max_dim=1, n_pca=30, subsample=100, summary features=16). 5-fold stratified CV logistic regression.

### Three Experiments

| Exp | What | Key Result |
|-----|------|------------|
| 1 | Code difficulty topology | p=0.811, z=-0.89 — **not significant** (unlike MATH z=8.11) |
| 2 | MATH vs Code comparison | code_easy↔code_hard=86, math_easy↔code_easy=778, math_hard↔code_hard=2349 |
| 3 | Correctness prediction | **AUROC=0.772** — strong predictive power from topology |

### Key Findings

**Exp 1 — H1 entropy IS non-monotonic but NOT significant:**
- Easy: H1=2.349, H1 features=14
- Medium: H1=2.188, H1 features=11 (minimum)
- Hard: H1=2.627, H1 features=18
- Pattern: medium < easy < hard (non-monotonic, dip at medium)
- Permutation test: observed=85.85, null=109.44±26.64, z=-0.89, p=0.811
- **Interpretation**: With only 164 problems split into 3 bins (~55 per bin), there's insufficient statistical power for the permutation test. The non-monotonicity pattern exists but the small group sizes (54-55 points per level) produce highly variable PH. MATH-500 had 43-134 points per level across 5 levels. The medium-dip pattern is interesting: medium-complexity code may produce more concentrated (less topologically complex) representations than either simple or complex code.

**Exp 2 — Code and MATH occupy different topological manifolds:**
- Within code: code_easy↔code_hard = 85.82 (small — similar topology)
- Within MATH: math_easy↔math_hard = 2125.61 (large — very different topology)
- Cross-domain: math_easy↔code_easy = 778.11, math_hard↔code_hard = 2349.05
- **Interpretation**: Code representations are much more topologically compact than MATH — all code difficulty levels live in a similar region of representation space. MATH problems create far more varied attractor geometries across difficulty levels. The cross-domain distances are dominated by the math_hard component, suggesting hard MATH problems create unusually complex representations that have no analogue in code processing. The model likely uses different processing strategies for the two domains.

**Exp 3 — Topological features strongly predict code correctness (AUROC=0.772):**
- AUROC: 0.772 ± 0.097 (5-fold CV)
- Accuracy: 73.8% ± 6.4%
- Top features: H1_n_features (0.999), H0_max_lifetime (0.855), H1_mean_birth (0.577)
- **Interpretation**: Comparable to MATH AUROC (0.787) — topology predicts correctness across domains. The top feature H1_n_features captures the number of 1-cycles in the token trajectory point cloud, which reflects the structural complexity of the model's internal processing. More H1 features correlate with incorrect solutions, suggesting the model's representation becomes more topologically fragmented when it fails to produce correct code.

### Interpretation

**Overall verdict: TDA transfers partially from MATH to code.**

1. **Difficulty discrimination does NOT transfer** — permutation test is non-significant (z=-0.89 vs z=8.11 for MATH). This likely reflects both smaller dataset size (164 vs 500) and the fundamentally different nature of code difficulty vs math difficulty.
2. **Correctness prediction DOES transfer** — AUROC 0.772 for code vs 0.787 for MATH. Topological features of token trajectories predict success regardless of domain.
3. **Cross-domain topology is distinct** — code representations are topologically compact (small within-domain distances), while MATH creates far more topological variation across difficulty levels.
4. **Non-monotonicity is present but weak** — the medium-dip pattern exists but lacks statistical power.

### Where This Approach Works Best

1. **Correctness prediction** — AUROC 0.772 from 16 TDA features, comparable to MATH
2. **Domain comparison** — Wasserstein distances clearly separate code vs MATH processing regimes
3. **Representation compactness** — within-code distance (86) vs within-MATH distance (2126) quantifies how differently the model distributes representations

### Where It Falls Short

1. **Difficulty discrimination** — only 164 problems split into 3 bins is insufficient for permutation-test significance
2. **Difficulty binning** — canonical solution complexity is a noisy proxy; actual pass@k rates would be better
3. **Small dataset** — HumanEval's 164 problems provide ~55 points per group, compared to MATH's ~100 per level

### Caveats

- HumanEval has only 164 problems (vs 500 for MATH) — smaller sample limits statistical power
- Difficulty bins are based on canonical solution complexity (lines + control flow + nesting), not empirical pass@k rates
- Correctness is from single greedy decoding (pass@1), not multiple samples
- 25.6% overall pass@1 is reasonable for a 1.5B model but creates class imbalance in Exp 3
- Code generation uses raw prompt continuation; an instruction-wrapped approach gave 0% correctness
- Shared PCA in Exp 2 may not fully capture domain-specific geometric structure
- The Wasserstein distances in Exp 2 are large and hard to interpret in absolute terms — the relative ordering is what matters

## Files

- `scripts/branches/code_comprehension_topology.py` — full analysis script
- `data/code/code_hidden_states.npz` — extracted hidden states + correctness labels (98MB)
- `data/code/results.json` — all numeric results
- `figures/code/` — 4 figures:
  - `overview.png` — 4-panel summary
  - `exp1_code_difficulty_topology.png` — H1 entropy + features + Wasserstein matrix + permutation test
  - `exp2_math_vs_code_topology.png` — cross-domain Wasserstein bar chart
  - `exp3_correctness_prediction.png` — ROC curve + feature importance

## Technical Notes

- Extraction uses raw HumanEval prompt (function signature + docstring) — instruction wrapping caused 0% correctness
- Post-processing strips markdown code blocks and stops at next top-level `def`/`class`
- Code correctness evaluated via subprocess execution with 10s timeout per problem
- Difficulty complexity score = n_lines + 2*n_control + 3*n_nested + 0.3*n_calls
- Hidden states: (164, 1536) last_hidden, (164, 29, 1536) layer_hidden
- Total runtime: 627s (569s extraction + 58s analysis)
- Ripser warning about columns > rows for math_easy (43×50) is expected and harmless

## Potential Follow-ups

- Use MBPP (974 problems) for larger dataset with more statistical power
- Use 5 difficulty levels for better comparison with MATH
- Empirical difficulty binning from model pass@10 rates
- Per-layer analysis (like MATH Direction 1) to find which layers discriminate code difficulty
- CROCKER matrices for code (finer-grained topological comparison)
- Cross-model replication (Phi-2, Pythia, Gemma on code)
- Combine MATH + Code features for universal correctness predictor
- Token-region analysis: do "code body" tokens carry more topological signal than "signature" tokens?
