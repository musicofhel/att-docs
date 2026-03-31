# Phase 5: Transformer Hidden State Topological Sparsity Analysis

**Date**: 2026-03-30
**Branch**: `experiment/neuromorphic-snn`
**Commit**: `0b213c8`
**Status**: Code complete, not yet run

## What was done

Implemented the Phase 5 "Transformer Hidden States" entry point from the ATT roadmap. Two files:

### `scripts/extract_hidden_states.py` — GPU extraction
- Default model: `Qwen/Qwen2.5-1.5B-Instruct` (~3GB float16, fits RTX 2060 Super 8GB)
- Alt model via `--model meta-llama/Llama-3.2-1B-Instruct`
- Dataset: `hendrycks/competition_math` test split, samples 100 per difficulty level (seed=42)
- Extracts three data types per problem:
  - Last hidden state at final token, final layer → `(N, d)`
  - All layers at final token → `(N, L+1, d)`
  - Token trajectory at final layer → variable-length `(T_i, d)` per problem
- OOM recovery, `torch.cuda.empty_cache()` every iteration, `max_length=512` truncation
- Output: `data/transformer/math500_hidden_states.npz`

### `notebooks/topo_sparsity_analysis.ipynb` — CPU analysis (19 cells)
- **Cell 0**: Sparsity baselines reproducing Jin et al. (L1 norm, Top-10% Energy, Effective Rank). Spearman sanity check gates further experiments.
- **Exp 1**: Point cloud PH per difficulty level. PCA→50d, `PersistenceAnalyzer(max_dim=2, backend="ripser")`, subsample=200. Outputs: 5-panel diagrams, entropy vs difficulty, PCA variance curves, PI(L5)-PI(L1) heatmap, Pearson correlations with sparsity metrics.
- **Exp 2**: Token trajectory PH. 20 problems/level, PCA→10d, `validate_embedding()` + `PersistenceAnalyzer(max_dim=1)`. Outputs: entropy/condition/degeneracy vs difficulty.
- **Exp 3**: Layer-wise bottleneck distance profile. Level 1 vs Level 5, 30 problems each, PCA→20d per layer cloud. Consecutive `pa.distance(pa_next, metric="bottleneck")`. Tests "terminal behavior" claim topologically.
- **Exp 4**: 5x5 Wasserstein distance matrix + 200-permutation significance test. Shuffles difficulty labels as null.
- **Summary**: Consolidated table + full Pearson correlation matrix, all figures saved to `figures/` at 300 DPI.

## Design decisions

1. **No BindingDetector** — it expects coupled time series, not independent point clouds. Replaced with Wasserstein distance + permutation null (Exp 4).
2. **No TransitionDetector** — it does internal Takens embedding on scalar time series. Replaced with direct `PersistenceAnalyzer.distance()` between consecutive layers (Exp 3).
3. **No TakensEmbedder** — token sequences aren't dynamical systems. Trajectory analysis (Exp 2) feeds PCA-reduced `(T, k)` directly as point clouds.
4. **1.5B models only** — 7B OOMs on 2060 Super even in float16.

## Next steps

1. **Run extraction**: `cd ~/att-docs && python scripts/extract_hidden_states.py` (needs GPU, ~10-20 min for 500 problems on 2060 Super)
2. **Run notebook**: Open `notebooks/topo_sparsity_analysis.ipynb`, execute all cells
3. **Interpret**: Key question is whether PH metrics correlate with sparsity (r>0.8 = convergent validation) or diverge (r<0.5 = complementary information)
4. **Cross-architecture**: Re-run extraction with `--model meta-llama/Llama-3.2-1B-Instruct --output data/transformer/llama_hidden_states.npz`, update `DATA_PATH` in notebook
5. **If results are positive**: Apply same analysis to pretraining checkpoints (Jin et al. Section 4) for U-shaped learning dynamic

## Dependencies not in ATT's pyproject.toml

The extraction script needs: `torch`, `transformers`, `datasets`
The notebook needs: `scipy`, `scikit-learn` (likely already installed)

## Files touched

- `scripts/extract_hidden_states.py` (new, 275 lines)
- `notebooks/topo_sparsity_analysis.ipynb` (new, 19 cells)
- `data/transformer/` directory exists but is empty (placeholder for `.npz` output)
