# TDA-LLM Wave 5-7 Pipeline Execution — 2026-03-31

## Branch
`experiment/neuromorphic-snn` @ `ed229b1` (no new commits yet — scripts only produce output files)

## What was done

Ran the full TDA-LLM analysis pipeline (built in Waves 1-4) on real `math500_hidden_states.npz` data (500 problems, Qwen2.5-1.5B-Instruct). Fixed 3 bugs discovered during execution.

### Wave 5 Results (6 of 7 complete)

| Direction | Script | Status | Key Finding |
|-----------|--------|--------|-------------|
| D3 Spectral PH | `run_spectral_ph.py` | **Done** | Spectral PH collapses H1 — 0 features after layer 0. Euclidean finds 27-44 H1 features. 1-cycles are geometric, not graph-topological. |
| D5 CROCKER | `run_crocker.py` | **Done** | L1 distance L1↔L5 = 43, L3↔L4 = 20 (closest pair). Non-monotonic difficulty-topology structure confirmed at Betti level. |
| D7 Intrinsic Dim | `run_intrinsic_dim.py` | **Done** | Terminal-layer TwoNN ID increases monotonically: 6.67 → 7.42 → 10.07 → 11.65 → 12.01. Layer 0 numerical overflow for Level 1 (2.5B). |
| D8 Token Topology | `run_token_topology.py` | **Done** | Problem tokens carry difficulty signal (|Δ|=0.815, entropy 2.15→2.97). Instruction tokens show ~0 signal (<0.03). |
| D9 Compression | `run_compression_resistance.py` | **Done** | Mixed pattern: features increase L1→L4 (27→51) but lifetimes decrease (2.27→1.84). L5 drops both (44 features, 1.84 lifetime). |
| D10 Binding Proxy | `run_attention_binding.py` | **Done** | Proxy mode produces ~0 signal. Split-half token clouds lack H1 structure for PI correlation. Needs real attention data. |
| D1 Z-score | `run_perlayer_zscore.py` | **RUNNING** | ~95 min wall, 100 perms × 29 layers × max_dim=2. Still in Step 4 permutation test. |

### Wave 6 Completed

- **Bug fixes** (3 bugs):
  1. Token topology partition: replaced `"x"` placeholder with fixed 15/9 token counts for instruction prefix/suffix
  2. Token topology persistence_entropy: `dim in list` (value check) → `dim < len(list)` (index check)
  3. Binding proxy: passed distance→similarity conversion before `compute_binding` (which internally calls `attention_to_distance`)
- **Results compilation**: `scripts/compile_tda_llm_results.py` aggregates all JSON results → `tda_llm_summary.json` + `tda_llm_table.tex`
- **Model swap**: Replaced gated Gemma-2-2B with non-gated StableLM-2-1.6B in multi-model scripts

### Wave 7 In Progress

- **D2 Correctness labels**: `evaluate_correctness.py` running on GPU (~25 min CPU, generating 500 answers)
- Phi-2, Pythia-1.4B, StableLM-2-1.6B tokenizers verified accessible

## Output Files

### Figures (18 total in `figures/llm/`)
- `spectral_comparison_level{1,5}.png` — Euclidean vs spectral entropy profiles
- `crocker_difficulty_h{0,1}.png` — CROCKER by difficulty
- `crocker_layer_h1_level{1,5}.png` — CROCKER by layer
- `crocker_all_levels_comparison.png` — side-by-side
- `id_profile_twonn.png`, `id_diff_hard_vs_easy_twonn.png`
- `compression_total_persistence.png`, `compression_count_vs_lifetime.png`, `compression_layerwise.png`
- `token_region_entropy.png`, `token_region_easy_vs_hard.png`
- `attention_binding_heatmap.png`, `attention_binding_easy_vs_hard.png`, `attention_binding_significance_level{1,5}.png`

### JSON Results (9 in `data/transformer/`)
- `spectral_ph_results.json`, `crocker_results.json`, `intrinsic_dim_twonn_results.json`
- `compression_resistance_results.json`, `token_topology_results.json`, `attention_binding_results.json`
- `tda_llm_summary.json`, `tda_llm_table.tex`
- `baseline_results.json`, `lindner_results.json` (from Phase 5)

## Remaining Work

1. **Z-score script** still running — will produce `perlayer_zscore_results.json` + 3 figures
2. **Correctness evaluation** still running — will produce `math500_correctness.npz`
3. After correctness: run `run_correctness_prediction.py` (D2)
4. Multi-model extraction: `extract_hidden_states_multimodel.py` (D6) — 3 models × 250 problems
5. Cross-model analysis: `run_cross_model.py` (D6)
6. Attention extraction: `extract_attention_weights.py` (D10) — GPU needed
7. Full attention binding: `run_attention_binding.py` with real data (D10)
8. Cherry-pick Lindner changes to master (#49)
9. Final compilation rerun after all results in

## Cross-Direction Findings (so far)

1. **Terminal-layer effect** — confirmed via intrinsic dimension (monotonic increase) and prior Phase 5 z=8.11
2. **Geometric not spectral** — Euclidean PH captures H1 structure that spectral PH misses entirely
3. **Problem tokens carry signal** — 25x stronger difficulty signal in problem vs instruction tokens
4. **Non-monotonic CROCKER** — L3-L4 are topologically closest, not L4-L5
5. **Mixed compression** — features increase with difficulty but lifetimes decrease (neither pure compression nor pure resistance)
6. **Proxy binding fails** — split-half token clouds can't proxy attention coupling
