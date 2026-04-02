# TDA-LLM All 10 Directions Complete — 2026-04-02

## Branch
`experiment/neuromorphic-snn` — all work committed in this session

## Summary

Completed the final remaining TDA-LLM analysis tasks: multi-model hidden state extraction (D6), cross-model universality analysis (D6), real attention PH extraction (D10), and full attention-hidden binding analysis (D10). All 10 research directions now have results, figures, and compiled summary.

## Session Work

### Multi-model extraction (D6, Task #53)
- Previous session extracted Phi-2 (166MB) before PC froze from GPU overload
- Re-ran Pythia-1.4B and StableLM-2-1.6B one at a time (5-6s each, ~3GB VRAM)
- Output: `data/transformer/{pythia14b,stablelm16b}_hidden_states.npz`
- **Fix for stability**: run models individually rather than back-to-back

### Cross-model universality (D6, Task #54)
- Ran `scripts/run_cross_model.py` (~80 min, CPU-bound permutation tests)
- **Fix**: `persistence_entropy` returns list not dict — changed `.get(1, 0.0)` to index-based access
- Results:
  - Terminal-layer effect: only Qwen (1/4 models) — model-specific, not universal
  - H1 non-monotonic entropy: all 4 models — universal signature
  - Peak z-scores: Qwen=8.00 (layer 28), Phi2=4.22 (layer 2), Pythia=2.23 (layer 2), StableLM=2.89 (layer 9)
- Output: `data/transformer/cross_model_results.json`, 3 figures

### Attention PH extraction (D10, Task #55)
- **Critical bug found**: Qwen2.5 with float16 produces ALL-NaN attention weights (100% NaN)
- Root cause: softmax underflow in half precision; SDPA/flash attention also returns NaN
- **Fix**: `torch_dtype=torch.float32` + `attn_implementation="eager"` in extraction script
- Re-extracted 125 problems (25/level), ~10s, non-trivial PH (H0: 35-63 features, H1: 0-52 features)

### Attention-hidden binding (D10, Task #56)
- **Scale mismatch bug**: attention distances in [0,1] vs hidden Euclidean in [0,178] made shared PI range collapse both to zeros
- **Fix**: added `_normalize_diagrams()` to scale both to [0,1] before PI computation; added `_diagrams_to_images()` for computing PIs from raw diagrams; added `compute_binding_from_diagrams()` method
- Results — monotonic coupling decrease with difficulty:
  - Level 1 (Easy): binding = 0.683
  - Level 2: binding = 0.650
  - Level 3: binding = 0.599
  - Level 4: binding = 0.540
  - Level 5 (Hard): binding = 0.465
- This is arguably the strongest new finding: easy problems have tightly coupled attention/hidden topologies

### Compilation updated
- Added `summarize_d6_cross_model()` to `compile_tda_llm_results.py`
- Updated `summarize_d10_binding()` to handle real attention mode
- Added D6 and D10 to LaTeX table and cross-direction findings
- Final output: `tda_llm_summary.json`, `tda_llm_table.tex`

## Final 10-Direction Results

| Dir | Analysis | Key Result |
|-----|----------|-----------|
| D1 | Per-layer z-score | Peak z=8.12 at terminal layer (28/28) |
| D2 | Correctness prediction | AUROC=0.580, top: H0_total_persistence |
| D3 | Spectral PH | Euclidean captures richer H1 than spectral |
| D5 | CROCKER matrix | Monotonic difficulty gradient (L1 max=57.0) |
| D6 | Cross-model (4 models) | H1 non-monotonicity universal; terminal effect Qwen-only |
| D7 | Intrinsic dimension | ID increases with difficulty |
| D8 | Token topology | Region-based signal detected |
| D9 | Compression resistance | Compression pattern |
| D10 | Attention-hidden binding | Coupling 0.683→0.465 (monotonic decrease) |

## Cross-Direction Findings
1. Terminal-layer effect confirmed (D1) but model-specific (D6 — 1/4 models)
2. H1 non-monotonic entropy is universal across all 4 architectures (D6)
3. Attention-hidden coupling monotonically weakens with difficulty (D10)
4. Topological features predict correctness above chance (D2)
5. CROCKER L1 distances show monotonic difficulty gradient (D5)
6. Intrinsic dimensionality increases with difficulty at terminal layer (D7)
7. Euclidean PH captures richer H1 than spectral PH (D3)

## Files Modified
- `att/llm/attention_binding.py` — added `_normalize_diagrams`, `_diagrams_to_images`, `compute_binding_from_diagrams`
- `scripts/extract_attention_weights.py` — float32 + eager attention fix
- `scripts/run_attention_binding.py` — real attention mode, `compute_real_attention_binding()`
- `scripts/run_cross_model.py` — persistence_entropy list vs dict fix
- `scripts/compile_tda_llm_results.py` — D6 + D10 summarizers, LaTeX table, findings

## Data Files (all in `data/transformer/`)
- `pythia14b_hidden_states.npz` (121MB), `stablelm16b_hidden_states.npz` (119MB)
- `attention_ph_diagrams.npz` — real attention PH (125 problems, layers 23-27)
- `cross_model_results.json`, `attention_binding_results.json`
- `tda_llm_summary.json`, `tda_llm_table.tex` — final compiled output

## 28 figures in `figures/llm/`
All generated, including 3 new cross-model and 2 new real attention binding plots.

## Tests
97/97 LLM tests pass. No regressions.

## Remaining
- Task #49: Cherry-pick Lindner library changes from this branch to master (code-only, no data)
- Consider writing up findings for preprint update
- The Qwen float16 attention NaN issue should be noted for any future attention extraction

## GPU Notes (for stability)
- RTX 2060 SUPER 8GB: float16 models fit comfortably (~3GB), float32 Qwen uses ~6GB
- Previous freeze caused by running 3 models back-to-back at 7.5GB VRAM + 92% CPU
- Safe pattern: run one model at a time, verify completion, then next
