# Branch 6: Multilingual Math — Topology of LLM Representations Across Languages — 2026-04-04

## Branch
`experiment/tda-multilingual` (from `experiment/neuromorphic-snn` @ `6e60605`)

## What Was Done

Created `scripts/branches/multilingual_math_topology.py` — extracts hidden states from Qwen2.5-1.5B-Instruct on MGSM (250 grade school math problems) in 4 languages (English, Chinese, Spanish, Japanese), and runs 3 topology experiments to test whether mathematical reasoning representations are language-agnostic.

### Data

- Dataset: juletxara/mgsm (Multilingual Grade School Math, TSV files)
- Model: Qwen/Qwen2.5-1.5B-Instruct (same as MATH + Code experiments)
- Languages: en, zh, es, ja (250 problems each, same problems translated)
- Hidden states: last token, final layer (1536-dim) + all 29 layers + token trajectories
- Total: 1000 hidden state vectors (250 × 4 languages)

### Config

Exp 1: PCA→50, PH(max_dim=1, subsample=200), 200-permutation Wasserstein test, seed=42.
Exp 2: TopologicalFeatureExtractor(max_dim=1, n_pca=30, subsample=100, summary features=16). Per-problem Pearson correlation across language pairs.
Exp 3: Persistence image subtraction (joint vs marginals). PCA→50, PH(max_dim=1), PI(50×50, sigma=0.1). 50-surrogate test with shuffled correspondence.

### Three Experiments

| Exp | What | Key Result |
|-----|------|------------|
| 1 | Cross-language topological distance | **z=27.37, p=0.005** — languages ARE topologically distinct |
| 2 | Per-problem fingerprint consistency | **r=0.9998** — near-perfect per-problem preservation |
| 3 | Cross-lingual binding (EN-ZH) | z=0.62, p=0.255 — NOT significant |

### Key Findings

**Exp 1 — Languages occupy distinct topological manifolds (z=27.37):**
- 4×4 Wasserstein-1 distance matrix:
  - en-zh: 1747, en-es: 3089, en-ja: 2509
  - zh-es: 1362, zh-ja: 763, es-ja: 605
- Closest pair: **es-ja (605)** — Spanish and Japanese closest
- Farthest pair: **en-es (3089)** — English and Spanish farthest
- Cluster structure: es-ja-zh form a tight cluster (605-1362); en is far from all (1747-3089)
- H1 entropy: en=4.78, zh=4.84, ja=4.60, es=4.50
- Permutation test: observed=1679, null=299±50, z=27.37, p=0.005
- **Interpretation**: English math representations are topologically isolated from the other three languages. The es-ja-zh cluster may reflect Qwen's multilingual training distribution — the model developed similar processing strategies for non-English languages. The extreme en distance suggests English math is processed through a fundamentally different representational geometry, possibly because English dominates the pretraining data and developed specialized circuitry.

**Exp 2 — Per-problem topological features are near-perfectly preserved (r=0.9998):**
- All 6 language pairs: r ∈ [0.9998, 0.9999]
- en-zh: 0.99982, en-es: 0.99986, en-ja: 0.99978
- zh-es: 0.99987, zh-ja: 0.99987, es-ja: 0.99989
- **Interpretation**: While global point cloud topology differs across languages (Exp 1), the per-problem topological fingerprint is nearly identical. The model assigns the same "topological complexity" to the same math problem regardless of input language. This is consistent with language-agnostic mathematical reasoning: the model's internal representation of problem difficulty/structure is preserved across languages, even though the overall geometric arrangement of the representation manifold differs. The correlation is dominated by total_persistence (the largest-magnitude feature), but the rank ordering is preserved across all 16 features.

**Exp 3 — Cross-lingual binding is NOT significant (z=0.62, p=0.255):**
- Observed EN-ZH binding: 42.10
- Null (shuffled correspondence): 37.02 ± 8.23
- z=0.62, p=0.255
- **Interpretation**: The joint EN-ZH representation (concatenating matched problem pairs) does not exhibit more topological structure than random pairings. This means the two languages don't create a shared topological "signature" that links matched problems beyond what each language has individually. The per-problem correlation (Exp 2) is high but the binding test measures something different: whether the paired representation has emergent structure. The null result suggests that while per-problem features are correlated (Exp 2), the paired representation doesn't create new topological features — the languages maintain independent representational geometries that happen to produce similar per-problem statistics.

### Interpretation

**Overall verdict: Mathematical representations are partially language-agnostic.**

1. **Global topology IS language-dependent** (z=27.37) — the model uses different geometric arrangements for different languages. English is topologically isolated from {zh, es, ja}.
2. **Per-problem topology IS language-agnostic** (r=0.9998) — the same math problem produces nearly identical topological features regardless of language.
3. **Cross-lingual binding is absent** (p=0.255) — languages maintain independent representational geometries; no emergent paired structure.
4. **Paradox resolution**: The model can have language-dependent global geometry while preserving language-agnostic per-problem structure. Analogy: two maps of the same city can use different projections (different global geometry) while preserving the same neighborhood relationships (same per-location features).

### Where This Approach Works Best

1. **Language comparison** — Wasserstein distances clearly separate processing regimes per language
2. **Universality testing** — r=0.9998 provides strong evidence for language-agnostic mathematical reasoning
3. **Cluster discovery** — reveals the es-ja-zh cluster vs English isolation

### Where It Falls Short

1. **Binding test** — persistence image subtraction on small (250-point) clouds may lack power
2. **Per-problem correlation ceiling** — r≈1 means we can't discriminate fine-grained differences
3. **Cluster interpretation** — es-ja proximity is surprising and may reflect training data artifacts

### Caveats

- MGSM has only 250 problems (vs 500 for MATH, 164 for HumanEval)
- Grade school math is simpler than MATH-500; difficulty variation is limited
- Tokenization differences across languages affect sequence lengths and thus token trajectories
- Qwen is trained heavily on Chinese — results may not generalize to English-dominant models
- The r≈0.9998 correlation is dominated by total_persistence; per-feature analysis might reveal more nuance
- Binding test uses only 50 surrogates (limited power for p-values near significance)
- MGSM TSV files loaded directly (dataset script deprecated on HuggingFace)

## Files

- `scripts/branches/multilingual_math_topology.py` — full analysis script
- `data/multilingual/en_hidden_states.npz` — English hidden states (19MB)
- `data/multilingual/zh_hidden_states.npz` — Chinese hidden states (19MB)
- `data/multilingual/es_hidden_states.npz` — Spanish hidden states (20MB)
- `data/multilingual/ja_hidden_states.npz` — Japanese hidden states (19MB)
- `data/multilingual/results.json` — all numeric results
- `figures/multilingual/` — 4 figures:
  - `overview.png` — 4-panel summary
  - `exp1_cross_language_topology.png` — Wasserstein heatmap + H1 entropy + permutation test
  - `exp2_cross_lang_correlation.png` — per-problem correlation matrix
  - `exp3_cross_lingual_binding.png` — binding bar chart + surrogate distribution

## Technical Notes

- MGSM TSV loaded via `huggingface_hub.hf_hub_download` (dataset script no longer supported)
- Extraction: 250 problems × 4 languages × Qwen2.5-1.5B-Instruct = ~116s on GPU
- Analysis: ~460s (dominated by 200-permutation test in Exp 1)
- Mean sequence length: ~61 tokens (range 24-134)
- `--skip-extraction` flag reuses cached NPZ files
- Persistence images: 50×50 resolution, sigma=0.1
- Binding = Σ_dim ||PI_joint[dim] - avg(PI_en[dim], PI_zh[dim])||

## Potential Follow-ups

- Add more languages (MGSM has 11: de, fr, ru, th, te, bn, sw)
- Per-layer analysis: which layers show language differentiation vs which are universal?
- Use harder math (MATH-500 translated) instead of grade school problems
- Feature-level analysis: which of the 16 TDA features drive the r≈0.9998?
- Cross-model: does the English isolation hold for Llama/Mistral (English-dominant models)?
- Investigate es-ja proximity: is it a Qwen artifact or a general phenomenon?
- Increase surrogate count for Exp 3 to test if binding reaches significance with more power
