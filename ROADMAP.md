# ROADMAP.md

## Philosophy

Each phase ships something usable. No phase depends on future work to be valuable. If the project stops after any phase, what exists is complete and presentable.

**Updated March 2026**: Deep research confirmed the joint-vs-marginal PH construction is novel. This changes the approach: Phase 2 is the priority deliverable. A clean synthetic validation with benchmarks, surrogate testing, and a methods preprint IS a publishable contribution even without EEG data. The preprint is the organizing goal — every experiment should produce a figure for the paper.

**Scope discipline**: Phase 2 is the core contribution and gets the most time. Diagram matching (Hungarian method) and twin surrogates are deferred to Phase 2b — they are nice-to-haves that don't affect the primary contribution. The essential path is: persistence image subtraction + phase-randomized surrogates + benchmark sweep + preprint draft. Everything else is layered on top.

**Compounding principle**: Invest hours where the output keeps generating value after the work is done. The benchmark framework (extensible, cited by future work) compounds. The preprint (indexed, found in literature searches) compounds. The blog (shared, linked, forwarded) compounds. A custom React frontend does not compound — it decays. Allocate accordingly.

---

## Phase 1 — Foundation (Weeks 1–2)

**Scope**: Core embedding + topology on synthetic data. Prove the math works in code. Includes the heterogeneous delay machinery and reproducibility infrastructure needed for Phase 2. Witness complexes deferred to Phase 3 — not needed until EEG-scale point clouds.

### Tasks

| # | Task | Module | Est. Hours |
|---|------|--------|-----------|
| 1.1 | Lorenz, Rössler, coupled Lorenz, coupled Rössler-Lorenz, switching Rössler generators (with seed params) | `synthetic` | 3 |
| 1.2 | `set_seed()`, `load_config()`, `save_config()` | `config` | 2 |
| 1.3 | TakensEmbedder with manual params | `embedding` | 3 |
| 1.4 | AMI-based delay estimation | `embedding` | 3 |
| 1.5 | FNN-based dimension estimation | `embedding` | 3 |
| 1.6 | JointEmbedder with per-channel delay/dimension | `embedding` | 4 |
| 1.7 | `validate_embedding()` — condition number, effective rank, degenerate flag | `embedding` | 2 |
| 1.8 | Ripser wrapper — persistence diagrams, Betti curves | `topology` | 4 |
| 1.9 | Persistence image + landscape computation (via persim) | `topology` | 3 |
| 1.10 | Bottleneck + Wasserstein distance | `topology` | 2 |
| 1.11 | Persistence diagram + barcode + Betti curve + PI plots | `viz` | 3 |
| 1.12 | 3D attractor point cloud plot (Plotly) | `viz` | 2 |
| 1.13 | Notebook: Lorenz end-to-end walkthrough | `notebooks` | 2 |
| 1.14 | Notebook: heterogeneous timescale demo (Rössler-Lorenz) | `notebooks` | 2 |
| 1.15 | Pytest suite: synthetic ground truth + embedding validation + reproducibility | `tests` | 4 |

**Subtotal: 42 hours**

### Completion Criteria

- [ ] `set_seed(42)` followed by full pipeline produces bitwise-identical results on repeat runs
- [ ] YAML config round-trips correctly (save → load → run = same results)
- [ ] `TakensEmbedder("auto", "auto")` correctly estimates τ=15±5 and d=3 for Lorenz x-component
- [ ] `JointEmbedder("auto", "auto")` produces different delays for Rössler and Lorenz channels
- [ ] Shared-delay joint embedding of Rössler-Lorenz has higher condition number (more degenerate) than per-channel embedding
- [ ] `validate_embedding()` returns `degenerate=True` for the shared-delay case and `degenerate=False` for the per-channel case
- [ ] `PersistenceAnalyzer` recovers 2 dominant H1 features for Lorenz
- [ ] Persistence images are non-zero and vary between Lorenz and Rössler
- [ ] Bottleneck distance between two Lorenz runs with same params < 0.1
- [ ] Bottleneck distance between Lorenz and Rössler > 1.0
- [ ] All plots render without error
- [ ] Notebooks run top-to-bottom with `make notebook-test`
- [ ] 100% test pass, >80% line coverage on `embedding` + `topology` + `config`

### Deliverable

A working Python library with deterministic reproducibility that takes any 1D or multi-channel time series, reconstructs attractors with appropriate per-channel parameters and quality validation, and produces publication-quality topological analysis including persistence images. Installable via `pip install -e .`

---

## Phase 2 — Binding Detection + Benchmarks + Preprint (Weeks 2–4)

**Scope**: The novel contribution. Cross-attractor topological binding via persistence image subtraction with configurable baselines and embedding quality gating. Surrogate-tested significance. Head-to-head benchmark against TE, PAC, CRQA with explicit normalization. Extensible benchmark framework with plugin interface. Preprint draft started here — every experiment is designed to produce a paper figure.

**Why the preprint starts here**: In academia, a preprint IS the product. The library is supplementary material. A short methods paper titled "Topological Binding Detection in Coupled Dynamical Systems via Persistent Homology" — with the benchmark sweep figure, surrogate validation, unimodal coupling response, and baseline comparison — is the single highest-compounding artifact this project produces. It gets indexed, cited, and found by every future researcher working on coupling detection. Starting the draft in Phase 2 means experiments are designed for figures, not just notebooks.

### Tasks (Essential Path)

| # | Task | Module | Est. Hours |
|---|------|--------|-----------|
| 2.1 | BindingDetector: persistence image subtraction with configurable baseline (max, sum) | `binding` | 6 |
| 2.2 | Embedding quality gate: validate all three embeddings, raise EmbeddingDegeneracyWarning | `binding` | 3 |
| 2.3 | Binding score computation (L1 norm of positive residual) | `binding` | 2 |
| 2.4 | Binding image visualization (residual heatmap) | `viz` | 2 |
| 2.5 | Phase-randomized surrogates (AAFT) with seed support | `surrogates` | 3 |
| 2.6 | Time-shuffle surrogates with seed support | `surrogates` | 1 |
| 2.7 | BindingDetector.test_significance() with embedding quality in output | `binding` | 3 |
| 2.8 | Transfer entropy wrapper (PyInform or Kraskov fallback) | `benchmarks` | 4 |
| 2.9 | Phase-amplitude coupling (modulation index) | `benchmarks` | 3 |
| 2.10 | Cross-recurrence quantification (PyRQA or manual) | `benchmarks` | 5 |
| 2.11 | CouplingBenchmark with normalization + `register_method()` plugin interface | `benchmarks` | 5 |
| 2.12 | CouplingBenchmark.sweep() with seed propagation + transient discard | `benchmarks` | 3 |
| 2.13 | `att benchmark run` CLI entry point for coupling sweeps | `cli` | 3 |
| 2.14 | Coupled Lorenz coupling sweep — PAPER FIGURE 1 (unimodal binding curve) | `notebooks` | 4 |
| 2.15 | Baseline comparison experiment: max vs sum — PAPER FIGURE 2 | `notebooks` | 3 |
| 2.16 | Benchmark sweep overlay (all methods, rank-normalized) — PAPER FIGURE 3 | `notebooks` | 4 |
| 2.17 | Surrogate null distribution — PAPER FIGURE 4 | `notebooks` | 2 |
| 2.18 | Coupled Rössler-Lorenz sweep (heterogeneous timescales) — PAPER FIGURE 5 | `notebooks` | 3 |
| 2.19 | Binding comparison visualization (3-panel) | `viz` | 2 |
| 2.20 | Benchmark sweep plot (all methods overlaid, normalized) | `viz` | 2 |
| 2.21 | Tests: binding (incl. quality gate, both baselines, subsample consistency), surrogates, benchmarks, normalization | `tests` | 6 |
| 2.22 | Preprint draft: introduction, method, experiments 1-5, discussion outline | `paper` | 8 |

**Subtotal: 67 hours**

### Tasks (Phase 2b — Stretch)

| # | Task | Module | Est. Hours |
|---|------|--------|-----------|
| 2b.1 | BindingDetector: diagram matching method (Hungarian) | `binding` | 4 |
| 2b.2 | Twin surrogates (recurrence-based, Thiel et al.) | `surrogates` | 3 |
| 2b.3 | Kuramoto-chaotic coupled oscillator generator | `synthetic` | 3 |
| 2b.4 | N-body binding test (3+ oscillators) | `notebooks` | 3 |

**Subtotal: 13 hours (stretch)**

### Completion Criteria (Essential)

- [ ] `BindingDetector.binding_score()` returns <0.05 for uncoupled systems (both baselines)
- [ ] `BindingDetector.test_significance()` returns p > 0.05 at coupling=0 (no false positive, both baselines)
- [ ] `BindingDetector.test_significance()` returns p < 0.05 at coupling≥0.3 (detects coupling)
- [ ] Binding score increases monotonically from coupling=0 to coupling≈0.5 (R² > 0.9 on lower half of sweep)
- [ ] Binding score decreases at coupling→1 due to synchronization collapse (unimodal shape confirmed)
- [ ] Max baseline produces lower false positive rate than sum baseline at coupling=0 (documented)
- [ ] Embedding quality gate fires on deliberately degenerate joint embedding (shared delay on heterogeneous system)
- [ ] Embedding quality gate does NOT fire on well-conditioned joint embedding
- [ ] `embedding_quality()` dict is included in all binding results and significance test output
- [ ] Binding image clearly shows emergent features that are absent in marginal images
- [ ] Benchmark sweep shows binding score curve qualitatively tracks TE curve (both monotone)
- [ ] Benchmark comparison figure includes all 4 methods on one plot per system type
- [ ] Rank normalization preserves ordering. Minmax normalization maps to [0, 1]. Raw scores preserved.
- [ ] `register_method()` works: user can add a 5th method and it appears in sweep output
- [ ] `att benchmark run` CLI produces a sweep DataFrame from a YAML config
- [ ] Per-channel delay binding on Rössler-Lorenz outperforms shared-delay binding (higher R²)
- [ ] All surrogate methods produce valid null distributions (normally distributed, centered near zero for uncoupled)
- [ ] Full sweep is reproducible: same seed → same DataFrame
- [ ] Five paper-quality figures exported at 2x resolution
- [ ] Preprint draft has complete methods section, experiment descriptions, and figure captions
- [ ] >80% coverage on `binding` + `benchmarks` + `surrogates`

### Deliverable

A validated method for topological binding detection, benchmarked against three established coupling measures, with a preprint draft containing five figures. The benchmark framework is extensible via `register_method()` so future researchers can plug in their own coupling measures. This is the core contribution and stands alone as a published methods paper.

---

## Phase 3 — Neural Data + Preprint Completion (Weeks 5–6)

**Scope**: Apply the toolkit to real EEG data. Scoped to a 1-subject proof-of-concept, not a full multi-subject analysis. Goal: demonstrate the pipeline works on real data and either produce a positive EEG figure for the preprint or document the failure mode honestly. Complete and submit the preprint.

**Why 1 subject**: The EEG result is a lottery ticket — it either works or it doesn't, and N=1 tells you which. If topology predicts perceptual switches for one subject, scaling to 3-5 subjects is straightforward and can happen post-submission. If it doesn't work for one subject, running more subjects won't fix a methodological problem. Invest the saved hours in the preprint.

### Tasks

| # | Task | Module | Est. Hours |
|---|------|--------|-----------|
| 3.1 | Locate + download EEG dataset (Katyal, fallback to ds003505) | `data` | 2 |
| 3.2 | Pre-screen: verify event labels, channel count, data quality | `data` | 2 |
| 3.3 | MNE-based EEG loader + preprocessing pipeline | `neuro` | 4 |
| 3.4 | `get_fallback_params()` with literature-grounded defaults per band | `neuro` | 2 |
| 3.5 | `embed_channel()`: auto→validate→fallback, silent in batch, metadata audit trail | `neuro` | 3 |
| 3.6 | Witness complex option (GUDHI) — needed at EEG scale | `topology` | 2 |
| 3.7 | Sliding-window PH + CUSUM changepoint detection | `transitions` | 4 |
| 3.8 | 1-subject experiment: topology transitions vs perceptual switches — PAPER FIGURE 6 | `notebooks` | 4 |
| 3.9 | 1-subject experiment: cross-region binding (occipital-parietal) | `notebooks` | 3 |
| 3.10 | Preprint: finalize results, write discussion, respond to own objections | `paper` | 6 |
| 3.11 | Preprint: format for arXiv, generate final figures | `paper` | 2 |
| 3.12 | Tests: sliding window on switching_rossler ground truth + fallback params validation | `tests` | 3 |

**Subtotal: 37 hours**

### Completion Criteria

- [ ] EEG data loads and preprocesses for at least 1 subject
- [ ] `embed_channel()` correctly uses auto params when they pass quality check and fallback params when they don't, with metadata audit trail
- [ ] Fallback parameters produce non-degenerate embeddings on tested EEG channels
- [ ] On `switching_rossler`: changepoints detected within ±1 window of true switch times
- [ ] On real EEG: either (a) topological transitions correlate with reported perceptual switches (p < 0.05), producing PAPER FIGURE 6, or (b) failure mode documented with condition numbers, embedding metadata, and proposed next steps
- [ ] Preprint is complete and submitted to arXiv
- [ ] All EEG experiments have YAML configs for reproducibility

### Deliverable

A submitted preprint. If EEG works: the paper has 6 figures (5 synthetic + 1 neural). If EEG doesn't work: the paper has 5 figures (all synthetic) with a discussion section addressing the EEG attempt honestly. Either way, the preprint ships.

---

## Phase 4 — Distribution + Blog + Polish (Weeks 6–7)

**Scope**: Make the project findable, installable, and understandable. The blog post is the primary distribution channel — invest in it. The interactive demo is a lightweight Streamlit app or static HTML exports, not a custom React build.

**Why no custom React frontend**: A Three.js + Plotly + React app costs 23 hours to build, serves one audience (someone clicking a link for 90 seconds), and decays immediately as dependencies rot. A Streamlit app or a set of static HTML exports from Plotly achieves 80% of the visual impact in 20% of the time. The saved hours go into the blog (which reaches 100x the audience) and PyPI packaging (which makes the tool actually usable).

### Tasks

| # | Task | Module | Est. Hours |
|---|------|--------|-----------|
| 4.1 | JSON export from all pipeline stages | `viz` | 2 |
| 4.2 | Lightweight interactive demo (Streamlit app or static Plotly HTML exports) | `demo` | 6 |
| 4.3 | PyPI packaging: pyproject.toml, `pip install att-toolkit` | `packaging` | 3 |
| 4.4 | `att benchmark run --config sweep.yaml` CLI polish + docs | `cli` | 2 |
| 4.5 | Integration docs: how to add a new coupling method via `register_method()` | `docs` | 2 |
| 4.6 | Sphinx docs: API reference + quickstart tutorial | `docs` | 4 |
| 4.7 | Blog post: "Your Brain Is a Matrix of Chaos Attractors" — full draft + figures | `blog` | 8 |
| 4.8 | Blog post: publication on personal site + dev.to cross-post | `blog` | 2 |
| 4.9 | README polish, badges, GIF demo, one-liner install | root | 2 |
| 4.10 | CI: GitHub Actions (lint, test, build demo) | root | 2 |

**Subtotal: 33 hours**

### Completion Criteria

- [ ] `pip install att-toolkit` works from PyPI
- [ ] `att benchmark run --config sweep.yaml` produces a DataFrame and sweep figure
- [ ] Demo loads and renders key panels (attractor, persistence diagram, binding image, sweep) without errors
- [ ] Integration docs show a complete example of adding a custom coupling method
- [ ] Docs build cleanly, all public API methods documented
- [ ] Blog post includes 6+ figures, distinguishes validated results from speculative directions
- [ ] Blog post published on personal site and cross-posted to dev.to
- [ ] GitHub repo public with MIT license, CI green
- [ ] README includes one-liner install, quickstart, and demo GIF

### Deliverable

A findable, installable, documented open-source project. `pip install att-toolkit` works. The blog post connects the intuition to the math to the code. The preprint is already on arXiv from Phase 3.

---

## Phase 5 — Extensions (Post-Launch, Ongoing)

Ranked by compounding potential:

1. **Full EEG analysis**: If 1-subject proof-of-concept worked, scale to 3-5 subjects. Multi-region binding. Submit as a separate neuroscience short paper.
2. **Phase 2b completion**: Diagram matching method, twin surrogates, N-body binding.
3. **Cross-barcode integration**: R-Cross-Barcode on VR complexes. Theoretically cleaner than PI subtraction.
4. **Community benchmark contributions**: Invite others to register methods. Maintain a leaderboard of coupling measures on standard systems.
5. **Transformer hidden states**: Attractor topology from LLM intermediate representations.
6. **Real-time streaming**: WebSocket pipeline for live EEG topology. Requires Ripser++ GPU.
7. **Multi-scale analysis**: Wavelet decomposition before embedding.
8. **Directed PH integration**: Combine with Xi et al.'s TE-network PH.

---

## Risk Register (Updated)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Ripser too slow for sliding-window at scale | Medium | High | Witness complexes (added in Phase 3), Ripser++ GPU, subsample, restrict to H0+H1 |
| Per-channel delay estimation disagrees with literature values | Medium | Medium | Validate against published τ for Lorenz/Rössler. Manual override available. |
| Degenerate embeddings produce false binding signal | High | High | Quality gate checks all 3 clouds. Mandatory. Cannot bypass without explicit override. |
| PI subtraction produces false positives from finite-sample effects | Medium | High | Surrogate testing mandatory. Subsample consistency enforced (same seed for all 3 clouds). |
| Baseline choice (max vs sum) affects conclusions | Medium | Medium | Both tested on coupled Lorenz. Results documented in preprint. Exposed as parameter. |
| Benchmark normalization distorts visual comparison | Medium | Medium | Three methods available. Raw scores always preserved. Default (rank) is most honest. |
| PyInform or PyRQA installation/compatibility issues | High | Medium | Budget extra hours. PyInform fallback: direct Kraskov estimator. PyRQA fallback: minimal CRQA from scratch. |
| Binding detection finds no signal in real EEG | Medium | Low | Scoped to 1-subject proof-of-concept. Preprint stands on synthetic results regardless. Full EEG deferred to Phase 5. |
| Katyal SSVEP dataset unavailable or poorly labeled | Medium | Medium | Pre-screen in Phase 3 task 3.1-3.2. Three backup datasets. Resting state as fallback paradigm. |
| AMI/FNN estimation unreliable on noisy EEG | High | Medium | `embed_channel()` auto-with-fallback. Literature-grounded defaults. Quality gate catches failures. |
| TE/PAC benchmarks show binding score is strictly inferior | Low | Medium | Report honestly. Complementarity is also publishable. |
| Non-reproducible results | Medium | High | `set_seed()` + YAML configs. Tested in Phase 1. |
| Nobody finds the project | Medium | High | PyPI packaging, blog post, arXiv preprint, CLI. Multiple discovery channels. |

---

## Time Budget (Updated)

Assumes ~30 productive hours/week (6 hours/day, 5 days) alongside job applications.

| Phase | Hours | Calendar | Primary Output |
|-------|-------|----------|----------------|
| Phase 1: Foundation | 42 | Mid-Week 2 | Working library |
| Phase 2: Binding + Benchmarks + Preprint Draft | 67 | End of Week 4 | 5 paper figures + draft |
| Phase 2b: Stretch | 13 | Week 4 (if time) | Diagram matching, twin surrogates |
| Phase 3: EEG + Preprint Submission | 37 | Mid-Week 6 | Submitted preprint |
| Phase 4: Distribution + Blog | 33 | End of Week 7 | PyPI, blog, docs |
| **Total (essential)** | **179** | **7 weeks** | |
| **Total (with stretch)** | **192** | **7 weeks** | |

Buffer: 13 hours from stretch + natural slack. Realistic total: 8 weeks.

**Where the hours compound**:
- Preprint (14h across Phases 2-3): indexed forever, found by literature search, cited
- Benchmark framework (8h): becomes the comparison baseline for future coupling papers
- Blog (10h): shared, linked, forwarded — 100x the audience of a frontend
- PyPI + CLI (5h): makes the tool actually reachable
- Synthetic test systems (3h): reused by every future method

**Where hours are consumed, not invested**:
- Benchmark wrappers (12h): necessary to produce the comparison figure, but the wrappers themselves are thin and disposable
- EEG preprocessing (11h): useful only if EEG result is positive; scoped to minimize downside
