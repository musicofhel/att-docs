# Ten experimental directions for TDA of LLM hidden states under distribution shift

**Persistent homology applied to LLM representations under difficulty-induced distribution shift is a genuinely novel research direction with strong recent literature support.** Your confirmed results — perfect sparsity-difficulty correlation, z=8.11 topological significance, non-monotonic H1 entropy, and the 5–6× terminal-layer effect — place this work at the intersection of three rapidly converging fields: topological data analysis for neural networks (Gardinazzi et al., ICML 2025; Fay et al., 2025), representation geometry under distribution shift (Jin et al., 2026; Datta et al., 2025), and geometric interpretability of transformers (Gurnee et al., Anthropic 2025; Park et al., ICLR 2025). No existing paper applies H1 persistence entropy to mathematical reasoning representations across controlled difficulty tiers. The following ten directions are ordered by estimated impact-to-effort ratio, each grounded in 2024–2026 literature and feasible on your hardware.

---

## 1. Per-layer decomposition of the z=8.11 topological signal

Your headline result — that persistence diagrams of hidden-state ensembles differ across difficulty levels with z=8.11 — aggregates across all layers. **Decomposing this signal layer by layer would reveal where topological discrimination actually lives** and connect directly to the terminal-layer finding.

Run the existing permutation test (Wasserstein distance, 200-permutation null) independently at each transformer layer. Produce a "topological discriminability profile" — z-score as a function of layer index. The BLOOD framework (Jelenić et al., ICLR 2024; arXiv 2310.02832) found that **later transformer layers are most discriminative for OOD detection** because learning smooths between-layer transformations for in-distribution data while leaving OOD regions unchanged. Lad et al. (arXiv 2406.19384) independently confirmed that early and terminal layers are most sensitive to perturbation, while middle layers are remarkably robust (72–95% accuracy retention after deletion). Gardinazzi et al. (arXiv 2410.11042, ICML 2025) identified four topological processing phases via zigzag persistence: initial rapid rearrangement, stable middle phase, transition phase, and final rearrangement.

The experiment is straightforward: you already have layer-wise hidden states and the permutation testing infrastructure. Expect the z-score profile to be bimodal — low in the stable middle phase, peaking at terminal layers — which would provide a quantitative topological counterpart to the four-phase model. This profile alone is a strong figure for the preprint. Additionally, computing the profile separately for H0, H1, and H2 would reveal whether different homological dimensions carry the signal at different depths. **Estimated effort: 1–2 days using existing code.**

---

## 2. Correct versus incorrect: topological predictors of solve success

The most impactful extension would be showing that **topological features of hidden states predict whether the model solves a given problem correctly**, not just its difficulty label. This transforms TDA from a descriptive tool into a predictive one.

Split your 500 problems into correct and incorrect subsets (per difficulty level), compute persistence diagrams for the hidden states of each subset, and test whether topological features discriminate between them. Multiple recent papers support this approach. Yin et al. (ICML 2024; arXiv 2402.18048) showed that **local intrinsic dimension (LID) of activations distinguishes truthful from hallucinated outputs**, outperforming entropy-based uncertainty by up to 8% AUROC. Fay et al. (arXiv 2505.20435) demonstrated that adversarial inputs induce "topological compression" — fewer, larger-scale features — detectable via 41-dimensional barcode summary vectors with near-perfect PCA separation. The EDTR framework (NeurIPS 2025; arXiv 2511.06437) achieved **41% better calibration** than competing methods by extracting 8 topological risk features from reasoning distributions.

Concretely: for each problem, extract the last-token hidden state at the final layer, label it correct/incorrect, compute persistence diagrams (H0, H1) on difficulty-stratified subsets, and vectorize using your existing persistence images. Then train a logistic regression on persistence image features to predict correctness. The non-monotonic H1 entropy may partly decompose: if Level 2–5 problems have similar topology overall but differ sharply between correct and incorrect subsets, the plateau would be explained by averaging over a bimodal distribution. **This experiment is high-impact because a "topological signature of failure" is directly publishable and practically useful.**

---

## 3. Spectral-distance persistent homology for high-dimensional activations

Your current PH computations likely use Euclidean distances between hidden-state vectors in R^1536 (Qwen2.5-1.5B hidden dimension). A NeurIPS 2024 paper on persistent homology for high-dimensional data demonstrated that **standard Euclidean PH fails to detect correct topology in high dimensions**, while spectral distances — effective resistance and diffusion distances computed on kNN graphs — robustly recover true topological features despite high-dimensional noise.

The method is: (1) construct a kNN graph on hidden-state point clouds (k ≈ 15–30), (2) compute the graph Laplacian, (3) derive effective resistance distances between all pairs via the pseudoinverse (or approximate via a few eigenvectors), (4) use these spectral distances as the filtration metric for Ripser. This directly addresses a potential weakness in your current pipeline — the curse of dimensionality may be suppressing genuine H1 and H2 features. Gardinazzi et al. (ICML 2025) used kNN-based filtrations rather than raw Euclidean distances for precisely this reason, finding that kNN filtrations produce consistent topological signatures across models.

The computational cost is moderate: eigendecomposition of a sparse kNN Laplacian for ~100 points is trivial, and the resulting distance matrix feeds directly into Ripser. **If spectral PH reveals richer H1/H2 structure than Euclidean PH — particularly at middle layers where you currently see less topological change — it would both strengthen existing results and resolve whether middle-layer "topological simplicity" is real or an artifact of distance metric choice.** This experiment could also sharpen the non-monotonic H1 signal, potentially turning the plateau into a more interpretable pattern.

---

## 4. Zigzag persistence to track feature evolution across layers under shift

Rather than comparing independent persistence diagrams layer by layer, **zigzag persistence tracks the same topological features as they are born, persist, and die across the full layer sequence**. Gardinazzi et al. (ICML 2025) applied exactly this to LLMs and found four universal processing phases. Your project can extend this by comparing zigzag barcodes between difficulty levels.

The method treats the hidden-state point cloud at each layer as a snapshot in a time-varying system. Zigzag persistence (available in GUDHI via `SimplexTree` with zigzag capabilities, or via the Dionysus2 library) constructs a zigzag filtration: X₁ ← X₁∪X₂ → X₂ ← X₂∪X₃ → X₃ ← ···, tracking which topological features persist across layer transitions. The output is a zigzag barcode where each bar represents a feature's lifetime across layers.

For each difficulty level, compute zigzag barcodes and then compare using the persistence similarity metric from Gardinazzi et al. The terminal-layer finding (5–6× more bottleneck distance) predicts that zigzag features will have dramatically shorter lifetimes in final layers. **The key novel question: does distribution shift (higher difficulty) cause topological features to die earlier (compressed processing), persist longer (failed simplification), or birth new features in terminal layers?** Datta et al. (arXiv 2501.12522) found that OOD data resists topological simplification — their average persistence lifetime for OOD features is statistically longer. If this holds in your LLM data, you'd expect harder problems to produce longer-lived zigzag features, particularly in terminal layers. This directly tests a concrete theoretical prediction. **Estimated effort: 3–5 days including learning the zigzag API.**

---

## 5. CROCKER plots for visualizing the topological landscape of difficulty

The non-monotonic H1 entropy — **2.93, 3.56, 3.40, 3.54, 3.45** across Levels 1–5 — demands richer visualization than scalar summaries. CROCKER (Contour Realization Of Computed K-dimensional hole Evolution in the Rips complex) plots, introduced by Güzel, Munch & Khasawneh (Chaos, 2022), display Betti numbers as a 2D heatmap with filtration scale ε on one axis and a varying parameter (here, difficulty level or layer index) on the other.

Construct one CROCKER matrix per difficulty level: for each layer l and filtration radius ε, record β₁(l, ε). Stack these five matrices side by side or compute their pairwise L¹ distances. The L¹ norm of the Betti curve difference is closely related to the 1-Wasserstein distance between persistence diagrams, giving a principled scalar summary. Barrios, Echávez & Álvarez (arXiv 2603.27395, 2026) extended this by using the maximum persistence of H₁ classes as a scalar topological functional that detects transitions Lyapunov exponents miss.

**The CROCKER format may resolve the non-monotonicity**: if Level 1 has a simple CROCKER pattern (few features, small radii), Level 2 introduces loops at specific scales, and Levels 3–5 redistribute loop activity across scales without changing entropy, the scalar H1 entropy would plateau while the CROCKER structure shifts. This 2D view separates scale-dependent effects from aggregate summaries. It also produces compelling figures: a clear visual progression from topologically simple (Level 1) to topologically complex (Level 2+), with the specific scales and layers where complexity appears highlighted. **Estimated effort: 1–2 days; requires only Betti number computation at multiple radii, which is fast.**

---

## 6. Cross-model universality on Phi-2, Pythia-1.4B, and Gemma-2-2B

A single-model result on Qwen2.5-1.5B, however significant, invites the critique that findings are architecture-specific. Gardinazzi et al. (ICML 2025) found that their topological processing phases are **consistent across Llama-2-7B, Llama-3-8B, Mistral-7B, and Pythia-6.9B**, suggesting universality. Testing whether your sparsity-topology-difficulty relationships replicate across architectures would substantially strengthen the preprint.

Three models fit comfortably on an RTX 2060 Super in float16:

- **Phi-2** (2.7B parameters, Microsoft): Different architecture family, strong math performance for its size, 32 layers
- **Pythia-1.4B** (EleutherAI): Well-studied in interpretability literature, 24 layers, trained on The Pile with known training dynamics
- **Gemma-2-2B** (Google): Recent architecture with grouped-query attention, 26 layers

For each model, run the core pipeline on MATH-500 (or a subset, e.g., 50 problems per level): extract hidden states, compute persistence diagrams, measure H1 entropy per difficulty level, run the Wasserstein permutation test, and compute bottleneck distances between consecutive layers. **The minimum viable universality claim requires reproducing two findings: (1) statistically significant topological differences between difficulty levels, and (2) greater topological change in terminal layers.** If the non-monotonic H1 pattern replicates across architectures, it becomes a robust empirical finding demanding theoretical explanation. If it doesn't, the architecture-dependence itself is interesting. **Estimated effort: 5–7 days including inference runs.**

---

## 7. Intrinsic dimension profiles as a complementary geometric probe

Persistent homology and intrinsic dimension (ID) measure different but related aspects of manifold geometry. ID captures the effective degrees of freedom; PH captures holes and voids. **Computing both provides a richer geometric characterization and connects your work to a large parallel literature.**

Multiple papers converge on the finding that ID tracks difficulty: Valeriani et al. (NeurIPS 2023; arXiv 2302.00294) showed transformer representations expand in early layers and compress in later ones. Baroni et al. (arXiv 2601.03779, 2026) demonstrated that **ID of LLM representations serves as a marker of linguistic complexity** — more complex data produces higher ID. The LADE group (Ital-IA 2025) found ID directly correlates with cross-entropy loss and prompt difficulty. Yin et al. (ICML 2024) showed LID distinguishes truthful from hallucinated outputs.

Use the TwoNN estimator (fast, requires only nearest-neighbor distances) to compute local ID at each layer for each difficulty level. Compare the layer-wise ID profile to your layer-wise bottleneck distance profile. **The key hypothesis: layers where ID changes rapidly should coincide with layers showing large bottleneck distance.** If PH captures topological features (loops, voids) that ID misses, the two profiles will diverge at specific layers — identifying where topology provides information beyond dimensionality. This comparison also connects your PCA variance concentration finding (94.5% → 91.6% from Level 1 to 5) to formal ID estimation. The PHD (Persistent Homology Dimension) estimator from Birdal et al. (NeurIPS 2021; arXiv 2111.13171) uses your existing PH infrastructure to estimate fractal dimension, bridging the two approaches with a single tool. **Estimated effort: 2–3 days.**

---

## 8. Token-position-resolved topology within mathematical reasoning sequences

Your current analysis uses last-token hidden states, collapsing all token-position information into a single representation. But mathematical reasoning unfolds across the sequence — problem parsing, strategy selection, computation, and answer formation occupy different token spans. **Analyzing how topology varies across token positions could reveal where the model's internal geometry reorganizes during reasoning.**

Viswanathan et al. (arXiv 2501.10573, 2025) computed token-level intrinsic dimension across LLM layers and found that **ID for randomized text is significantly higher than for coherent text** — models struggle to compress it, directly paralleling the OOD resistance-to-simplification finding. Gurnee et al. (Anthropic, arXiv 2601.04480, 2025) discovered that character counts are represented on **1-dimensional curved manifolds (helices)** in the residual stream, with attention heads performing geometric rotations — demonstrating that specific token positions carry distinct geometric structure.

Partition token positions into functional regions: problem statement tokens, operator/equation tokens, and generated-answer tokens. For each region, extract hidden states at a fixed layer (e.g., the terminal layer where topological change is maximal), compute persistence diagrams, and compare. **If answer-generation tokens show topology that differs sharply between difficulty levels while problem-statement tokens do not, it localizes the topological signature of difficulty to the reasoning phase.** This also tests whether the non-monotonic H1 pattern is driven by problem encoding or solution generation. For MATH-500 with Qwen2.5-1.5B-Instruct, the generated CoT tokens are likely the richest signal. **Estimated effort: 3–4 days, primarily data extraction and reorganization.**

---

## 9. Topological compression versus resistance to simplification

Two recent papers make seemingly contradictory claims about how distribution shift affects representation topology. Fay et al. (arXiv 2505.20435, 2025) found that adversarial inputs induce **"topological compression"** — the latent space becomes structurally simpler, collapsing from varied small-scale features into fewer dominant large-scale ones. Datta et al. (arXiv 2501.12522, 2025) found that **OOD data resists topological simplification** — average persistence lifetimes are statistically longer for OOD than for in-distribution data. These may not conflict: adversarial inputs (targeted perturbations) could compress topology, while OOD inputs (naturally harder problems) could resist simplification. **Your MATH difficulty gradient provides a clean test of which regime mathematical reasoning occupies.**

The experiment: for each difficulty level, compute the total persistence (sum of all feature lifetimes) in H0 and H1 at each layer. If harder problems show higher total persistence (resistance to simplification), the Datta et al. pattern holds. If they show lower total persistence with fewer but longer-lived features (compression), the Fay et al. pattern holds. Also compute the number of features versus their average lifetime — these decompose total persistence into "how many" and "how long," distinguishing the two mechanisms. **Your sparsity result (higher difficulty → sparser representations) suggests the compression hypothesis**, since sparsification concentrates information into fewer dimensions. But your H1 entropy shows non-trivial loop structure persists even at high difficulty, suggesting partial resistance to simplification in the topological sense.

This experiment directly engages two high-profile papers and resolves an apparent tension in the literature using your unique controlled-difficulty dataset. **Estimated effort: 1–2 days with existing tools.**

---

## 10. Binding detection between attention topology and hidden-state topology

Your toolkit includes a BindingDetector for measuring topological coupling between paired time series. **Applying this to test whether attention-pattern topology and hidden-state topology are coupled — and whether that coupling changes under distribution shift — would produce a genuinely novel finding that no existing paper addresses.**

Pollano et al. (IJCAI 2024; arXiv 2311.13102) computed PH on attention weight matrices for OOD detection and found attention-derived features outperform CLS embedding features for far-OOD detection. The hallucination detection paper (arXiv 2504.10063, 2025) showed training-free TDA on attention graphs outperforms other methods. But no paper has studied the **coupling** between attention topology and hidden-state topology, or how this coupling varies with difficulty.

For each problem at each layer: (1) compute the persistence diagram of the attention weight matrix (treating the attention matrix as a distance/similarity matrix and applying sublevel-set filtration), and (2) compute the persistence diagram of the hidden-state point cloud. Use your BindingDetector (joint vs. marginal persistence image subtraction) to measure topological coupling between these two views. Run surrogate tests (phase randomization, time shuffle) to assess significance. **The hypothesis: for easy problems where the model is confident, attention and hidden-state topology should be tightly coupled (coordinated processing). For harder problems where the model struggles, coupling should weaken as the attention mechanism fails to organize hidden states effectively.** This leverages your unique surrogate-testing infrastructure — the CouplingBenchmark with transfer entropy and PAC comparisons — to provide rigorous statistical assessment of a novel coupling measure. **Estimated effort: 4–5 days, primarily adapting the BindingDetector to attention matrices.**

---

## How these directions form a coherent research program

The ten experiments fall into three natural clusters that together constitute a comprehensive extension of the preprint. **Deepening existing results** (Directions 1, 5, 9) resolve the non-monotonic H1, localize the z=8.11 signal, and adjudicate between competing theoretical predictions — all achievable in a week of work. **New geometric perspectives** (Directions 2, 3, 7) add predictive power, address methodological limitations of Euclidean PH in high dimensions, and connect to the large intrinsic dimension literature — strengthening the paper's methodological rigor. **Exploratory frontiers** (Directions 4, 6, 8, 10) test universality, introduce zigzag persistence, probe token-level structure, and study cross-modal topological coupling — any one of which could become a paper's strongest contribution.

The priority ordering reflects impact-to-effort: Direction 1 (per-layer decomposition) and Direction 9 (compression vs. resistance) require minimal new code and directly strengthen the paper's narrative. Direction 2 (correct vs. incorrect) has the highest standalone impact. Direction 3 (spectral PH) addresses the most serious methodological concern. Directions 4–10 progressively expand scope. On your hardware, the full program is achievable in approximately 4–6 weeks, with the first three directions completable in the first week.
