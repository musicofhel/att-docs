Experimental Results
====================

ATT has been validated across synthetic, neural, cardiac, and LLM domains through 14
experiments. The first 9 appear in the original preprint; Experiments 10--12 were
discovered via automated multi-agent search using
`CORAL <https://github.com/Human-Agent-Society/CORAL>`_.
Experiments 13--14 extend ATT to transformer hidden-state analysis and LLM
correctness prediction.

.. contents:: On this page
   :local:
   :depth: 2

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 5 30 40 25

   * - #
     - Experiment
     - Key Result
     - Domain
   * - 1
     - Coupling sweep
     - Binding rises from baseline, collapses at synchronization
     - Coupled Lorenz
   * - 2
     - Baseline comparison
     - Max baseline more conservative than sum baseline
     - Coupled Lorenz
   * - 3
     - Benchmark comparison
     - Binding score > TE, PAC, CRQA on heterogeneous timescales
     - Rossler-Lorenz
   * - 4
     - Surrogate validation
     - Controlled FPR with AAFT surrogates
     - Coupled Lorenz
   * - 5
     - Heterogeneous timescales
     - Per-channel delay estimation handles timescale mismatch
     - Rossler-Lorenz
   * - 6
     - Sliding-window transitions
     - CUSUM detects topological regime changes
     - Synthetic bistable EEG
   * - 7
     - Real EEG validation
     - 94.1% precision, 40.6% recall on perceptual switches (N=80)
     - Binocular rivalry SSVEP
   * - 8
     - Cross-region binding
     - Oz-Pz coupling modulates with perception (N=79, p=0.001)
     - Binocular rivalry SSVEP
   * - 9
     - Z-score calibration
     - Selective sensitivity to heterogeneous-timescale coupling
     - Lorenz-Lorenz vs Rossler-Lorenz
   * - 10
     - **Cardiac arrhythmia**
     - **AUROC = 0.8014** (mixed TDA + signal features)
     - MIT-BIH ECG
   * - 11
     - **Künneth confirmation**
     - **Inverted binding F1 = 0.984** (first Künneth test)
     - Reservoir computing
   * - 12
     - **Perfect rivalry detection**
     - **F1 = 1.0** (adaptive per-subject strategy)
     - Binocular rivalry SSVEP
   * - 13
     - **LLM hidden-state topology**
     - **z = 8.12** terminal-layer significance; correctness AUROC = 0.580
     - Qwen2.5-1.5B on MATH-500
   * - 14
     - **Topo-confidence validation**
     - **Holdout AUROC = 0.948** [0.898, 0.986]; orthogonal to entropy (r = 0.062)
     - Qwen2.5-1.5B on MATH-500


Experiment 10: Cardiac Arrhythmia Detection
-------------------------------------------

**Domain**: MIT-BIH Arrhythmia Database (records 200, 201, 207, 210, 217)

**Method**: 21-feature mixed approach -- 9 topological features (H0/H1 counts,
persistence statistics, entropy) + 12 signal/HRV features (RR-interval
variability, spectral power, statistical moments).

**Result**: Mean AUROC = **0.8014** via leave-one-record-out cross-validation.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Rec. 200
     - Rec. 201
     - Rec. 207
     - Rec. 210
     - Rec. 217
   * - 0.916
     - 0.714
     - 0.622
     - 0.963
     - 0.792

**Key insight**: Pure TDA features alone achieve AUROC ~ 0.60 because persistence
entropy at 10s window scale is rhythm-class-invariant. But combining topological
features with classical signal/HRV features lifts the classifier above the
publishable threshold (0.75). Topological features capture complementary geometric
information.


Experiment 11: Künneth Confirmation via Inverted Binding
--------------------------------------------------------

**Domain**: Reservoir computing (echo state network, 100 neurons)

**Setup**: 3 failure modes (spectral radius drift, noise injection, connection
death) × 3 random seeds. Monitor binding score between PC1 and PC2 of reservoir
state.

**Result**: Overall F1 = **0.985** via hybrid detection.

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 20

   * - Failure Mode
     - Detection Method
     - Mean F1
     - Surrogate F1
   * - Spectral radius drift
     - Inverted binding
     - 0.984
     - 0.986
   * - Noise injection
     - Crocker CUSUM
     - 0.988
     - --
   * - Connection death
     - Crocker CUSUM
     - 0.984
     - --

**Key finding**: The Künneth cross-term prediction states that when coupled systems
synchronize, the joint topology approaches a product of marginals, and the binding
residual vanishes. Experiment 11 confirms this via inverted binding detection:
spectral radius drift drives synchronization, the binding score *decreases*, and an
inverted alarm (trigger on decrease) detects the onset with F1 = 0.984. This is the
**first direct experimental test** of the Künneth cross-term prediction.

The surrogate ablation (F1 = 0.986 on phase-randomized surrogates) confirms that
the detector responds to genuine topological change, not mere non-stationarity.


Experiment 12: Perfect Perceptual Switch Detection
---------------------------------------------------

**Domain**: Binocular rivalry SSVEP (same dataset as Experiments 7-8)

**Baseline**: Experiment 7 achieved 94.1% precision but only 40.6% recall.

**Adaptive strategy**:

1. Classify subjects by switch density: dense (≥ 20 switches) vs sparse
2. Dense subjects: Oz alpha power (8-13 Hz) with peak detection
3. Sparse subjects: Windowed TDA constrained to behavioral switch region

**Result**: F1 = **1.000** (perfect detection across all test subjects)

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Method
     - Mean Recall
     - Mean F1
     - Strategy
   * - TDA-only (Exp. 7)
     - 40.6%
     - ~0.57
     - Uniform sliding-window PH
   * - Adaptive (Exp. 12)
     - 100%
     - **1.000**
     - Alpha power (dense) + TDA (sparse)

**Key insight**: The recall gap was caused by over-reliance on topology for a signal
with a clean spectral signature. Alpha power on Oz captures perceptual switches
directly for dense-switching subjects. TDA is still needed for sparse/ambiguous
cases. Topological and spectral methods are complementary.


Experiment 13: LLM Hidden-State Topology (Directions 1--10)
-----------------------------------------------------------

**Domain**: Qwen2.5-1.5B-Instruct on MATH-500 (5 difficulty levels, 100 problems
each), cross-validated on Phi-2, Pythia-1.4B, and StableLM-2-1.6B.

**Motivation**: Phase 5 extension item #5 asked whether attractor topology applies to
transformer hidden states. Ten research directions systematically answered this.

**Method**: Extract hidden states at all 29 layers for 500 math problems. Compute
persistent homology (H0, H1, H2) on per-layer point clouds, intrinsic dimension
profiles, CROCKER matrices, zigzag persistence, and attention-hidden coupling.

**Key Results**:

.. list-table::
   :header-rows: 1
   :widths: 10 35 30 25

   * - Dir.
     - Question
     - Key Metric
     - Finding
   * - D1
     - Where does topological discrimination live?
     - Peak z-score = 8.12
     - Terminal layer (28/28)
   * - D2
     - Can topology predict correctness?
     - AUROC = 0.580 (16 features)
     - Above chance; H0_total_persistence best
   * - D5
     - CROCKER visualization of difficulty
     - L1 distance = 43.0 (easy vs hard)
     - Monotonic gradient confirmed
   * - D6
     - Cross-model universality
     - H1 non-monotonic entropy
     - Universal across 4/4 models
   * - D7
     - Intrinsic dimension profiles
     - Terminal ID: 6.7 → 12.0 (L1 → L5)
     - ID increases with difficulty
   * - D9
     - Compression vs resistance
     - Pattern: compression
     - Harder problems compress topology
   * - D10
     - Attention-hidden coupling
     - Binding: 0.683 → 0.465 (L1 → L5)
     - Monotonic decrease with difficulty

**Key insights**:

1. **Terminal-layer concentration.** The topological signal is overwhelmingly
   concentrated in the final transformer layer (z = 8.12 at layer 28; all 28 layers
   significant). This parallels BLOOD-framework findings on OOD detection.

2. **H1 non-monotonic entropy is universal.** All four tested models show a jump in
   H1 persistence entropy from Level 1 to Level 2, then a plateau through Levels 3--5.
   The terminal-layer effect is Qwen-specific (1/4 replicate).

3. **Attention-hidden decoupling under difficulty.** The binding score between attention
   patterns and hidden-state topology monotonically decreases as problem difficulty
   increases (0.683 → 0.465). This is a novel application of ATT's binding framework
   to transformer internals.

**Implementation**: ``att/llm/`` subpackage (8 modules). See API reference for
``HiddenStateLoader``, ``LayerwiseAnalyzer``, ``TopologicalFeatureExtractor``,
``CROCKERMatrix``, ``ZigzagLayerAnalyzer``, ``TokenPartitioner``,
``AttentionHiddenBinding``.


Experiment 14: Topo-Confidence Validation (Pathway 1)
-----------------------------------------------------

**Domain**: Qwen2.5-1.5B-Instruct on MATH-500 (same dataset as Experiment 13).

**Background**: Experiment 13 Direction 2 achieved AUROC = 0.580 with 16 topological
features. The `topo-confidence <https://github.com/musicofhel/topo-confidence>`_
project applied CORAL multi-agent feature engineering to push this substantially
higher: 78 features across 4 tiers (A/B/C/D), reaching CV AUROC = 0.977 in the
CORAL optimization loop. Pathway 1 answers: how much of this survives honest holdout
evaluation?

**Method**: Stratified 400/100 train/holdout split. Train-only PCA to eliminate
statefulness leakage (31/78 features flagged). Tier ablation (cumulative
A → A+B → A+B+C → A+B+C+D) locked before consulting holdout. Bootstrap 95% CI.

**Result**: Holdout AUROC = **0.948** [0.898, 0.986].

.. list-table::
   :header-rows: 1
   :widths: 20 10 20 25

   * - Evaluation
     - N
     - AUROC
     - Note
   * - CV-500 (CORAL)
     - 500
     - 0.977
     - PCA leakage inflated
   * - CV-train-400
     - 400
     - 0.932
     - Honest train CV
   * - **Holdout-100**
     - 100
     - **0.948** [0.898, 0.986]
     - Holdout exceeds train CV

.. list-table::
   :header-rows: 1
   :widths: 20 15 20 20

   * - Tier
     - N features
     - CV-train-400
     - Holdout
   * - A (baseline PH)
     - 9
     - 0.789
     - 0.827
   * - A+B (+ layer dynamics)
     - 39
     - 0.908
     - 0.932
   * - A+B+C
     - 44
     - 0.922
     - 0.941
   * - A+B+C+D (full)
     - 78
     - 0.927
     - 0.950

**Key findings**:

1. **Tier B is the value jump.** Adding layer cosine similarities and dynamics features
   (Tier B, 30 features) increases holdout AUROC by +0.105 over baseline PH alone.
   Tiers C and D contribute marginal additional signal (+0.009 each on holdout).

2. **Orthogonal to output entropy.** Correlation between topo-confidence and output
   entropy: r = 0.062. Entropy-based routing approaches random baseline accuracy on
   MATH-500; topology captures genuinely different signal.

3. **Routing works.** At 10% coverage, topo-confidence achieves 50% trusted accuracy
   vs 0% for entropy routing. At 39% coverage: 28.2% vs 10.3%. Topology dominates
   entropy at every meaningful operating point.

**Decision**: GO for Pathways 2--4 (activation steering, topology-aware fine-tuning,
topologically-informed distillation).

Full validation protocol and artifacts:
`topo-confidence/pathway1/ <https://github.com/musicofhel/topo-confidence/tree/main/pathway1>`_.


Practical Lessons
-----------------

Key methodological takeaways from Experiments 10--14 (see individual
``coral_archive/*/SUMMARY.md`` for full details):

1. **Pure TDA is not always enough.** Cardiac arrhythmia detection achieves only
   AUROC ~ 0.60 with topological features alone -- persistence entropy at 10s
   window scale is rhythm-class-invariant. Combining TDA with classical signal/HRV
   features lifts performance to 0.80. Always consider hybrid feature sets for
   real-world classification.

2. **Adaptive beats uniform.** Per-subject strategy selection (dense-switching
   subjects use spectral power; sparse subjects use windowed TDA) closes the
   rivalry recall gap from 40.6% to 100%. Uniform pipelines leave performance
   on the table when subjects differ in signal characteristics.

3. **Binding can decrease.** The Künneth cross-term prediction: when coupled
   systems synchronize, the joint attractor collapses toward a product of
   marginals and the binding residual vanishes. Detection requires an *inverted*
   alarm (trigger on binding score drop, not rise). Experiment 11 is the first
   direct experimental test.

4. **Always validate with surrogates.** Phase-randomized surrogate ablation
   distinguishes genuine topological coupling from mere non-stationarity. All
   CORAL experiments used this, and it should be standard practice for any
   real-world binding claim.

5. **Implementation matters.** The vineyard tracking concept (persistence
   diagram matching over time) went from F1 = 0.898 (Hungarian matching) to
   F1 = 1.0 (greedy top-K) with parameter tuning alone. Sound theory with
   poor implementation underperforms.

6. **Feature engineering compounds.** The jump from AUROC 0.580 (Experiment 13,
   D2, raw PH features) to 0.948 (Experiment 14, CORAL-engineered features)
   demonstrates that topology provides the *right signal* but extracting it
   requires systematic feature search. The key insight: layer-to-layer dynamics
   (cosine similarities, curvature, SVD spectrum) carry more discriminative power
   than single-layer PH summaries. CORAL's competitive multi-agent approach found
   this in hours.


How These Experiments Were Run
------------------------------

Experiments 10--12 were discovered by `CORAL <https://github.com/Human-Agent-Society/CORAL>`_,
a multi-agent competitive experiment runner. CORAL is the **orchestrator**; ATT is
the **subject under test**.

Each CORAL run launches 4 autonomous AI agents (Claude Code instances) in isolated
git worktrees. The agents write experiment scripts that call ATT's Python API
directly:

- ``att.embedding.TakensEmbedder`` -- delay embedding with auto AMI/FNN estimation
- ``att.topology.PersistenceAnalyzer`` -- persistent homology via Ripser/GUDHI
- ``att.binding.BindingDetector`` -- joint-vs-marginal persistence image subtraction
- ``att.transitions.TransitionDetector`` -- sliding-window PH with CUSUM changepoint detection
- ``att.surrogates`` -- phase-randomized surrogate generation for significance testing

Agents independently explore strategies (feature combinations, detection thresholds,
per-subject adaptation), and CORAL scores each attempt against a target metric. The
winning strategies validate ATT's theoretical claims -- the Künneth cross-term
prediction (Experiment 11), binding score sensitivity to heterogeneous timescales
(Experiment 10), and complementarity of topological and spectral methods
(Experiment 12).

Experiments 13--14 extend ATT's scope from dynamical systems to transformer hidden
states. Experiment 13 was conducted via a 10-direction research roadmap, each
direction building on ATT infrastructure (``PersistenceAnalyzer``, ``BindingDetector``)
adapted for point clouds extracted from LLM layers. Experiment 14 was conducted in a
standalone repository
(`topo-confidence <https://github.com/musicofhel/topo-confidence>`_) using CORAL
for feature engineering, followed by a rigorous 4-phase validation protocol
(reproduction, holdout evaluation, tier ablation, seed sensitivity, routing prototype).


CORAL Archive
-------------

Full artifacts from the automated experiment search are stored in
``coral_archive/`` at the repository root:

- **108 scored attempts** (JSON) with commit hashes, scores, and strategy descriptions
- **53 agent notes** (Markdown) capturing discovery insights and synthesis
- **4 task summaries** with winning strategies and key findings

See `coral_archive/README.md <../coral_archive/README.md>`_ for details.
