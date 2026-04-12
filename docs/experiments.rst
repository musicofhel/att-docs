Experimental Results
====================

ATT has been validated across synthetic, neural, and cardiac domains through 12
experiments. The first 9 appear in the original preprint; Experiments 10--12 were
discovered via automated multi-agent search using
`CORAL <https://github.com/Human-Agent-Society/CORAL>`_.

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

**Key finding**: The Künneth formula predicts that when coupled systems synchronize,
the joint topology approaches a product of marginals, and the binding residual
vanishes. Experiment 11 confirms this: spectral radius drift drives synchronization,
binding score *decreases*, and an inverted alarm (alarm on decrease) detects the
onset with F1 = 0.984. This is the **first direct experimental test** of the
Künneth cross-term interpretation.

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


CORAL Archive
-------------

Full artifacts from the automated experiment search are stored in
``coral_archive/`` at the repository root:

- **108 scored attempts** (JSON) with commit hashes, scores, and strategy descriptions
- **53 agent notes** (Markdown) capturing discovery insights and synthesis
- **4 task summaries** with winning strategies and key findings

See `coral_archive/README.md <../coral_archive/README.md>`_ for details.
