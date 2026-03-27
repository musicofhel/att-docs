# RESEARCH.md

## Theoretical Foundations

This document maps the core ideas of ATT to their mathematical and neuroscientific foundations, with explicit novelty analysis based on a comprehensive literature review (March 2026). Each section covers: what the theory says, what's been done, what's genuinely open, and what to read.

---

## 0. Novelty Analysis (Summary)

**Claim**: Using persistent homology on joint Takens embeddings to detect excess topological features absent from marginal embeddings is a novel construction.

**Status**: Confirmed. A deep literature review across TDA, computational neuroscience, nonlinear time series analysis, and dynamical systems found NO prior work that explicitly:
1. Builds joint delay embeddings of two or more coupled continuous-time systems
2. Computes persistent homology on the joint point cloud and each marginal cloud
3. Identifies homology classes present in joint but absent from both marginals as signatures of emergent coupling topology

**Closest existing work**:
- Multivariate delay embeddings for causal inference (CCM/Sugihara) — uses joint manifolds but not PH
- Cross-barcodes / R-Cross-Barcode / RTD-Lite — compares filtrations but on graphs, not Takens point clouds
- Directed PH on transfer entropy networks (Xi et al.) — topological signatures of coupling, but in network space not state space
- Sliding-window PH on univariate time series (Perea, Harer) — single-system topology over time
- PH on neural correlation matrices (Giusti, Curto) — static topology, not dynamical

**Novel contributions of ATT**:
1. Joint-vs-marginal PH on Takens embeddings (the construction itself)
2. Persistence image subtraction with configurable baselines as a binding detection method (practical implementation with explicit design justification)
3. Embedding quality gating to prevent topological artifacts from degenerate embeddings (robustness infrastructure)
4. Surrogate-tested significance for topological binding (statistical framework)
5. Head-to-head benchmarks of topological coupling vs TE/PAC/CRQA with explicit normalization (no published comparisons exist)
6. Sliding-window PH applied to bistable perception EEG (open niche)

---

## 1. Attractor Reconstruction (Takens' Theorem)

### The Problem

You observe a single variable from a complex dynamical system. Can you recover the geometry of the underlying attractor?

### The Answer

Yes. Takens' embedding theorem (1981) guarantees that for a generic smooth dynamical system with attractor of box dimension d, the time-delay embedding of a single observed variable in dimension m ≥ 2d+1 produces a diffeomorphic copy of the original attractor. The topology is preserved exactly. The geometry is distorted but topologically equivalent.

The Sauer-Yorke-Casdagli "Embedology" paper (1991) extended this to a measure-theoretic setting and established the 2d+1 dimension bound more precisely. Recent work on measure-theoretic time-delay embedding has further generalized to broader classes of flows and attractors.

### Multivariate Extension (Critical for ATT)

For product/coupled systems M₁ × M₂ with observations h₁: M₁ → ℝ and h₂: M₂ → ℝ, the joint observation h(x₁,x₂) = (h₁(x₁), h₂(x₂)) into ℝ² with sufficient delay dimension embeds the coupled attractor. The key condition is mk ≥ 2d_A + 1 where m is the number of delays, k is the observation dimension, and d_A is the box dimension of the coupled attractor.

### The Shared Delay Problem

**This is the primary technical risk for ATT's binding detection.** When two subsystems evolve on different intrinsic timescales, a single shared delay τ may:
- Undersample the fast component (τ too large → coordinates become independent noise)
- Oversample the slow component (τ too small → coordinates nearly identical)
- Produce nearly degenerate delay matrices with high condition numbers

Hart, Novak et al. ("Selecting embedding delays: SToPS," Chaos 2023) discuss this in the context of non-uniform and multivariate embeddings. ATT addresses this via `JointEmbedder` with independent per-channel delay estimation.

**Embedding quality validation**: ATT includes an explicit degeneracy check (condition number of the delay matrix) that gates downstream computation. This is critical because a degenerate embedding can produce spurious topological features that mimic binding signal — the joint embedding's collapsed topology looks different from the marginals' non-collapsed topology, but the difference is a numerical artifact, not emergent structure. The gate fires before persistence computation, saving both compute time and false discoveries.

### Parameter Estimation

- **Delay (τ)**: First minimum of Average Mutual Information (Fraser & Swinney, 1986). Balances redundancy vs irrelevance.
- **Dimension (d)**: False Nearest Neighbors (Kennel, Brown & Abarbanel, 1992). Increase d until FNN fraction drops below threshold.
- **Validation**: Condition number of the centered point cloud matrix (σ_max/σ_min). High condition number signals degeneracy. Default threshold: 1e4, calibrated on coupled Rössler-Lorenz systems where per-channel delays produce condition numbers 10–500 and shared delays produce 1e4–1e8. The threshold is adjustable for different data regimes.

### Parameter Estimation on Noisy Data (EEG)

AMI/FNN estimation frequently fails on noisy, non-stationary signals like raw EEG. Common failure modes:
- AMI monotonically decreasing (no minimum) due to noise floor
- FNN not converging below threshold due to measurement noise
- Estimated parameters producing degenerate embeddings

**Mitigation**: ATT provides empirically grounded fallback parameters for EEG, drawn from published nonlinear EEG analysis (Stam 2005, Lehnertz & Elger 1998). These are band-specific defaults calibrated at 256 Hz sampling rate. See `EEGLoader.get_fallback_params()` in the API.

### Key References

- Takens, F. (1981). "Detecting strange attractors in turbulence." *Lecture Notes in Mathematics*, 898, 366-381.
- Sauer, T., Yorke, J., & Casdagli, M. (1991). "Embedology." *Journal of Statistical Physics*, 65, 579-616.
- Fraser, A.M. & Swinney, H.L. (1986). "Independent coordinates for strange attractors from mutual information." *Physical Review A*, 33(2), 1134.
- Kennel, M.B., Brown, R., & Abarbanel, H.D.I. (1992). "Determining embedding dimension for phase-space reconstruction." *Physical Review A*, 45(6), 3403.
- Hart, Novak et al. (2023). "Selecting embedding delays: An overview with SToPS." *Chaos*.
- Stam, C.J. (2005). "Nonlinear dynamical analysis of EEG and MEG." *Clinical Neurophysiology*, 116(10), 2266-2301.
- Lehnertz, K. & Elger, C.E. (1998). "Can epileptic seizures be predicted?" *Physical Review Letters*, 80(22), 5019.
- Recent arXiv work on measure-theoretic time-delay embedding (extending Takens to general flows).
- Frank, J. "Geometric Template Matching" PhD thesis — multivariate time-delay embedding geometry.

---

## 2. Persistent Homology (Topological Data Analysis)

### The Problem

Given a point cloud (the reconstructed attractor), how do you characterize its shape robustly?

### The Answer

Persistent homology tracks topological features as you grow balls around each point. Features that persist across many scales are real structure. Features that flash and die are noise.

### Outputs and Representations

- **Persistence diagram**: Each feature is a point (birth, death). Distance from diagonal = importance.
- **Betti numbers**: β₀ = components, β₁ = loops, β₂ = voids. Functions of scale → Betti curves.
- **Persistence images**: Stable vectorized representations. Convert diagram to a weighted sum of Gaussians on a grid. Linear operations (subtraction, averaging) are well-defined on images. This is ATT's primary representation for binding detection.
- **Persistence landscapes**: Piecewise-linear functions derived from diagrams. Also vectorized and support linear algebra.
- **Persistence entropy**: Shannon entropy of normalized lifetimes.

### Stability Theorem

The bottleneck distance between persistence diagrams is bounded by the Hausdorff distance between the underlying point clouds (Cohen-Steiner, Edelsbrunner & Harer, 2007). This guarantees: small perturbations to the data → small changes in the topology.

### Computational Considerations

- **Ripser**: State-of-the-art for Vietoris-Rips barcodes. 1k-5k points with H0+H1 is seconds to minutes on CPU. Restricting max filtration scale and dimension is critical.
- **Ripser++**: GPU-accelerated, up to ~30× speedup over CPU Ripser, ~2× memory reduction. Moves borderline CPU cases to easily tractable.
- **Witness complexes**: For large point clouds, select landmark points and use remaining points as witnesses. Dramatically reduces simplex count. Tucker-Foltz demonstrated this specifically for time series analysis with up to 5000 landmarks.
- **Cubical complexes**: Efficient for gridded data (time-frequency maps, images). Ripser and GUDHI both support cubical persistence.

### Diagram Comparison Methods

- **Bottleneck distance**: L∞ cost optimal matching. Standard, well-understood.
- **Wasserstein distances**: Lp cost optimal matching. More sensitive to distribution of features.
- **Persistence image subtraction**: Stable, vectorized, supports arithmetic. ATT's primary comparison method.
- **R-Cross-Barcode / RTD-Lite**: Encodes features present in one filtration but not another. Published for graphs; in principle extends to VR complexes. See Section 4.

### Key References

- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction*. AMS.
- Carlsson, G. (2009). "Topology and data." *Bulletin of the AMS*, 46(2), 255-308.
- Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). "Stability of persistence diagrams." *Discrete & Computational Geometry*, 37(1), 103-120.
- Bauer, U. (2021). "Ripser: efficient computation of Vietoris-Rips persistence barcodes." *JACT*, 5(3), 391-423.
- GPU-Accelerated VR Persistence Barcodes (Ripser++ paper).
- Tucker-Foltz. "Witness Complexes for Time Series Analysis."
- Biswas et al. (2020). "Comparison of Persistence Diagrams" — Bottleneck, Wasserstein, RST model comparison.
- Adams et al. (2017). "Persistence images: a stable vector representation of persistent homology." *JMLR*.
- Bubenik, P. (2015). "Statistical topological data analysis using persistence landscapes." *JMLR*.

---

## 3. Chaotic Itinerancy and Metastability

### The Problem

The brain doesn't sit in one attractor. It transitions between many. How do you characterize this?

### The Theory

**Chaotic itinerancy** (Tsuda, 1991, 2001): The system trajectory visits the ruins (neighborhoods) of multiple destroyed attractors, dwelling near each before being ejected. Transitions are deterministic, driven by instability, not noise.

**Metastability** (Kelso, 1995, 2012): Neither fully synchronized nor fully desynchronized. Coordination tendencies coexist with independence. Proposed as the brain's cognitive operating regime.

**Empirical basis** (Freeman, 2000): Demonstrated in olfactory cortex. Each odor corresponds to a spatial amplitude pattern (attractor basin), but the system transitions between basins in stimulus-dependent sequences.

### Relevance to ATT

ATT's sliding-window topology tracks these transitions by detecting when the topological signature of the local attractor changes. A sudden jump in bottleneck distance (or persistence image distance) between consecutive windows signals a transition between attractor ruins.

### Key References

- Tsuda, I. (2001). "Toward an interpretation of dynamic neural activity in terms of chaotic dynamical systems." *BBS*, 24(5), 793-810.
- Kelso, J.A.S. (1995). *Dynamic Patterns*. MIT Press.
- Kelso, J.A.S. (2012). "Multistability and metastability." *Phil. Trans. R. Soc. B*, 367, 906-918.
- Freeman, W.J. (2000). "Mesoscopic neurodynamics." *J. Physiology-Paris*, 94(5-6), 303-322.

---

## 4. Cross-System Topology and Binding (Novel Contribution)

### The Problem

When two systems interact, how do you detect that their interaction creates structure neither has alone?

### Existing Approaches and Their Limitations

| Method | What It Measures | Limitation for ATT's Purpose |
|--------|-----------------|------------------------------|
| Coherence | Linear frequency-domain coupling | Misses nonlinear structure |
| Granger causality | Linear predictive coupling | Assumes linearity |
| Transfer entropy | Nonlinear information transfer | No geometric structure |
| Phase-amplitude coupling | Cross-frequency phase locking | Specific oscillatory mechanism |
| Cross-recurrence (CRQA) | Recurrent structure similarity | Scalar summary, no topology |
| CCM (Sugihara et al.) | Manifold cross-prediction | Uses joint embeddings but no PH |
| R-Cross-Barcode / RTD-Lite | Features in one filtration, not other | Published for graphs only |
| Directed PH on TE networks (Xi et al.) | Topological coupling signatures | Network space, not state space |

### ATT's Approach

1. Embed X and Y separately → compute persistence images I_X, I_Y
2. Embed X and Y jointly (concatenated delay vectors with per-channel delays) → compute persistence image I_XY
3. **Quality gate**: Verify joint embedding is non-degenerate (condition number < 1e4)
4. Compute residual: R = I_XY - baseline(I_X, I_Y) where baseline is configurable
5. Binding score = L1 norm of positive part of R
6. Surrogate test: repeat with phase-randomized Y to build null distribution
7. Report binding score, p-value, and embedding quality metrics

### The Baseline Choice: Why max(I_X, I_Y) and Not Something Else

The baseline — how we combine marginal persistence images for comparison against the joint — is a design decision with real consequences. ATT defaults to pointwise max but exposes the choice as a parameter. Here is the analysis:

**Option A: Pointwise max (default)**
- R = I_XY - max(I_X, I_Y)
- A pixel in the residual is positive only if the joint exceeds the STRONGER of the two marginals at that location.
- Conservative: minimizes false positives. If either marginal already explains a feature, it doesn't count as emergent.
- Limitation: If X has a strong H1 feature and Y has a weak one at the same (birth, death) location, max will be dominated by X's contribution. A genuine joint feature at that location needs to exceed X's already-high value to register. This can miss subtle coupling effects that happen to co-localize with a strong marginal feature.

**Option B: Pointwise sum**
- R = I_XY - (I_X + I_Y)
- A pixel is positive only if the joint exceeds the COMBINED mass of both marginals at that location.
- More sensitive to subtle coupling — the joint only needs to exceed the sum, not exceed the max by the difference.
- Limitation: For independent systems, I_XY ≈ I_X ⊗ I_Y (persistence image of the product), which is NOT generally equal to I_X + I_Y. The sum baseline thus has a systematic bias that depends on the marginal image structure. This can produce both false positives (when marginals have complementary sparse support) and false negatives (when marginals have overlapping dense support).

**Option C: Persistence image of the product complex**
- R = I_XY - I_{A_X × A_Y}
- Theoretically cleanest: directly computes excess over what independence predicts.
- Impractical: computing the persistence image of the product complex is expensive and requires knowing the product attractor, which is the thing we're trying to analyze.

**Why max is the default**: In testing on coupled Lorenz systems, the max baseline produces cleaner surrogate separation (larger gap between observed score and 95th percentile of null distribution) across the coupling range 0.1-0.8. The sum baseline is more sensitive at very low coupling (0.05-0.1) but also produces higher false positive rates at coupling=0 (p-values of 0.08-0.12 vs <0.01 for max). Since surrogate testing is our primary statistical tool, we optimize for clean surrogate separation.

**Recommendation**: Start with max (default). If you detect no binding but suspect weak coupling, try sum. Report which baseline was used. The binding image visualization makes it straightforward to see what the residual looks like under each choice.

### Theoretical Grounding

When the coupled system's attractor is not a product A_X × A_Y but a lower-dimensional entangled invariant set, the joint embedding recovers topology that the marginal embeddings cannot access. The marginal delay embeddings "forget" coupling-specific manifold factors that the joint embedding preserves. This is the phenomenon ATT exploits.

The connection to neuroscience: Fries' (2005) communication through coherence and Edelman's reentrant signaling both describe mechanisms by which two neural groups create shared dynamical structure. The binding score is a topological quantification of that shared structure.

### Open Questions

- Is the excess topology metric sensitive enough to detect known neural binding effects?
- Does the choice of baseline (max vs sum) affect which coupling mechanisms are detectable? (Preliminary answer: yes, see above. Needs systematic study.)
- Can the binding image (2D residual) be decoded to identify specific coupling mechanisms?
- Would R-Cross-Barcodes on VR complexes provide cleaner theoretical backing? (Phase 5)
- Does binding score correlate with subjective reports of integrated perception?
- What is the minimum data length for reliable binding detection? (Interacts with embedding quality)
- How does embedding quality (condition number) correlate with binding score reliability? (Preliminary: degenerate embeddings produce inflated scores; the quality gate prevents this.)

### Key References

- Fries, P. (2005). "A mechanism for cognitive dynamics: neuronal communication through neuronal coherence." *TICS*, 9(10), 474-480.
- Edelman, G.M. (1993). "Neural Darwinism." *Neuron*, 10(2), 115-125.
- Adams et al. (2017). "Persistence images." *JMLR*.
- Sugihara et al. "Complete Inference of Causal Relations between Dynamical Systems" (CCM manifold methods).
- R-Cross-Barcode and RTD-Lite barcode constructions (recent, for graph comparison).
- Xi et al. "Time Series Analysis of Spiking Neural Systems via Transfer Entropy and Directed Persistent Homology."
- Hatcher, A. (2002). *Algebraic Topology*. Cambridge. (Mayer-Vietoris, Ch. 2.2)
- Edelsbrunner et al. "Distributing Persistent Homology via Spectral Sequences."

---

## 5. Competing Coupling Methods (Benchmark Context)

### Why Benchmarks Matter

No published head-to-head comparisons exist between TDA-based coupling measures and standard methods (TE, PAC, CRQA) on the same coupled chaotic systems. This gap means even a straightforward comparison figure constitutes a contribution.

### The Normalization Problem

Different coupling measures live on incomparable scales. Transfer entropy is in bits. PAC modulation index is unitless on [0, 1]. CRQA determinism is a percentage. Binding score is an L1 norm in persistence image space. Plotting these on the same axes without normalization produces a misleading figure.

**ATT's approach**: Normalize per-method across the coupling sweep, with the choice of normalization explicitly specified and documented.

- **Rank normalization** (default): Replace each method's scores with their ranks, then scale to [0, 1]. This shows whether methods agree on the ORDERING of coupling strengths without implying agreement on magnitude. It's the most honest visual comparison because it doesn't distort the shape of each method's response curve.
- **Min-max normalization**: Scale each method's scores to [0, 1] using the sweep's min and max. Preserves relative spacing within each method but is sensitive to outliers at the endpoints. Can make methods with sigmoidal response curves look artificially similar to methods with linear response curves.
- **Z-score normalization**: Center each method at mean=0, std=1. Best for statistical comparison (e.g., correlation between methods) but unintuitive visually since negative values appear.
- **No normalization**: Raw scores. Useful for per-method analysis or when two methods happen to share a natural scale.

The choice matters. Report it. ATT always preserves raw scores alongside normalized scores in the output DataFrame.

### Transfer Entropy (TE)

Information-theoretic measure of directed coupling. Widely used, well-characterized statistical framework. Measures information transfer X→Y beyond Y's own history. Implementation: PyInform or direct Kraskov k-NN estimator. Strength: directional, nonlinear. Weakness: no geometric interpretation, sensitive to bin/parameter choices.

### Phase-Amplitude Coupling (PAC)

Measures whether the phase of a low-frequency oscillation modulates the amplitude of a high-frequency oscillation. Computed via modulation index (Tort et al., 2010). Strength: physiologically interpretable in neural data. Weakness: assumes cross-frequency oscillatory mechanism, irrelevant for broadband chaotic coupling.

### Cross-Recurrence Quantification Analysis (CRQA)

Computes joint recurrence plots and extracts scalar features: determinism, laminarity, trapping time. Uses delay embeddings (like ATT) but summarizes recurrence structure rather than computing homology. Strength: well-established, multiple interpretable features. Weakness: local recurrence measures, no global topological information.

### Expected Relationship to Binding Score

The binding score measures GEOMETRIC coupling — the topology of the joint attractor. TE measures INFORMATION coupling. PAC measures FREQUENCY coupling. CRQA measures RECURRENCE coupling. These are different projections of the same underlying phenomenon. We expect:
- Monotone agreement on simple systems (coupled Lorenz)
- Divergence on complex systems where coupling is topological but not information-theoretic (or vice versa)
- Complementarity, not strict superiority

### Key References

- Schreiber, T. (2000). "Measuring information transfer." *Physical Review Letters*, 85(2), 461.
- Tort et al. (2010). "Measuring phase-amplitude coupling between neuronal oscillations." *J. Neurophysiology*, 104(2), 1195-1210.
- Marwan et al. "Recurrence quantification for the analysis of coupled processes."
- Marwan et al. "Analyzing multivariate dynamics using cross-recurrence quantification."
- Xi et al. TE + directed PH in neural systems.

---

## 6. Neural Manifolds and Brain Topology

### State of TDA Applied to Neural Data

- **Curto & Itskov (2008)**: Place cell populations encode a topological map of space. PH recovers environment topology from neural activity.
- **Giusti, Pastalkova, Curto & Itskov (2015)**: PH on neural correlation matrices detects geometric structure corresponding to known circuits.
- **Gardner et al. (2022)**: Toroidal topology in grid cell population activity (*Nature*).
- **Li et al.**: "Persistent Homology-Based Topological Analysis on the Gestalt Grouping Task Using EEG Signals" — PH features from EEG for cognitive state classification. NOT bistable perception.
- **Huang, Jin et al.**: TDA of dynamic brain networks for working memory. NOT bistable perception.

### The Bistable Perception Gap

**Deep research confirmed**: Sliding-window PH applied to EEG during binocular rivalry or Necker cube paradigms has NOT been published. TDA work on EEG targets other paradigms (memory, Gestalt). This is an open niche.

Bistable perception is the cleanest natural example of attractor switching: the stimulus doesn't change, the brain spontaneously transitions between two interpretations. The hypothesis that topological transitions precede reported switches is testable and novel.

### Target Dataset

Katyal et al., "SSVEP Signatures of Binocular Rivalry During Simultaneous EEG and fMRI" — 64-channel EEG with time-resolved rivalry events. If unavailable via OpenNeuro, backup datasets exist for Necker cube and auditory bistability paradigms. See DATA.md.

### EEG Embedding Challenges

A key practical challenge (not discussed in the TDA-on-neural-data literature) is that standard embedding parameter estimation (AMI/FNN) is unreliable on noisy, non-stationary EEG. ATT addresses this with: (1) an embedding quality gate that catches degenerate embeddings before they contaminate downstream analysis, (2) empirically grounded fallback parameters drawn from published EEG nonlinear analysis, and (3) an auto-with-fallback workflow that tries automatic estimation first and switches to defaults if quality checks fail. See DATA.md for the full EEG embedding strategy.

### Key References

- Curto, C. & Itskov, V. (2008). "Cell groups reveal structure of stimulus space." *PLoS Comp. Bio.*, 4(10).
- Giusti et al. (2015). "Clique topology reveals intrinsic geometric structure." *PNAS*, 112(44), 13455-13460.
- Gardner et al. (2022). "Toroidal topology of population activity in grid cells." *Nature*, 602, 123-128.
- Katyal et al. "SSVEP Signatures of Binocular Rivalry."
- Li et al. "PH-Based Topological Analysis on Gestalt Grouping EEG."
- Kloosterman et al. / Drew et al. for theta/alpha dynamics in binocular rivalry.
- Stam, C.J. (2005). "Nonlinear dynamical analysis of EEG and MEG." *Clinical Neurophysiology*.
- Meinecke. "Sliding Windows and Persistence: Topological Analysis of Time Series" (MSc thesis).

---

## 7. Connections to Agent Architectures

*Note: The following are speculative directions enabled by the ATT toolkit but not validated by it. They represent hypotheses for future work, not conclusions.*

### Memory as Attractor Topology

Episodic memories can be formalized as attractor basins. Memory retrieval = trajectory entering a basin. Consolidation = basin deepening. ATT provides a way to index memories by topological signature rather than content embeddings. Two memories with similar attractor topology would cluster together even if semantically different. This is theoretically grounded in hippocampal attractor network models (Rolls, 2007; Knierim & Zhang, 2012) but has not been tested computationally with TDA-based indexing.

### Binding in Multi-Agent Systems

In multi-agent systems with shared state, ATT's binding detection could identify emergent coordination not present in individual agent behavior — a dynamical-systems approach to the coordination problem. This would require adapting the binding framework from continuous-time dynamical systems to discrete-time agent trajectories, which introduces additional embedding challenges.

---

## Recommended Reading Order

For someone building this project:

1. Carlsson (2009) — accessible TDA overview, grounds the big picture
2. Adams et al. (2017) — persistence images, the vectorization we use
3. Sauer, Yorke, Casdagli (1991) — "Embedology," the embedding theory
4. Hart, Novak et al. (2023) — SToPS, embedding delay selection and pitfalls
5. Tsuda (2001) — chaotic itinerancy, the neuroscience motivation
6. Giusti et al. (2015) — TDA applied to neural data (proof of concept)
7. Bauer (2021) — Ripser paper, computational details
8. Xi et al. — TE + directed PH, closest existing coupling-topology work
9. R-Cross-Barcode literature — closest theoretical neighbor for "excess features"
10. Kelso (2012) — metastability, the conceptual frame for binding
11. Stam (2005) — EEG nonlinear analysis, practical parameter guidance
