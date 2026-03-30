# Cone Prototype: 2-Column, 3-Layer Directed Attractor Projection

**v2 — revised to address common-driver confound, statistical power, and observational blind spots**

## Purpose

Test a single hypothesis: does a directed projection from a source attractor through receiver attractors at different layers produce measurable geometric structure (a cone) in the joint state space that is absent from the marginal state spaces? This is the minimal experiment that validates or kills the construction before scaling to grids.

---

## Architecture

```
Layer 2 (source):     [C]
                     /   \
Layer 3 (receiver): [A3]  [B3]
                     |      |
Layer 5 (receiver): [A5]  [B5]
```

Six attractor nodes total. Two columns (A, B), three layers (2, 3, 5). The source C sits in layer 2 and projects downward. Each column has a receiver in layer 3 and layer 5.

Coupling is directed and asymmetric. C drives A3 and B3. A3 drives A5. B3 drives B5. No upward projections. No lateral coupling between columns A and B. This isolates the cone — any emergent cross-column structure must come through the shared source C.

The layer numbering follows cortical convention: layer 2 is superficial (fast timescale), layer 5 is deep (slow timescale). The timescale asymmetry is the regime where ATT's binding detection has demonstrated power.

---

## Node Dynamics

Each node is an Aizawa attractor. The Aizawa's roughly spherical geometry with a helical escape tube produces cross-sections (circles, deformed annuli) when intersected by a projection — cleaner cone geometry than Lorenz butterflies or Rossler bands.

```python
def aizawa(x, y, z, alpha=0.95, beta=0.7, gamma=0.6,
           delta=3.5, epsilon=0.25, zeta=0.1):
    dx = (z - beta) * x - delta * y
    dy = delta * x + (z - beta) * y
    r2 = x * x + y * y
    dz = gamma + alpha * z - z**3 / 3 - r2 * (1 + epsilon * z) + zeta * z * x**3
    return dx, dy, dz
```

Timescale separation is implemented by scaling dt per layer. Layer 2 runs at dt = 0.005 (fast). Layer 3 runs at dt = 0.008 (medium). Layer 5 runs at dt = 0.012 (slow). This mirrors the empirical finding that superficial cortical layers exhibit faster oscillatory dynamics than deep layers.

---

## Coupling Structure

Coupling is diffusive on the x and y components only (not z), following the convention that lateral/horizontal information flow in cortex is carried by different projection types than vertical/radial flow.

Note on observability: coupling on xy indirectly drives z through the nonlinear terms (z appears in dx/dt and dy/dt), so the effect propagates through the full dynamics. However, the cone will be most visible in the xy subspace of the joint embedding. If the Takens embedding from x-only time series reconstructs primarily z-dynamics (which carry the helical escape tube), the cone could be masked. Experiment 2 addresses this with a coupling-influence subspace projection.

---

## Experiments

### Experiment 0: Sanity check — verify directed cascade

Before any topology, verify that coupling works as intended. Compute lagged cross-correlation between C's x-series and each receiver's x-series. Expected results:

The lag at peak correlation should increase monotonically along the directed chain: C leads A3 leads A5 (and C leads B3 leads B5). The peak correlation magnitude should decrease monotonically: corr(C, A3) > corr(C, A5) and corr(C, B3) > corr(C, B5), reflecting the attenuation across each coupling step. The cross-column correlations corr(A3, B3) should be non-zero (common driver) but should show zero lag (both driven simultaneously by C).

If these cross-correlations don't show a directed cascade, the coupling parameters are wrong and no amount of topological analysis will produce interpretable results. This takes 15 minutes to implement and prevents wasting hours on a misconfigured system. Run at 3-4 coupling strengths to find the regime where the cascade is clean but receivers haven't fully synchronized.

### Experiment 1: Does the cone exist?

**Revised to avoid the common-driver confound.** The v1 hypothesis compared cross-layer binding ([A3; A5]) against within-layer binding ([A3; B3]) and predicted the former would be larger. This is wrong: A3 and B3 both receive the same driving signal from C on the same components with the same strength. Common-cause correlation is one of the strongest statistical dependencies in coupled systems, so within-layer binding could easily exceed cross-layer binding without invalidating the cone construction. The cone is about the *shape* of the joint manifold (expanding cross-section with depth), not about whether cross-layer binding exceeds within-layer binding in magnitude.

**Revised design:** test for the cone's existence via *asymmetry of binding along the directed chain*.

Generate 80,000 time steps (increased from 10,000 for statistical power in later experiments; computationally negligible for a 6-node system). Extract x-component of each node as a 1D time series. Compute the following joint embeddings and their persistent homology using ATT's existing machinery:

**Depth-stratified pairwise joints** (probing cone geometry along the chain):
- [C; A3] — source to first receiver (shallow)
- [C; A5] — source to deep receiver (deep)
- [C; B3] — same, other column
- [C; B5] — same, other column

**Full-chain joint** (probing emergent structure):
- [C; A3; A5] — full directed chain, column A
- [C; B3; B5] — full directed chain, column B

**Cross-column joints** (separating common-driver from cone):
- [A3; B3] — within-layer, common driver
- [A5; B5] — within-layer, common driver (attenuated)

For each joint embedding, compute residual persistence images against the max of constituent marginals using ATT's PI subtraction.

**Primary hypothesis:** the joint [C; A5] manifold will have *different* (richer) topology than the joint [C; A3] manifold. If the cone exists, the further projection should "open up" more state space — expect a shift in the persistence image toward higher persistence values or different Betti numbers. This directly probes directed projection geometry without the common-driver confound.

**Secondary hypothesis:** the full-chain [C; A3; A5] binding score should exceed max([C; A3], [C; A5], [A3; A5]). This tests whether the complete directed chain has emergent topology beyond any pairwise combination — the topological signature of the cone as a 3-stage structure.

**Control:** run with coupling_source = 0. All excess binding should vanish. Run surrogate tests using ATT's phase randomization framework on the coupled case.

### Experiment 2: Does the cone have direction?

This is the core experiment. It requires extending ATT with directional filtration.

Generate 80,000 time steps. Compute the joint embedding of all four receivers [A3; B3; A5; B5] using ATT's JointEmbedder with per-channel delay estimation. This produces a point cloud in ~12-20 dimensions depending on estimated embedding parameters.

**Axis estimation (revised from centroid-to-centroid).** The v1 spec defined the projection axis as the line from C's mean state to the receiver centroid. This assumes a linear axis in a nonlinear system. Instead, use the conditional-mean principal component: for each value of C's state (binned into 20 quantiles of C's x-component), compute the conditional mean of the receiver joint cloud. Take the first principal component of those 20 conditional means as the projection axis. This captures the direction along which C's variation maximally structures the receiver cloud, even if that direction curves in the full state space.

**Depth slicing.** Project all points in the receiver joint cloud onto the estimated axis. Bin into 5 depth slices (reduced from 10 to ensure ~3,200+ points per bin at 80,000 total steps after transient removal and embedding). For each depth slice, compute the persistence diagram of the cross-section restricted to H0 and H1. Once a trend is confirmed at 5 bins, increase resolution to 8-10.

**Coupling-influence subspace analysis (new).** In parallel, estimate the coupling-influence subspace: the 2-3 dimensions of the receiver joint cloud along which C's state has maximal predictive power, computed via canonical correlation analysis (CCA) between C's embedded state and the receiver joint cloud. Recompute the depth-sliced persistence within this projected subspace. If the cone appears in the coupling-influence subspace but not the full Takens embedding, it means the cone is a low-dimensional feature embedded in a high-dimensional attractor state space. This would actually strengthen the theoretical claim — the cone is a submanifold, consistent with Burak and Bhatt's finding that coupling restricts dynamics to a low-dimensional submanifold of the product space.

**Primary hypothesis:** the cross-section topology should change systematically with depth along the projection axis. Near the source injection (shallow depth), expect simpler topology (fewer H1 features — the cone is narrow). At greater depth, expect richer topology (more H1 features — the cone has expanded to illuminate more of the joint state space). This progressive topological enrichment IS the cone.

**Output:** an availability profile — Betti1 as a function of depth along the projection axis. Two versions: one in the full Takens embedding, one in the coupling-influence subspace. This is the core deliverable: the shape of the cone expressed as topology-vs-depth.

**Failure mode:** if cross-section topology is uniform across depth, the projection is isotropic, not conical, and the construction needs revision. If topology is non-monotonic but structured (e.g., peaks at intermediate depth), the geometry may be more complex than a simple cone — worth characterizing but requiring a revised theoretical model.

### Experiment 3: Coupling sweep

Sweep coupling_source from 0.0 to 0.5 in 10 steps. Hold coupling_down equal to coupling_source (uniform coupling chain). At each coupling value, compute the depth-asymmetry binding measure from Experiment 1 (the topological difference between [C; A3] and [C; A5]) and the availability profile from Experiment 2.

Expected outcome: an inverted-U in the depth-asymmetry measure. No binding at zero coupling (no cone). Peak asymmetry at intermediate coupling (the metastable regime where C's influence structures the receivers without enslaving them). Collapse at strong coupling (receivers synchronize to C and each other, destroying the cone — the joint manifold degenerates to a low-dimensional synchronized subspace).

The family of availability profiles across coupling strengths should show cone narrowing at weak coupling (few features illuminated) and widening at moderate coupling, with potential collapse to a line at strong coupling (all receivers locked to source).

### Experiment 4: Timescale ratio sweep

Hold coupling constant at the peak value from Experiment 3. Vary the timescale ratio between layers by changing dt_layer5 while keeping dt_layer2 and dt_layer3 fixed. Test ratios of 1.0 (same timescale), 1.5, 2.0, 2.4, 3.0, 4.0, 5.0.

The upper range (4.0-5.0) covers the biologically relevant ratio between superficial gamma oscillations (~40 Hz) and deep beta/alpha oscillations (~10-15 Hz). The 2025 Scientific Reports bistable perception paper found that L5/L6 gain modulation of L2/3 attractor depth was most effective with substantial timescale separation, suggesting the interesting regime may be at the high end.

**Hypothesis:** the cone should be most detectable when the timescale ratio is large, consistent with ATT's finding that binding detection works for heterogeneous timescales. If the cone is equally detectable at ratio 1.0, the construction generalizes beyond ATT's known operating regime — a significant positive result.

### Experiment 5: Directed vs undirected comparison

Replace the directed coupling topology (C->A3->A5, C->B3->B5, no reverse) with symmetric bidirectional coupling (all six nodes coupled to all others). Match total coupling energy by equalizing the Frobenius norm of the coupling matrix: the directed topology has 4 nonzero coupling entries (C->A3, C->B3, A3->A5, B3->B5), so the symmetric topology with 30 nonzero entries (6x5 bidirectional pairs) gets per-edge coupling scaled by sqrt(4/30) relative to the directed case, ensuring ||K_directed||_F = ||K_symmetric||_F. This prevents the symmetric network from trivially differing due to weaker per-edge coupling.

Recompute the availability profile for the symmetric case.

**Hypothesis:** the directed architecture should produce an asymmetric availability profile (topology varies with depth along the projection axis) while the symmetric architecture should produce a flat profile (no preferred direction). If both produce asymmetric profiles, directionality is not the operative variable and the construction needs to be reformulated as a general coupling geometry result rather than a directed projection result. If the directed profile is asymmetric and the symmetric profile is flat, the cone is a genuine consequence of directed coupling — the central theoretical claim.

---

## Implementation Plan

### What exists in ATT and can be reused directly

The `TakensEmbedder` and `JointEmbedder` with per-channel delay estimation handle all embedding. The `PersistenceAnalyzer` with persistence image computation handles all topology. The `BindingDetector` handles the marginal-vs-joint comparison for Experiment 1. The surrogate framework handles statistical testing. The visualization code handles persistence diagram and image plotting.

### What needs to be written from scratch

**`att/synthetic/aizawa.py`**: Aizawa attractor generator. Approximately 30 lines. Follows the same pattern as `lorenz_system` and `rossler_system` in the existing codebase.

**`att/synthetic/layered_network.py`**: The 2-column, 3-layer network integrator with directed coupling. Approximately 80 lines. Returns a dict of time series keyed by node name. Parameterized by coupling strengths, timescale ratios, and initial conditions.

**`att/cone/detector.py`**: The `ConeDetector` class. This is the primary new contribution. It wraps the existing `BindingDetector` but adds directional filtration. Key methods: `estimate_projection_axis()` using conditional-mean PCA; `slice_at_depth()` for cross-section extraction; `availability_profile()` for Betti-vs-depth curves; `coupling_influence_subspace()` using CCA. The axis estimation, adaptive binning, confidence intervals for the availability profile, and CCA-based subspace projection push this to approximately 220 lines.

**`att/cone/visualize.py`**: Plotting functions for the availability profile (Betti-vs-depth curve), the coupling sweep with availability profiles, cross-section topology at each depth, and the coupling-influence subspace comparison. Approximately 100 lines.

**`notebooks/cone_prototype.ipynb`**: The notebook that runs all six experiments (0-5) and produces the key figures. This is the deliverable that determines whether to scale up. Approximately 350 lines, budgeted at 10 hours to account for debugging the depth-sliced analysis.

### Estimated effort

| Component | Lines | Hours |
|---|---|---|
| `aizawa.py` | 30 | 1 |
| `layered_network.py` | 80 | 2 |
| `cone/detector.py` | 220 | 6 |
| `cone/visualize.py` | 100 | 2 |
| `cone_prototype.ipynb` (6 experiments) | 350 | 10 |
| Tests | 120 | 2 |
| **Total** | **900** | **23** |

---

## Success Criteria

The prototype produces a clear answer to one question: does directed cross-layer coupling through attractor networks create measurable conical geometry in the joint state space?

**Positive result (proceed to grid scale)**: Experiment 0 shows a clean directed cascade. Experiment 1 shows that [C; A5] has different (richer) topology than [C; A3], and the full chain [C; A3; A5] has emergent topology beyond pairwise combinations. Experiment 2 shows a non-flat availability profile with systematic topology enrichment along the projection axis (in at least one of: full embedding or coupling-influence subspace). Experiment 5 shows that directed coupling produces asymmetric profiles while symmetric coupling produces flat profiles. All results survive surrogate testing.

**Negative result (revise or abandon)**: the availability profile is flat in both the full embedding and the coupling-influence subspace, or symmetric coupling produces the same cone as directed coupling. Each of these failures points to a different revision. If the profile is flat everywhere, the cone may not be a topological object and requires different detection methods (information-geometric, spectral). If symmetric coupling produces the same cone, the directionality claim is wrong and the construction should be reformulated as a general coupling geometry result.

**Ambiguous result (needs more investigation)**: [C; A3] and [C; A5] differ but not monotonically with depth, or the availability profile shows non-monotonic structure that doesn't fit a simple cone model. This would motivate scaling to 4x4 grids with more statistical power and exploring whether the cone is a limiting case of a more complex projection geometry.

---

## Key References for Implementation

**Burak and Bhatt (2020), eLife** — the mathematical template for joint attractor state-space analysis. Read Section 3 on coupled attractor manifold geometry before implementing Experiment 2. Their framework for characterizing how coupling restricts dynamics to a low-dimensional submanifold of the product space is directly transferable from hippocampus-MEC to L2/3-L5.

**Khona and Fiete (2024), eLife preprint** — the MADE framework for constructing continuous attractors of arbitrary topology. Relevant if the Aizawa's discrete attractor dynamics don't produce clean cones and you need to engineer the attractor topology directly.

**Kang et al. (2021), PLoS Computational Biology** — persistent cohomology detecting product topologies in combined neural populations. Their multi-population analysis is the validation precedent for Experiment 1's full-chain emergent topology test.

**Chung, Lee and Sompolinsky (2018), Physical Review X** — conic decomposition in neural state space. Read Section 4 on manifold capacity and cone geometry before implementing Experiment 2's depth-sliced analysis. The math for cones in neural state space exists here — unused for cross-layer projections.

**Scientific Reports (2025), bistable perception model** — L5/L6 modulating L2/3 attractor depth through interlaminar loops. Provides the neuroscience motivation for why cross-layer timescale asymmetry should matter. Read to calibrate timescale ratios for Experiment 4, particularly the gain-control mechanism at large timescale separation.

---

## Relationship to ATT

This prototype extends ATT along one axis: from pairwise symmetric binding to directed multi-node projection geometry. If the prototype succeeds, the next step is `att/cone/` as a new module alongside `att/binding/`, with `ConeDetector` inheriting from or composing with `BindingDetector`. The existing benchmark framework extends to compare cone detection against directed transfer entropy and directed CRQA. The existing surrogate framework extends without modification.

If the prototype fails, the failure characterization itself is publishable as a companion to the ATT preprint: "Topological binding detection does not generalize to directed cross-layer projections in layered attractor networks." Honest negative results in this space are rare and valuable, as the deep research confirmed that nobody has attempted this experiment.
