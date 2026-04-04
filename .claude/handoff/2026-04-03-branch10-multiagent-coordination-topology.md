# Branch 10: Multi-Agent Coordination — Topological Binding — 2026-04-03

## Branch
`experiment/tda-multiagent` (from `experiment/neuromorphic-snn` @ `6e60605`)

## What Was Done

Created `scripts/branches/multiagent_coordination_topology.py` — tests whether ATT's BindingDetector captures emergent coordination topology in multi-agent dynamical systems. Three synthetic systems (Vicsek flocking, Kuramoto oscillators, coupled Lorenz chain) across four experiments testing pairwise binding, synchronization transitions, coupling structure recovery, and N-body emergence.

### Synthetic Systems

| System | Agents | Steps | Parameters |
|--------|--------|-------|-----------|
| Vicsek flocking | 30 | 3000 | noise=[0.05–3.0], interaction_radius=1.0, speed=0.5, box_size=10 |
| Kuramoto oscillators | 20 | 5000 | coupling=[0.0–3.0], omega_spread=1.0 |
| 3-node Lorenz chain | 3 | 10000 | coupling_12=0.5, coupling_23=0.5, dt=0.01 |

### Config

BindingDetector: max_dim=1, method="persistence_image", baseline="max". PH subsample=200, n_surrogates=50 (phase-randomize), seed=42.

### Four Experiments

| Exp | What | Key Result |
|-----|------|------------|
| 1 | Vicsek flocking: flocked vs disordered binding + noise sweep | Flocked binding LOWER than disordered (93.5 vs 113.0); neither significant |
| 2 | Kuramoto synchronization binding sweep | Binding DECREASES with coupling (12184→337); K=0 significant (z=2.64), K=3 not (z=-1.13) |
| 3 | 3-node Lorenz: direct vs indirect coupling | Direct > Indirect (81.4 vs 73.8) but no pair significantly above surrogates |
| 4 | Population-level 5-agent joint topology | Joint H1=250 < max pairwise H1=316; zero emergent features |

### Key Findings

**Exp 1 — Vicsek flocking binding is NOT regime-discriminating:**
- Flocked (η=0.1): binding=93.46, p=1.00, z=-3.90 (surrogates produce MORE binding)
- Disordered (η=2.0): binding=112.98, p=0.45, z=-0.06 (indistinguishable from null)
- Order parameter: flocked=0.92, disordered=0.16 (flocking IS occurring)
- Noise sweep shows no monotonic binding-noise relationship
- **Position-based time series from Vicsek agents do not produce topological binding that differs from phase-randomized surrogates**

**Exp 2 — Kuramoto binding reveals a Künneth effect (the critical finding):**
- K=0 (uncoupled): binding=12184, z=2.64, p=0.02 — SIGNIFICANT binding
- K=3 (synchronized): binding=337, z=-1.13, p=0.92 — NOT significant
- Binding monotonically DECREASES from K=0 to K≈2.0 (12184→101), then slightly increases
- This is the opposite of the initial hypothesis but makes mathematical sense:
  - Independent oscillators produce a joint attractor that is the Cartesian product of marginals
  - By the Künneth formula, H1(X×Y) ≈ H1(X)⊕H1(Y)⊕(H0(X)⊗H0(Y)) — genuine excess H1 from cross-terms
  - Synchronized oscillators collapse to effectively 1D dynamics — joint ≈ marginal, no excess topology
- **Binding score measures "topological independence" (Künneth cross-terms), not coupling strength**
- Known K_c=0.637 was NOT detected as a binding transition (binding drops smoothly, no sharp knee)

**Exp 3 — Lorenz coupling structure weakly recovered:**
- Pair 1↔2 (direct): binding=99.73, p=0.45, z=0.37
- Pair 2↔3 (direct): binding=63.06, p=0.92, z=-1.51
- Pair 1↔3 (indirect): binding=73.80, p=0.73, z=-0.78
- Mean direct (81.39) > indirect (73.80) — correct direction, but not significant
- No pair shows binding significantly above phase-randomized surrogates
- The 1↔2 vs 2↔3 asymmetry (99.7 vs 63.1) likely reflects different initial conditions/attractor basins despite equal coupling strength

**Exp 4 — No emergent N-body topology:**
- Joint 5-agent: H1=250 features, entropy=5.20
- Max pairwise: H1=316 features (pair 2-4), entropy=5.44
- Emergent features: 0 (joint LOWER than several pairs)
- Disordered joint: H1=160 (lower than flocked, expected — less structured dynamics)
- **CAVEAT**: subsample=200 on a 50-dimensional joint cloud is far too sparse. With 200 points in 50D, the Vietoris-Rips complex captures noise topology, not signal. A fair test would need subsample=2000+, but computational cost scales cubically.

### Interpretation

**Overall verdict: binding measures independence, not coordination.**

The key insight from this branch is that ATT's persistence image subtraction detects Künneth cross-terms — the excess topology arising from taking products of independent spaces — rather than emergent coordination topology. This produces a binding score that is:

1. **High** for independent systems (joint = product space, cross-terms produce excess H1)
2. **Low** for synchronized systems (joint ≈ single marginal, no excess)
3. **Intermediate** for partially coupled systems

This is mathematically correct behavior for the persistence image subtraction method, but it means the BindingDetector measures "topological independence" rather than "coupling strength." For multi-agent coordination detection, one would need to invert the interpretation: LOW binding = high synchronization, HIGH binding = independence.

The Kuramoto result is the clearest demonstration: uncoupled oscillators (K=0) show significant binding (z=2.64) while fully synchronized oscillators (K=3) show none (z=-1.13). This aligns with the original ATT paper's coupled-Lorenz results where binding was detected between chaotic (non-synchronized) coupled systems.

### Where This Approach MIGHT Work

1. **Detecting desynchronization events** — binding should spike when a synchronized flock breaks apart
2. **Temporal binding dynamics** — sliding-window binding on a system transitioning between synchronized and independent states
3. **Partial synchronization clusters** — if a subgroup synchronizes while the rest remains independent, binding between the subgroup and outsiders should differ
4. **Heterogeneous coupling** — systems with asymmetric coupling matrices where some pairs are strongly coupled and others weakly

### Caveats

- Vicsek model uses periodic boundary conditions (box_size=10), which creates trajectory wrapping that may confound embedding
- Only tested x-coordinate of positions; heading angles might produce different results
- Kuramoto uses sin(θ) as signal, not θ directly — this may mask binding in the phase space
- 3-node Lorenz with coupling=0.5 may be too strongly coupled (near synchronization), reducing binding
- Exp4 subsample=200 in 50D is insufficient — joint cloud needs much larger subsample for meaningful PH
- n_surrogates=50 gives limited p-value resolution (minimum p ≈ 0.02)

## Files

- `scripts/branches/multiagent_coordination_topology.py` — full analysis script
- `data/multiagent/results.json` — all numeric results
- `figures/multiagent/` — 5 figures:
  - `overview.png` — 4-panel summary (noise sweep, Kuramoto sweep, Lorenz bars, N-body bars)
  - `exp1_vicsek_flocking.png` — trajectories + noise sweep + order parameter
  - `exp2_kuramoto_binding.png` — coupling sweep + surrogate z-scores
  - `exp3_lorenz_3node.png` — time series + binding bars + z-score bars
  - `exp4_population_binding.png` — joint vs pairwise H1 + flocked vs disordered

## Technical Notes

- Vicsek model implemented from scratch (not in ATT's synthetic module)
- 3-node Lorenz chain implemented from scratch (ATT only has 2-node coupled_lorenz)
- Kuramoto model implemented from scratch (ATT's kuramoto_oscillators has different return format)
- BindingDetector.test_significance reuses cached embedding params (delay, dimension) across surrogates
- Surrogate testing only on key samples (not every sweep point) for performance
- Total runtime: 56.3s (~1 minute)

## Potential Follow-ups

- Test with heading angles instead of positions for Vicsek model
- Implement sliding-window binding to detect synchronization transitions in real time
- Test at coupling=0.1 for Lorenz (weak coupling, non-synchronized regime) where binding should be higher
- Increase subsample to 2000 for Exp4 joint cloud (requires ~30min compute)
- Compare BindingDetector methods: "persistence_image" vs "diagram_matching" on same data
- Add a "binding transition detector" that looks for binding score drops as a synchronization indicator
- Test on real multi-agent data: bird flocking GPS tracks, fish schooling, drone swarm telemetry
