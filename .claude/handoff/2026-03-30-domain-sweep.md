# Domain Sweep: Topology Probe on 10 Dynamical Systems — Handoff

**Date**: 2026-03-30
**Branch**: `experiment/neuromorphic-snn`
**Commit**: `9b7facc` — Domain sweep: topology probe on 10 dynamical systems beyond reservoirs
**Status**: COMPLETE — 8/10 pass, committed

## What Was Done

Built `scripts/probe_domain_sweep.py` (995 lines) — sweeps the PCA-population PH probe across 10 dynamical systems to test whether topology predicts functional properties beyond ESN memory capacity.

### Protocol
- Each domain: sweep a control parameter (8 values), 3 seeds
- PCA(3) on state trajectory, PH via `PersistenceAnalyzer(max_dim=1, subsample=400)`
- Discard first 20% as transient
- Spearman rho between best topology feature and ground truth
- Pass gate: |rho| > 0.6

### Results

| # | Domain | Ground Truth | Best Feature | |rho| | Pass |
|---|--------|-------------|-------------|-------|------|
| 1 | Kuramoto oscillators | Order parameter | betti_1 | 0.883 | PASS |
| 2 | Lotka-Volterra ecosystem | Lyapunov stability | betti_0 | 0.709 | PASS |
| 3 | Coupled map lattice | Spatial corr. length | betti_1 | 0.633 | PASS |
| 4 | Hopfield network | Retrieval accuracy | betti_1 | 0.756 | PASS |
| 5 | FitzHugh-Nagumo population | Info transfer | persistence_entropy | 0.655 | PASS |
| 6 | Boolean network (NK) | Attractor count | betti_0 | 0.889 | PASS |
| 7 | Lorenz-96 weather | Lyapunov exponent | betti_0 | 0.408 | FAIL |
| 8 | Spiking STDP network | Weight entropy | persistence_entropy | 0.500 | FAIL |
| 9 | Power grid stability | Max freq deviation | total_H1_pers | 0.687 | PASS |
| 10 | Gene regulatory network | Oscillation CV | max_H1_pers | 0.877 | PASS |

**Runtime**: 1.9 minutes

### Bugs Fixed During Development

1. **Kuramoto sin/cos embedding** (0.000 → 0.883): Raw phase values grow unbounded, PCA sees near-linear trajectories. Fixed by storing `[cos(theta), sin(theta)]` as the 2N-dimensional state instead of raw phases.

2. **Gene regulatory even-cycle bug** (0.000 → 0.877): A 6-gene repression ring has an even-length negative feedback cycle, which converges to alternating high/low fixed points instead of oscillating. Repressilators require odd-length cycles. Fixed with 3-gene repressilator core (genes 0→1→2→0, odd cycle) plus 3 downstream genes activated by the core.

3. **Spiking STDP dead network** (0.000 → 0.500): Poisson rate=0.5 with 0.3 scaling was too weak — no neuron ever reached threshold, giving all-zero firing rates. Fixed by increasing to rate=3.0 with 0.5 scaling, using membrane voltage (pre-threshold) as the continuous state variable instead of binary spike trains.

### Failure Analysis

Both failures are near-misses, not collapses:

- **Lorenz-96** (|rho|=0.408): 40-dimensional state → PCA(3) captures too little of the manifold structure. The Lyapunov exponent varies smoothly with forcing F, but the 3D projection may not preserve the topologically relevant structure. Higher PCA dim (5-10) may fix this.

- **Spiking STDP** (|rho|=0.500): Membrane voltage resets (v → v_reset on spike) create discontinuities in the trajectory that break PCA's linear projection. The topology-entropy chain (spike timing → weight structure → weight entropy) has too many lossy steps.

### Key Finding

**The probe is a general tool for dynamical systems diagnostics**, not reservoir-specific. 8/10 systems pass, spanning oscillators, ecosystems, memory networks, discrete automata, power grids, and gene circuits. All 5 topology features (betti_0, betti_1, persistence_entropy, total_H1_pers, max_H1_pers) appear as "best" across different domains — no single feature dominates.

### Branch Status (experiment/neuromorphic-snn)

5 commits ahead of master on this branch:
1. `1c831d7` Screen SNN topology
2. `f2e3c3d` Deepen SNN topology
3. `df3b983` Reservoir degradation monitoring
4. `bec80cb` SNN probe battery (20 tests)
5. `9b7facc` Domain sweep (this session)

### Possible Next Steps

- **Lorenz-96 rescue**: Try PCA(5) or PCA(10) to capture more of the 40D manifold
- **Spiking rescue**: Use subthreshold voltage trace with spike times removed, or population-level order parameter instead of raw voltage
- **Cross-domain meta-analysis**: Which topology features predict which types of functional property? (betti_1 → phase/synchronization, betti_0 → complexity/stability, persistence_entropy → information transfer?)
- **Write up**: This is a strong generality result — 8/10 with diverse system types
