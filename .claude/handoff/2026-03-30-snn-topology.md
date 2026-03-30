# SNN Topology: Reservoir Quality Prediction — March 30, 2026

Screen testing whether topological features predict reservoir computing
quality (memory capacity, NARMA-10) without training a readout. Branch:
`experiment/neuromorphic-snn` (commit `1c831d7`, pushed to origin).

---

## Motivation

The reservoir experiment (`experiment/neuromorphic-reservoir`) showed topology
sees the ordered-to-chaotic phase transition in ESNs (Part 1 Betti bifurcation
at SR=1.0). Monitoring didn't pan out — variance is faster. Different question
here: can persistence features computed on a single neuron's trajectory
**predict** reservoir computational quality?

If yes, topology provides a training-free probe for reservoir quality — useful
for neuromorphic hardware where you can't easily train readouts in real time.

---

## What Was Built

**Script**: `scripts/screen_snn_topology.py` (~813 lines)

- MinimalESN (copied from neuromorphic-reservoir): 100 neurons, 10% connectivity, tanh
- MinimalLIF: Leaky integrate-and-fire population, 100 neurons, membrane voltage readout
- Memory capacity: linear readout reconstruction R^2 summed over lags 1-50 (Jaeger 2001)
- NARMA-10: standard nonlinear benchmark, linear readout, NRMSE metric
- Topological features via TakensEmbedder + PersistenceAnalyzer on neuron 0
- Spearman correlation analysis across SR sweep x 3 seeds

**No new library code** — uses existing TakensEmbedder and PersistenceAnalyzer as-is.

---

## Results (92s runtime)

### Part 1: Topology Predicts Memory Capacity — PASS

MC peaks at SR=1.0 (MC=20.9), matching expected edge-of-chaos behavior.

| Feature              | MC Spearman rho | p-value  | Sig |
|----------------------|-----------------|----------|-----|
| persistence_entropy  | +0.72           | <0.0001  | *** |
| total_H1_pers        | +0.51           | 0.0032   | **  |
| max_H1_pers          | +0.57           | 0.0008   | *** |
| betti_0              | +0.30           | 0.1029   |     |
| betti_1              | +0.77           | <0.0001  | *** |

**persistence_entropy** and **betti_1** both exceed the rho > 0.7 gate.
Higher H1 entropy and more 1-loops in the attractor correspond to higher
memory capacity. This makes physical sense: richer attractor topology
= more distinct dynamical states = better short-term memory.

### Part 2: Topology Predicts NARMA-10 — PASS

Best NARMA at SR=0.8 (NRMSE=0.322), different from MC peak — as expected.

| Feature              | NARMA Spearman rho | p-value  | Sig |
|----------------------|--------------------|----------|-----|
| persistence_entropy  | -0.81              | <0.0001  | *** |
| total_H1_pers        | -0.62              | 0.0002   | *** |
| max_H1_pers          | +0.89              | <0.0001  | *** |
| betti_0              | -0.25              | 0.1818   |     |
| betti_1              | -0.80              | <0.0001  | *** |

Note: NRMSE is error, so negative rho = lower error = better performance.
**persistence_entropy** (rho=-0.81), **max_H1_pers** (rho=+0.89), and
**betti_1** (rho=-0.80) all pass the gate.

Key insight: **max_H1_pers** has opposite sign for MC (+0.57) vs NARMA
(+0.89). For NARMA, larger maximum H1 persistence means worse performance.
This suggests NARMA prefers a "many small loops" attractor (high entropy,
many Betti-1 features) over a "few large loops" attractor. Different tasks
benefit from different topological signatures.

### Part 3: Spiking LIF Network — WEAK

All neurons active (bias=0.8, membrane voltage readout). MC decreases
monotonically with connection strength (3.2 at CS=0.5 to 2.0 at CS=15).

| Feature              | LIF MC Spearman rho | p-value |
|----------------------|---------------------|---------|
| persistence_entropy  | -0.42               | 0.0428  |
| total_H1_pers        | -0.52               | 0.0098  |
| max_H1_pers          | -0.51               | 0.0111  |
| betti_0              | -0.22               | 0.2975  |
| betti_1              | -0.46               | 0.0241  |

Moderate correlations (max |rho| = 0.52) but all below the 0.7 gate.
**Topology-MC link does NOT generalize from rate to spiking at these
parameters.**

---

## Honest Assessment

**What works**: Parts 1-2 are genuinely novel and useful. persistence_entropy
and betti_1 are *shared* strong predictors of reservoir quality across two
different tasks (MC and NARMA-10). This means topology can serve as a
training-free probe for reservoir quality — compute PH on a single neuron's
trajectory, and the H1 features tell you how good the reservoir is, without
ever training a readout.

**What's scientifically interesting**: MC and NARMA-10 peak at different
spectral radii (1.0 vs 0.8), and different topological features are most
predictive for each. This suggests topology captures multiple aspects of
computational capacity, not just one.

**What doesn't generalize**: The rate-to-spiking transition breaks the
correlation. LIF membrane voltage dynamics produce fundamentally different
attractor geometry than tanh-based rate coding. The continuous tanh produces
smooth attractors with clear H1 features; LIF voltage resets to 0 at each
spike, creating a discontinuous trajectory that disrupts Takens embedding.

**Comparison to ATT's other results**:
- Reservoir regime detection: topology sees the bifurcation (static), but
  variance wins for monitoring (dynamic)
- Reservoir quality prediction: topology predicts MC and NARMA well in rate
  ESNs (this experiment), but doesn't generalize to spiking

**Pattern update**: Topology works when:
1. The system has smooth, continuous dynamics (brain EEG, rate ESN)
2. The topological change is structural, not just amplitude-based
3. The observable captures the attractor geometry (neuron trajectory, EEG channel)

Topology fails when:
1. Dynamics are discontinuous (LIF resets, spike trains)
2. Regime change is primarily amplitude/variance-based
3. Time series is near-white-noise (financial returns)

---

## Branch Status

- Branch: `experiment/neuromorphic-snn` (pushed to origin)
- Commit: `1c831d7`
- NOT merged to master
- Script is self-contained — no library changes, no test changes
- No data files to track

---

## If Continuing This Work

1. **Rate-filtered LIF**: Instead of membrane voltage, use exponentially
   filtered spike rates (tau_filter=50-100ms). This recovers a smooth
   signal that might restore topology-MC correlation. The current script
   uses raw membrane voltage for maximum information, but smoothed rates
   might embed better.

2. **Multi-neuron PCA embedding**: Instead of neuron 0's trajectory, take
   the first 3-5 PCA components of the full 100-neuron state. This gives
   Takens a higher-quality representation of the reservoir dynamics.

3. **Ridge regression MC**: The current lstsq can overfit at high SR
   (100 neurons, many near-zero eigenvalues). Ridge regularization might
   give more reliable MC estimates and sharpen the correlation.

4. **Preprint angle**: "Persistence features predict reservoir computing
   quality without readout training" — could be a short paper or workshop
   contribution. The MC correlation (rho=0.77 for betti_1) and the MC
   vs NARMA dissociation are publishable findings.

5. **Don't bother with**: Spiking networks using membrane voltage readout.
   The spike-reset discontinuity fundamentally breaks Takens embedding.
   Either smooth the observable (rate filtering) or use a fundamentally
   different topological approach (filtration on spike times, not Takens).
