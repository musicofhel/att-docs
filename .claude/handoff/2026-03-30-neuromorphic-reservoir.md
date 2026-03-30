# Neuromorphic Reservoir Regime Detection — March 30, 2026

Experiment testing whether ATT's topological changepoint detection works on
echo state network (ESN) dynamics. Branch: `experiment/neuromorphic-reservoir`
(commit `3836312`, pushed to origin).

---

## Motivation

An ESN's computational power depends on its dynamical regime, controlled by
the spectral radius (SR) of the recurrent weight matrix:
- SR < 0.8: ordered (signals decay, fading memory)
- SR 0.9–1.1: edge of chaos (maximum computational power)
- SR > 1.2: chaotic (signals explode)

Detecting this transition matters for tuning neuromorphic hardware where you
can't compute Lyapunov exponents in real time.

---

## What Was Built

**Script**: `scripts/reservoir_regime_detection.py` (~600 lines)

- Self-contained MinimalESN class (100 neurons, 10% connectivity, tanh activation)
- All 4 comparison methods copied from `benchmark_changepoint_methods.py`
  (topological, spectral, variance, BOCPD)
- 4-part experiment with 3 seeds each, total runtime 49s

**No new library code** — uses existing `PersistenceAnalyzer`, `TakensEmbedder`,
`BindingDetector` as-is.

---

## Results

### Part 1: Static Classification — PASS

Topological features correlate significantly with spectral radius.

Key findings:
- **Noise input**: Max persistence H1 increases monotonically (Spearman rho=+1.0,
  p<0.001). Entropy H1 *decreases* at high SR (rho=-0.82, p=0.023) because the
  attractor structure becomes dominated by fewer, larger features.
- **Sine input**: Sharp phase transition at SR=1.0 — Betti-0 jumps from ~50 to 399
  and H1 features appear for the first time. This is a genuine topological
  bifurcation: the reservoir transitions from input-driven (simple attractor) to
  internally-driven (complex attractor) exactly at the edge of chaos.
- At SR<1 with sine input, the attractor is essentially 1-dimensional (the input
  cycle), so PH finds ~50 connected components and zero loops. At SR>=1.0, internal
  dynamics create a high-dimensional attractor with many loops.

**This is the cleanest result.** Topology captures something variance/spectral cannot:
the qualitative change in attractor structure at the edge of chaos.

### Part 2: Continuous Ramp — FAIL (all methods)

SR ramped linearly 0.5→1.3 over 20k steps. No method works:

| Method | F1 |
|--------|-----|
| Variance | 0.24 |
| BOCPD | 0.21 |
| Spectral | 0.18 |
| Topological | 0.06 |

The gradual transition is too smooth. There are no sharp changepoints to detect —
all methods produce many false alarms. Topology is worst because the PI L2 distance
signal is noisy across windows with only incremental SR changes.

### Part 3: Discrete Switches — Topology detects but doesn't win

Abrupt SR switches 0.7↔1.1 every 3000 steps, 4 transitions.

| Method | F1 | Mean Lag |
|--------|-----|----------|
| **Variance** | **1.00** | +17 |
| Spectral | 0.98 | +92 |
| Topological | 0.64 | +14 |
| BOCPD | 0.43 | -118 |

- Topology has the smallest positive lag (+14 samples) but low precision (47%).
  It fires 14 times for 4 true transitions — many false alarms.
- Variance is perfect: F1=1.00 with similar lag (+17). The SR 0.7→1.1 transition
  is primarily a variance signature (chaotic regime has higher state variance).
- BOCPD fires early (-118) but with even worse false alarm rate.

### Part 4: Coupled Reservoirs — No signal

Two ESNs with different SRs, coupled via shared input vs independent input.
BindingDetector cannot distinguish them. Scores are noisy and non-significant
(Mann-Whitney p=0.65). This is expected: shared input coupling is weak (it only
affects the input term, not the recurrent dynamics), and the two reservoirs have
very different timescales.

---

## Honest Assessment

**What works**: Part 1 is genuinely interesting. The topological bifurcation at
SR=1.0 (visible in the sine-input Betti numbers) is a *qualitative* change that
variance/spectral methods cannot capture. This has scientific value even if it
doesn't translate to better monitoring.

**What doesn't work**: For practical regime *detection* (Parts 2-3), variance
and spectral methods beat topology handily. The SR transition changes variance
more than it changes topology. This makes sense: the ordered→chaotic transition
primarily increases signal amplitude/variance, and the topological change
(more loops) is a secondary effect that's harder to detect in sliding windows.

**Comparison to ATT's other results**:
- Rivalry EEG: topology F1=0.98 (beats all) — the perceptual switch is
  genuinely topological
- Seizure onset: topology 9/9 detections, earliest by 2.6s — brain state
  transitions have rich topological signatures
- Reservoir switches: topology F1=0.64 (3rd of 4) — the regime change is
  primarily a variance phenomenon, not a topological one

**Pattern**: Topology wins when the dynamical change is primarily structural
(brain state transitions) and loses when it's primarily amplitude-based
(reservoir regime switches). This is a useful insight for positioning ATT.

---

## Branch Status

- Branch: `experiment/neuromorphic-reservoir` (pushed to origin)
- Commit: `3836312`
- NOT merged to master
- Script is self-contained — no library changes, no test changes
- No data files to track

---

## If Continuing This Work

1. **Use PCA components instead of single neuron**: The script uses neuron 0's
   activation. Using the first 3 PCA components of the full 100-neuron state
   might give topology more to work with (higher-dimensional embedding of the
   reservoir state).

2. **Cross-reservoir topology**: Instead of BindingDetector (which uses
   marginal vs joint PH), try computing PH of the *combined* state of two
   reservoirs and looking for topological features that appear only when
   they're coupled.

3. **Slower transitions**: The Part 2 ramp might work with a longer time series
   (100k steps) and larger windows. The current 500-sample window sees very
   little SR change per window (~0.02 SR units).

4. **Real neuromorphic data**: The MinimalESN is a toy model. Real neuromorphic
   hardware (memristive crossbar arrays, spintronic oscillators) might have
   richer dynamics. Contact a neuromorphic lab for data.

5. **Don't bother with**: Part 4 shared-input coupling — it's too weak a
   coupling mechanism for BindingDetector. Would need direct cross-connections
   between reservoir weight matrices to create meaningful topological binding.
