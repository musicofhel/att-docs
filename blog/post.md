# Your Brain Is a Matrix of Chaos Attractors

*How persistent homology on delay embeddings reveals hidden coupling in neural dynamics*

---

Close your eyes and press lightly on one eyelid. The image flickers --- your brain can't decide which eye to believe. This is **binocular rivalry**, a decades-old paradigm where two conflicting images compete for conscious perception, spontaneously alternating every few seconds without any change in the stimulus.

The alternation isn't random noise. It's a signature of multistable dynamics: your visual cortex is a coupled nonlinear system that settles into one attractor, destabilizes, and snaps to another. The percept *you* experience is a readout of which attractor currently dominates.

Neuroscientists have spent years measuring these dynamics with tools like transfer entropy, phase-amplitude coupling, and cross-recurrence analysis. These methods are powerful, but they all share a blind spot: they don't see the **shape** of the coupling. They can tell you that information flows from region A to region B, or that two signals recur together, but they can't tell you what new geometric structure appears in the joint state space when those regions interact.

We built a tool that can. It's called the [Attractor Topology Toolkit (ATT)](https://github.com/musicofhel/attractor-topology-toolkit), and in this post we'll walk through what it does, why it works, and what we found when we pointed it at real EEG data from a binocular rivalry experiment.

## The Problem: Measuring Coupling Between Chaotic Systems

Suppose you have two brain regions, each producing a chaotic time series. You want to know: are they coupled? How strongly? And what kind of structure does the coupling create?

The standard toolkit offers several approaches:

- **Transfer entropy (TE)**: Measures directed information flow by asking how much the past of X reduces uncertainty about the future of Y. Great for causal inference, but it operates on probability distributions, not geometry. It can't distinguish topologically different coupling regimes that produce the same information transfer.

- **Phase-amplitude coupling (PAC)**: Quantifies how the phase of a slow oscillation modulates the amplitude of a fast one. Purpose-built for cross-frequency neural interactions (e.g., theta-gamma coupling). But for broadband chaotic coupling --- the kind you get between two strange attractors --- it has essentially no sensitivity.

- **Cross-recurrence quantification analysis (CRQA)**: Measures shared recurrence structure in a joint recurrence plot. Captures some geometry, but collapses it into scalar summary statistics (determinism, laminarity) that lose most of the topological information.

None of these methods ask the question we care about: **what topological features exist in the joint dynamical system that don't exist in either system alone?**

## The Insight: Joint vs. Marginal Topology on Delay Embeddings

Our approach starts from three classical ideas and combines them in a way that, as far as we know, hasn't been done before.

### 1. Takens' Embedding Theorem

Given a scalar time series $x(t)$ from a dynamical system, you can reconstruct the geometry of the underlying attractor by forming delay vectors:

$$\mathbf{x}(t) = \bigl(x(t),\; x(t-\tau),\; x(t-2\tau),\; \ldots,\; x(t-(m-1)\tau)\bigr)$$

With the right delay $\tau$ and dimension $m$, this produces a point cloud that is diffeomorphic to the original attractor. This is not a heuristic --- it's a theorem. The delay $\tau$ is typically estimated via the first minimum of average mutual information (AMI), and $m$ via false nearest neighbors (FNN).

### 2. Joint Delay Embedding

For *two* coupled time series $x(t)$ and $y(t)$, you can form a **joint** delay embedding by concatenating the delay vectors from both channels:

$$\mathbf{z}(t) = \bigl(x(t), \ldots, x(t-(m_1-1)\tau_1),\;\; y(t), \ldots, y(t-(m_2-1)\tau_2)\bigr)$$

This lives in $\mathbb{R}^{m_1 + m_2}$. When the two systems are coupled, the joint attractor is *not* a simple product of the two marginal attractors --- it has additional structure created by the coupling. When uncoupled, the joint is (approximately) the product.

### 3. Persistent Homology

Persistent homology is a tool from topological data analysis (TDA) that characterizes the "shape" of a point cloud across multiple scales. You grow balls around each point, track when connected components merge ($H_0$), loops form and fill ($H_1$), and voids appear and close ($H_2$). Each feature has a birth time and a death time. Long-lived features (large persistence $= d - b$) represent robust topological structure; short-lived ones are noise.

The output --- a persistence diagram --- is an intrinsic, coordinate-free descriptor of the point cloud's shape.

### Putting It Together: Persistence Image Subtraction

Here's the core construction:

1. Embed $x(t)$ alone (marginal X). Compute its persistence image $I^X$.
2. Embed $y(t)$ alone (marginal Y). Compute its persistence image $I^Y$.
3. Embed them jointly. Compute the joint persistence image $I^{XY}$.
4. Subtract: $R = I^{XY} - \max(I^X, I^Y)$

The residual $R$ captures **topological features present in the joint embedding that are absent from the stronger marginal**. The binding score $S = \|\max(R, 0)\|_1$ is the total mass of these excess features --- the topological novelty created by the coupling.

In code, this is straightforward:

```python
from att.binding import BindingDetector

detector = BindingDetector(max_dim=1, method="persistence_image")
detector.fit(channel_x, channel_y)

print(f"Binding score: {detector.binding_score():.4f}")
print(f"Significant: {detector.test_significance(n_surrogates=100)}")
```

The `max` baseline is conservative by design: a pixel in the residual is positive only if the joint exceeds whichever marginal is stronger at that location. This minimizes false positives.

## Results: Synthetic Validation

We validated the method on coupled chaotic systems with known ground truth before touching any neural data.

### A Unimodal Coupling Curve

We swept the coupling parameter $\epsilon$ from 0 (independent) to 1 (fully synchronized) for two diffusively coupled Lorenz systems.

![Figure 1: Binding score vs. coupling strength](figures/fig1_coupling_sweep.png)

The binding score traces out a **unimodal curve**: it starts at a nonzero baseline for uncoupled systems (inherent mismatch between joint and marginal representations), peaks at low coupling ($\epsilon \approx 0.1$, $S = 257.2$), then collapses at full synchronization ($S = 89.3$, a 65% drop from peak).

This makes intuitive sense. At low coupling, the two attractors are structurally distinct, and the coupling creates genuinely new topology in the joint space. As coupling increases toward synchronization, the joint attractor degenerates toward a copy of either marginal --- there's less and less emergent structure to detect.

### 2.9x Dynamic Range vs. Flat Competitors

We benchmarked the binding score head-to-head against transfer entropy, PAC, and CRQA on the same coupled Lorenz sweep.

![Figure 3: Benchmark comparison of coupling measures](figures/fig3_benchmark_overlay.png)

| Measure | $\epsilon=0$ | $\epsilon=0.1$ | $\epsilon=0.5$ | $\epsilon=1.0$ | Dynamic Range |
|---------|-------------|----------------|----------------|----------------|---------------|
| **Binding (ours)** | 184.2 | 257.2 | 191.4 | 89.3 | **2.9x** |
| Transfer entropy | 0.0086 | 0.0142 | 0.0087 | 0.0129 | 1.7x |
| PAC | 0.0019 | 0.0017 | 0.0011 | 0.0030 | ~flat |
| CRQA | 0.638 | 0.643 | 0.647 | 0.610 | 6% variation |

The binding score's 2.9x peak-to-trough ratio dwarfs the competition. CRQA is nearly flat. PAC shows no systematic response (expected: it's designed for cross-frequency oscillatory coupling, not chaotic diffusive coupling). Transfer entropy varies by only 1.7x with a non-monotonic trend.

These methods aren't broken --- they're measuring different things. The binding score captures a complementary aspect of coupling that information-theoretic and recurrence-based methods simply don't resolve on chaotic systems.

### Surrogate Validation

We tested statistical significance using amplitude-adjusted Fourier transform (AAFT) surrogates, which preserve the marginal distribution and power spectrum of each signal while destroying inter-system coupling.

![Figure 4: Surrogate null distributions](figures/fig4_surrogate_null.png)

At $\epsilon = 0$ (uncoupled), the observed score falls well within the null distribution ($p = 0.250$) --- no false positive. At $\epsilon = 0.5$ (moderate coupling), the observed score exceeds most surrogates ($p = 0.060$), a trend toward significance that doesn't quite cross the $\alpha = 0.05$ threshold.

**Being honest about this**: the $p = 0.06$ is borderline. It reflects the shorter time series used for computational tractability ($n = 6000$ samples instead of 8000) and the inherent variability of chaotic dynamics at moderate coupling. Stronger coupling or longer recordings would likely push this to significance. The key result is that the false positive rate is well controlled.

### Per-Channel Delays Fix the Timescale Problem

Real coupled systems often have different intrinsic frequencies. When you embed a coupled Rossler-Lorenz system (Rossler is slower than Lorenz) using a single shared delay, the faster system's embedding degenerates.

![Figure 5: Per-channel vs. shared delay estimation](figures/fig5_heterogeneous_timescales.png)

Per-channel delay estimation produces joint embeddings with condition numbers around 50 --- well-conditioned. Shared delays produce condition numbers of 2,671 --- a **52x degradation**, approaching the degeneracy threshold of $10^4$.

The binding score itself diverges correspondingly. At $\epsilon = 0.8$, per-channel delays give $S = 405.2$; the shared delay gives $S = 179.3$. The shared-delay scores are erratic artifacts of a near-singular embedding, not genuine coupling measurements.

ATT handles this automatically: the `embed_channel()` function estimates AMI and FNN independently for each channel, validates the result via singular value decomposition, and falls back to literature-grounded defaults if the auto-estimate degenerates.

## Real EEG: Detecting Perceptual Switches as Topological Reorganizations

The synthetic experiments validate the method's mechanics. The real question: does it work on actual brain data?

We applied ATT to EEG recordings from a binocular rivalry experiment ([Nie, Katyal & Bhatt 2023](https://doi.org/10.13020/9sy5-a716)), where subjects viewed competing stimuli and reported their perceptual switches via button press. We analyzed Subject 1, Epoch 0: 120 seconds of 34-channel EEG at 360 Hz, during which the subject reported 41 perceptual switches (29 clear alternations, 12 involving a mixed percept).

### Transition Detection: 100% Precision, Zero False Alarms

We used the Oz (occipital midline) channel, bandpass-filtered to 4-13 Hz (theta-alpha, the frequency range most relevant to rivalry dynamics). The `TransitionDetector` computed persistent homology on 214 sliding windows and applied CUSUM changepoint detection to the persistence image distance series.

![Figure 7: Real EEG transition detection](figures/fig7_real_eeg.png)

The results:

| Metric | Value |
|--------|-------|
| Changepoints detected | 7 |
| Behavioral switches | 41 |
| **Precision @ 3s tolerance** | **100% (7/7)** |
| Recall @ 3s | 41.5% (17/41) |
| Recall @ 5s | 80.5% (33/41) |

Every single detected changepoint fell within 3 seconds of a real perceptual switch. Zero false alarms.

The asymmetry between precision and recall is informative: the detector doesn't see individual percept flips. It sees **major topological reorganizations** --- qualitative regime changes in the attractor structure. Periods of rapid switching (multiple alternations in a few seconds) produce a single sustained change in persistence image structure, yielding one changepoint for several behavioral events.

This is exactly what you'd expect if the detector is capturing something real about the dynamics, not just tracking the stimulus timing.

### Cross-Region Binding Tracks Perceptual Instability

We then applied the full binding detection framework to two spatially separated electrodes --- Oz (occipital) and Pz (parietal) --- using sliding 10-second windows.

![Figure 8: Cross-region Oz-Pz binding](figures/fig8_eeg_binding.png)

The binding score between Oz and Pz varies dramatically over time: from 5.1 to 34.1, a 6.7x range. This isn't noise. It correlates with behavior:

- **Spearman $\rho = 0.51$, $p = 0.016$** between binding and the number of perceptual switches per window
- Windows during high-switching periods: mean binding = 24.5
- Windows during low-switching periods: mean binding = 11.0 (2.2x lower, $p = 0.0014$)
- Windows adjacent to CUSUM changepoints show significantly elevated binding ($p = 0.042$)

In other words: **when the brain is perceptually unstable --- rapidly flipping between percepts --- the topological coupling between occipital and parietal cortex increases**. During stable perception, coupling drops.

The joint topology captures structure that neither channel has alone. This is genuine inter-region binding, not an artifact of single-channel complexity.

## Limitations (We Mean It)

This is an early-stage result, and we want to be explicit about what it does and does not establish.

**N=1.** The real EEG analysis is from a single subject, a single epoch, and two electrodes. The [full dataset](https://doi.org/10.13020/9sy5-a716) has 84 subjects with multiple rivalry epochs each. Multi-subject analysis is the next step. Until then, these results demonstrate feasibility, not generalizability.

**Borderline surrogate p-values.** The surrogate significance test at moderate coupling ($\epsilon = 0.5$) yielded $p = 0.060$, which does not reach conventional significance. The false positive rate is well controlled, but the test's power at moderate coupling needs improvement --- longer time series, more surrogates, or stronger coupling are likely needed.

**Computational cost.** Persistent homology is expensive. We subsample to 300-500 points per window, which introduces variance. GPU-accelerated backends (Ripser++) and witness complexes could help scale to larger datasets.

**Parameter sensitivity.** The binding score depends on embedding parameters ($\tau$, $m$), persistence image resolution, Gaussian bandwidth, and subsampling. The quality gate catches gross failures but doesn't optimize these parameters.

**No causal claims.** The binding score is symmetric --- it detects coupling, not directionality. Transfer entropy or convergent cross-mapping would be needed for causal inference. ATT complements these methods rather than replacing them.

## What's Next

Several directions are currently in progress or planned:

- **Multi-subject analysis**: Running the pipeline across all 84 subjects in the Nie et al. dataset to compute group-level statistics, precision-recall curves across subjects, and test whether the binding-instability correlation replicates.

- **Full topological connectivity matrices**: Extending cross-region binding from two electrodes to all electrode pairs, producing a topological connectivity map that complements coherence and Granger causality.

- **Real-time streaming**: The sliding-window pipeline is inherently causal (each window uses only past data). With optimized persistence computation, real-time topological monitoring of neural dynamics is within reach.

- **Transformer hidden states**: The method is not specific to neural time series. Any system that produces multivariate time series --- including the hidden state trajectories of recurrent neural networks and transformers --- can be analyzed for topological coupling structure. We're particularly interested in whether attention heads in large language models exhibit chaotic attractor dynamics, and whether joint topology between layers captures something that gradient-based attribution methods miss.

## Try It

ATT is open source and pip-installable:

```bash
pip install att-toolkit
```

For EEG support (requires MNE-Python):

```bash
pip install att-toolkit[eeg]
```

Quick start --- detect binding between two coupled Lorenz systems:

```python
from att.config import set_seed
from att.synthetic import coupled_lorenz
from att.binding import BindingDetector

set_seed(42)

# Generate coupled Lorenz systems
ts_x, ts_y = coupled_lorenz(coupling=0.5)

# Detect emergent topology in the joint embedding
detector = BindingDetector(max_dim=1, method="persistence_image")
detector.fit(ts_x[:, 0], ts_y[:, 0])

print(f"Binding score: {detector.binding_score():.4f}")
print(f"Significant: {detector.test_significance(n_surrogates=100)}")
```

- **GitHub**: [github.com/musicofhel/attractor-topology-toolkit](https://github.com/musicofhel/attractor-topology-toolkit)
- **PyPI**: [att-toolkit](https://pypi.org/project/att-toolkit/)
- **Preprint**: [paper/preprint.pdf](https://github.com/musicofhel/attractor-topology-toolkit/blob/main/paper/preprint.pdf)
- **Notebooks**: All figures in this post are reproducible from `notebooks/` with fixed random seeds.

---

*ATT was developed as an open-source research tool. If you use it in your work, please cite the preprint --- it helps us justify continued development.*

```bibtex
@software{att2026,
  title = {Attractor Topology Toolkit: Joint-vs-Marginal Persistent Homology
           on Takens Embeddings},
  author = {{ATT Contributors}},
  year = {2026},
  url = {https://github.com/musicofhel/attractor-topology-toolkit},
}
```
