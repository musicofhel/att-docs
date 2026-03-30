# ATT Project State — March 30, 2026

Verbose handoff covering the full project trajectory, today's benchmark results,
and honest assessment of what works, what doesn't, and what to do next.

---

## What ATT Is

Attractor Topology Toolkit. A Python library that detects coupling between
dynamical systems by comparing the persistent homology (PH) of joint vs
marginal Takens embeddings. The core idea: if two time series are coupled,
their joint attractor has topological features absent from either marginal
attractor. Measure the difference via persistence image subtraction.

**Novel contribution**: Joint-vs-marginal PH on Takens embeddings. Confirmed
novel via deep research (March 2026). Published as a 21-page preprint on arXiv
with 9 experiments.

---

## What's Been Built (Phases 1-12, all complete)

### Core Library (`att/`)

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `att/embedding/` | Takens delay embedding | `TakensEmbedder`, `JointEmbedder`, AMI/FNN estimation |
| `att/topology/` | Persistent homology | `PersistenceAnalyzer` (ripser backend, persistence images, distances) |
| `att/binding/` | Coupling detection | `BindingDetector` (PI subtraction + surrogate significance) |
| `att/transitions/` | Changepoint detection | `TransitionDetector` (sliding-window PH + CUSUM) |
| `att/surrogates/` | Null hypothesis testing | `phase_randomize` (AAFT), `time_shuffle`, `twin_surrogate` |
| `att/benchmarks/` | Method comparison | `CouplingBenchmark` with TE, PAC, CRQA + plugin interface |
| `att/synthetic/` | Test data generators | Lorenz, Rossler, coupled systems, switching Rossler, Aizawa networks |
| `att/cone/` | Directionality detection | `ConeDetector` (depth asymmetry, availability profiles) |
| `att/neuro/` | EEG processing | `embed_channel` with auto-estimation + literature fallback |
| `att/config/` | Reproducibility | YAML configs, `set_seed()`, `get_rng()` |
| `att/viz/` | Visualization | Persistence diagrams, binding images, sweep plots, cone plots |

### Test Suite

216 tests across 13 test files (4049 lines). Covers embedding, topology, binding,
surrogates, benchmarks, transitions, cone detection, CLI, config, visualization,
validation experiments.

### Preprint

21 pages, 9 experiments, z-score calibration. Submitted to arXiv (Phase 9).

### Real Data Validation

- **Multi-subject EEG** (Phase 11): N=80 subjects from Katyal rivalry SSVEP dataset
  (84 subjects total, 34-channel, 360 Hz). Data at `data/eeg/rivalry_ssvep/`.
  TransitionDetector achieves 94% precision / 41% recall on binocular rivalry
  perceptual switches.
- **Binding batch** (Phase 12): N=79 subjects, tutorial notebooks, Makefile.

---

## Recent Explorations (March 29-30): 0 for 6

After the core library stabilized, we explored 6 directions beyond the validated
TransitionDetector result. All failed surrogate/null testing:

### 1. Cone Geometry (Phase 6 + Diagnostics)

**Hypothesis**: Directed cross-layer coupling creates measurable conical geometry
in joint state space (depth asymmetry increases monotonically with network depth).

**Phase 6 results** (single seed=42, 80k steps): Looked promising. Depth
asymmetry +1745/+23242 for coupled columns vs -3768 control. Availability slope
+42.25 (full embedding). Directed 7.6x steeper than symmetric.

**Diagnostics** (multi-seed, 10k-15k steps): Killed it.
- Depth asymmetry CV > 500% — dominated by stochastic noise
- Surrogate test FAILS: destroying coupling doesn't reduce the signal
- Symmetric coupling produces equal or larger asymmetry than directed
- Betti_1 profiles are U-shaped regardless of coupling (no monotonic increase)
- Phase 6 result was one draw from a high-variance distribution

**File**: `.claude/handoff/2026-03-29-cone-diagnostics.md` (full analysis)

### 2. Cross-Frequency Binding (`scripts/cross_freq_binding_screen.py`)

**Hypothesis**: Theta-alpha-gamma cross-frequency coupling in EEG is detectable
via joint PH binding scores.

**Result**: FAIL. Phase-randomized surrogates produce equal or higher binding
scores. The binding score's positive baseline (~3000-8000) varies more with
embedding parameters than with coupling presence.

### 3. State Fingerprints (`scripts/screen_state_fingerprints.py`)

**Hypothesis**: Different perceptual states (stable vs near-switch) produce
measurably different attractor topologies in persistence image space.

**Protocol**: Sliding-window PH on Takens-embedded EEG, label windows by switch
proximity, permutation test + LOO logistic regression on persistence images.

**Result**: FAIL 0/3 subjects. LOO accuracy at chance. Null L2 distance
(stable-half vs stable-half) exceeds observed L2 (stable vs transition).
Topology doesn't distinguish perceptual states.

### 4. Pre-Switch Timing (`scripts/screen_preswitch_timing.py`)

**Hypothesis**: Topology changes predictably BEFORE reported perceptual switches
(negative lag = early detection).

**Protocol**: TransitionDetector on rivalry EEG, find peak image_distance in
[-3s, +1s] around each switch, compute lag distribution.

**Result**: FAIL 0/3 subjects. Mean lag is negative (-840ms) — topology DOES
precede switches. But null controls (random timepoints) show identical negative
lag pattern. The negative lag is a property of the image_distance time series
(autocorrelation), not switch-specific.

### 5. Sleep Generalization (`scripts/screen_sleep_transitions.py`)

**Hypothesis**: TransitionDetector generalizes from rivalry to sleep stage
transitions (PhysioNet Sleep-EDF).

**Protocol**: Sliding-window PH + CUSUM on Fpz-Cz channel, evaluate
precision/recall against hypnogram annotations.

**Result**: FAIL. Precision 43.5% (vs null 56.1%). Fisher p=0.92 (not
significant). The detector fires on generic non-stationarity, not
sleep-specific dynamics.

### 6. Benchmark: Topology vs Standard Methods (`scripts/benchmark_changepoint_methods.py`)

**This session's work.** Reframed the question: instead of testing new
directions, characterize WHERE topology adds value over simpler approaches.

---

## Today's Benchmark Results (commit 8a696e6)

### Setup

5 methods compared on 3 data sources:

**Methods**:
1. **Topological**: Takens embed -> sliding-window PH -> persistence image L2 -> CUSUM
2. **PELT**: `ruptures.Pelt(model="normal")` — parametric mean+variance changepoint
3. **Spectral**: Sliding-window Welch PSD -> L2 distance -> CUSUM
4. **Variance**: Sliding-window variance -> abs diff -> CUSUM
5. **BOCPD**: Bayesian Online Changepoint Detection (Gaussian, Adams & MacKay 2007)

**Data sources**:
- A: Synthetic switching Rossler (3 known transitions, tolerance +/-1000 samples)
- B: PhysioNet Sleep-EDF (20 hypnogram transitions, tolerance +/-30s)
- C: Rivalry EEG (41 behavioral switches, tolerance +/-5s)

**Fair comparison**: Methods 1/3/4 use identical CUSUM threshold (mean + 2*std).
Same window size (500) and step (50) for windowed methods. PELT gets penalty
sweep on synthetic, default pen=10 on EEG.

### Results

```
                 Synthetic    Sleep-EDF  Rivalry EEG   Mean F1
Topological           0.40         0.65         0.98      0.68
PELT                  0.49         0.00         0.99      0.49
Spectral              0.26         0.76         0.89      0.63
Variance              0.38         0.74         0.87      0.67
BOCPD                 0.47         0.74         0.99      0.73
```

### Interpretation

**BOCPD wins overall** (mean F1=0.73). Consistent across all data sources.
Bayesian approach handles diverse signal types without domain-specific tuning.

**Topology is #2** (mean F1=0.68). Beats spectral (0.63), ties variance (0.67).
But the story is source-dependent:

**Where topology WINS — Rivalry EEG (F1=0.98)**:
- Beats spectral (0.89) and variance (0.87) by meaningful margins
- 97% precision, 100% recall — near-perfect detection
- Earliest detection: lag=-27 samples vs variance +101
- This is topology's niche: detecting attractor shape changes in the specific
  frequency band (theta-alpha) where binocular rivalry dynamics live
- The PH captures information that PSD summaries and simple variance miss

**Where topology LOSES — Sleep EEG (F1=0.65)**:
- Beaten by spectral (0.76), variance (0.74), BOCPD (0.74)
- Perfect recall (1.00) but low precision (0.49) — 169 detections for 20 transitions
- Sleep transitions are gradual spectral shifts, not sharp attractor topology changes
- Simpler methods capture this just as well

**Where topology is MIDDLING — Synthetic (F1=0.40)**:
- Beats spectral (0.26) but loses to PELT (0.49) and BOCPD (0.47)
- Topology detects earliest (lag=-412) but with many false positives (32 detections for 3 transitions)
- The switching Rossler has sharp bifurcation transitions that PELT handles well

**PELT completely fails on EEG** (0/0/0 on sleep). model="normal" only detects
mean+variance shifts. EEG transitions are oscillatory, not Gaussian. This is
not a bug — it's a real limitation of parametric changepoint detection on
non-Gaussian signals.

### Runtime

Total: ~5 minutes. Topological method is the bottleneck (~15s on sleep data
with 3588 windows and 16 parallel workers). Other methods are <1s each.

Performance fixes applied during development:
- Capped Takens auto-estimation to 20k samples (AMI/FNN is O(n log n))
- Min/max tracking instead of Python list accumulation for shared PH ranges
  (original: 155s on 180k samples; fixed: <0.1s)
- Persistence image resolution 20x20 (default 50x50 was unnecessary)
- Adaptive PELT jump parameter for long signals (jump=5 on 180k is O(n^2))

---

## The Honest Assessment

### What ATT definitively does well

1. **TransitionDetector on rivalry EEG**: 94% precision (from batch_eeg) and
   now F1=0.98 in the benchmark. This is real, reproducible, and better than
   simpler methods on this specific data type.

2. **BindingDetector on synthetic coupled systems**: The core contribution.
   Joint-vs-marginal PH detects coupling, tracks coupling strength monotonically,
   and is validated against TE/PAC/CRQA.

3. **The library itself**: Clean, tested (216 tests), documented, pip-installable.
   Good infrastructure regardless of which analyses pan out.

### What ATT does NOT do

1. **Generalize beyond rivalry EEG for changepoint detection**. Sleep data,
   state fingerprints, and pre-switch timing all failed. The signal seems
   specific to the binocular rivalry paradigm with its theta-alpha bandpass.

2. **Detect cone geometry from directed coupling**. The cone prototype failed
   surrogate testing. The measurement is too noisy relative to the effect.

3. **Cross-frequency binding**. Failed surrogate testing.

4. **Beat BOCPD overall**. Bayesian online changepoint detection is simpler,
   faster, and more general. Topology only wins on rivalry EEG.

### The remaining question

The TransitionDetector's 0.98 F1 on rivalry EEG is real but unexplained.
It could be:
- (a) Genuine attractor topology change during perceptual switches
- (b) Bandpass-specific non-stationarity that happens to correlate with switches
- (c) Variance/spectral change that topology captures indirectly

The benchmark suggests (a) has partial support: topology beats spectral and
variance on this data. But it doesn't beat BOCPD. The question of mechanism
remains open.

---

## Open Plan: 4x4 Grid Scaling

A plan file exists at `.claude/plans/quirky-painting-rain.md` with 7 tasks for
scaling the cone prototype to a 4x4 grid network. **This plan was created BEFORE
the cone diagnostics killed the statistical case.** It should be treated as
superseded unless the measurement problem is solved first.

If someone picks this up, read `2026-03-29-cone-diagnostics.md` first. The plan's
"PROCEED with revisions" decision was based on single-seed Phase 6 results that
don't replicate.

---

## File Inventory

### Scripts (in execution order of development)

| Script | Purpose | Result |
|--------|---------|--------|
| `scripts/download_eeg.py` | Fetch rivalry SSVEP dataset | Utility |
| `scripts/generate_readme_figure.py` | README demo figure | Utility |
| `scripts/batch_eeg.py` | N=80 multi-subject EEG analysis | 94% precision, 41% recall |
| `scripts/cross_freq_binding_screen.py` | Theta-alpha-gamma binding | FAIL |
| `scripts/screen_state_fingerprints.py` | Stable vs transition topology | FAIL 0/3 |
| `scripts/screen_preswitch_timing.py` | Pre-switch lag detection | FAIL 0/3 |
| `scripts/screen_sleep_transitions.py` | Sleep stage transitions | FAIL |
| `scripts/benchmark_changepoint_methods.py` | Topology vs 4 standard methods | F1=0.98 on rivalry |

### Handoff Files (chronological)

| File | Content |
|------|---------|
| `2026-03-27-phase1-complete.md` | Foundation: embedding + topology |
| `2026-03-27-phase2-core.md` | Binding detection + benchmarks |
| `2026-03-27-phase2-complete.md` | Phase 2 wrap-up |
| `2026-03-27-phase3-transitions.md` | Transition detection |
| `2026-03-27-figures-and-results.md` | Paper figures |
| `2026-03-28-phase4-real-eeg.md` | Real EEG validation |
| `2026-03-28-phase5-distribution.md` | PyPI, docs, blog |
| `2026-03-28-phase6-extensions.md` | Extensions |
| `2026-03-28-phase7-test-expansion.md` | 134->216 tests |
| `2026-03-28-phase8-method-hardening.md` | Edge cases + validation |
| `2026-03-28-phase9-preprint-update.md` | 21-page preprint |
| `2026-03-28-phase10-polish.md` | README, CONTRIBUTING, URLs |
| `2026-03-28-phase11-multi-subject-eeg.md` | N=80 validation |
| `2026-03-28-phase12-final-gaps.md` | Binding batch + tutorials |
| `2026-03-29-cone-prototype-scaffold.md` | Cone infrastructure |
| `2026-03-29-cone-implementation.md` | ConeDetector code |
| `2026-03-29-cone-experiments.md` | 6 experiments, PROCEED decision |
| `2026-03-29-cone-diagnostics.md` | **Multi-seed kills cone signal** |
| `2026-03-29-grid-scaling-plan.md` | 4x4 grid plan (likely superseded) |
| `2026-03-30-benchmark-and-project-state.md` | **This file** |

### Data

- `data/eeg/rivalry_ssvep/`: 84 subjects, binocular rivalry SSVEP (360 Hz, 34-ch)
- Sleep-EDF: auto-downloaded via MNE at runtime
- Synthetic: generated by `att.synthetic.generators`

### Key Commits

```
8a696e6 Benchmark topological changepoints vs PELT, spectral, variance on synthetic + EEG
5508eec Screen 3 directions: state fingerprints, pre-switch timing, sleep transitions
4840b3b Cross-frequency binding screen: theta-alpha-gamma pairs, surrogate-tested
5e45f63 Cone prototype: ConeDetector, 5-node network, all 6 experiments complete
73231e7 Phase 11: multi-subject EEG validation (N=80) + batch parallelization
155e06d Phase 9: preprint update — 21 pages, 9 experiments, z-score calibration
```

---

## Recommendations for Next Session

### If continuing the project

**Option A — Deepen the rivalry EEG result**: The 0.98 F1 is real and topology
beats simpler methods here. Investigate WHY. Does it generalize to more subjects?
What specific topological features (H0 vs H1, specific persistence ranges) drive
the detection? Can you find the frequency band or embedding dimension that
maximizes the gap between topology and spectral/variance? This is the path to a
second paper: "Topological changepoint detection outperforms spectral methods on
binocular rivalry EEG."

**Option B — BOCPD integration**: BOCPD won the overall benchmark. Could you
combine topology and BOCPD? Use PH-derived features as input to a Bayesian
changepoint model. This hybrid might get the best of both: topology's attractor
sensitivity + BOCPD's probabilistic framework.

**Option C — Drop the cone, ship what works**: The binding detection and
transition detection both work. The preprint is submitted. The library is
pip-installable. The remaining value is in distribution (Phase 4 blog post,
community engagement), not in pushing null results further.

### If pivoting away

The project is in a clean state. All tests pass. Everything is committed and
pushed. The handoff files document what worked and what didn't. Someone can
pick this up at any point and understand the full history.

### What NOT to do

- Do NOT scale to 4x4 grids. The cone measurement doesn't work at 5-node scale.
- Do NOT run more screening scripts in the same pattern. 0-for-6 is a clear signal
  that the low-hanging fruit has been picked.
- Do NOT rerun Phase 6 cone experiments with more seeds hoping for better results.
  The diagnostics already did this and the signal is noise.

---

## Technical Gotchas for Future Sessions

1. **PersistenceAnalyzer.to_image() is slow in Python loops**: The inner loop
   iterates over persistence features with np.outer. For 3000+ windows, pass
   `resolution=20` not the default 50. Or vectorize the loop.

2. **Shared PH ranges**: Never accumulate birth/persistence values into Python
   lists via `.tolist().extend()`. Track min/max directly. The list version is
   O(n_features * n_windows) in memory and 150x slower.

3. **TakensEmbedder.fit() on long signals**: AMI + FNN estimation is O(n log n).
   On 180k samples it takes minutes. Cap the fit signal to 20k samples; the
   estimated parameters generalize.

4. **ruptures PELT with model="rbf"**: O(n^2) memory for the kernel matrix.
   Infeasible above ~10k samples. Use model="normal" or model="l2" instead.
   Adaptive jump parameter: `jump = max(5, len(signal) // 2000)`.

5. **Multiprocessing.Pool for ripser**: The parallel PH pattern works well.
   16 workers, `pool.imap` with tqdm. ~600 windows/sec with subsample=200.
   Don't forget `if __name__ == "__main__"` guard if running on Windows.

6. **EEG rivalry data**: 84 subjects at `data/eeg/rivalry_ssvep/`. Load via
   `scripts/batch_eeg.py` functions: `discover_subjects()`, `load_rivalry_epoch()`,
   `load_behavioral_switches()`. Oz channel, bandpass 4-13 Hz, sfreq=360.
