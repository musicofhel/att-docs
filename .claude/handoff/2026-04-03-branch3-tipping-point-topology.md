# Branch 3: Tipping Point Topology — 2026-04-03

## Branch
`experiment/tda-tipping` (from `experiment/neuromorphic-snn` @ `6e60605`)

## What Was Done

Created `scripts/branches/tipping_point_topology.py` — generates three synthetic dynamical systems with known catastrophic bifurcations and tests whether ATT's topological tools detect the approach of the tipping point BEFORE the system flips. Compares topological early warning signals (EWS) with classical EWS (variance, autocorrelation, skewness).

### Synthetic Models

| Model | Equation | Bifurcation | Tipping Index |
|-------|----------|-------------|---------------|
| Saddle-node (lake eutrophication) | dx/dt = -x³ + x + r(t) | r = 2/(3√3) ≈ 0.385 | 17698 |
| Hopf (oscillation onset) | dx/dt = μx - y - x(x²+y²), dy/dt = x + μy - y(x²+y²) | μ = 0 | 10000 |
| Double-well (regime switching) | dx/dt = x - x³ + 0.2·sin(2πt/20000) | First zero-crossing | 5891 |

All models: 20000 timesteps, dt=0.01, seed=42. Hopf analyzed via amplitude √(x²+y²) for 1D sliding-window compatibility.

### Takens Embedding

Fixed params: delay=4, dimension=5 (consistent with sleep branch). TransitionDetector: window_size=500, step_size=200, subsample=200, max_dim=1, Ripser backend.

### Three Experiments

| Exp | What | Key Result |
|-----|------|------------|
| 1 | Sliding-window topology lead times | SN: 2291, Hopf: 416, DW: 2357 steps before tipping |
| 2 | Topology vs classical EWS comparison | Topology earliest in 1/3 models; variance often detects earlier |
| 3 | Pre/post tipping cloud PH (saddle-node) | Wass=2.32, H1 counts similar (61 vs 59), entropy similar (3.67 vs 3.72) |

### Key Findings

**Exp 1 — Topology detects all three tipping points:**
- Saddle-node: 2291 steps (11.5% of series) lead time
- Hopf: 416 steps (2.1%) — weakest signal, expected since Hopf is a smooth bifurcation
- Double-well: 2357 steps (11.8%) — strong signal despite noise-driven transition
- All three show topology changes before the system flips

**Exp 2 — Classical EWS often detect earlier than topology:**
- Saddle-node: variance (6883) and autocorrelation (6738) detect much earlier than topology (2291). This is expected — critical slowing down produces a strong variance/AC signal in the saddle-node model
- Hopf: variance (728) > topology (416) > AC (0) = skewness (0). Topology outperforms AC and skewness
- Double-well: topology (2357) ties with variance (2357), beats skewness (1711) and AC (652)
- Topology is the earliest detector in 1/3 models (double-well, tied with variance)
- Classical EWS benefit from the clean critical-slowing-down signature in these canonical models. In real-world systems with non-stationary noise, topology may perform relatively better

**Exp 3 — Pre/post tipping topology is subtly different:**
- Wasserstein-1 distance = 2.32 between pre and post clouds
- H1 feature counts nearly identical (61 vs 59) — the bifurcation changes the *geometry* more than the *topology*
- H0 entropy drops post-tipping (5.08 → 4.50) — the point cloud becomes more clustered after the system jumps to the new attractor
- H1 entropy slightly increases (3.67 → 3.72) — marginal

### Interpretation

The results are honest: topology *works* as an EWS but doesn't universally beat classical methods on clean synthetic data. This is actually the expected finding — classical EWS are tuned for exactly these canonical bifurcation models (Scheffer et al. 2009). Topology's advantage should emerge in:
1. Systems without clean critical-slowing-down signatures
2. High-dimensional systems where variance/AC are undefined or misleading
3. Multivariate coupling scenarios (not tested here)

The double-well model, which has noise-driven transitions rather than parameter-driven, is the one where topology ties with the best classical method — hinting at topology's relative advantage in noise-dominated regimes.

### Caveats

- 20000 steps is relatively short; longer runs (50000+) would give smoother topology signals but take ~6x longer
- step_size=200 gives only ~98 windows per model — higher resolution (step_size=50) would improve temporal precision
- Hopf model uses amplitude √(x²+y²) for 1D analysis, losing phase information. 2D sliding-window PH could capture the bifurcation more directly
- Classical EWS use rolling window of 500 (same as topology window) for fair comparison
- Lead time detection uses 2σ above pre-bifurcation baseline with 3-consecutive-point criterion
- Double-well "tipping point" is the first zero-crossing, which is a noise-driven event, not a parameter-driven bifurcation like the other two
- subsample=200 for PH computation — larger subsamples would give more detailed persistence diagrams

## Files

- `scripts/branches/tipping_point_topology.py` — full analysis script
- `data/tipping/results.json` — all numeric results
- `figures/tipping/` — 10 figures:
  - `time_series_overview.png` — all 3 models with tipping points marked
  - `exp1_{saddle_node,hopf,double_well}.png` — time series + transition scores + H1 entropy
  - `exp2_{saddle_node,hopf,double_well}.png` — 5-panel EWS comparison
  - `exp3_saddle_node.png` — pre/post persistence diagrams + PI difference
  - `lead_time_comparison.png` — bar chart comparing all EWS methods

## Technical Notes

- No external data needed — fully synthetic
- TransitionDetector runs once per model; results shared between Exp 1 and Exp 2
- Classical EWS (rolling variance, lag-1 autocorrelation, skewness) computed with scipy.stats
- Lead time detection: interpolate topology scores to full-length signal, find first 3-consecutive-point exceedance of mean+2σ from pre-bifurcation baseline (first 60% of pre-tipping region)
- Total runtime: ~3 minutes

## Potential Follow-ups

- Increase n_steps to 50000 with step_size=50 for higher temporal resolution
- 2D sliding-window PH for Hopf model (use xy point cloud directly, not amplitude)
- Add May's fold catastrophe model (another canonical tipping point)
- Test on empirical regime-shift data (e.g., lake sediment cores, climate records)
- Multivariate topology: use 2+ coupled oscillators and BindingDetector to test cross-system coupling EWS
- Compare with more sophisticated classical EWS: detrended fluctuation analysis, spectral reddening
- Bootstrap confidence intervals on lead times
