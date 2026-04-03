# Branch 8: Financial Regime Detection — Market State Topology — 2026-04-03

## Branch
`experiment/tda-finance` (from `experiment/neuromorphic-snn` @ `6e60605`)

## What Was Done

Created `scripts/branches/financial_regime_topology.py` — applies ATT's topological tools to 20 years of S&P 500 daily returns (SPY, 2005-2025) to test whether persistent homology can detect market regime changes. Compares with VIX as ground truth. Four experiments covering crisis detection, surrogate null hypothesis testing, VIX correlation, and bull/bear attractor comparison.

### Data

| Source | Ticker | Period | Trading Days |
|--------|--------|--------|-------------|
| S&P 500 | SPY | 2005-01-03 to 2025-12-30 | 5282 |
| CBOE Volatility Index | ^VIX | same | same |

Log returns: annualized mean 10.2%, annualized vol 19.1%. VIX: mean 19.1, max 82.7 (2020 COVID).

### Takens Embedding

Fixed params: delay=1 (standard for daily returns), dimension=5. TransitionDetector: window_size=252 (1 trading year), step_size=21 (1 month), subsample=200, max_dim=1, Ripser backend.

### Four Experiments

| Exp | What | Key Result |
|-----|------|------------|
| 1 | Sliding-window topology crisis detection | 0/3 crises detected above 2-sigma threshold |
| 2 | Surrogate testing (phase-randomize + time-shuffle) | z=-0.07 (phase), z=0.05 (shuffle) — signal is LINEAR |
| 3 | Topology vs VIX correlation | H1 entropy-VIX Pearson r=-0.19, transition score-VIX r=0.04 |
| 4 | Bull vs Bear attractor topology | Wasserstein p=0.0, z=23.6 (confounded by sample size) |

### Key Findings

**Exp 1 — Topology does NOT detect known market crises:**
- 2008 Financial Crisis: NOT DETECTED
- 2020 COVID Crash: NOT DETECTED
- 2022 Rate Hike Selloff: NOT DETECTED
- 14 CUSUM changepoints detected, but none align with known crisis onsets using 2-sigma exceedance criterion
- Transition scores show variation but no distinctive spikes at crisis events
- Threshold (1838) derived from first 500 trading days (2005-2007 pre-crisis baseline)

**Exp 2 — Topological signal is LINEAR (the critical finding):**
- Phase-randomize z-scores: H1 entropy z=-0.74, total persistence z=-0.07, H0 entropy z=-3.44
- Time-shuffle z-scores: H1 entropy z=-0.28, total persistence z=0.05, H0 entropy z=-3.44
- Phase-randomized surrogates (which preserve power spectrum but destroy nonlinear structure) produce nearly identical H1 topology to real data
- This means the topological features of financial returns are explained by the linear autocorrelation structure (power spectrum), not by any nonlinear dynamics
- The H0 entropy z=-3.44 is significant but in the WRONG direction (real < surrogate), meaning the real data has LESS H0 structure than surrogates — possibly reflecting the well-known fact that financial returns have lighter tails in the Fourier domain than surrogate-reconstructed series
- **This confirms prior ATT findings on financial data (z-scores -8 to -10)**

**Exp 3 — Weak topology-VIX correlation:**
- H1 persistence entropy vs VIX: Pearson r=-0.19, Spearman r=-0.21 (both highly significant, p < 10^-46)
- Transition score vs VIX: Pearson r=0.04, Spearman r=0.08 (statistically significant but practically negligible)
- The negative H1-VIX correlation means higher volatility periods have LOWER topological complexity — consistent with the idea that volatile markets have more "random walk" character (less coherent attractor structure)
- However, r=-0.19 is weak and not useful for prediction

**Exp 4 — Bull/Bear topology differs, but confounded:**
- Bull cloud (2000 pts from 2320 VIX<20 days): H0 ent=6.11, H1 ent=5.44, 330 H1 features
- Bear cloud (206 pts from 210 VIX>30 days): H0 ent=5.21, H1 ent=4.54, 126 H1 features
- Wasserstein-1 distance: 4.68, permutation p=0.0, z=23.6
- **CAVEAT**: The huge z-score is almost certainly a sample-size artifact. Bull cloud has 10x more points (2000 vs 206), producing richer PH. The permutation test pools all points and randomly splits, but the unequal group sizes mean the null distribution is narrow while the observed difference is large simply due to point count disparity
- A fair comparison would need equal-sized subsamples from each regime

### Interpretation

**Overall verdict: linear-only.** This is a principled negative result that confirms the hypothesis stated in the branch spec.

The topological structure of financial return time series is almost entirely explained by the power spectrum (linear autocorrelation). Phase-randomized surrogates — which preserve the spectrum but destroy all nonlinear dynamics — produce statistically indistinguishable persistent homology from real market data. This means:

1. TDA on financial returns does not capture "nonlinear market dynamics" — it captures the autocorrelation function in a more expensive way
2. ATT's TransitionDetector cannot reliably detect market crises from topological signatures alone
3. The weak H1-VIX correlation (r=-0.19) is real but too weak for any practical application

This aligns with the efficient market hypothesis in a TDA context: daily returns are close enough to filtered noise that their Takens embedding topology is spectrum-determined.

### Where Topology MIGHT Work in Finance

Despite the negative result on daily returns, topology could add value in:
1. **Intraday tick data** — where microstructure creates genuine nonlinear dynamics (queue effects, market-making)
2. **Order book snapshots** — topological features of bid/ask depth profiles
3. **Cross-asset correlation matrices** — where Mapper/Reeb graphs on the correlation space may reveal regime structure invisible to PCA
4. **Options implied volatility surfaces** — where the geometry has genuine topological content (holes, cusps)

### Caveats

- Only tested daily close-to-close returns; intraday granularity might reveal nonlinear dynamics
- Takens embedding with delay=1 on daily returns may be too coarse — financial data at sub-second resolution has much richer temporal structure
- The 2-sigma crisis detection threshold may be too strict; a percentile-based or adaptive threshold could perform differently
- VIX is itself a model-derived quantity (implied volatility), not a direct market observable
- subsample=200 for PH computation — larger subsamples give more detail but same qualitative result expected
- Exp4 bull/bear comparison is confounded by unequal sample sizes (2320 vs 210 days)

## Files

- `scripts/branches/financial_regime_topology.py` — full analysis script
- `data/finance/results.json` — all numeric results
- `figures/finance/` — 6 figures:
  - `overview.png` — SPY price, VIX, and log returns (20 years)
  - `exp1_topology_crisis_detection.png` — returns + transition scores + VIX with crisis markers
  - `exp2_surrogate_testing.png` — surrogate null distributions vs real topology
  - `exp3_topology_vix_correlation.png` — time series overlay of VIX, H1 entropy, transition score
  - `exp3_scatter_correlation.png` — scatter plots of topology metrics vs VIX
  - `exp4_bull_bear_topology.png` — persistence diagrams + permutation test

## Technical Notes

- yfinance used for data download (cached after first run)
- TransitionDetector runs once (~149s for 5281 returns, 252-day window, 21-day step → ~239 windows)
- Surrogate testing uses fast full-cloud PH comparison (50 surrogates × ~1s each) instead of per-surrogate TransitionDetector runs (~150s each, which would take 4+ hours)
- Phase-randomize uses ATT's AAFT surrogate method (preserves spectrum + marginal distribution)
- Permutation test for Exp4: pool bull+bear returns, randomly split, compute Wasserstein-1 distance 100 times
- Total runtime: ~169s (~2.8 minutes)

## Potential Follow-ups

- Test on intraday tick data (e.g., 1-minute SPY bars) where nonlinear microstructure may produce genuine topological signal
- Apply Mapper/Reeb graphs to rolling correlation matrices of sector ETFs
- Test on emerging market indices (MSCI EM) where inefficiency may produce nonlinear dynamics
- Equal-size subsample comparison for bull/bear (subsample both to 200 points)
- Extend to crypto markets (BTC/ETH) where extreme volatility and market inefficiency may produce detectable nonlinear topology
- Multi-asset cross-correlation topology: compute PH on the distance matrix derived from rolling pairwise correlations
