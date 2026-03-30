# Financial Regime Detection Screen — March 30, 2026

Screen testing whether ATT's topological tools detect meaningful structure in
financial time series and can identify regime transitions. Branch:
`experiment/financial-dynamics` (commit `7ddb903`, pushed to origin).

---

## Motivation

Financial markets exhibit regime-switching behavior (bull/bear, low/high
volatility). If daily return series have genuine attractor geometry, ATT
could detect regime transitions topologically — using VIX > 30 as ground
truth for crisis onsets.

---

## What Was Built

**Script**: `scripts/screen_financial_regimes.py` (~585 lines)

- Data fetching via yfinance with disk cache (`data/financial/*.parquet`)
- Fallback to synthetic two-state Hamilton regime-switching model
- Part 1: Attractor structure detection (Takens embedding + PH + phase-randomized surrogates)
- Part 2: 4-method changepoint benchmark on SPY (topological, spectral, variance, BOCPD), VIX > 30 ground truth
- Part 3: Cross-asset binding (SPY-TLT, SPY-GLD via BindingDetector)
- Kill criterion: if z <= 0 on ALL series in Part 1, skip Parts 2-3

**Data**: 5 years daily OHLCV for SPY, TLT, GLD, ^VIX (1256 rows each,
2021-03-29 to 2026-03-30) cached as Parquet files.

**No new library code** — uses existing TakensEmbedder, PersistenceAnalyzer,
BindingDetector, plus copied benchmark methods.

---

## Results (23.4s runtime)

### Part 1: Attractor Structure Detection — ALL FAIL

| Ticker | Delay | Dim | B0   | B1   | Entropy | Mean Z-score | Significant |
|--------|-------|-----|------|------|---------|--------------|-------------|
| SPY    | 2     | 10  | 399  | 302  | 6.04    | -8.34        | NO          |
| TLT    | 3     | 10  | 399  | 309  | 6.11    | -3.05        | NO          |
| GLD    | 2     | 10  | 399  | 344  | 6.12    | -9.88        | NO          |

Kill criterion triggered. Parts 2 and 3 skipped.

All three series show *less* total persistence than phase-randomized surrogates
(strongly negative z-scores, 3 seeds x 15 surrogates each). The real data has
less topological complexity than the null, not more.

---

## Honest Assessment

**Why it fails**: Phase randomization preserves the power spectrum (linear
autocorrelation) but destroys nonlinear dynamics. If financial returns had
genuine attractor geometry, the real series would show *more* topological
features than surrogates. The negative z-scores indicate surrogates are
actually more complex, meaning the nonlinear structure in financial returns
(if any) acts to *reduce* topological features at daily frequency.

**Interpretation**: Daily financial log returns are well-described by their
linear autocorrelation structure. Takens embedding finds high embedding
dimensions (dim=10) but the resulting point clouds don't have topological
structure beyond what linear processes produce. This is consistent with the
efficient market hypothesis — daily returns are close to white noise with
some autocorrelation.

**Comparison to ATT's other results**:
- Rivalry EEG (neural oscillations): strong attractor structure, topology F1=0.98
- Lorenz/Rossler (deterministic chaos): strong attractor structure
- Financial returns: no detectable attractor structure at daily frequency
- The kill-early design worked correctly — saved ~10 min of compute

**What might help**:
- Higher-frequency data (tick/minute) where microstructure effects create
  deterministic dynamics
- Multivariate embedding (multiple assets jointly) rather than univariate
- Longer lookback periods (10+ years) for more data
- Different embedding approach (time-delay mutual information might not be
  appropriate for near-white-noise processes)

---

## Branch Status

- Branch: `experiment/financial-dynamics` (pushed to origin)
- Commit: `7ddb903`
- NOT merged to master
- Script is self-contained — no library changes, no test changes
- Data files cached in `data/financial/` (5 Parquet files, ~1MB total)

---

## If Continuing This Work

1. **Higher-frequency data**: Minute or tick data from a provider like
   Polygon.io. Microstructure effects at high frequency are genuinely
   nonlinear and might have attractor geometry.

2. **Realized volatility series**: Instead of raw returns, compute 5-min
   realized volatility (which has strong serial dependence). This is a
   positive-valued process with known long memory — more likely to have
   attractor structure.

3. **Multivariate Takens**: Embed multiple assets jointly. The joint
   embedding might reveal cross-asset structure invisible in marginals.
   This aligns with ATT's core joint-vs-marginal approach.

4. **Don't bother with**: Daily log returns — the kill criterion was
   definitive. The signal-to-noise ratio at daily frequency is too low
   for topological methods.
