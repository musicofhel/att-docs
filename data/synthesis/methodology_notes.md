# Methodology Notes: Cross-Domain Topological Analysis

## 1. What data properties make topology informative?

**Dimensionality matters most.** The strongest results come from high-dimensional
embeddings: LLM hidden states (d=1536-2560) and EEG Takens embeddings (d=7-10).
Low-dimensional financial return embeddings (d=5) yield topology that is entirely
explainable by linear spectral properties.

**Non-stationarity helps.** Systems with genuine regime changes (sleep stages,
tipping points, climate shifts) produce significant topological discrimination.
Quasi-stationary processes (financial returns over stable periods) do not.

**Sample size.** Permutation tests require ≥50 samples per condition for adequate
power. Cardiac (n=15/15) and hallucination (n=38 correct) both failed to reach
significance despite apparent effect sizes.

**Signal-to-noise ratio.** Synthetic data with known structure (tipping models,
Kuramoto oscillators, synthetic music) yields clean results. Real-world data
(cardiac ECG, financial returns) introduces noise that degrades topological
discrimination.

## 2. What are the failure modes?

1. **Linear dynamics dominate:** Finance topology is fully explained by phase-randomized
   surrogates (|z|<2). The power spectrum alone produces equivalent topology. This is the
   most important failure mode to test for.

2. **Insufficient data:** Cardiac and hallucination permutation tests fail despite visible
   differences in H1 counts. With more data, these might reach significance.

3. **High noise:** Real EEG has high noise floors. Sleep stages are discriminable (z=20.14)
   because the signal is strong, but transition detection precision is only 31%.

4. **Wrong embedding parameters:** Music genre boundary detection failed (F1=0.0). The
   Takens embedding parameters may not be appropriate for audio feature time series —
   genre changes are compositional, not dynamical.

5. **Weak coupling:** Binding detection fails for weakly coupled systems (ENSO-NAO r=0.0004,
   sleep cross-region). The joint-vs-marginal PH test appears to require coupling
   strengths above ~0.3 to detect.

## 3. Surrogate testing is non-negotiable

**Without surrogates, finance would be a false positive.** The bull/bear permutation test
gives z=23.6 (p=0.0), apparently a strong result. But phase-randomized surrogates produce
identical topology (z=-0.07). The entire topological signal is linear.

Domains that would have produced misleading results without surrogates:
- **Finance:** False positive (z=23.6 → surrogate z=−0.07)
- **Cardiac:** Would have appeared marginally significant based on RR intervals alone,
  but raw ECG permutation p=0.135 is correctly non-significant

Domains where surrogates confirmed genuine nonlinear topology:
- **LLM:** z=8.11, well beyond surrogate range
- **Sleep:** z=20.14
- **Multilingual:** z=27.37
- **Music:** z=11.58 (synthetic, so surrogates are less critical)

## 4. Per-channel delay estimation

**Needed when:**
- Multi-channel time series with different sampling rates (EEG: different electrode locations)
- Time series with known periodicity (climate ENSO: delay=9 months ≈ ENSO cycle)

**Not needed when:**
- Hidden state analysis (LLM, hallucination, code, multilingual) — no temporal embedding
- Simulated systems with known delay structure (tipping, multiagent)

Auto-estimated delays: cardiac (delay=8), climate (delay=9), sleep (auto per-channel).
Fixed delays: tipping (delay=4), finance (delay=1).

## 5. Binding detection vs standard coupling measures

**Where topology added value beyond TE/PAC/CRQA:**
- **LLM binding (D10):** Population-level topological similarity between problems at same
  difficulty has no standard coupling analogue. This is a genuinely novel measure showing
  r=-0.9 with difficulty level.
- **Multiagent:** Binding correctly distinguishes direct from indirect coupling
  (direct 81.4 > indirect 73.8) — validates the geometric intuition.

**Where it did NOT add value:**
- **Climate ENSO-NAO:** Pearson correlation (r=0.0004) already shows no coupling.
  Binding test (p=0.68) adds nothing.
- **Sleep cross-region:** Binding p=0.80 for REM EEG. Standard coherence measures
  would likely also fail at these sample sizes.
- **Multilingual:** En-Zh binding p=0.255. Cross-language correlation (r=0.9998)
  already shows the representation structure is nearly identical, making binding
  redundant.

**Bottom line:** Binding detection is ATT's most novel contribution but also its
most fragile. It works best when: (1) coupling is strong, (2) the coupling is
geometric (not merely correlational), and (3) sample sizes are adequate (>100 per
condition). For weak or correlational coupling, standard measures are sufficient.
