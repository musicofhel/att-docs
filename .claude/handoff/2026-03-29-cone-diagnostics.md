# Cone Diagnostic Results — March 29, 2026

All tests: 5-node Aizawa network, N=10k-15k steps (post-transient),
subsample=350-400 for persistence, multiple seeds.

---

## TEST 1: Uncoupled vs Directed — Depth Asymmetry (5 seeds)

Depth asymmetry = binding(C,A5) - binding(C,A3).
Positive = "cone opens with depth."

```
seed  OFF(asym)   ON(asym)
  0     -1141       -540
  1     -1931      -1066
  2      -283       -815
  3     -2700      +1446
  4     -1757      +2123

UNCOUPLED:  mean=-1562 ± 810
DIRECTED:   mean=+230  ± 1298
t=2.34, p=0.047, Cohen's d=1.48
Paired (deep>shallow, directed): t=0.35, p=0.74
CV of directed asymmetry: 566%
```

**Interpretation**: Uncoupled systems have NEGATIVE asymmetry (shallow > deep
consistently). Directed coupling shifts this toward zero. But:
- Only 2/5 directed seeds are positive
- The paired test (is deep > shallow within directed?) is p=0.74
- The asymmetry is 3.9% of the raw binding scores (scores are 3000-8000)
- CV of 566% means the measure is dominated by noise

The Cohen's d=1.48 looks impressive but is measuring "directed is less
negative than uncoupled" — not "directed produces a cone."

---

## TEST 2: Coupling Sweep (2 seeds per value)

```
  eps   seed   shallow     deep     asym    d/s
 0.00     0      4652      1297    -3355   0.28
 0.00     1      3097       907    -2190   0.29
 0.10     0      5147      4274     -874   0.83
 0.10     1      5074      5505     +431   1.08
 0.20     0      4824      5307     +483   1.10
 0.20     1      4532      5519     +986   1.22
 0.30     0      4384      3914     -469   0.89
 0.30     1      3947      5474    +1528   1.39
 0.50     0      1701      4311    +2610   2.53
 0.50     1      3746      3419     -327   0.91
```

**Interpretation**: No consistent monotonic trend. At ε=0.3, seed 0 is
negative while seed 1 is positive. At ε=0.5, seed 0 is strongly positive
while seed 1 is negative. The sign of the asymmetry depends more on the
seed than on the coupling strength.

The Phase 6 report claimed "asymmetry monotonic 0→+40889" from a single
seed=42 run at 80k steps. That was one draw from a high-variance distribution.

---

## TEST 3: Surrogate Test (seed=0, ε=0.15, 8 surrogates)

Phase-randomize A5 (destroy its coupling to A3), recompute asymmetry.
If the cone is real, observed asymmetry should exceed surrogate distribution.

```
Observed asymmetry: +588
Surrogate asymmetries: +183, +831, +1479, +263, +391, +3143, +68, +41
Surrogate mean: +800 ± 994
z = -0.21, p = 0.375
```

**Interpretation**: The observed "cone signal" is SMALLER than the surrogate
mean. Phase-randomizing the deep channel doesn't reduce the signal — it
sometimes INCREASES it (surr 5: +3143 vs observed +588). The "cone" is
indistinguishable from what you get with a random signal at the deep level.

This is the kill shot. If the cone were real coupling geometry, destroying
the coupling should destroy the asymmetry. It doesn't.

---

## TEST 4: Directed vs Symmetric vs Uncoupled (2 seeds)

```
seed=0:
   uncoupled: shallow=4652  deep=1297  asym=-3355  d/s=0.28
    directed: shallow=3202  deep=3790  asym=+588   d/s=1.18
   symmetric: shallow=4775  deep=5400  asym=+625   d/s=1.13

seed=1:
   uncoupled: shallow=3097  deep=907   asym=-2190  d/s=0.29
    directed: shallow=3819  deep=5050  asym=+1230  d/s=1.32
   symmetric: shallow=6829  deep=9757  asym=+2927  d/s=1.43
```

**Interpretation**: Symmetric coupling produces EQUAL OR LARGER asymmetry
than directed in both seeds. The Phase 6 report claimed directed was 7.6x
steeper than symmetric. That was one seed.

If the cone requires directed projection, symmetric (all-to-all) coupling
should NOT produce cone-like asymmetry. But it does. This means the
asymmetry is not about directionality — it's about coupling strength
affecting the deep node more than the shallow node (which makes sense:
A5 gets two hops of coupling influence, A3 gets one).

---

## TEST 5: Betti_1 Slope (Availability Profile, 3 seeds)

```
seed=0 uncoupled: slope=+3.27  β₁=[36, 30, 20, 21, 46]  monotonic=False
seed=0  directed: slope=+3.32  β₁=[89, 73, 64, 67, 106] monotonic=False
seed=1 uncoupled: slope=-3.93  β₁=[61, 28, 31, 27, 51]  monotonic=False
seed=1  directed: slope=-2.24  β₁=[100, 64, 63, 56, 94] monotonic=False
seed=2 uncoupled: slope=-4.81  β₁=[50, 37, 14, 18, 43]  monotonic=False
seed=2  directed: slope=-1.40  β₁=[81, 56, 39, 47, 78]  monotonic=False
```

**Interpretation**: The Betti_1 profiles are all U-shaped (high at edges,
low in middle) regardless of coupling. No monotonic increase with depth
in ANY condition. Slopes are small and sign-unstable. Directed is
indistinguishable from uncoupled.

The Phase 6 report of "slope=+42.25, β₁: 431→787" was from 80k steps.
Longer series produce more persistent features everywhere, inflating
absolute Betti counts. The slope might just scale with data length.

---

## DIAGNOSIS

### What's actually happening

1. The binding score has a large structural positive baseline (~3000-6000)
   that varies with embedding parameters, data length, and seed.

2. The asymmetry (deep minus shallow) is a TINY residual on top of these
   large scores — typically 3-5% of the raw score.

3. This residual is dominated by stochastic variation (CV > 500%).

4. The surrogate test fails: destroying coupling doesn't destroy the signal.

5. Symmetric coupling produces the same or larger "cone" as directed.

6. The Betti slope shows no directional structure.

### What went wrong in Phase 6

- Single seed (42), long time series (80k), one run per condition.
- Raw asymmetry numbers looked large (hundreds to thousands) but were
  never compared against a null distribution or error bars.
- Two predictions failed (inverted-U, 3-way emergence) and were reframed
  rather than treated as evidence against the hypothesis.
- The CCA vs full-embedding contradiction was resolved by picking the
  one that worked, rather than asking why they disagreed.

### What the cone idea needs to survive

If there IS a real cone-like structure from directed projection, the
current measurement approach (binding score asymmetry) cannot detect it.
The binding score's variance is too high relative to the expected signal.

Possible paths forward:
a) Different measurement — not binding scores, maybe direct geometric
   analysis of the joint point cloud (principal curves, cross-section
   areas, something more targeted than generic PH)
b) Much longer time series + ensemble averaging to beat down the variance
c) Stronger coupling where the effect is obvious before doing statistics
d) Accept that the cone might not exist at this network scale

### Recommendation

Do NOT scale to 4×4 grids. The 5-node network doesn't produce a
statistically detectable cone signal. Scaling up multiplies compute
cost without addressing the measurement problem.

If pursuing the cone further: first find a measurement that reliably
distinguishes directed from symmetric AND from uncoupled on the 5-node
network across 10+ seeds with p < 0.01. Then scale.
