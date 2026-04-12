---
creator: agent-1
created: 2026-04-12T08:35:00+00:00
---
# Eval 14 Reflection: sigma=1.9 for noise did NOT improve

## Result
Score: 0.9931 (unchanged). Per-mode: sr_drift=0.988, noise=0.9915, cd=1.0.

## What happened
Lowering sigma from 2.0 to 1.9 for noise Crocker CUSUM should have pushed threshold
from 14.48d to 14.10d. If the hard noise seed's CUSUM at window start=5200 was between
14.10d and 14.48d, it would have triggered. It didn't.

## Conclusion about the noise FN
The hard noise seed's CUSUM at window start=5200 is BELOW 14.10d (below even sigma=1.9
threshold). This means the noise signal accumulation in Crocker topology space is weaker
than estimated. The FN is NOT a threshold-tuning issue — it's a signal strength issue.

The CUSUM is cumulative. By window start=5200, only 2 drift windows have accumulated.
For the hard seed, the per-window drift signal may be genuinely weak (low CUSUM growth).

## Implication
Sigma reduction is NOT the path forward for noise improvement.

## Agent-2's key insight (from their notes)
Agent-2 achieved 0.9931668064 (slightly higher than my 0.9931349329) using:
- sr_drift: inverted binding STRIDE=200, k=2.0 → 0.988 (same as my Crocker approach!)
- noise: Crocker H1 STRIDE=400 cached → 0.9915 (same)
- cd: Crocker STRIDE=400 cached → 1.0 (same)
- uses_inverted_binding=True → leaderboard flags it, scientific value

The tiny score difference (0.000032) is within the per-seed F1 variation.
The significant difference: agent-2's approach has uses_inverted_binding=True.

## Next experiment: variance-based OR arm for noise
The noise FN is about weak Crocker signal accumulation. A DIRECT (non-CUSUM) 
variance detector might catch the hard seed:

- Noise adds weight perturbations → reservoir state variance increases
- Mean neuron std per window: var_score = mean(std(X, axis=0)) over window
- Direct alarm: var_score > baseline_mean + k * baseline_std (no CUSUM needed!)
- OR this with Crocker CUSUM for noise only

Expected: noise 0.9915 → 1.0 → mean 0.9931 → 0.9944

Risk: if variance increases gradually (not abrupt at window 5200), might not catch
the specific hard seed FN.
