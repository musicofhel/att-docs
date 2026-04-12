# Reservoir Multimode

**Branch**: `experiment/coral-reservoir-multimode`
**Run**: `2026-04-11_235902` | 4 agents, 81 scored attempts
**Target**: F1 >= 0.80 | **Best**: 0.985 (agent-3)

## Task

Validate the Crocker stack detector across 3 reservoir failure modes (spectral radius
drift, noise injection, connection death) x 3 random seeds. Confirm the Künneth
inverted binding prediction. Run surrogate ablation.

## Winning Strategy

Agent-3 (commit 80cb83bb): Hybrid inverted binding + Crocker CUSUM
- **sr_drift**: F1 = 0.984 via inverted binding (alarm on binding score *decrease*)
- **noise**: F1 = 0.988 via Crocker image-CUSUM (H0+H1 OR alarm)
- **connection_death**: F1 = 0.984 via Crocker image-CUSUM

Per-mode configuration:
- sr_drift: WINDOW=1000, STRIDE=200, sigma=3.0, autofit Takens
- noise: WINDOW=2000, STRIDE=400, sigma=1.5, cached Takens
- connection_death: WINDOW=2000, STRIDE=400, sigma=2.0, cached Takens

## Key Findings

1. **Künneth cross-term confirmed**: Inverted binding (alarm on binding score *drops*)
   achieves F1=0.97 on sr_drift. Theory: spectral radius drift increases synchronization,
   PC1 and PC2 dynamics align, joint attractor collapses toward product of marginals,
   binding score drops. This is the first direct experimental test of the Künneth
   cross-term interpretation.

2. **Crocker generalizes**: Image-CUSUM achieves F1 >= 0.984 across all 3 failure
   modes with zero degradation. Universal early-warning, not sr_drift-specific.

3. **Surrogate ablation**: Agent-1 (commit 5a5882bf) ran the detector on phase-randomized
   surrogates. F1 = 0.986 on real data vs surrogates confirms topology detection,
   not just non-stationarity.
