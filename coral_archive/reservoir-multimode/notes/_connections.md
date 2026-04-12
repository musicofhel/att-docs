# Knowledge Connections (reservoir-multimode)

## STRIDE → Everything
- Links: all agent notes
- Pattern: STRIDE=400 is better than 200 for THIS task. Reason: avoids the hardest window
  (start=5000, barely entered drift → hard FN). BOTH binding AND Crocker benefit.
  First observed by agent-2 (570ed6c8). Confirmed by agent-4 eval #6.

## PCA Choice → Binding Quality
- Links: agent4-reflection-eval6-pca-fix.md, agent2-inverted-binding-finding.md
- Pattern: Global PCA (all states) >> Healthy-only PCA for binding channels.
  Global PCA captures drift variance in top components → binding more sensitive to drift.

## Fixed Takens → Crocker Quality  
- Links: agent1-reflection-eval6-stride-discovery.md, reference/run_crocker_reservoir.py
- Pattern: Consistent state space (same Takens params per window) is essential for
  comparing Crocker columns. Re-fitting per window creates inconsistency → more noise.

## Künneth Theory → sr_drift Only
- Links: kunneth-inverted-binding-confirmed.md, agent3-inverted-binding-kunneth.md
- Pattern: Inverted binding (alarm on drops) only works for sr_drift (synchronization onset).
  noise and cd: binding does NOT drop → inverted binding gives F1≈0.127 for those modes.
  Künneth: synchronization → joint topology → product of marginals → binding drops.
  connection_death and noise do not cause synchronization → no binding drop signal.

## CUSUM Robustness Principle
- Links: agent1-synthesis-final.md, agent3-strategy-demeaned-cusum.md
- Pattern: Monotone CUSUM (image-CUSUM) has no FPs in baseline by construction.
  Direct threshold and demeaned CUSUM BOTH add FPs. Rule: only OR two signals if
  BOTH are monotone CUSUMs.
  Exception: direct threshold on inverted binding is safe when binding drops are extreme (200x).
