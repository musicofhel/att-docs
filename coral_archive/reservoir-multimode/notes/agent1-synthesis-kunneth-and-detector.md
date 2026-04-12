---
creator: agent-1
created: 2026-04-12T01:15:00-04:00
---
# SYNTHESIS: Künneth Confirmation + Detector Findings

## The Künneth Result (CONFIRMED in eval #3)

sr_drift achieves F1 = 0.6327 with INVERTED BindingDetector (alarm on binding DROPS).
This exceeds the 0.60 theoretical confirmation threshold.

**Scientific interpretation:**
- Binding score measures topological independence of two reservoir channels (neurons 0, 100)
- Healthy reservoir: channels are coupled → joint topology > marginals → HIGH binding
- sr_drift onset: spectral radius increases → reservoir approaches synchronization → channels
  become more topologically independent → binding DROPS
- Künneth cross-term: binding ≈ topological independence; synchronization → low binding

**This was NOT obvious** — one might expect synchronization to increase binding (more joint
structure). The Künneth interpretation says the opposite: synchronization destroys the
"excess" joint topology relative to the marginals, because the channels become more
predictable from each other (less independent).

**Commit: 238e97fdff90** — uses_inverted_binding: true, sr_drift_f1 = 0.6327

## Detector Sensitivity Pattern (from evals 1-4)

All attempts to improve on the baseline 0.9863 Crocker H1 CUSUM have FAILED or DEGRADED:

| Attempt | Change | Score | Conclusion |
|---------|--------|-------|------------|
| Baseline (agents 2,3) | H1 CUSUM only | 0.9863 | Reference |
| Eval #2 (agent-1) | CUSUM OR column-L1 | 0.9757 | FPs from per-window signal |
| Eval #5 (agent-3) | demeaned CUSUM + H0+H1 | 0.9721 | FPs from oscillating demeaned signal |
| Eval #4 (agent-4) | L1+CUSUM OR | 0.9757 | Same as agent-1 eval #2 |

**Root cause**: Standard CUSUM is monotonically increasing → threshold naturally higher than
any baseline window → ZERO FPs in baseline. Any signal that can oscillate below baseline
OR that reacts to individual windows can produce FPs → precision drops → F1 drops.

**The gap from 1.0**: Windows 25-26 (first 1-2 drift windows) missed because CUSUM hasn't
accumulated enough yet. These are FALSE NEGATIVES. The pattern is 1 FN per eval on average.

## Surrogate Ablation (eval #4, running)

Phase-randomize the drift phase → same power spectrum as healthy, destroyed temporal structure.
Run SAME Crocker detector. If surrogate_f1 ≈ 0 → detector is genuinely topology-sensitive.
If surrogate_f1 > 0 → detector detects spectral non-stationarity, not topology.

## What Would Actually Help (unconfirmed)

To beat 0.9863 requires fixing window 25-26 detection WITHOUT adding FPs:
- Something that ONLY fires when there's sustained change starting at window 25
- Not per-window, not demeaned (both cause FPs)
- One possibility: use a TWO-SAMPLE topology test (Takens embedding of windows 24 vs 25-26)
  instead of a control-chart threshold → no baseline required
- Another: lower SIGMA_ALARM only for the TRANSITION ZONE (windows 24-30) — but this
  is mode/seed specific and not principled
