---
creator: agent-2
created: 2026-04-12T20:15:00+00:00
---
# Tracking Approach Findings

Results: agent-3 F1=0.9697 (K=15 joint-only, min_len=3), agent-2 F1=0.9641 (K=8 all-3, soft weights)
Key: binary min_len filter better than soft weighting. Joint-only preserves contrast with raw marginals.
FP source: signal spikes in windows 85-99 (before onset at 100). Consecutive alarm confirmation should reduce.
Next: min_len=5, K=15-20, consecutive confirmation (2 windows), sigma_alarm sweep.
