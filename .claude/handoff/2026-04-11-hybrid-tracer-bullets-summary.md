# Hybrid Topology Tracer Bullets — Consolidated Summary

**Date**: 2026-04-11
**Plan**: `~/.claude/plans/hazy-wobbling-penguin.md`
**Branches**:
- `experiment/crocker-stack-reservoir` (Branch 3) — **PASS**
- `experiment/vineyard-binding-reservoir` (Branch 2) — **FAIL on strict criterion, partial PASS for arm A**
- `experiment/zigzag-reservoir` (Branch 4) — **FAIL**
- `experiment/tiered-sentinel-cardiac` (Branch 1) — **FAIL**

## TL;DR

Of four hybrid topological monitoring architectures from the 10-architecture document,
**one passes outright (Crocker stack)**, one **uncovers a substantial side-finding**
(dense windows alone, no smoothing or tracking, give F1=0.929 on the reservoir signal —
15× the existing baseline), and two are **architecturally killed** with informative
diagnostics. The document's premise — "hybrid acceleration helps ATT's known failure
modes" — is partially validated for the reservoir continuous-ramp problem and
empirically refuted for cardiac arrhythmia at the 10s window scale.

## Per-branch results

| # | Branch | Architecture | Domain | Best F1 | Verdict | Headline |
|---|---|---|---|---|---|---|
| **3** | crocker-stack-reservoir | Crocker stack process control | Reservoir SR drift | **0.988** | **PASS** | image CUSUM on H1 Betti curves; ALL three arms beat target |
| **2** | vineyard-binding-reservoir | Vineyard-tracked binding | Reservoir SR drift | **0.929 (A)** | **FAIL** strict | dense windows alone (arm A) hit F1=0.929; tracking *hurts* |
| **4** | zigzag-reservoir | Zigzag temporal | Reservoir SR drift | **n/a** | **FAIL** | every H1 feature has lifetime 0.5; no cross-regime tracking |
| **1** | tiered-sentinel-cardiac | Tiered sentinel | MIT-BIH cardiac | **0.250 (B)** | **FAIL** | persistence entropy is rhythm-class-invariant; topology gate kills everything |

Existing baselines being attacked:
- Reservoir continuous ramp: **F1 = 0.06** (per-window binding score, stride=200)
- Cardiac arrhythmia onset: **precision ≈ 20%** (negative mean lag — variance pre-empts)

## Key findings

### 1. Crocker stack passes decisively (Branch 3)

H1 feature count collapses from ~280 in healthy windows to ~30 in drift windows.
The Crocker stack preserves the per-epsilon Betti curves so this collapse is
trivially detected — even the dumbest arm (total Betti mass per column) gets
F1=0.91. Image CUSUM hits F1=0.988. **The 2D image-domain framing barely beats
1D column L1**, suggesting that for drift-style signals, the document's 2D framing
is overengineered — a column-vector L1 baseline is sufficient.

**Why ATT's existing F1=0.06 was so bad on this signal**: per-window binding
score integrates over the whole persistence image, washing out the H1 feature
*count* signal that the Crocker stack preserves directly.

### 2. Dense windows alone are 15× the baseline (Branch 2)

The Branch 2 vineyard-binding experiment was a strict FAIL — Hungarian-tracked
feature reconstruction made the binding score *worse*, not better. But the
control arm A (dense per-window binding score with stride=50 instead of the
existing experiment's stride=200) hit F1=0.929 with NO tracking and NO smoothing.
Savitzky-Golay smoothing on the dense scores added another +0.035 (arm B at
F1=0.964).

**The existing F1=0.06 reservoir baseline is pinned by stride=200, not by
per-window noise.** The sub-second binding score primitive works fine; the
existing experiment was just not running it often enough to catch the drift
trajectory. Stride=50 should be the new comparison baseline for any future
reservoir monitoring work.

### 3. Vineyard tracking hurts (Branch 2)

`F1(A=0.929) > F1(C=0.898)` with `F1(C) - F1(A) = -0.031`. Hungarian-L∞
matching of joint H1 features across windows + Savitzky-Golay smoothing of
the tracked persistence pairs + reconstructed PI: all of this *injects* noise
into a signal that's already very clean. The tracker creates 3631 mostly-short
tracks (mean length 15 windows out of 261), so smoothing barely smooths and
reconstruction loses information.

**Plan-agent's pessimism about real Dionysus vineyards is justified.** If the
dumb-vineyard proxy doesn't break even on this clean synthetic signal, the
boost-install yak-shave for real Dionysus is unlikely to recover the loss.

### 4. Zigzag fails because regime change expands the embedding (Branch 4)

Zigzag temporal persistence requires the underlying point cloud scale to be
*roughly constant* across the time axis, so that union complexes between
adjacent clouds are coherent. SR drift expands the embedding by ~30× (per-cloud
median pairwise distance: healthy ≈ 0.30, drift ≈ 10.5). A global VR-complex
threshold can never simultaneously serve both regimes:
- Low threshold (p30): healthy clouds form many edges, drift clouds form none.
- High threshold (p70): drift clouds form edges, but the union complexes between
  healthy and drift clouds are scale-incoherent.

**Every H1 feature found by zigzag, at any threshold, has lifetime 0.5** — i.e.
the feature exists at one zigzag time step and dies immediately. Zigzag is not
tracking anything.

Meanwhile, snapshot PH (the comparison baseline) at the same stride detects
**perfectly** at the exact onset cloud — total H1 persistence per cloud rises
monotonically 0.30 → 8.66 across the regime change. There is no architectural
benefit to zigzag for this signal class.

**Architectural lesson**: zigzag works on signals where the *topology changes
shape* without changing scale (e.g., LLM hidden states across layers — same
embedding dimension throughout). Drift signals where the state space expands
or contracts are not zigzag's natural regime.

### 5. Persistence entropy of cardiac windows is rhythm-class-invariant (Branch 1)

Empirically measured H1 persistence entropy on 60 normal + 43 arrhythmic
10-second ECG windows from MIT-BIH record 200:

```
baseline (normal): mean=3.311, std=0.363
arrhythmic:        mean=3.250, std=0.325
arrhythmic z-scores: mean=-0.17, n above +2 = 0/43, n below -2 = 2/43
```

**The distributions are statistically indistinguishable.** The Layer 2 entropy
gate in the tiered sentinel structurally kills every Layer 1 trigger, because
no arrhythmic window has entropy z > 2.

This **fully explains the existing cardiac experiment's negative mean detection
lag**: variance is the only discriminative source; topology converges to nothing
different. The original 20% precision is essentially noise, with whatever weak
correlation between topological features and arrhythmic windows comes from
amplitude transients, not rhythm structure.

**Architectural lesson**: tiered sentinels require the topology layer to be
**more selective** than variance for the events of interest. For cardiac
arrhythmia at 10s windows, persistence entropy is rhythm-invariant. A
beat-scale (~0.8s) window where the rhythm class is exposed in single-beat
embedding shape, or a different topological feature (max H1 lifetime, betti
curve shape, RR-interval topology), would be needed before tiered sentinel
has a chance.

## Decision matrix

| Direction | Status | Action |
|---|---|---|
| Crocker stack on reservoir | ✅ PASS | **Promote to branch-grade**: multi-seed (5-10), phase-randomized surrogates, all 3 degradation modes (`sr_drift`, `noise`, `connection_death`), ablations on window/stride. Should run in ~30 min total. Lift `build_crocker()` to `att/topology/crocker.py` if it replicates. |
| Dense binding on reservoir | ✅ partial | **Promote arm A specifically**: same multi-seed protocol, F1=0.929 must replicate. Existing reservoir code can adopt stride=50 directly. |
| Vineyard tracking | ❌ FAIL | **Killed**. Document the negative result; do not pursue real Dionysus vineyards for this signal class. The `FeatureTracker` class is reusable for any future use case where matching matters more (e.g. labelling features by ID across regimes), but should not be promoted to `att/` until then. |
| Zigzag temporal | ❌ FAIL | **Killed for expanding-scale signals**. One quick follow-up worth doing: run `--mode noise` on the same script. If `noise` mode also yields no cross-window features, the kill is complete. |
| Tiered sentinel cardiac | ❌ FAIL | **Killed at 10s window scale**. Two possible rescues worth considering before fully closing the architecture: (1) beat-scale ~0.8s windows; (2) different topological feature than entropy. Neither is in scope for tracer bullets. |

## What ATT should pick up immediately

1. **Update the reservoir monitoring docs** to flag that the F1=0.06 baseline is
   pinned by stride=200, not by topology — stride=50 + binding score gets to F1=0.929.
2. **The "rhythm-class-invariant entropy" finding** belongs in `RESEARCH.md` or
   the cardiac handoff note as a cautionary tale for future biosignal topology work.
3. **Crocker stack as a process-control primitive** is a publishable
   reservoir-monitoring result on its own (subject to multi-seed replication).
   It's also the first ATT result where total H1 mass / image structure beats
   the binding-score primitive on synthetic continuous drift.

## What ATT should NOT do

- Do not promote the inlined zigzag helper to `att/topology/` — it's a copy
  of `att/llm/zigzag.py` from the TDA-LLM branches; if zigzag becomes a
  cross-domain primitive in ATT, the LLM module should move to
  `att/topology/zigzag.py` independently.
- Do not lift `att/llm/crocker.py` to `att/topology/crocker.py` either —
  Branch 3's inlined `build_crocker()` is simpler and adequate for non-LLM
  signals. Merge them only when there's a graduated branch that *needs*
  cross-domain `CROCKERMatrix`.
- Do not pursue real Dionysus vineyards based on the Branch 2 result. The
  smoothing-only baseline (arm B at F1=0.964) is the new bar, and if
  Hungarian tracking can't break even with that baseline, no graph-tracked
  vineyard variant will.

## Operational summary

**Wallclock**:
- Branch 3: 31s
- Branch 2: 122s
- Branch 4: 14s (after threshold fix from 27s first-try)
- Branch 1: 66s (record 207) + 104s (record 200 retry) = 170s

Total: ~6 minutes of compute, ~5 hours of total agent time including planning,
verification, debugging, and writing handoffs.

**Branches created** (all forked from master, all committed locally only —
nothing pushed to origin):
- `experiment/crocker-stack-reservoir` (commit 7f08987)
- `experiment/vineyard-binding-reservoir` (commit da7cc60)
- `experiment/zigzag-reservoir` (commit b7e88a5)
- `experiment/tiered-sentinel-cardiac` (commit 41fb564)

Each branch has its own handoff doc at `.claude/handoff/2026-04-11-<branch>-tracer.md`
with full per-branch results, diagnostics, caveats, and decision notes. This summary
file is meant to sit alongside them as the cross-branch view.

## Next steps for the next agent

1. **Read this summary first** to know which branches passed and failed.
2. **Branch 3 graduation** is the highest-value follow-up: multi-seed Crocker
   stack on all 3 degradation modes + phase-randomized surrogates. Should be
   trivially scriptable from the existing tracer.
3. **Branch 2 arm A graduation** is the second-highest: same multi-seed
   protocol, F1=0.929 must replicate, then update the reservoir docs.
4. **Optional small experiments** documented per-branch:
   - Branch 4 with `--mode noise` to fully kill or partially rescue zigzag.
   - Branch 1 on record 100 (mostly normal) to confirm arm A behaviour with a clean baseline.
5. **Push branches to origin** — currently all 4 branches are local-only on this
   machine. The user's workflow appears to keep experiment branches as local
   records (cf. existing experiment branches), but if remote backup is desired,
   `git push -u origin experiment/<name>` per branch.

## One-line summary

> 1/4 PASS (Crocker stack F1=0.988), 1/4 partial PASS (dense binding F1=0.929
> from a control arm in a failed branch), 2/4 architecturally killed with
> informative diagnostics (zigzag needs scale-stable signals; persistence
> entropy is cardiac-rhythm-invariant). Net new finding worth promoting:
> the existing reservoir F1=0.06 baseline is just stride=200 — stride=50
> binding gets 0.929 with no other changes.
