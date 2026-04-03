# Branch 2: Sleep Stage Topology — 2026-04-03

## Branch
`experiment/tda-sleep` (from `experiment/neuromorphic-snn` @ `60e67f2`)

## What Was Done

Created `scripts/branches/sleep_stage_topology.py` — applies ATT's sliding-window PH, transition detector, and binding detector to Sleep-EDF (PhysioNet) EEG data. Tests whether topological structure discriminates sleep stages and whether transitions in topology align with expert-annotated stage changes.

### Data

Sleep-EDF Expanded dataset (PhysioNet) via `mne.datasets.sleep_physionet.age.fetch_data`. 3 subjects (0, 1, 2), recording 1 each. Channels: Fpz-Cz (primary), Pz-Oz (secondary for binding). Preprocessing: 0.5–30 Hz bandpass, resample to 100 Hz. 30-second epochs aligned to hypnogram annotations.

Epoch distribution: W=5760, N1=261, N2=1357, N3=420, REM=458 (total 8256 across 3 subjects).

### Critical Fix: Voltage Scaling

MNE stores EEG in SI units (Volts), typically 10⁻⁵ to 10⁻⁴ V. Without scaling, all persistence features are microscopically small, producing Wasserstein distances that round to 0.00. **Must multiply by 1e6** (convert to μV) before Takens embedding. This was the main debugging issue in this session.

### Takens Embedding

Used literature fallback params from `att/neuro/eeg_params.py`: broadband at 100 Hz → delay=4, dimension=5 (Stam 2005). All experiments use these fixed params rather than auto-estimation (AMI/FNN on 180k+ samples was prohibitively slow).

### Three Experiments

| Exp | What | Result |
|-----|------|--------|
| 1 | Per-stage point cloud PH + permutation test | **z=20.1, p=0.01** — topology strongly discriminates stages |
| 2 | Sliding-window transition detection vs annotations | P=0.31, R=0.16, median lag=**-5s** |
| 3 | Cross-region binding (Fpz-Cz ↔ Pz-Oz): REM vs N3 | REM=16.4 (p=0.80), N3=7.7 (p=0.47) — not significant |

### Key Findings

**Exp 1 — Topology discriminates sleep stages:**
- Wasserstein-1 distances reveal clear structure: N1↔REM closest (287), N3↔REM most distant (3462)
- N1-REM proximity is physiologically expected — both involve desynchronized cortical activity
- N3-REM maximal distance reflects the deep-sleep/REM contrast (high-amplitude delta vs desynchronized)
- W↔N2 relatively close (1069) — light sleep topologically resembles wakefulness more than deep sleep
- H1 persistence entropy: REM highest (5.35), Wake lowest (4.99) — REM has richest loop structure
- Permutation test z=20.1, p<0.01 (100 permutations) — highly significant

**Exp 2 — Transition detection precedes annotations:**
- Median lag = -5s (negative = topology changes BEFORE the annotation)
- Precision 31%: about 1 in 3 detected changepoints corresponds to a real stage transition
- Recall 16%: most real transitions are missed (typical for broadband detection without stage-specific tuning)
- Subject 0 shows strongest leading signal (median lag -15s)

**Exp 3 — Cross-region binding not significant:**
- REM binding score (16.4) > N3 (7.7), direction consistent with hypothesis (REM has more coupling)
- But neither is significant vs phase-randomized surrogates (p=0.80, 0.47)
- Likely underpowered: only 20 epochs (10 min) per condition due to compute constraints
- Also: same embedding params for both channels may not capture channel-specific dynamics

### Caveats

- Sleep-EDF recordings start hours before sleep onset (subject 0: 8.5h of wake before N1). Exp 2 extracts 1h around sleep onset, not the full recording.
- Exp 1 pools epochs across all 3 subjects. Between-subject variability is not modeled.
- Exp 3 uses pre-configured embedders (no auto AMI/FNN) — necessary for speed but may miss optimal embedding for each channel/condition.
- Permutation test p=0.0099 with 100 perms means it's at the resolution floor (1/(100+1)). More perms would give a tighter estimate.
- Exp 2 step=15s (not 3s as originally specified) to keep runtime manageable (~240 windows/subject vs ~2400).
- H0 entropy is nearly constant across stages (5.94–6.10). All discrimination is in H1 (loop structure).

## Files

- `scripts/branches/sleep_stage_topology.py` — full analysis script
- `data/sleep/results.json` — all numeric results
- `figures/sleep/` — 8 figures:
  - `epoch_distribution.png` — epoch counts per stage
  - `entropy_by_stage.png` — H0 and H1 persistence entropy per stage
  - `wasserstein_matrix.png` — 5×5 pairwise Wasserstein-1 heatmap
  - `permutation_test.png` — null distribution with observed value
  - `transition_timeline_subj{0,1,2}.png` — transition scores + detected/annotated changepoints
  - `binding_rem_vs_n3.png` — REM vs N3 binding score comparison

## Technical Notes

- Data cached at `~/mne_data/physionet-sleep-data/` (3 subjects, ~150MB)
- Stage mapping: Sleep stage 4 merged into N3 (standard AASM convention)
- Permutation test computes PH at 200-point subsample (not 500) for speed; observed total recomputed at same subsample for fair comparison
- Transition detector: 30s window, 15s step, CUSUM changepoint detection
- Binding detector: `embedding_quality_gate=False` since we're using fixed params
- Exp 3 surrogate method: `phase_randomize` (AAFT, preserves power spectrum)

## Potential Follow-ups

- Increase Exp 2 step to 3s (as originally specified) with longer runtime budget — would improve temporal resolution
- Increase Exp 3 to 60+ epochs per condition for better power
- Band-specific analysis (delta for N3, theta-alpha for REM) instead of broadband
- Per-subject Exp 1 analysis to assess between-subject consistency
- Use auto-estimation with smaller data chunks for Exp 3 (embed 30s chunks, aggregate scores)
