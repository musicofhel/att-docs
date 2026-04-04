# Branch 7: Musical Topology — Genre Fingerprints — 2026-04-03

## Branch
`experiment/tda-music` (from `experiment/neuromorphic-snn` @ `6e60605`)

## What Was Done

Created `scripts/branches/music_genre_topology.py` — generates synthetic genre audio (jazz, classical, electronic, rock, ambient), extracts MFCCs and raw waveforms, applies Takens embedding + persistent homology to test whether genres have distinct topological fingerprints and whether structural boundaries correspond to topological transitions.

### Data

Synthetic audio generated via genre-specific signal models:
- 5 genres: jazz, classical, electronic, rock, ambient
- 15 tracks per genre, 30 seconds each, 22050 Hz
- Jazz: complex harmonics, walking bass, swing cymbal, improvised melody
- Classical: strings with vibrato, woodwind melody, timpani, evolving dynamics
- Electronic: 128 BPM kick/hihat, sawtooth bass, pad with filter sweep
- Rock: distorted power chords, kick/snare pattern, bass
- Ambient: slowly detuning drones, granular texture, reverb diffusion

### Config

Exp 1: MFCCs (13 coeff, hop=512) → MFCC1 as scalar → TakensEmbedder(delay=auto, dim=auto) → PH(max_dim=1, subsample=500). Permutation test: 200 perms, seed=42.
Exp 2: Structured track (intro-verse-chorus-verse-chorus) → MFCC1 → TransitionDetector(window=2s=86 frames, step=0.5s=21 frames, subsample=200, embedding τ=6, d=7).
Exp 3: Raw audio downsampled to 8kHz, truncated to 5s → Takens(delay=5, dim=7) → PH(max_dim=1, subsample=500).

### Three Experiments

| Exp | What | Key Result |
|-----|------|------------|
| 1 | Genre topology from MFCCs | p=0.000, z=11.58 — **highly significant** genre discrimination |
| 2 | Structural boundary detection | P=0.000, R=0.000 — detected transitions don't align with true boundaries |
| 3 | Raw waveform attractor complexity | jazz(419) > ambient(277) > rock(263) > electronic(237) > classical(234) |

### Key Findings

**Exp 1 — MFCC topology strongly discriminates genres (key positive result):**
- Ambient: H1=553±29, entropy=6.005±0.055 (highest — slowly evolving, high-dimensional attractor)
- Classical: H1=361±30, entropy=5.562±0.095 (rich harmonics, pitch variation)
- Rock: H1=198±6, entropy=4.498±0.105 (moderate — repetitive but harmonically dense)
- Jazz: H1=192±15, entropy=4.708±0.353 (similar to rock — complex but irregular)
- Electronic: H1=130±9, entropy=4.277±0.097 (lowest — repetitive, simple harmonics)
- Permutation test: observed Wasserstein=0.789, null=0.183±0.052, p=0.000, z=11.58
- **Interpretation**: The MFCC time series captures timbral evolution, which differs dramatically between genres. Ambient's slow modulation and granular texture creates the richest attractor; electronic's rigid repetition creates the simplest. The classical-ambient similarity (Wasserstein=0.051) reflects shared properties: both have slow dynamics, rich harmonics, and no strong beat.

**Exp 2 — Structural boundary detection fails on this synthetic track:**
- True boundaries: 4s, 12s, 18s (intro→verse, verse→chorus, chorus→verse)
- Detected: 9.5s, 22.7s — neither within 2s tolerance of any true boundary
- Precision=0.000, Recall=0.000
- **Interpretation**: The CUSUM-based changepoint detection on MFCC1 transition scores doesn't capture the structural boundaries in this synthetic track. The sections have different harmonic content but the MFCC1 (energy) contour doesn't show sharp enough transitions. Multi-MFCC or chromagram-based detection might perform better. The TransitionDetector was designed for continuous attractor transitions (e.g., EEG regime changes), not discrete musical section boundaries.

**Exp 3 — Raw waveform complexity: jazz leads, hypothesis partially confirmed:**
- Jazz: H1=419±16, entropy=5.700 (highest — complex harmonics + irregular rhythm)
- Ambient: H1=277±3, entropy=5.288
- Rock: H1=263±6, entropy=5.195
- Electronic: H1=237±5, entropy=5.034
- Classical: H1=234±11, entropy=5.054 (lowest — smooth, fewer transients)
- Hypothesis (jazz > classical > electronic): **partially holds** — jazz is highest, electronic is near-lowest, but classical is actually the lowest due to smooth legato and fewer transient events in the raw waveform
- **Interpretation**: In the raw time-domain signal, transient density and harmonic complexity drive H1 features. Jazz's swing rhythm, chromatic passing tones, and cymbal noise create the richest waveform attractor. Classical's smooth strings and gradual dynamics produce simpler raw topology despite rich harmonic structure (which is captured better by MFCCs).

### Interpretation

**Overall verdict: TDA effectively fingerprints musical genres via MFCC topology.**

1. **MFCC topology is a strong genre discriminator** (z=11.58) — timbral evolution patterns create genre-specific attractor shapes
2. **Structural boundary detection needs refinement** — CUSUM on single-MFCC transition scores misses discrete section changes
3. **Raw waveform topology captures transient density** — jazz's rhythmic complexity and harmonic richness produce the most complex attractor

The MFCC result (Exp 1) is the most informative: it demonstrates that persistent homology on MFCC-derived attractors can recover genre-level structural differences without any supervised learning. The genre distance matrix reveals natural clusters: ambient-classical (both slow, harmonic) vs electronic-rock (both rhythmic, repetitive) with jazz intermediate.

### Where This Approach Works Best

1. **Genre discrimination** — MFCC attractor topology produces highly significant genre fingerprints (z=11.58)
2. **Timbral complexity quantification** — H1 feature counts and persistence entropy rank genres by sonic complexity
3. **Genre similarity mapping** — Wasserstein distance matrix reveals natural genre clusters

### Where It Falls Short

1. **Structural boundary detection** — CUSUM on MFCC1 doesn't detect verse→chorus transitions
2. **Synthetic audio limitations** — real music has far more variability within genres
3. **Raw waveform interpretation** — raw H1 reflects transient density more than harmonic complexity

### Caveats

- Synthetic audio only — real GTZAN or other datasets would provide stronger validation
- 15 tracks per genre is adequate for permutation testing but limited for generalization claims
- Fixed embedding params (τ=5, d=7) for Exp 3 raw waveform may not be optimal for all genres
- Exp 2 uses a single synthetic track with idealized section boundaries
- MFCC hop_length=512 at 22050 Hz gives ~43 Hz temporal resolution for the MFCC time series
- Genre ordering may change with different synthesis parameters or real audio

## Files

- `scripts/branches/music_genre_topology.py` — full analysis script
- `data/music/results.json` — all numeric results
- `figures/music/` — 4 figures:
  - `overview.png` — 4-panel summary
  - `exp1_genre_mfcc_topology.png` — H1 features + entropy boxplots + Wasserstein matrix + permutation test
  - `exp2_structural_boundaries.png` — transition scores with section shading + boundary comparison
  - `exp3_raw_audio_complexity.png` — raw waveform H1 and entropy by genre

## Technical Notes

- Synthetic generators use deterministic seeds (42 + i*100 + hash(genre)%1000) for reproducibility
- MFCC extraction via librosa (n_mfcc=13, hop_length=512, sr=22050)
- Exp 3 raw audio: downsampled 22050→8000 Hz, truncated to 5s (40K samples) for computational feasibility
- Auto-embedding (Exp 1) typically estimates τ=3-14, d=4-10 depending on genre (ambient gets highest τ=13-14)
- TransitionDetector operates on MFCC frames (~43 Hz), not raw audio samples
- Total runtime: 85.6s

## Potential Follow-ups

- Use real GTZAN dataset or Creative Commons music for validation
- Multi-MFCC embedding (all 13 coefficients as multivariate time series) for richer attractor
- Chromagram-based topology for harmonic analysis (12-dimensional time series)
- Compare to standard MIR features (spectral centroid, zero-crossing rate, tempo)
- Use max_dim=2 for richer topological characterization
- Improve boundary detection with multi-scale scoring or self-similarity matrix approaches
- Tempo-normalized analysis to separate rhythmic from harmonic topology
- Cross-genre transfer: how well do topological features generalize?
