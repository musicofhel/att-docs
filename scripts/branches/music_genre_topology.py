#!/usr/bin/env python3
"""Branch 7: Musical Topology — Genre Fingerprints from Audio Shape.

Applies Takens embedding + persistent homology to audio signals to test:
1. Whether genres have distinct topological fingerprints (via MFCCs)
2. Whether structural boundaries correspond to topological transitions
3. Whether raw audio attractor complexity varies by genre

Uses synthetic genre-like audio (jazz, classical, electronic, rock, ambient)
with librosa for feature extraction and ATT for TDA.
"""

import argparse
import functools
import json
import os
import sys
import warnings

import numpy as np

print = functools.partial(print, flush=True)

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TopologyDimensionalityWarning.*")
warnings.filterwarnings("ignore", message=".*n_components.*")

# ── Constants ──────────────────────────────────────────────────────────────
SR = 22050           # sample rate for feature extraction
SR_RAW = 8000        # sample rate for raw waveform embedding (Exp 3)
DURATION = 30        # seconds per track
N_MFCC = 13          # MFCC coefficients
HOP_LENGTH = 512     # librosa hop
N_TRACKS = 15        # tracks per genre
GENRES = ["jazz", "classical", "electronic", "rock", "ambient"]


# ══════════════════════════════════════════════════════════════════════════
# SYNTHETIC AUDIO GENERATORS
# ══════════════════════════════════════════════════════════════════════════

def synthetic_jazz(duration=DURATION, sr=SR, seed=42):
    """Complex harmonics, irregular rhythm, walking bass, swing feel."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr

    # Walking bass line with chromatic passing tones
    bass_freq = 110 + 20 * np.sin(2 * np.pi * 0.3 * t) + 10 * rng.standard_normal(len(t)).cumsum() * 0.0001
    bass = np.sin(2 * np.pi * np.cumsum(bass_freq) / sr) * 0.25

    # Complex chord voicings with extensions (7ths, 9ths, 13ths)
    chord_changes = np.sin(2 * np.pi * 0.067 * t)  # ~4 bar phrases at 120 bpm
    root = 440 * (1 + 0.12 * chord_changes)
    signal = np.sin(2 * np.pi * np.cumsum(root) / sr) * 0.2          # root
    signal += np.sin(2 * np.pi * np.cumsum(root * 1.26) / sr) * 0.15  # maj 3rd
    signal += np.sin(2 * np.pi * np.cumsum(root * 1.498) / sr) * 0.15 # 5th
    signal += np.sin(2 * np.pi * np.cumsum(root * 1.782) / sr) * 0.12 # 7th
    signal += np.sin(2 * np.pi * np.cumsum(root * 2.245) / sr) * 0.08 # 9th

    # Swing rhythm cymbal pattern
    beat_period = sr * 60 / 140  # 140 bpm
    cymbal_env = np.zeros(len(t))
    for i in range(int(duration * 140 / 60)):
        pos = int(i * beat_period)
        swing_offset = int(beat_period * 0.66) if i % 2 == 1 else 0
        idx = pos + swing_offset
        if idx < len(t):
            end = min(idx + int(0.05 * sr), len(t))
            cymbal_env[idx:end] = 0.8 * np.exp(-np.arange(end - idx) / (0.01 * sr))
    cymbal = cymbal_env * rng.standard_normal(len(t)) * 0.15

    # Improvised melody line with chromatic runs
    melody_freq = 880 + 200 * np.sin(2 * np.pi * 1.5 * t) + 100 * np.sin(2 * np.pi * 0.7 * t)
    melody_env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t)) * (rng.random(len(t)) > 0.7).astype(float)
    melody_env = np.convolve(melody_env, np.ones(int(0.05 * sr)) / int(0.05 * sr), mode='same')
    melody = np.sin(2 * np.pi * np.cumsum(melody_freq) / sr) * melody_env * 0.18

    signal = signal + bass + cymbal + melody
    signal += 0.02 * rng.standard_normal(len(t))
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal, sr


def synthetic_classical(duration=DURATION, sr=SR, seed=42):
    """Evolving dynamics, smooth pitch variation, orchestral texture."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr

    # Slow tempo dynamics (crescendo / decrescendo)
    dynamics = 0.3 + 0.2 * np.sin(2 * np.pi * 0.033 * t)  # ~30s phrase

    # String section — smooth legato with vibrato
    vib_rate = 5.5 + 0.5 * rng.standard_normal(1)[0]
    vib_depth = 3 + rng.random(1)[0]
    base_freq = 440 + 200 * np.sin(2 * np.pi * 0.1 * t) + 100 * np.sin(2 * np.pi * 0.033 * t)
    freq = base_freq + vib_depth * np.sin(2 * np.pi * vib_rate * t)
    strings = np.sin(2 * np.pi * np.cumsum(freq) / sr) * dynamics
    # Add harmonics for richness
    strings += 0.5 * np.sin(2 * np.pi * np.cumsum(freq * 2) / sr) * dynamics
    strings += 0.25 * np.sin(2 * np.pi * np.cumsum(freq * 3) / sr) * dynamics
    strings *= 0.3

    # Woodwind melody — independent pitch contour
    ww_freq = 880 + 300 * np.sin(2 * np.pi * 0.15 * t) + 50 * np.sin(2 * np.pi * 0.5 * t)
    ww_env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
    woodwind = np.sin(2 * np.pi * np.cumsum(ww_freq) / sr) * ww_env * dynamics * 0.2

    # Timpani rolls at phrase boundaries
    timp_env = np.zeros(len(t))
    for i in range(0, int(duration), 8):
        idx = i * sr
        if idx < len(t):
            end = min(idx + int(1.5 * sr), len(t))
            roll = np.sin(2 * np.pi * 80 * t[:end - idx]) * np.exp(-np.arange(end - idx) / (0.5 * sr))
            timp_env[idx:end] = roll * 0.15

    signal = strings + woodwind + timp_env
    signal += 0.01 * rng.standard_normal(len(t))
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal, sr


def synthetic_electronic(duration=DURATION, sr=SR, seed=42):
    """Repetitive beats, simple harmonics, steady tempo, synthesizer."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr

    bpm = 128
    beat_period = 60.0 / bpm
    beat_samples = int(beat_period * sr)

    # Kick drum — 4/4 pattern
    kick_env = np.zeros(len(t))
    for i in range(int(duration / beat_period)):
        idx = int(i * beat_samples)
        if idx < len(t):
            end = min(idx + int(0.15 * sr), len(t))
            kick_freq = 150 * np.exp(-np.arange(end - idx) / (0.03 * sr))
            kick_env[idx:end] = np.sin(2 * np.pi * np.cumsum(kick_freq) / sr) * np.exp(-np.arange(end - idx) / (0.05 * sr))
    kick = kick_env * 0.4

    # Hi-hat — 8th notes
    hihat_env = np.zeros(len(t))
    for i in range(int(duration / beat_period * 2)):
        idx = int(i * beat_samples / 2)
        if idx < len(t):
            end = min(idx + int(0.03 * sr), len(t))
            hihat_env[idx:end] = np.exp(-np.arange(end - idx) / (0.01 * sr))
    hihat = hihat_env * rng.standard_normal(len(t)) * 0.1

    # Bass synth — simple sawtooth approximation
    bass = np.zeros(len(t))
    for harmonic in range(1, 8):
        bass += np.sin(2 * np.pi * 55 * harmonic * t) * (0.3 / harmonic)
    bass *= 0.5 * (1 + np.sign(np.sin(2 * np.pi * (bpm / 60) * t))) / 2  # gate to beat

    # Pad synth — sustained chord
    pad = np.sin(2 * np.pi * 220 * t) * 0.15
    pad += np.sin(2 * np.pi * 277 * t) * 0.12
    pad += np.sin(2 * np.pi * 330 * t) * 0.12

    # Filter sweep
    sweep = 0.5 + 0.5 * np.sin(2 * np.pi * 0.067 * t)  # 15s cycle
    pad *= sweep

    signal = kick + hihat + bass + pad
    signal += 0.01 * rng.standard_normal(len(t))
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal, sr


def synthetic_rock(duration=DURATION, sr=SR, seed=42):
    """Power chords, driving drums, distortion-like harmonics."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr

    bpm = 120
    beat_period = 60.0 / bpm
    beat_samples = int(beat_period * sr)

    # Distorted power chords — root + 5th with harmonic saturation
    chord_root = 146.83  # D3
    guitar = np.zeros(len(t))
    for h in range(1, 12):
        guitar += np.sin(2 * np.pi * chord_root * h * t) * (0.5 / h)
        guitar += np.sin(2 * np.pi * chord_root * 1.498 * h * t) * (0.4 / h)
    # Soft clipping for distortion
    guitar = np.tanh(3 * guitar) * 0.3

    # Chord changes every 2 bars
    chord_env = np.ones(len(t))
    changes = [0, 4, 8, 12, 16, 20, 24, 28]
    freqs = [146.83, 196.0, 164.81, 130.81, 146.83, 196.0, 164.81, 130.81]
    for i, (bar, freq) in enumerate(zip(changes, freqs)):
        start = int(bar * beat_period * sr * 4 / len(changes))
        end = min(int((bar + 4) * beat_period * sr * 4 / len(changes)), len(t))
        ratio = freq / 146.83
        if start < len(t):
            seg_t = t[start:end] - t[start]
            seg = np.zeros(end - start)
            for h in range(1, 12):
                seg += np.sin(2 * np.pi * freq * h * seg_t) * (0.5 / h)
                seg += np.sin(2 * np.pi * freq * 1.498 * h * seg_t) * (0.4 / h)
            guitar[start:end] = np.tanh(3 * seg) * 0.3

    # Drums — kick on 1,3 + snare on 2,4
    drums = np.zeros(len(t))
    for i in range(int(duration / beat_period)):
        idx = int(i * beat_samples)
        if idx < len(t):
            end = min(idx + int(0.1 * sr), len(t))
            if i % 2 == 0:  # kick
                kf = 100 * np.exp(-np.arange(end - idx) / (0.02 * sr))
                drums[idx:end] += np.sin(2 * np.pi * np.cumsum(kf) / sr) * np.exp(-np.arange(end - idx) / (0.04 * sr)) * 0.35
            else:  # snare
                drums[idx:end] += rng.standard_normal(end - idx) * np.exp(-np.arange(end - idx) / (0.03 * sr)) * 0.25

    # Bass follows root
    bass = np.sin(2 * np.pi * 73.42 * t) * 0.2  # D2

    signal = guitar + drums + bass
    signal += 0.015 * rng.standard_normal(len(t))
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal, sr


def synthetic_ambient(duration=DURATION, sr=SR, seed=42):
    """Slowly evolving drones, minimal rhythm, pad textures."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr

    # Drone — slowly detuning unison
    drone1 = np.sin(2 * np.pi * 220 * t) * 0.2
    drone2 = np.sin(2 * np.pi * (220 + 0.5 * np.sin(2 * np.pi * 0.02 * t)) * t) * 0.2
    drone3 = np.sin(2 * np.pi * (220 * 1.5 + 0.3 * np.sin(2 * np.pi * 0.015 * t)) * t) * 0.15

    # Evolving pad with very slow modulation
    pad_freq = 330 + 30 * np.sin(2 * np.pi * 0.01 * t)
    pad = np.sin(2 * np.pi * np.cumsum(pad_freq) / sr) * 0.15
    pad *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.025 * t)

    # Granular texture — random short grains
    grains = np.zeros(len(t))
    n_grains = 100
    grain_dur = int(0.1 * sr)
    for _ in range(n_grains):
        pos = rng.integers(0, len(t) - grain_dur)
        freq = rng.uniform(200, 800)
        amp = rng.uniform(0.02, 0.08)
        grain_t = np.arange(grain_dur) / sr
        grain = np.sin(2 * np.pi * freq * grain_t) * np.hanning(grain_dur) * amp
        grains[pos:pos + grain_dur] += grain

    # Reverb-like diffusion (simple convolution)
    reverb_ir = np.exp(-np.arange(int(0.5 * sr)) / (0.15 * sr))
    reverb_ir = reverb_ir / np.sum(reverb_ir)
    grains_rev = np.convolve(grains, reverb_ir, mode='same')

    signal = drone1 + drone2 + drone3 + pad + grains_rev
    signal += 0.005 * rng.standard_normal(len(t))
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal, sr


GENRE_GENERATORS = {
    "jazz": synthetic_jazz,
    "classical": synthetic_classical,
    "electronic": synthetic_electronic,
    "rock": synthetic_rock,
    "ambient": synthetic_ambient,
}


def generate_tracks(n_tracks=N_TRACKS, seed=42):
    """Generate multiple tracks per genre with varying seeds."""
    tracks = {}
    for genre in GENRES:
        tracks[genre] = []
        gen_func = GENRE_GENERATORS[genre]
        for i in range(n_tracks):
            sig, sr_ = gen_func(seed=seed + i * 100 + hash(genre) % 1000)
            tracks[genre].append({"signal": sig, "sr": sr_, "genre": genre, "track_id": i})
    return tracks


# ══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

def extract_mfcc(signal, sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH):
    """Extract MFCCs from audio signal."""
    import librosa
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfccs  # (n_mfcc, n_frames)


def extract_chroma(signal, sr, hop_length=HOP_LENGTH):
    """Extract chromagram from audio signal."""
    import librosa
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, hop_length=hop_length)
    return chroma  # (12, n_frames)


# ══════════════════════════════════════════════════════════════════════════
# TDA HELPERS
# ══════════════════════════════════════════════════════════════════════════

def embed_and_ph(signal, delay="auto", dimension="auto", max_dim=1,
                 subsample=None, seed=42, fallback_delay=8, fallback_dim=5):
    """Takens embed + PH with fallback for short/problematic signals."""
    from att.embedding.takens import TakensEmbedder
    from att.topology.persistence import PersistenceAnalyzer

    embedder = TakensEmbedder(delay=delay, dimension=dimension)
    used_fallback = False

    try:
        embedder.fit(signal)
    except Exception:
        embedder = TakensEmbedder(delay=fallback_delay, dimension=fallback_dim)
        embedder.fit(signal)
        used_fallback = True

    if embedder.delay_ is None or embedder.delay_ < 1:
        embedder.delay_ = fallback_delay
        used_fallback = True
    if embedder.dimension_ is None or embedder.dimension_ < 2:
        embedder.dimension_ = fallback_dim
        used_fallback = True

    cloud = embedder.transform(signal)
    pa = PersistenceAnalyzer(max_dim=max_dim)
    result = pa.fit_transform(cloud, subsample=subsample, seed=seed)
    return cloud, result, embedder, used_fallback


def count_features(diagrams, dim):
    """Count finite-lifetime features in dimension dim."""
    if dim >= len(diagrams) or len(diagrams[dim]) == 0:
        return 0
    dgm = diagrams[dim]
    finite = dgm[:, 1] < np.inf
    return int(np.sum(finite))


def get_lifetimes(diagrams, dim):
    """Get finite lifetimes for dimension dim."""
    if dim >= len(diagrams) or len(diagrams[dim]) == 0:
        return np.array([])
    dgm = diagrams[dim]
    finite = dgm[:, 1] < np.inf
    if not np.any(finite):
        return np.array([])
    return dgm[finite, 1] - dgm[finite, 0]


def wasserstein_1d(dgm1, dgm2, dim=1):
    """1D Wasserstein distance on lifetimes."""
    from scipy.stats import wasserstein_distance
    l1 = get_lifetimes(dgm1, dim)
    l2 = get_lifetimes(dgm2, dim)
    if len(l1) == 0 and len(l2) == 0:
        return 0.0
    if len(l1) == 0:
        l1 = np.array([0.0])
    if len(l2) == 0:
        l2 = np.array([0.0])
    return wasserstein_distance(l1, l2)


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Genre topology from MFCCs
# ══════════════════════════════════════════════════════════════════════════

def run_exp1(tracks, subsample=500, n_perms=200, seed=42):
    """Genre discrimination via MFCC-derived topological features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Genre Topology from MFCCs")
    print("=" * 70)

    genre_results = {}

    for genre in GENRES:
        print(f"\n  Processing {genre} ({len(tracks[genre])} tracks)...")
        h1_counts = []
        h1_entropies = []
        all_lifetimes = []

        for track in tracks[genre]:
            # Extract MFCC1 (energy contour) as scalar time series
            mfccs = extract_mfcc(track["signal"], track["sr"])
            mfcc1 = mfccs[0]  # first coefficient = energy

            try:
                cloud, ph, emb, fallback = embed_and_ph(
                    mfcc1, delay="auto", dimension="auto",
                    max_dim=1, subsample=subsample, seed=seed
                )
                h1 = count_features(ph["diagrams"], 1)
                ent = ph["persistence_entropy"][1] if len(ph["persistence_entropy"]) > 1 else 0.0
                lifetimes = get_lifetimes(ph["diagrams"], 1)

                h1_counts.append(h1)
                h1_entropies.append(ent)
                all_lifetimes.append(lifetimes)

                if track["track_id"] == 0:
                    print(f"    Track 0: τ={emb.delay_}, d={emb.dimension_}, "
                          f"cloud={cloud.shape}, H1={h1}, entropy={ent:.3f}"
                          f"{' (fallback)' if fallback else ''}")
            except Exception as e:
                print(f"    Track {track['track_id']} FAILED: {e}")
                h1_counts.append(0)
                h1_entropies.append(0.0)
                all_lifetimes.append(np.array([]))

        genre_results[genre] = {
            "h1_counts": h1_counts,
            "h1_entropies": h1_entropies,
            "h1_mean": float(np.mean(h1_counts)),
            "h1_std": float(np.std(h1_counts)),
            "entropy_mean": float(np.mean(h1_entropies)),
            "entropy_std": float(np.std(h1_entropies)),
            "all_lifetimes": all_lifetimes,
        }
        print(f"    → H1={np.mean(h1_counts):.1f}±{np.std(h1_counts):.1f}, "
              f"entropy={np.mean(h1_entropies):.3f}±{np.std(h1_entropies):.3f}")

    # Pairwise Wasserstein distance matrix between genres
    print("\n  Computing pairwise Wasserstein distances...")
    n_genres = len(GENRES)
    wass_matrix = np.zeros((n_genres, n_genres))

    for i, g1 in enumerate(GENRES):
        for j, g2 in enumerate(GENRES):
            if i < j:
                # Pool lifetimes across tracks for each genre
                pool1 = np.concatenate([l for l in genre_results[g1]["all_lifetimes"] if len(l) > 0])
                pool2 = np.concatenate([l for l in genre_results[g2]["all_lifetimes"] if len(l) > 0])
                if len(pool1) > 0 and len(pool2) > 0:
                    from scipy.stats import wasserstein_distance
                    d = wasserstein_distance(pool1, pool2)
                else:
                    d = 0.0
                wass_matrix[i, j] = d
                wass_matrix[j, i] = d

    print("\n  Wasserstein distance matrix:")
    print(f"  {'':>12}", end="")
    for g in GENRES:
        print(f" {g:>12}", end="")
    print()
    for i, g1 in enumerate(GENRES):
        print(f"  {g1:>12}", end="")
        for j in range(n_genres):
            print(f" {wass_matrix[i, j]:>12.4f}", end="")
        print()

    # Permutation test: are genre labels informative?
    print(f"\n  Running permutation test ({n_perms} permutations)...")
    rng = np.random.default_rng(seed)

    # Observed: mean off-diagonal Wasserstein
    obs_stat = np.mean(wass_matrix[np.triu_indices(n_genres, k=1)])

    # Pool all lifetimes with genre labels
    all_pools = []
    all_labels = []
    for gi, genre in enumerate(GENRES):
        for lifetimes in genre_results[genre]["all_lifetimes"]:
            all_pools.append(lifetimes)
            all_labels.append(gi)

    null_stats = []
    for perm in range(n_perms):
        perm_labels = rng.permutation(all_labels)
        # Recompute pooled lifetimes per shuffled genre
        perm_pools = {g: [] for g in range(n_genres)}
        for li, label in enumerate(perm_labels):
            perm_pools[label].append(all_pools[li])

        perm_matrix = np.zeros((n_genres, n_genres))
        for i in range(n_genres):
            for j in range(i + 1, n_genres):
                p1 = np.concatenate([l for l in perm_pools[i] if len(l) > 0]) if any(len(l) > 0 for l in perm_pools[i]) else np.array([0.0])
                p2 = np.concatenate([l for l in perm_pools[j] if len(l) > 0]) if any(len(l) > 0 for l in perm_pools[j]) else np.array([0.0])
                from scipy.stats import wasserstein_distance
                perm_matrix[i, j] = wasserstein_distance(p1, p2)

        null_stats.append(np.mean(perm_matrix[np.triu_indices(n_genres, k=1)]))

    null_stats = np.array(null_stats)
    p_value = float(np.mean(null_stats >= obs_stat))
    z_score = float((obs_stat - np.mean(null_stats)) / (np.std(null_stats) + 1e-10))

    print(f"  Observed mean Wasserstein = {obs_stat:.6f}")
    print(f"  Null mean = {np.mean(null_stats):.6f} ± {np.std(null_stats):.6f}")
    print(f"  p = {p_value:.4f}, z = {z_score:.2f}")

    # Genre ordering by H1 complexity
    ordering = sorted(GENRES, key=lambda g: genre_results[g]["h1_mean"], reverse=True)
    ordering_str = " > ".join([f"{g}({genre_results[g]['h1_mean']:.0f})" for g in ordering])
    print(f"\n  Genre ordering by H1: {ordering_str}")

    return {
        "genre_results": genre_results,
        "wass_matrix": wass_matrix,
        "p_value": p_value,
        "z_score": z_score,
        "obs_stat": obs_stat,
        "null_stats": null_stats,
        "ordering": ordering_str,
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Structural boundary detection
# ══════════════════════════════════════════════════════════════════════════

def create_structured_track(sr=SR, seed=42):
    """Create a track with known structural boundaries.

    Structure: intro(4s) - verse(8s) - chorus(6s) - verse(8s) - chorus(4s)
    Total: 30s
    """
    rng = np.random.default_rng(seed)
    t_total = np.arange(int(30 * sr)) / sr

    sections = []
    boundaries = []
    current = 0

    # Intro: sparse, low energy
    dur = 4
    t = np.arange(int(dur * sr)) / sr
    intro = np.sin(2 * np.pi * 220 * t) * 0.1 + 0.02 * rng.standard_normal(len(t))
    sections.append(("intro", intro))
    current += dur
    boundaries.append(current)

    # Verse 1: medium energy, melodic
    dur = 8
    t = np.arange(int(dur * sr)) / sr
    verse = np.sin(2 * np.pi * 330 * t) * 0.3
    verse += np.sin(2 * np.pi * 440 * t * (1 + 0.02 * np.sin(2 * np.pi * 2 * t))) * 0.2
    verse += np.sin(2 * np.pi * 110 * t) * 0.15  # bass
    verse += 0.03 * rng.standard_normal(len(t))
    sections.append(("verse", verse))
    current += dur
    boundaries.append(current)

    # Chorus 1: high energy, dense harmonics, drums
    dur = 6
    t = np.arange(int(dur * sr)) / sr
    chorus = np.zeros(len(t))
    for h in range(1, 8):
        chorus += np.sin(2 * np.pi * 440 * h * t) * (0.4 / h)
    # Add beat
    beat_period = int(sr * 60 / 130)
    for i in range(int(dur * 130 / 60)):
        idx = i * beat_period
        if idx < len(t):
            end = min(idx + int(0.05 * sr), len(t))
            chorus[idx:end] += 0.5 * np.exp(-np.arange(end - idx) / (0.01 * sr))
    chorus += 0.05 * rng.standard_normal(len(t))
    sections.append(("chorus", chorus))
    current += dur
    boundaries.append(current)

    # Verse 2: similar to verse 1 but different melody
    dur = 8
    t = np.arange(int(dur * sr)) / sr
    verse2 = np.sin(2 * np.pi * 370 * t) * 0.3
    verse2 += np.sin(2 * np.pi * 494 * t * (1 + 0.015 * np.sin(2 * np.pi * 1.5 * t))) * 0.2
    verse2 += np.sin(2 * np.pi * 110 * t) * 0.15
    verse2 += 0.03 * rng.standard_normal(len(t))
    sections.append(("verse", verse2))
    current += dur
    boundaries.append(current)

    # Chorus 2 (final)
    dur = 4
    t = np.arange(int(dur * sr)) / sr
    chorus2 = np.zeros(len(t))
    for h in range(1, 8):
        chorus2 += np.sin(2 * np.pi * 440 * h * t) * (0.4 / h)
    beat_period = int(sr * 60 / 130)
    for i in range(int(dur * 130 / 60)):
        idx = i * beat_period
        if idx < len(t):
            end = min(idx + int(0.05 * sr), len(t))
            chorus2[idx:end] += 0.5 * np.exp(-np.arange(end - idx) / (0.01 * sr))
    chorus2 += 0.05 * rng.standard_normal(len(t))
    sections.append(("chorus", chorus2))

    # Concatenate
    signal = np.concatenate([s[1] for s in sections])
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    boundary_times = boundaries[:-1]  # exclude end

    section_labels = [s[0] for s in sections]
    print(f"    Structure: {' → '.join(section_labels)}")
    print(f"    Boundaries at: {boundary_times} seconds")

    return signal, sr, boundary_times, section_labels


def run_exp2(subsample=200, seed=42):
    """Structural boundary detection via TransitionDetector on MFCCs."""
    from att.transitions.detector import TransitionDetector

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Structural Boundary Detection")
    print("=" * 70)

    signal, sr_, boundary_times, section_labels = create_structured_track(sr=SR, seed=seed)

    # Extract MFCC1 for transition detection
    mfccs = extract_mfcc(signal, sr_)
    mfcc1 = mfccs[0]
    mfcc_sr = sr_ / HOP_LENGTH  # effective sample rate of MFCC frames

    print(f"    MFCC1 length: {len(mfcc1)} frames ({len(mfcc1) / mfcc_sr:.1f}s)")
    print(f"    MFCC sample rate: {mfcc_sr:.1f} Hz")

    # Estimate embedding parameters on the full MFCC1
    from att.embedding.takens import TakensEmbedder
    emb = TakensEmbedder(delay="auto", dimension="auto")
    try:
        emb.fit(mfcc1)
        e_delay = max(emb.delay_, 1)
        e_dim = max(emb.dimension_, 3)
    except Exception:
        e_delay = 3
        e_dim = 5

    print(f"    Embedding: τ={e_delay}, d={e_dim}")

    # TransitionDetector: window=2s of MFCC frames, step=0.5s
    window_frames = max(int(2.0 * mfcc_sr), 20)
    step_frames = max(int(0.5 * mfcc_sr), 5)

    print(f"    Window: {window_frames} frames ({window_frames / mfcc_sr:.1f}s), "
          f"Step: {step_frames} frames ({step_frames / mfcc_sr:.1f}s)")

    td = TransitionDetector(
        window_size=window_frames,
        step_size=step_frames,
        max_dim=1,
        subsample=subsample,
    )

    result = td.fit_transform(mfcc1, seed=seed, embedding_dim=e_dim, embedding_delay=e_delay)

    scores = result["transition_scores"]
    centers = result["window_centers"]

    # Convert window centers to time
    if len(scores) < len(centers):
        score_times = ((centers[:-1] + centers[1:]) / 2.0) / mfcc_sr
    else:
        score_times = centers / mfcc_sr

    print(f"    Transition scores: {len(scores)} values")

    # Detect changepoints
    changepoints = td.detect_changepoints(method="cusum")
    detected_times = [float(score_times[cp]) for cp in changepoints if cp < len(score_times)]

    print(f"    Detected transitions at: {[f'{t:.1f}s' for t in detected_times]}")
    print(f"    True boundaries at: {boundary_times}")

    # Compute precision/recall with 2s tolerance
    tolerance = 2.0
    tp = 0
    matched_true = set()
    matched_det = set()

    for di, dt in enumerate(detected_times):
        for bi, bt in enumerate(boundary_times):
            if abs(dt - bt) <= tolerance and bi not in matched_true:
                tp += 1
                matched_true.add(bi)
                matched_det.add(di)
                break

    precision = tp / max(len(detected_times), 1)
    recall = tp / max(len(boundary_times), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    print(f"    TP={tp}, FP={len(detected_times) - tp}, FN={len(boundary_times) - tp}")
    print(f"    Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    # Also use librosa onset detection as reference
    import librosa
    onset_frames = librosa.onset.onset_detect(y=signal, sr=SR, hop_length=HOP_LENGTH)
    onset_times = librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP_LENGTH)
    print(f"    Librosa detected {len(onset_frames)} onsets")

    return {
        "scores": scores,
        "score_times": score_times,
        "boundary_times": boundary_times,
        "section_labels": section_labels,
        "detected_times": detected_times,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "onset_times": onset_times,
        "embedding": {"delay": e_delay, "dim": e_dim},
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Raw audio attractor complexity by genre
# ══════════════════════════════════════════════════════════════════════════

def run_exp3(tracks, subsample=500, seed=42):
    """Compare raw waveform attractor topology across genres."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Raw Audio Attractor Complexity by Genre")
    print("=" * 70)

    import librosa

    genre_results = {}

    for genre in GENRES:
        print(f"\n  Processing {genre}...")
        h1_counts = []
        h1_entropies = []

        for track in tracks[genre]:
            # Downsample to 8kHz and truncate to 5s for computational feasibility
            signal_8k = librosa.resample(track["signal"], orig_sr=track["sr"], target_sr=SR_RAW)
            signal_8k = signal_8k[:5 * SR_RAW]  # 5s = 40000 samples

            try:
                # Use fixed embedding params to avoid slow auto-estimation on raw audio
                cloud, ph, emb, fallback = embed_and_ph(
                    signal_8k, delay=5, dimension=7,
                    max_dim=1, subsample=subsample, seed=seed
                )
                h1 = count_features(ph["diagrams"], 1)
                ent = ph["persistence_entropy"][1] if len(ph["persistence_entropy"]) > 1 else 0.0
                h1_counts.append(h1)
                h1_entropies.append(ent)

                if track["track_id"] == 0:
                    print(f"    Track 0: τ={emb.delay_}, d={emb.dimension_}, "
                          f"cloud={cloud.shape}, H1={h1}, entropy={ent:.3f}"
                          f"{' (fallback)' if fallback else ''}")
            except Exception as e:
                print(f"    Track {track['track_id']} FAILED: {e}")
                h1_counts.append(0)
                h1_entropies.append(0.0)

        genre_results[genre] = {
            "h1_counts": h1_counts,
            "h1_entropies": h1_entropies,
            "h1_mean": float(np.mean(h1_counts)),
            "h1_std": float(np.std(h1_counts)),
            "entropy_mean": float(np.mean(h1_entropies)),
            "entropy_std": float(np.std(h1_entropies)),
        }
        print(f"    → H1={np.mean(h1_counts):.1f}±{np.std(h1_counts):.1f}, "
              f"entropy={np.mean(h1_entropies):.3f}±{np.std(h1_entropies):.3f}")

    # Genre ordering
    ordering = sorted(GENRES, key=lambda g: genre_results[g]["h1_mean"], reverse=True)
    ordering_str = " > ".join([f"{g}({genre_results[g]['h1_mean']:.0f})" for g in ordering])
    print(f"\n  Genre ordering by raw H1: {ordering_str}")

    # Hypothesis check: jazz > classical > electronic
    jazz_h1 = genre_results["jazz"]["h1_mean"]
    classical_h1 = genre_results["classical"]["h1_mean"]
    electronic_h1 = genre_results["electronic"]["h1_mean"]
    print(f"\n  Hypothesis (jazz > classical > electronic):")
    print(f"    jazz={jazz_h1:.1f}, classical={classical_h1:.1f}, electronic={electronic_h1:.1f}")
    hypothesis_holds = jazz_h1 > classical_h1 > electronic_h1
    print(f"    Hypothesis holds: {hypothesis_holds}")

    return {
        "genre_results": genre_results,
        "ordering": ordering_str,
        "hypothesis_holds": hypothesis_holds,
    }


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def plot_exp1(exp1_results, output_dir):
    """Plot Experiment 1: Genre MFCC topology comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment 1: Genre Topology from MFCCs", fontsize=14, fontweight="bold")

    genres = GENRES
    gr = exp1_results["genre_results"]

    # Panel A: H1 features by genre
    ax = axes[0, 0]
    data = [gr[g]["h1_counts"] for g in genres]
    bp = ax.boxplot(data, tick_labels=genres, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(genres)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("H1 Features")
    ax.set_title("A) H1 Feature Count by Genre")
    ax.grid(True, alpha=0.3)

    # Panel B: Persistence entropy by genre
    ax = axes[0, 1]
    data = [gr[g]["h1_entropies"] for g in genres]
    bp = ax.boxplot(data, tick_labels=genres, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("H1 Persistence Entropy")
    ax.set_title("B) Persistence Entropy by Genre")
    ax.grid(True, alpha=0.3)

    # Panel C: Wasserstein distance matrix
    ax = axes[1, 0]
    im = ax.imshow(exp1_results["wass_matrix"], cmap="YlOrRd")
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=45, ha="right")
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(genres)
    for i in range(len(genres)):
        for j in range(len(genres)):
            ax.text(j, i, f"{exp1_results['wass_matrix'][i, j]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if exp1_results["wass_matrix"][i, j] > exp1_results["wass_matrix"].max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Wasserstein Distance")
    ax.set_title("C) Pairwise Wasserstein Distance")

    # Panel D: Permutation test null distribution
    ax = axes[1, 1]
    ax.hist(exp1_results["null_stats"], bins=30, alpha=0.7, color="steelblue", edgecolor="black", label="Null")
    ax.axvline(exp1_results["obs_stat"], color="red", linewidth=2, label=f"Observed (p={exp1_results['p_value']:.3f})")
    ax.set_xlabel("Mean Pairwise Wasserstein")
    ax.set_ylabel("Count")
    ax.set_title(f"D) Permutation Test (z={exp1_results['z_score']:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "exp1_genre_mfcc_topology.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_exp2(exp2_results, output_dir):
    """Plot Experiment 2: Structural boundary detection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Experiment 2: Structural Boundary Detection", fontsize=14, fontweight="bold")

    scores = exp2_results["scores"]
    score_times = exp2_results["score_times"]
    boundary_times = exp2_results["boundary_times"]
    detected_times = exp2_results["detected_times"]
    section_labels = exp2_results["section_labels"]

    # Panel A: Transition scores with boundaries
    ax = axes[0]
    ax.plot(score_times, scores, color="steelblue", linewidth=1, label="Transition Score")

    # Shade sections
    section_starts = [0] + list(boundary_times)
    section_ends = list(boundary_times) + [30]
    section_colors = {"intro": "#e8e8e8", "verse": "#c8e6c9", "chorus": "#ffcdd2"}
    for i, (start, end, label) in enumerate(zip(section_starts, section_ends, section_labels)):
        ax.axvspan(start, end, alpha=0.3, color=section_colors.get(label, "#e0e0e0"), label=f"{label}" if i < 4 else "")

    for bt in boundary_times:
        ax.axvline(bt, color="green", linewidth=2, linestyle="--", alpha=0.8)
    for dt in detected_times:
        ax.axvline(dt, color="red", linewidth=1.5, linestyle=":", alpha=0.8)

    ax.set_ylabel("Transition Score")
    ax.set_title(f"A) Transition Scores (P={exp2_results['precision']:.2f}, R={exp2_results['recall']:.2f}, F1={exp2_results['f1']:.2f})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Detected vs true boundaries
    ax = axes[1]
    for bt in boundary_times:
        ax.axvline(bt, color="green", linewidth=3, alpha=0.7, label="True Boundary" if bt == boundary_times[0] else "")
    for dt in detected_times:
        ax.axvline(dt, color="red", linewidth=2, linestyle="--", alpha=0.7, label="Detected" if dt == detected_times[0] else "")

    # Mark sections
    for i, (start, end, label) in enumerate(zip(section_starts, section_ends, section_labels)):
        mid = (start + end) / 2
        ax.text(mid, 0.5, label.upper(), ha="center", va="center", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=section_colors.get(label, "#e0e0e0"), alpha=0.8))
        ax.axvspan(start, end, alpha=0.2, color=section_colors.get(label, "#e0e0e0"))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("B) Section Structure: True Boundaries (green) vs Detected (red)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 30)

    plt.tight_layout()
    path = os.path.join(output_dir, "exp2_structural_boundaries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_exp3(exp3_results, output_dir):
    """Plot Experiment 3: Raw audio attractor complexity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 3: Raw Audio Attractor Complexity by Genre", fontsize=14, fontweight="bold")

    genres = GENRES
    gr = exp3_results["genre_results"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(genres)))

    # Panel A: H1 features
    ax = axes[0]
    means = [gr[g]["h1_mean"] for g in genres]
    stds = [gr[g]["h1_std"] for g in genres]
    bars = ax.bar(genres, means, yerr=stds, capsize=5, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Mean H1 Features")
    ax.set_title("A) Raw Waveform H1 by Genre")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Entropy
    ax = axes[1]
    means = [gr[g]["entropy_mean"] for g in genres]
    stds = [gr[g]["entropy_std"] for g in genres]
    bars = ax.bar(genres, means, yerr=stds, capsize=5, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Mean H1 Persistence Entropy")
    ax.set_title("B) Raw Waveform Entropy by Genre")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "exp3_raw_audio_complexity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_overview(exp1, exp2, exp3, output_dir):
    """4-panel overview figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Branch 7: Musical Topology — Genre Fingerprints from Audio Shape",
                 fontsize=14, fontweight="bold")

    genres = GENRES
    colors = plt.cm.Set2(np.linspace(0, 1, len(genres)))
    gr1 = exp1["genre_results"]
    gr3 = exp3["genre_results"]

    # Panel A: MFCC H1 by genre
    ax = axes[0, 0]
    means = [gr1[g]["h1_mean"] for g in genres]
    stds = [gr1[g]["h1_std"] for g in genres]
    ax.bar(genres, means, yerr=stds, capsize=5, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("H1 Features")
    ax.set_title("A) MFCC Topology by Genre")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Transition detection
    ax = axes[0, 1]
    score_times = exp2["score_times"]
    scores = exp2["scores"]
    ax.plot(score_times, scores, color="steelblue", linewidth=1)
    for bt in exp2["boundary_times"]:
        ax.axvline(bt, color="green", linewidth=2, linestyle="--", alpha=0.7)
    for dt in exp2["detected_times"]:
        ax.axvline(dt, color="red", linewidth=1.5, linestyle=":", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_title(f"B) Boundary Detection (F1={exp2['f1']:.2f})")
    ax.grid(True, alpha=0.3)

    # Panel C: Raw H1 by genre
    ax = axes[1, 0]
    means = [gr3[g]["h1_mean"] for g in genres]
    stds = [gr3[g]["h1_std"] for g in genres]
    ax.bar(genres, means, yerr=stds, capsize=5, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("H1 Features")
    ax.set_title("C) Raw Waveform Topology by Genre")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel D: Entropy comparison (MFCC vs Raw)
    ax = axes[1, 1]
    x = np.arange(len(genres))
    width = 0.35
    mfcc_ent = [gr1[g]["entropy_mean"] for g in genres]
    raw_ent = [gr3[g]["entropy_mean"] for g in genres]
    ax.bar(x - width / 2, mfcc_ent, width, label="MFCC", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, raw_ent, width, label="Raw Waveform", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(genres)
    ax.set_ylabel("H1 Persistence Entropy")
    ax.set_title("D) Entropy: MFCC vs Raw Waveform")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Branch 7: Music Genre Topology")
    parser.add_argument("--subsample", type=int, default=500)
    parser.add_argument("--n-perms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-tracks", type=int, default=N_TRACKS)
    args = parser.parse_args()

    print("=" * 70)
    print("BRANCH 7: Musical Topology — Genre Fingerprints from Audio Shape")
    print("=" * 70)
    print(f"Genres: {GENRES}")
    print(f"Tracks per genre: {args.n_tracks}")
    print(f"Subsample: {args.subsample}, Permutations: {args.n_perms}, Seed: {args.seed}")

    import time
    t0 = time.time()

    # Generate synthetic audio
    print("\nGenerating synthetic audio tracks...")
    tracks = generate_tracks(n_tracks=args.n_tracks, seed=args.seed)
    for genre in GENRES:
        print(f"  {genre}: {len(tracks[genre])} tracks × {DURATION}s @ {SR} Hz")

    # Output directories
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "music")
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "music")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Run experiments
    exp1_results = run_exp1(tracks, subsample=args.subsample, n_perms=args.n_perms, seed=args.seed)
    exp2_results = run_exp2(subsample=min(args.subsample, 200), seed=args.seed)
    exp3_results = run_exp3(tracks, subsample=args.subsample, seed=args.seed)

    runtime = time.time() - t0

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_json = {
        "branch": "experiment/tda-music",
        "data_type": "synthetic",
        "n_genres": len(GENRES),
        "genres": GENRES,
        "n_tracks_per_genre": args.n_tracks,
        "exp1_h1_entropy_per_genre": {g: exp1_results["genre_results"][g]["entropy_mean"] for g in GENRES},
        "exp1_h1_features_per_genre": {g: exp1_results["genre_results"][g]["h1_mean"] for g in GENRES},
        "exp1_wasserstein_p": exp1_results["p_value"],
        "exp1_wasserstein_z": exp1_results["z_score"],
        "exp1_genre_ordering": exp1_results["ordering"],
        "exp2_transition_precision": exp2_results["precision"],
        "exp2_transition_recall": exp2_results["recall"],
        "exp2_transition_f1": exp2_results["f1"],
        "exp2_detected_boundaries": exp2_results["detected_times"],
        "exp2_true_boundaries": exp2_results["boundary_times"],
        "exp3_raw_h1_per_genre": {g: exp3_results["genre_results"][g]["h1_mean"] for g in GENRES},
        "exp3_raw_entropy_per_genre": {g: exp3_results["genre_results"][g]["entropy_mean"] for g in GENRES},
        "exp3_genre_ordering": exp3_results["ordering"],
        "exp3_hypothesis_holds": exp3_results["hypothesis_holds"],
        "runtime_seconds": round(runtime, 1),
    }

    results_path = os.path.join(data_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {results_path}")

    # Plot
    print("\nGenerating figures...")
    plot_exp1(exp1_results, fig_dir)
    plot_exp2(exp2_results, fig_dir)
    plot_exp3(exp3_results, fig_dir)
    plot_overview(exp1_results, exp2_results, exp3_results, fig_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Exp 1 — Genre MFCC Topology:")
    print(f"    Permutation test: p={exp1_results['p_value']:.4f}, z={exp1_results['z_score']:.2f}")
    print(f"    Genre ordering: {exp1_results['ordering']}")

    print(f"\n  Exp 2 — Structural Boundary Detection:")
    print(f"    Precision={exp2_results['precision']:.3f}, Recall={exp2_results['recall']:.3f}, F1={exp2_results['f1']:.3f}")
    print(f"    Detected: {[f'{t:.1f}s' for t in exp2_results['detected_times']]}")
    print(f"    True: {exp2_results['boundary_times']}")

    print(f"\n  Exp 3 — Raw Audio Attractor Complexity:")
    print(f"    Genre ordering: {exp3_results['ordering']}")
    print(f"    Hypothesis (jazz > classical > electronic): {exp3_results['hypothesis_holds']}")

    print(f"\n  Runtime: {runtime:.1f}s")
    print("  Done.")


if __name__ == "__main__":
    main()
