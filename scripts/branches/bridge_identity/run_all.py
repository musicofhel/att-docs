#!/usr/bin/env python3
"""Identify the singleton bridge token across 500 MATH-500 problems.

Branch: experiment/tda-bridge-identity
Parent: experiment/tda-bridge-topology
"""

import numpy as np
import json
import os
import hashlib
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

# =============================================================================
# Step 1: Load data and build hash-based problem lookup
# =============================================================================
print("=" * 80)
print("STEP 1: Load data and build hash-based problem lookup")
print("=" * 80)

data = np.load("data/transformer/math500_hidden_states_aligned.npz", allow_pickle=True)
token_trajs = data["token_trajectories"]
levels = data["difficulty_levels"]
stored_hashes = data["problem_hashes"]
seq_lengths = data["seq_lengths"]

print(f"Loaded {len(token_trajs)} problems")
print(f"Stored hashes: {len(stored_hashes)}")
print(f"First 3 hashes: {stored_hashes[:3]}")

# Load MATH-500 and build hash -> problem text lookup
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

def problem_hash(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]

hash_to_problem = {}
for row in ds:
    h = problem_hash(row["problem"])
    hash_to_problem[h] = row["problem"]

# Verify matching — handle potential bytes/numpy string dtype
def to_str(h):
    """Convert hash to plain Python string regardless of stored dtype."""
    if isinstance(h, bytes):
        return h.decode("utf-8")
    return str(h)

matched = sum(1 for h in stored_hashes if to_str(h) in hash_to_problem)
print(f"Hash matches: {matched}/{len(stored_hashes)}")
assert matched > 400, f"Only {matched} hash matches — something is wrong with problem lookup"

# =============================================================================
# Step 2: Load tokenizer and define prompt template
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: Load tokenizer and define prompt template")
print("=" * 80)

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)

print(f"Loaded tokenizer: {MODEL_NAME}")

# =============================================================================
# Step 3: For every problem, find bridge tokens and decode their text
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: For every problem, find bridge tokens and decode their text")
print("=" * 80)

results = []
token_count_mismatches = 0

for idx in range(len(token_trajs)):
    traj = token_trajs[idx]
    h = to_str(stored_hashes[idx])

    # Look up problem text via hash
    if h not in hash_to_problem:
        continue

    problem_text = hash_to_problem[h]
    prompt = PROMPT_TEMPLATE.format(problem=problem_text)

    # Tokenize using the SAME call as extraction script
    inputs = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()

    # Verify token count matches stored seq_length
    expected_len = int(seq_lengths[idx])
    actual_len = len(token_ids)
    if expected_len != actual_len:
        token_count_mismatches += 1
        if token_count_mismatches <= 3:
            print(f"  WARNING: Problem {idx} token count mismatch: stored={expected_len}, re-tokenized={actual_len}")

    # Cluster and find bridges
    if traj.shape[0] < 10:
        continue

    n_comp = min(30, traj.shape[0] - 1, traj.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(traj)

    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(reduced)

    if len(set(labels)) < 2:
        continue

    per_tok_sil = silhouette_samples(reduced, labels)
    bridge_mask = np.abs(per_tok_sil) < 0.1
    bridge_positions = np.where(bridge_mask)[0].tolist()

    # Decode each bridge token
    n_toks = min(traj.shape[0], len(token_ids))
    bridge_details = []
    for pos in bridge_positions:
        if pos < n_toks:
            tok_id = token_ids[pos]
            tok_text = tokenizer.decode([tok_id])
            bridge_details.append({
                "position": pos,
                "token_id": tok_id,
                "text": tok_text,
                "fraction": round(pos / n_toks, 4),
            })

    results.append({
        "idx": idx,
        "level": int(levels[idx]),
        "n_tokens": n_toks,
        "n_bridges": len(bridge_positions),
        "bridges": bridge_details,
    })

print(f"\nProcessed {len(results)} problems with bridge tokens")
print(f"Token count mismatches: {token_count_mismatches}")

# =============================================================================
# Step 4: Print the full report
# =============================================================================
print("\n" + "=" * 80)
print("BRIDGE TOKEN IDENTITY — FIRST 30 PROBLEMS")
print("=" * 80)

for r in results[:30]:
    print(f"\nProblem {r['idx']} (Level {r['level']}, {r['n_tokens']} tok, {r['n_bridges']} bridges):")
    for bt in r["bridges"][:10]:
        print(f"  pos={bt['position']:>3d} ({bt['fraction']:.1%})  id={bt['token_id']:>6d}  text='{bt['text']}'")

# --- Aggregates ---
all_texts = [bt["text"] for r in results for bt in r["bridges"]]
all_stripped = [t.strip() for t in all_texts]
all_positions = [bt["position"] for r in results for bt in r["bridges"]]
all_fractions = [bt["fraction"] for r in results for bt in r["bridges"]]

print("\n" + "=" * 80)
print(f"AGGREGATE: {len(all_texts)} bridge token occurrences across {len(results)} problems")
print("=" * 80)

# Top texts
text_counts = Counter(all_stripped)
print(f"\nTop 20 bridge token texts (stripped):")
for text, count in text_counts.most_common(20):
    print(f"  '{text}': {count} ({count/len(all_texts):.1%})")

# Position stats
pos0_count = sum(1 for p in all_positions if p == 0)
last_count = sum(1 for r in results for bt in r["bridges"] if bt["position"] == r["n_tokens"] - 1)

print(f"\nPosition statistics:")
print(f"  Mean absolute position: {np.mean(all_positions):.1f}")
print(f"  Median absolute position: {np.median(all_positions):.1f}")
print(f"  Mean fractional position: {np.mean(all_fractions):.4f}")
print(f"  Std fractional position: {np.std(all_fractions):.4f}")
print(f"  Position 0 (attention sink): {pos0_count}/{len(all_positions)} ({pos0_count/len(all_positions):.1%})")
print(f"  Last position: {last_count}/{len(all_positions)} ({last_count/len(all_positions):.1%})")

# Token type classification
def classify(text):
    # Check newlines BEFORE stripping — strip destroys them
    if text in ["\n", "\n\n", "\r\n", " \n", "\n "]:
        return "newline"
    t = text.strip()
    if not t:
        return "empty/whitespace"
    if t in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<s>", "</s>", "<pad>", "<|end|>"]:
        return "special_token"
    if t in [".", ",", ":", ";", "?", "!"]:
        return "punctuation"
    if t.replace(".", "").replace("-", "").replace(",", "").isdigit():
        return "number"
    if t in list("+-*/=^()[]{}\\<>|"):
        return "math_operator"
    if t.startswith("\\"):
        return "latex_command"
    return "word"

type_counts = Counter(classify(t) for t in all_texts)
print(f"\nBridge token type distribution:")
for typ, count in type_counts.most_common():
    print(f"  {typ}: {count} ({count/len(all_texts):.1%})")

# Per-level breakdown
print(f"\nPer-level top 3 bridge tokens:")
for level in range(1, 6):
    level_texts = [bt["text"].strip() for r in results if r["level"] == level for bt in r["bridges"]]
    if level_texts:
        top3 = Counter(level_texts).most_common(3)
        print(f"  Level {level} ({len(level_texts)} bridges): " + ", ".join(f"'{t}'={c}" for t, c in top3))

# =============================================================================
# Step 5: Singleton analysis
# =============================================================================
print("\n" + "=" * 80)
print("SINGLETON BRIDGE ANALYSIS")
print("=" * 80)

singleton_results = [r for r in results if r["n_bridges"] == 1]
print(f"Problems with exactly 1 bridge token: {len(singleton_results)}/{len(results)}")

if len(singleton_results) > 0:
    s_texts = [r["bridges"][0]["text"].strip() for r in singleton_results]
    s_positions = [r["bridges"][0]["position"] for r in singleton_results]
    s_fractions = [r["bridges"][0]["fraction"] for r in singleton_results]

    s_text_counts = Counter(s_texts)
    print(f"\nSingleton bridge token texts:")
    for text, count in s_text_counts.most_common(10):
        print(f"  '{text}': {count} ({count/len(s_texts):.1%})")

    print(f"\nSingleton position stats:")
    print(f"  Mean position: {np.mean(s_positions):.1f}")
    print(f"  Mean fraction: {np.mean(s_fractions):.4f}")
    print(f"  Unique positions: {len(set(s_positions))}")
    if len(set(s_positions)) <= 10:
        print(f"  Position set: {sorted(set(s_positions))}")

    s_pos0 = sum(1 for p in s_positions if p == 0)
    print(f"  Singleton at position 0: {s_pos0}/{len(s_positions)} ({s_pos0/len(s_positions):.1%})")
else:
    s_texts = []
    print("No singleton problems found at the final layer.")
    print("(The singleton emerges at INTERMEDIATE layers during crystallization — see Step 6.)")

# =============================================================================
# Step 6: Per-layer bridge identity for crystallized regime
# =============================================================================
perlayer_path = "data/bridge_topology/perlayer_token_states.npz"

print("\n" + "=" * 80)
print("PER-LAYER BRIDGE IDENTITY (CRYSTALLIZED REGIME)")
print("=" * 80)

crystallized_bridge_tokens = []

if not os.path.exists(perlayer_path):
    print(f"ERROR: {perlayer_path} not found. Cannot analyze per-layer bridge identity.")
else:
    perlayer_data = np.load(perlayer_path, allow_pickle=True)
    perlayer_entries = perlayer_data["results"]
    target_layers = perlayer_data["target_layers"].tolist()

    for entry_raw in perlayer_entries:
        entry = entry_raw.item() if hasattr(entry_raw, 'item') else entry_raw
        if not isinstance(entry, dict):
            continue

        prob_idx = entry["problem_idx"]
        h_str = to_str(stored_hashes[prob_idx])
        if h_str not in hash_to_problem:
            continue

        problem_text = hash_to_problem[h_str]
        prompt = PROMPT_TEMPLATE.format(problem=problem_text)
        inputs = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
        token_ids = inputs["input_ids"][0].tolist()

        print(f"\nProblem {prob_idx} (Level {entry['level']}, {len(token_ids)} tokens):")

        per_layer = entry["per_layer"]
        # Keys may be strings after npz round-trip — cast to int for correct sort
        if isinstance(per_layer, dict):
            per_layer = {int(k): v for k, v in per_layer.items()}
        for l in sorted(per_layer.keys()):
            h = per_layer[l]
            n_tokens = h.shape[0]
            if n_tokens < 10:
                continue

            n_comp = min(30, n_tokens - 1, h.shape[1])
            if n_comp < 2:
                continue

            pca_l = PCA(n_components=n_comp)
            reduced_l = pca_l.fit_transform(h)

            km_l = KMeans(n_clusters=2, n_init=10, random_state=42)
            labels_l = km_l.fit_predict(reduced_l)

            if len(set(labels_l)) < 2:
                print(f"  Layer {l:>2d}: single cluster")
                continue

            sil_l = silhouette_samples(reduced_l, labels_l)
            sil_mean = float(np.mean(np.abs(sil_l)))
            bridge_pos = np.where(np.abs(sil_l) < 0.1)[0]

            bridge_info = []
            for pos in bridge_pos:
                if pos < len(token_ids):
                    tid = token_ids[pos]
                    txt = tokenizer.decode([tid])
                    bridge_info.append(f"pos={pos} '{txt}'")

                    if 4 <= l <= 24 and len(bridge_pos) <= 3:
                        crystallized_bridge_tokens.append({
                            "problem_idx": int(prob_idx),
                            "layer": int(l),
                            "position": int(pos),
                            "token_id": int(tid),
                            "text": txt,
                            "fraction": round(pos / len(token_ids), 4),
                        })
                else:
                    bridge_info.append(f"pos={pos} [OOB]")

            print(f"  Layer {l:>2d} (sil={sil_mean:.3f}): {len(bridge_pos)} bridges: {', '.join(bridge_info[:5])}")

    # Summarize crystallized-regime bridge tokens
    print("\n" + "=" * 80)
    print("CRYSTALLIZED REGIME (layers 4-24) BRIDGE TOKENS")
    print("=" * 80)

    if crystallized_bridge_tokens:
        cryst_texts = [bt["text"].strip() for bt in crystallized_bridge_tokens]
        cryst_positions = [bt["position"] for bt in crystallized_bridge_tokens]
        cryst_fractions = [bt["fraction"] for bt in crystallized_bridge_tokens]

        cryst_text_counts = Counter(cryst_texts)
        print(f"\nTotal crystallized-regime bridge occurrences: {len(crystallized_bridge_tokens)}")
        print(f"\nTop 10 crystallized bridge token texts:")
        for text, count in cryst_text_counts.most_common(10):
            print(f"  '{text}': {count} ({count/len(cryst_texts):.1%})")

        print(f"\nCrystallized bridge position stats:")
        print(f"  Mean position: {np.mean(cryst_positions):.1f}")
        print(f"  Mean fraction: {np.mean(cryst_fractions):.4f}")
        cryst_pos0 = sum(1 for p in cryst_positions if p == 0)
        print(f"  Position 0: {cryst_pos0}/{len(cryst_positions)} ({cryst_pos0/len(cryst_positions):.1%})")
        print(f"  Unique positions: {len(set(cryst_positions))}")

        # Per-problem consistency
        by_problem = defaultdict(list)
        for bt in crystallized_bridge_tokens:
            by_problem[bt["problem_idx"]].append(bt["position"])

        print(f"\nPer-problem consistency (same token at every crystallized layer?):")
        consistent_count = 0
        for prob_idx in sorted(by_problem.keys()):
            positions = by_problem[prob_idx]
            unique = set(positions)
            is_consistent = len(unique) == 1
            if is_consistent:
                consistent_count += 1
            print(f"  Problem {prob_idx}: {len(positions)} observations, {len(unique)} unique positions: {sorted(unique)} -> {'YES' if is_consistent else 'NO'}")
        print(f"\nConsistent across layers: {consistent_count}/{len(by_problem)} problems")
    else:
        print("No crystallized-regime bridge tokens found.")

# =============================================================================
# Step 7: Final answer
# =============================================================================
print("\n" + "=" * 80)
print("FINAL ANSWER")
print("=" * 80)

top_text, top_count = text_counts.most_common(1)[0] if text_counts else ("?", 0)
top_frac = top_count / len(all_texts) if all_texts else 0
print(f"\nMost common bridge token (final layer): '{top_text}' = {top_count}/{len(all_texts)} ({top_frac:.1%})")

if crystallized_bridge_tokens:
    cryst_top, cryst_top_count = cryst_text_counts.most_common(1)[0]
    cryst_frac = cryst_top_count / len(cryst_texts)
    print(f"Most common crystallized bridge (L4-24): '{cryst_top}' = {cryst_top_count}/{len(cryst_texts)} ({cryst_frac:.1%})")

print(f"\nHypothesis verdicts:")
if all_positions:
    pos0_frac = pos0_count / len(all_positions)
    print(f"  Attention sink (position 0): {pos0_frac:.1%} -> {'CONFIRMED' if pos0_frac > 0.5 else 'PARTIAL' if pos0_frac > 0.2 else 'REJECTED'}")

    most_common_type = type_counts.most_common(1)[0][0] if type_counts else "?"
    print(f"  Dominant token type: {most_common_type}")

    if most_common_type == "special_token":
        print(f"  -> REGISTER TOKEN: model uses a special/structural token as computational workspace")
    elif most_common_type in ["punctuation", "newline", "empty/whitespace"]:
        print(f"  -> DELIMITER: model uses a semantically empty token as computational workspace")
    elif most_common_type in ["number", "math_operator", "latex_command"]:
        print(f"  -> CONTENT PIVOT: model routes computation through a math-content token")
    elif most_common_type == "word":
        print(f"  -> LANGUAGE TOKEN: model routes computation through a natural language token")

print("=" * 80)

# =============================================================================
# Step 8: Save all results
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: Save all results")
print("=" * 80)

output = {
    "n_problems": len(results),
    "total_bridge_occurrences": len(all_texts),
    "top_20_texts": [{"text": t, "count": int(c), "fraction": round(c/max(len(all_texts),1), 4)} for t, c in text_counts.most_common(20)],
    "type_distribution": {str(t): int(c) for t, c in type_counts.most_common()},
    "position_stats": {
        "mean": round(float(np.mean(all_positions)), 2) if all_positions else 0,
        "median": round(float(np.median(all_positions)), 2) if all_positions else 0,
        "mean_fraction": round(float(np.mean(all_fractions)), 4) if all_fractions else 0,
        "position_0_fraction": round(float(pos0_count / max(len(all_positions), 1)), 4),
    },
    "n_singleton_problems": len(singleton_results),
    "singleton_top_texts": [{"text": t, "count": int(c)} for t, c in Counter(s_texts).most_common(5)] if s_texts else [],
    "token_count_mismatches": token_count_mismatches,
}

if crystallized_bridge_tokens:
    output["crystallized_regime"] = {
        "n_observations": len(crystallized_bridge_tokens),
        "top_10_texts": [{"text": t, "count": int(c)} for t, c in cryst_text_counts.most_common(10)],
        "mean_position": round(float(np.mean(cryst_positions)), 2),
        "position_0_fraction": round(sum(1 for p in cryst_positions if p == 0) / max(len(cryst_positions), 1), 4),
        "per_problem_consistent": {str(k): bool(len(set(v)) == 1) for k, v in by_problem.items()},
        "n_consistent": int(consistent_count),
        "n_problems_with_data": len(by_problem),
    }

os.makedirs("data/bridge_topology", exist_ok=True)
with open("data/bridge_topology/bridge_identity.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: data/bridge_topology/bridge_identity.json")
