#!/usr/bin/env python3
"""Causation experiments: is the bridge token positional or token-specific?

Experiment 1: Swap first word ('Please' at pos 0 instead of 'You')
Experiment 2: Verify position 0 contents (already done inline, recorded here)
Experiment 3: Base model with no system prompt (first word varies per problem)

Branch: experiment/tda-bridge-identity
"""

import numpy as np
import json
import os
import hashlib
import gc
import torch
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# =============================================================================
# Setup: load problem hashes and build lookup
# =============================================================================
print("=" * 80)
print("SETUP: Load data and build problem lookup")
print("=" * 80)

data = np.load("data/transformer/math500_hidden_states_aligned.npz", allow_pickle=True)
stored_hashes = data["problem_hashes"]
levels = data["difficulty_levels"]

def problem_hash(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]

def to_str(h):
    if isinstance(h, bytes):
        return h.decode("utf-8")
    return str(h)

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
hash_to_problem = {}
for row in ds:
    h = problem_hash(row["problem"])
    hash_to_problem[h] = row["problem"]

# Get the same 50 problem indices used in perlayer analysis
perlayer_data = np.load("data/bridge_topology/perlayer_token_states.npz", allow_pickle=True)
problem_indices = []
for entry_raw in perlayer_data["results"]:
    entry = entry_raw.item() if hasattr(entry_raw, 'item') else entry_raw
    if isinstance(entry, dict):
        problem_indices.append(entry["problem_idx"])
problem_indices = sorted(problem_indices)
target_layers = perlayer_data["target_layers"].tolist()

print(f"Using {len(problem_indices)} problems, layers {target_layers}")

# Resolve problem texts
problems = {}
for idx in problem_indices:
    h = to_str(stored_hashes[idx])
    if h in hash_to_problem:
        problems[idx] = hash_to_problem[h]
print(f"Resolved {len(problems)}/{len(problem_indices)} problem texts")

# =============================================================================
# Helper: extract per-layer hidden states for a set of problems
# =============================================================================
def extract_hidden_states(model, tokenizer, problems, template, target_layers, label):
    """Extract per-token hidden states at target layers for each problem."""
    print(f"\n{'='*80}")
    print(f"EXTRACTING: {label}")
    print(f"{'='*80}")

    results = {}
    device = next(model.parameters()).device

    for i, (idx, problem_text) in enumerate(problems.items()):
        prompt = template.format(problem=problem_text)
        inputs = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        if i == 0:
            # Print first 10 tokens for verification
            token_ids = input_ids[0].tolist()
            print(f"\nFirst problem (idx={idx}) — first 10 tokens:")
            for j, tid in enumerate(token_ids[:10]):
                print(f"  pos={j:>3d}  id={tid:>6d}  text={repr(tokenizer.decode([tid]))}")
            print(f"  ... total {len(token_ids)} tokens")

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
        token_ids_list = input_ids[0].tolist()

        per_layer = {}
        for l in target_layers:
            if l < len(hidden_states):
                per_layer[l] = hidden_states[l][0].cpu().float().numpy()  # (seq_len, hidden_dim)

        results[idx] = {
            "token_ids": token_ids_list,
            "per_layer": per_layer,
            "level": int(levels[idx]),
            "n_tokens": len(token_ids_list),
        }

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(problems)} problems")

        torch.cuda.empty_cache()

    print(f"  Done: {len(results)} problems extracted")
    return results


# =============================================================================
# Helper: cluster and find bridge tokens at each layer
# =============================================================================
def analyze_bridges(hidden_data, tokenizer, target_layers, label):
    """Run clustering on extracted hidden states and identify bridge tokens."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {label}")
    print(f"{'='*80}")

    crystallized_bridges = []  # bridges in layers 4-24
    per_problem_report = []

    for idx in sorted(hidden_data.keys()):
        entry = hidden_data[idx]
        token_ids = entry["token_ids"]
        per_layer = entry["per_layer"]
        n_tokens = entry["n_tokens"]

        print(f"\nProblem {idx} (Level {entry['level']}, {n_tokens} tokens):")

        problem_bridges = []
        for l in sorted(per_layer.keys()):
            h = per_layer[l]
            if h.shape[0] < 10:
                continue

            n_comp = min(30, h.shape[0] - 1, h.shape[1])
            if n_comp < 2:
                continue

            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(h)

            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            labels = km.fit_predict(reduced)

            if len(set(labels)) < 2:
                print(f"  Layer {l:>2d}: single cluster")
                continue

            sil = silhouette_samples(reduced, labels)
            sil_mean = float(np.mean(np.abs(sil)))
            bridge_pos = np.where(np.abs(sil) < 0.1)[0]

            bridge_info = []
            for pos in bridge_pos:
                if pos < len(token_ids):
                    tid = token_ids[pos]
                    txt = tokenizer.decode([tid])
                    bridge_info.append(f"pos={pos} '{txt}'")

                    if 4 <= l <= 24 and len(bridge_pos) <= 3:
                        crystallized_bridges.append({
                            "problem_idx": int(idx),
                            "layer": int(l),
                            "position": int(pos),
                            "token_id": int(tid),
                            "text": txt,
                            "fraction": round(pos / n_tokens, 4),
                        })
                else:
                    bridge_info.append(f"pos={pos} [OOB]")

            print(f"  Layer {l:>2d} (sil={sil_mean:.3f}): {len(bridge_pos)} bridges: {', '.join(bridge_info[:5])}")

            problem_bridges.append({
                "layer": int(l),
                "n_bridges": len(bridge_pos),
                "bridge_positions": [int(p) for p in bridge_pos],
                "sil_mean": round(sil_mean, 4),
            })

        per_problem_report.append({
            "problem_idx": int(idx),
            "level": int(entry["level"]),
            "n_tokens": n_tokens,
            "layers": problem_bridges,
        })

    # Summarize crystallized regime
    print(f"\n{'='*80}")
    print(f"CRYSTALLIZED REGIME (layers 4-24) — {label}")
    print(f"{'='*80}")

    if crystallized_bridges:
        cryst_texts = [bt["text"].strip() for bt in crystallized_bridges]
        cryst_positions = [bt["position"] for bt in crystallized_bridges]

        cryst_text_counts = Counter(cryst_texts)
        print(f"\nTotal crystallized-regime bridge occurrences: {len(crystallized_bridges)}")
        print(f"\nTop 10 crystallized bridge token texts:")
        for text, count in cryst_text_counts.most_common(10):
            print(f"  '{text}': {count} ({count/len(cryst_texts):.1%})")

        print(f"\nCrystallized bridge position stats:")
        print(f"  Mean position: {np.mean(cryst_positions):.1f}")
        cryst_pos0 = sum(1 for p in cryst_positions if p == 0)
        print(f"  Position 0: {cryst_pos0}/{len(cryst_positions)} ({cryst_pos0/len(cryst_positions):.1%})")
        print(f"  Unique positions: {len(set(cryst_positions))}")

        # Per-problem consistency
        by_problem = defaultdict(list)
        for bt in crystallized_bridges:
            by_problem[bt["problem_idx"]].append(bt["position"])

        consistent_count = 0
        for prob_idx in sorted(by_problem.keys()):
            positions = by_problem[prob_idx]
            unique = set(positions)
            is_consistent = len(unique) == 1
            if is_consistent:
                consistent_count += 1
            print(f"  Problem {prob_idx}: {len(positions)} obs, positions={sorted(unique)} -> {'CONSISTENT' if is_consistent else 'VARIES'}")
        print(f"\nConsistent across layers: {consistent_count}/{len(by_problem)} problems")

        return {
            "n_observations": len(crystallized_bridges),
            "top_texts": [{"text": t, "count": int(c)} for t, c in cryst_text_counts.most_common(10)],
            "position_0_count": cryst_pos0,
            "position_0_fraction": round(cryst_pos0 / len(cryst_positions), 4),
            "n_consistent": consistent_count,
            "n_problems": len(by_problem),
            "unique_positions": sorted(set(cryst_positions)),
            "details": crystallized_bridges,
        }
    else:
        print("No crystallized-regime bridge tokens found.")
        return {"n_observations": 0}


# =============================================================================
# Experiment 2: Already done — record result
# =============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: Position 0 verification (already completed)")
print("=" * 80)
print("Position 0 = 'You' (token ID 2610)")
print("Qwen tokenizer has no BOS token, no chat template was applied")
print("The extraction script used raw tokenization: tokenizer(prompt, ...)")
print("RESULT: Position 0 is literally the word 'You', not a special token")

exp2_result = {
    "experiment": "position_0_verification",
    "position_0_token": "You",
    "position_0_token_id": 2610,
    "has_bos": False,
    "chat_template_used": False,
    "conclusion": "Position 0 is a plain word token, not a special/register token",
}

# =============================================================================
# Experiment 1: Swap first word
# =============================================================================
TEMPLATE_SWAP = "Please help me solve this math problem.\n\n{problem}\n\nPlease provide the final answer."

print("\n" + "=" * 80)
print("EXPERIMENT 1: Swap first word ('Please' at position 0)")
print("=" * 80)

tokenizer_instruct = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
if tokenizer_instruct.pad_token is None:
    tokenizer_instruct.pad_token = tokenizer_instruct.eos_token

# Verify new template tokenization
test_prompt = TEMPLATE_SWAP.format(problem="What is 2+2?")
test_ids = tokenizer_instruct(test_prompt, return_tensors="pt")["input_ids"][0].tolist()
print(f"Swap template — first 5 tokens:")
for j, tid in enumerate(test_ids[:5]):
    print(f"  pos={j}  id={tid}  text={repr(tokenizer_instruct.decode([tid]))}")
assert tokenizer_instruct.decode([test_ids[0]]).strip() == "Please", f"Expected 'Please' at pos 0, got '{tokenizer_instruct.decode([test_ids[0]])}'"
print("VERIFIED: 'Please' is at position 0")

model_instruct = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model_instruct.eval()

hidden_exp1 = extract_hidden_states(model_instruct, tokenizer_instruct, problems, TEMPLATE_SWAP, target_layers, "Exp1: Swap first word")
exp1_result = analyze_bridges(hidden_exp1, tokenizer_instruct, target_layers, "Exp1: Swap first word")

# Free GPU memory
del model_instruct
gc.collect()
torch.cuda.empty_cache()

# =============================================================================
# Experiment 3: Base model, no system prompt
# =============================================================================
TEMPLATE_BASE = "{problem}\n\nPlease provide the final answer."

print("\n" + "=" * 80)
print("EXPERIMENT 3: Base model, no system prompt")
print("=" * 80)

tokenizer_base = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
if tokenizer_base.pad_token is None:
    tokenizer_base.pad_token = tokenizer_base.eos_token

# Show what position 0 looks like for a few problems
print("Position 0 tokens for first 5 problems:")
for idx in sorted(problems.keys())[:5]:
    prompt = TEMPLATE_BASE.format(problem=problems[idx])
    ids = tokenizer_base(prompt, truncation=True, max_length=512, return_tensors="pt")["input_ids"][0].tolist()
    pos0_text = tokenizer_base.decode([ids[0]])
    print(f"  Problem {idx}: pos0 = {repr(pos0_text)} (id={ids[0]}), {len(ids)} tokens total")

model_base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model_base.eval()

hidden_exp3 = extract_hidden_states(model_base, tokenizer_base, problems, TEMPLATE_BASE, target_layers, "Exp3: Base model, no system prompt")
exp3_result = analyze_bridges(hidden_exp3, tokenizer_base, target_layers, "Exp3: Base model, no system prompt")

del model_base
gc.collect()
torch.cuda.empty_cache()

# =============================================================================
# Final comparison
# =============================================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

# Load original result for comparison
orig = json.load(open("data/bridge_topology/bridge_identity.json"))
orig_cryst = orig.get("crystallized_regime", {})

print(f"\n{'Experiment':<45} {'Pos-0 Bridge %':>15} {'Consistent':>12} {'Top Token':>15}")
print("-" * 90)

# Original
orig_pos0 = orig_cryst.get("position_0_fraction", 0)
orig_cons = f"{orig_cryst.get('n_consistent', 0)}/{orig_cryst.get('n_problems_with_data', 0)}"
orig_top = orig_cryst.get("top_10_texts", [{}])[0].get("text", "?")
print(f"{'Original (You at pos 0, instruct)':<45} {orig_pos0:>14.1%} {orig_cons:>12} {repr(orig_top):>15}")

# Exp 1
if exp1_result["n_observations"] > 0:
    e1_pos0 = exp1_result["position_0_fraction"]
    e1_cons = f"{exp1_result['n_consistent']}/{exp1_result['n_problems']}"
    e1_top = exp1_result["top_texts"][0]["text"]
    print(f"{'Exp1: Swap (Please at pos 0, instruct)':<45} {e1_pos0:>14.1%} {e1_cons:>12} {repr(e1_top):>15}")
else:
    print(f"{'Exp1: Swap (Please at pos 0, instruct)':<45} {'N/A':>15} {'N/A':>12} {'N/A':>15}")

# Exp 3
if exp3_result["n_observations"] > 0:
    e3_pos0 = exp3_result["position_0_fraction"]
    e3_cons = f"{exp3_result['n_consistent']}/{exp3_result['n_problems']}"
    e3_top = exp3_result["top_texts"][0]["text"]
    print(f"{'Exp3: Base model (varying pos 0, base)':<45} {e3_pos0:>14.1%} {e3_cons:>12} {repr(e3_top):>15}")
else:
    print(f"{'Exp3: Base model (varying pos 0, base)':<45} {'N/A':>15} {'N/A':>12} {'N/A':>15}")

# Verdicts
print(f"\n{'='*80}")
print("VERDICTS")
print("=" * 80)

if exp1_result["n_observations"] > 0 and exp1_result["position_0_fraction"] > 0.9:
    print("Exp1: POSITIONAL — changing 'You' to 'Please' still yields pos-0 bridge")
    print("      The attention sink effect is about POSITION, not token identity")
    verdict1 = "positional"
elif exp1_result["n_observations"] > 0:
    print(f"Exp1: TOKEN-SPECIFIC — 'Please' at pos 0 does NOT become the bridge (pos-0 = {exp1_result['position_0_fraction']:.1%})")
    print("      Something about the token 'You' specifically matters")
    verdict1 = "token_specific"
else:
    print("Exp1: INCONCLUSIVE — no crystallized bridges found")
    verdict1 = "inconclusive"

if exp3_result["n_observations"] > 0 and exp3_result["position_0_fraction"] > 0.9:
    print("Exp3: ARCHITECTURAL — base model with varying pos-0 content still uses pos-0 as bridge")
    print("      This is a fundamental property of causal attention, not instruction tuning")
    verdict3 = "architectural"
elif exp3_result["n_observations"] > 0 and exp3_result["position_0_fraction"] > 0.5:
    print(f"Exp3: MOSTLY ARCHITECTURAL — pos-0 is bridge {exp3_result['position_0_fraction']:.1%} of the time")
    print("      Some problems route through different tokens")
    verdict3 = "mostly_architectural"
elif exp3_result["n_observations"] > 0:
    unique_pos = exp3_result.get("unique_positions", [])
    print(f"Exp3: CONTENT-DEPENDENT — pos-0 is only {exp3_result['position_0_fraction']:.1%} of bridges")
    print(f"      Unique bridge positions: {unique_pos[:10]}")
    print("      The model routes computation through specific semantic positions")
    verdict3 = "content_dependent"
else:
    print("Exp3: INCONCLUSIVE — no crystallized bridges found")
    verdict3 = "inconclusive"

print("=" * 80)

# =============================================================================
# Save results
# =============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output = {
    "experiment_2_position_check": exp2_result,
    "experiment_1_swap_first_word": {
        "template": TEMPLATE_SWAP,
        "position_0_token": "Please",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "n_problems": len(problems),
        "verdict": verdict1,
        "crystallized_regime": {k: v for k, v in exp1_result.items() if k != "details"},
    },
    "experiment_3_base_model": {
        "template": TEMPLATE_BASE,
        "position_0_token": "varies_per_problem",
        "model": "Qwen/Qwen2.5-1.5B",
        "n_problems": len(problems),
        "verdict": verdict3,
        "crystallized_regime": {k: v for k, v in exp3_result.items() if k != "details"},
    },
    "comparison": {
        "original_pos0_fraction": orig_pos0,
        "exp1_pos0_fraction": exp1_result.get("position_0_fraction", None),
        "exp3_pos0_fraction": exp3_result.get("position_0_fraction", None),
    },
}

os.makedirs("data/bridge_topology", exist_ok=True)
with open("data/bridge_topology/causation_experiments.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved: data/bridge_topology/causation_experiments.json")
