#!/usr/bin/env python3
"""Causal mechanism test: Does attention asymmetry cause the singleton bridge?

Tests three attention configurations:
1. Causal (control): Standard causal mask — pos 0 uniquely visible
2. Bidirectional: No causal mask — no position special
3. Dual-visible: Causal mask + pos 5 also visible to all tokens
"""

import numpy as np
import json
import os
import hashlib
import torch
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 1: Load model, data, and tokenizer ──────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load with eager attention so we can intercept the attention mask
# Use bfloat16 — float16 overflows when causal mask is removed (bidirectional mode)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    output_hidden_states=True,
    attn_implementation="eager",
    trust_remote_code=True,
)
model.eval()

# Load problem data
data = np.load("data/transformer/math500_hidden_states_aligned.npz", allow_pickle=True)
levels = data["difficulty_levels"]
stored_hashes = data["problem_hashes"]

from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

def problem_hash(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]

def to_str(h):
    if isinstance(h, bytes):
        return h.decode("utf-8")
    return str(h)

hash_to_problem = {}
for row in ds:
    h = problem_hash(row["problem"])
    hash_to_problem[h] = row["problem"]

# Select 30 problems (6 per level)
rng = np.random.default_rng(42)
selected = []
for lv in range(1, 6):
    lv_idx = np.where(levels == lv)[0]
    chosen = rng.choice(lv_idx, size=min(6, len(lv_idx)), replace=False)
    selected.extend(chosen.tolist())

problems = []
for idx in selected:
    h = to_str(stored_hashes[idx])
    if h in hash_to_problem:
        problems.append({"idx": idx, "level": int(levels[idx]), "text": hash_to_problem[h]})

print(f"Selected {len(problems)} problems for causal mechanism test")


# ── Step 2: Define the three attention modes ──────────────────────────────────

def run_with_attention_mode(model, tokenizer, prompt, mode="causal"):
    """Run forward pass with specified attention mode.

    mode: "causal" | "bidirectional" | "dual_visible_5"
    Returns: final-layer hidden states (n_tokens, hidden_dim) and crystal-layer states
    """
    inputs = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]

    hooks = []

    if mode == "causal":
        # Standard — don't modify anything
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

    elif mode == "bidirectional":
        # Override causal mask: make every position attend to every other position
        def make_bidi_hook():
            def hook_fn(module, args, kwargs):
                # Try kwargs first
                if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    kwargs["attention_mask"] = torch.zeros_like(kwargs["attention_mask"])
                # Also check positional args for 4D tensors (the attention mask)
                new_args = list(args)
                for i in range(len(new_args)):
                    if isinstance(new_args[i], torch.Tensor) and new_args[i].dim() == 4:
                        new_args[i] = torch.zeros_like(new_args[i])
                return tuple(new_args), kwargs
            return hook_fn

        for layer in model.model.layers:
            h = layer.self_attn.register_forward_pre_hook(make_bidi_hook(), with_kwargs=True)
            hooks.append(h)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

    elif mode == "dual_visible_5":
        # Modify causal mask: position 5 is ALSO visible to all tokens
        def make_dual_hook(visible_pos=5):
            def hook_fn(module, args, kwargs):
                # Try kwargs
                if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    mask = kwargs["attention_mask"].clone()
                    if mask.dim() == 4 and visible_pos < mask.shape[-1]:
                        mask[:, :, :, visible_pos] = 0.0
                    kwargs["attention_mask"] = mask
                # Also check positional args
                new_args = list(args)
                for i in range(len(new_args)):
                    if isinstance(new_args[i], torch.Tensor) and new_args[i].dim() == 4:
                        mask = new_args[i].clone()
                        if visible_pos < mask.shape[-1]:
                            mask[:, :, :, visible_pos] = 0.0
                        new_args[i] = mask
                return tuple(new_args), kwargs
            return hook_fn

        for layer in model.model.layers:
            h = layer.self_attn.register_forward_pre_hook(make_dual_hook(5), with_kwargs=True)
            hooks.append(h)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

    # Clean up hooks
    for h in hooks:
        h.remove()

    final_hidden = outputs.hidden_states[-1][0].cpu().float().numpy()
    # Collect hidden states at crystallized layers
    crystal_layers = [4, 7, 10, 14, 17, 21, 24]
    layer_states = {}
    for l in crystal_layers:
        if l < len(outputs.hidden_states):
            layer_states[l] = outputs.hidden_states[l][0].cpu().float().numpy()

    torch.cuda.empty_cache()
    return final_hidden, layer_states


# ── Step 3: Test that hooks actually modify attention ─────────────────────────

test_prompt = PROMPT_TEMPLATE.format(problem=problems[0]["text"])
print(f"Testing hooks on problem {problems[0]['idx']}...")

h_causal, _ = run_with_attention_mode(model, tokenizer, test_prompt, "causal")
h_bidi, _ = run_with_attention_mode(model, tokenizer, test_prompt, "bidirectional")
h_dual, _ = run_with_attention_mode(model, tokenizer, test_prompt, "dual_visible_5")

# If hooks work, hidden states should differ between modes
diff_bidi = float(np.mean(np.abs(h_causal - h_bidi)))
diff_dual = float(np.mean(np.abs(h_causal - h_dual)))

print(f"  Causal vs Bidirectional mean abs diff: {diff_bidi:.6f}")
print(f"  Causal vs Dual-visible mean abs diff: {diff_dual:.6f}")

bidi_nan_rate = float(np.mean(~np.isfinite(h_bidi)))
dual_nan_rate = float(np.mean(~np.isfinite(h_dual)))
print(f"  Bidirectional NaN rate: {bidi_nan_rate:.1%}")
print(f"  Dual-visible NaN rate: {dual_nan_rate:.1%}")

hooks_working = True
if np.isnan(diff_bidi) and bidi_nan_rate > 0:
    # NaN means hooks ARE working (changed the computation), just numerically unstable
    print(f"  WARNING: Bidirectional produces NaN ({bidi_nan_rate:.1%}), but hooks ARE working")
    print(f"  Proceeding — NaN values will be zeroed before clustering")
elif diff_bidi < 1e-4:
    hooks_working = False
    print("  FATAL: Bidirectional hook not working — hidden states identical to causal!")

if not hooks_working:
    print("  The attention mask is not being intercepted by the pre-hook.")

    # Debug: print what args the attention layer actually receives
    debug_args = {}
    def debug_hook(module, args, kwargs):
        debug_args["args_types"] = [type(a).__name__ for a in args]
        debug_args["args_shapes"] = [a.shape if hasattr(a, 'shape') else str(a)[:50] for a in args]
        debug_args["kwargs_keys"] = list(kwargs.keys())
        if "attention_mask" in kwargs:
            debug_args["mask_shape"] = kwargs["attention_mask"].shape if kwargs["attention_mask"] is not None else None
        return args, kwargs

    dbg_hook = model.model.layers[0].self_attn.register_forward_pre_hook(debug_hook, with_kwargs=True)
    with torch.no_grad():
        inputs_dbg = tokenizer(test_prompt, truncation=True, max_length=512, return_tensors="pt")
        model(input_ids=inputs_dbg["input_ids"].to(model.device))
    dbg_hook.remove()

    print(f"  Debug — attention layer receives:")
    print(f"    args types: {debug_args.get('args_types')}")
    print(f"    args shapes: {debug_args.get('args_shapes')}")
    print(f"    kwargs keys: {debug_args.get('kwargs_keys')}")
    print(f"    mask shape: {debug_args.get('mask_shape')}")
    raise RuntimeError("Cannot modify attention mask. Experiment cannot proceed.")

print(f"  Hooks verified — bidirectional diff={diff_bidi}, dual diff={diff_dual}\n")


# ── Step 4: Run all three modes on all 30 problems ───────────────────────────

def analyze_bridges(hidden_states, n_pca=30):
    """Run k=2 clustering and identify bridge tokens."""
    n_tokens = hidden_states.shape[0]
    if n_tokens < 10:
        return None

    # Replace NaN/Inf with 0 — bidirectional attention can produce outliers
    if not np.isfinite(hidden_states).all():
        nan_count = np.sum(~np.isfinite(hidden_states))
        total = hidden_states.size
        print(f"    WARNING: {nan_count}/{total} non-finite values ({nan_count/total:.1%}), replacing with 0")
        hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)

    n_comp = min(n_pca, n_tokens - 1, hidden_states.shape[1])
    if n_comp < 2:
        return None

    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(hidden_states)

    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(reduced)

    if len(set(labels)) < 2:
        return None

    sil_samples = silhouette_samples(reduced, labels)
    sil_mean = float(silhouette_score(reduced, labels))
    bridge_mask = np.abs(sil_samples) < 0.1
    bridge_positions = np.where(bridge_mask)[0].tolist()

    return {
        "silhouette": sil_mean,
        "n_bridges": len(bridge_positions),
        "bridge_positions": bridge_positions,
        "pos0_is_bridge": 0 in bridge_positions,
    }

all_results = {"causal": [], "bidirectional": [], "dual_visible_5": []}

for i, prob in enumerate(problems):
    prompt = PROMPT_TEMPLATE.format(problem=prob["text"])
    token_ids = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")["input_ids"][0].tolist()

    print(f"Problem {i+1}/{len(problems)} (idx={prob['idx']}, Level {prob['level']}, {len(token_ids)} tokens)")

    for mode in ["causal", "bidirectional", "dual_visible_5"]:
        final_hidden, layer_states = run_with_attention_mode(model, tokenizer, prompt, mode)

        # Analyze final layer
        final_result = analyze_bridges(final_hidden)

        # Analyze crystallized layers
        crystal_results = {}
        for l, h in layer_states.items():
            crystal_results[l] = analyze_bridges(h)

        # Decode bridge token texts at the most crystallized layer (layer 14)
        bridge_texts = []
        if 14 in crystal_results and crystal_results[14] is not None:
            for pos in crystal_results[14]["bridge_positions"][:10]:
                if pos < len(token_ids):
                    bridge_texts.append({"pos": pos, "text": tokenizer.decode([token_ids[pos]])})

        result = {
            "problem_idx": prob["idx"],
            "level": prob["level"],
            "mode": mode,
            "n_tokens": len(token_ids),
            "final_layer": final_result,
            "crystal_layer_14": crystal_results.get(14),
            "crystal_bridge_texts": bridge_texts,
        }
        all_results[mode].append(result)

        # Print summary for this mode
        if crystal_results.get(14) is not None:
            cr = crystal_results[14]
            pos0 = "YES" if cr["pos0_is_bridge"] else "NO"
            texts = ", ".join(f"pos={bt['pos']}'{bt['text']}'" for bt in bridge_texts[:3])
            print(f"  {mode:>20s}: sil={cr['silhouette']:.3f}, {cr['n_bridges']} bridges, pos0={pos0}, [{texts}]")
        else:
            print(f"  {mode:>20s}: no valid clustering")

    print()


# ── Step 5: Aggregate and print results ──────────────────────────────────────

print("\n" + "="*80)
print("AGGREGATE RESULTS")
print("="*80)

for mode in ["causal", "bidirectional", "dual_visible_5"]:
    results = all_results[mode]

    # Crystal layer 14 statistics
    valid = [r for r in results if r["crystal_layer_14"] is not None]

    if not valid:
        print(f"\n{mode}: No valid clustering results")
        continue

    n_with_bridge = sum(1 for r in valid if r["crystal_layer_14"]["n_bridges"] > 0)
    pos0_bridge = sum(1 for r in valid if r["crystal_layer_14"]["pos0_is_bridge"])
    mean_sil = np.mean([r["crystal_layer_14"]["silhouette"] for r in valid])
    mean_n_bridges = np.mean([r["crystal_layer_14"]["n_bridges"] for r in valid])

    # Singleton count (exactly 1 bridge)
    singletons = [r for r in valid if r["crystal_layer_14"]["n_bridges"] == 1]
    singleton_at_pos0 = sum(1 for r in singletons if r["crystal_layer_14"]["pos0_is_bridge"])

    # Position 5 bridge count (for dual_visible_5)
    pos5_bridge = sum(1 for r in valid if 5 in r["crystal_layer_14"]["bridge_positions"])

    # Collect all bridge positions
    all_bridge_pos = [pos for r in valid for pos in r["crystal_layer_14"]["bridge_positions"]]
    pos_counts = Counter(all_bridge_pos)

    print(f"\n{'='*40}")
    print(f"MODE: {mode}")
    print(f"{'='*40}")
    print(f"Valid problems: {len(valid)}/{len(results)}")
    print(f"Mean silhouette: {mean_sil:.3f}")
    print(f"Mean bridges per problem: {mean_n_bridges:.1f}")
    print(f"Problems with any bridge: {n_with_bridge}/{len(valid)}")
    print(f"Pos 0 is bridge: {pos0_bridge}/{len(valid)} ({pos0_bridge/max(len(valid),1):.1%})")
    print(f"Pos 5 is bridge: {pos5_bridge}/{len(valid)} ({pos5_bridge/max(len(valid),1):.1%})")
    print(f"Singleton bridges: {len(singletons)}/{len(valid)}")
    if singletons:
        print(f"  Singletons at pos 0: {singleton_at_pos0}/{len(singletons)}")
    print(f"Top 5 bridge positions: {pos_counts.most_common(5)}")

    # Bridge token texts at layer 14
    all_texts = [bt["text"].strip() for r in valid for bt in r["crystal_bridge_texts"]]
    if all_texts:
        text_counts = Counter(all_texts)
        print(f"Top 5 bridge token texts: {text_counts.most_common(5)}")


# ── Step 6: Print the verdict ────────────────────────────────────────────────

print("\n" + "="*80)
print("VERDICT")
print("="*80)

causal_valid = [r for r in all_results["causal"] if r["crystal_layer_14"] is not None]
bidi_valid = [r for r in all_results["bidirectional"] if r["crystal_layer_14"] is not None]
dual_valid = [r for r in all_results["dual_visible_5"] if r["crystal_layer_14"] is not None]

# Test 1: Does bidirectional kill the singleton bridge at pos 0?
causal_pos0 = sum(1 for r in causal_valid if r["crystal_layer_14"]["pos0_is_bridge"]) / max(len(causal_valid), 1)
bidi_pos0 = sum(1 for r in bidi_valid if r["crystal_layer_14"]["pos0_is_bridge"]) / max(len(bidi_valid), 1)

print(f"\nTest 1: Bidirectional attention")
print(f"  Causal pos-0 bridge rate:        {causal_pos0:.1%}")
print(f"  Bidirectional pos-0 bridge rate:  {bidi_pos0:.1%}")
if bidi_pos0 < 0.2:
    print(f"  -> CONFIRMED: Removing causal asymmetry eliminates the pos-0 bridge")
elif bidi_pos0 < 0.8:
    print(f"  -> PARTIAL: Bridge weakened but not eliminated")
else:
    print(f"  -> REJECTED: Bridge persists without causal attention")

# Test 2: Does dual-visible create a second bridge at pos 5?
dual_pos0 = sum(1 for r in dual_valid if r["crystal_layer_14"]["pos0_is_bridge"]) / max(len(dual_valid), 1)
dual_pos5 = sum(1 for r in dual_valid if 5 in r["crystal_layer_14"]["bridge_positions"]) / max(len(dual_valid), 1)

print(f"\nTest 2: Dual-visible (pos 0 + pos 5)")
print(f"  Pos-0 bridge rate: {dual_pos0:.1%}")
print(f"  Pos-5 bridge rate: {dual_pos5:.1%}")
if dual_pos5 > 0.5:
    print(f"  -> CONFIRMED: Making pos 5 universally visible creates a second bridge")
elif dual_pos5 > 0.2:
    print(f"  -> PARTIAL: Pos 5 sometimes becomes a bridge")
else:
    print(f"  -> REJECTED: Universal visibility alone insufficient to create bridge")

# Test 3: Does the two-cluster structure itself survive?
bidi_sil = np.mean([r["crystal_layer_14"]["silhouette"] for r in bidi_valid]) if bidi_valid else 0
causal_sil = np.mean([r["crystal_layer_14"]["silhouette"] for r in causal_valid]) if causal_valid else 0

print(f"\nTest 3: Two-cluster structure")
print(f"  Causal mean silhouette:       {causal_sil:.3f}")
print(f"  Bidirectional mean silhouette: {bidi_sil:.3f}")
if bidi_sil < 0.1:
    print(f"  -> Clusters DISAPPEAR under bidirectional attention")
elif bidi_sil < causal_sil * 0.5:
    print(f"  -> Clusters WEAKEN under bidirectional attention")
else:
    print(f"  -> Clusters PERSIST under bidirectional attention")

# Overall
print(f"\nOVERALL CAUSAL MECHANISM:")
if bidi_pos0 < 0.2 and (dual_pos5 > 0.3 or bidi_sil < 0.1):
    print(f"  CONFIRMED: Causal attention asymmetry causes the singleton bridge at position 0.")
    print(f"  The model uses the uniquely-visible position as a computational register.")
elif bidi_pos0 < 0.5:
    print(f"  PARTIALLY CONFIRMED: Causal attention contributes to the bridge but other factors also play a role.")
else:
    print(f"  NOT CONFIRMED: The bridge persists without causal asymmetry. Other mechanisms are responsible.")

print("="*80)


# ── Step 7: Save results ─────────────────────────────────────────────────────

output = {
    "branch": "experiment/tda-bridge-causal",
    "n_problems": len(problems),
    "causal": {
        "pos0_bridge_rate": float(causal_pos0),
        "mean_silhouette": float(causal_sil),
        "mean_n_bridges": float(np.mean([r["crystal_layer_14"]["n_bridges"] for r in causal_valid])) if causal_valid else 0,
    },
    "bidirectional": {
        "pos0_bridge_rate": float(bidi_pos0),
        "mean_silhouette": float(bidi_sil),
        "mean_n_bridges": float(np.mean([r["crystal_layer_14"]["n_bridges"] for r in bidi_valid])) if bidi_valid else 0,
    },
    "dual_visible_5": {
        "pos0_bridge_rate": float(dual_pos0),
        "pos5_bridge_rate": float(dual_pos5),
        "mean_silhouette": float(np.mean([r["crystal_layer_14"]["silhouette"] for r in dual_valid])) if dual_valid else 0,
        "mean_n_bridges": float(np.mean([r["crystal_layer_14"]["n_bridges"] for r in dual_valid])) if dual_valid else 0,
    },
    "hook_verified": True,
    "causal_mechanism_confirmed": bool(bidi_pos0 < 0.2),
}

os.makedirs("data/bridge_causal", exist_ok=True)
with open("data/bridge_causal/results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: data/bridge_causal/results.json")
