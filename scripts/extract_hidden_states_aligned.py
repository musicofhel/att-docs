"""Re-extract Qwen hidden states from MATH-500 to align with attention PH and correctness data.

AUDIT FIX: The original extract_hidden_states.py loaded from the full MATH dataset
(~5000 problems via hendrycks/competition_math or EleutherAI/hendrycks_math), while
extract_attention_weights.py and evaluate_correctness.py fell through to
HuggingFaceH4/MATH-500 (500 problems). This caused a cross-file problem mismatch:
the 100 problems per level from the full MATH were entirely different from the
25 per level from MATH-500.

This script extracts hidden states from MATH-500 directly, using the SAME loading
and sampling logic as extract_attention_weights.py, so that:
1. The problem pool is identical (MATH-500, 500 problems)
2. For the attention-matched subset: sample 25/level with seed=42 (same as attention)
3. For the full MATH-500: sample all available per level (43/90/105/128/134)
4. Problem text hashes are stored in the .npz for cross-file verification

Usage:
    python scripts/extract_hidden_states_aligned.py
    python scripts/extract_hidden_states_aligned.py --subset attention  # only 125 problems matching attention PH
    python scripts/extract_hidden_states_aligned.py --subset full       # all 500 MATH-500 problems
"""

import argparse
import hashlib
import os
import re
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "transformer")

PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)


def problem_hash(text: str) -> str:
    """SHA-256 hash of problem text for cross-file verification."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def load_math500(n_per_level: int | None = None, seed: int = 42):
    """Load MATH-500 problems, optionally sampling n_per_level.

    Uses the SAME loading logic as extract_attention_weights.py to ensure
    identical problem selection when n_per_level and seed match.

    Parameters
    ----------
    n_per_level : int or None
        If None, take ALL problems at each level (no sampling).
        If int, sample min(n_per_level, pool_size) per level with given seed.
    seed : int
        Random seed for sampling (must match other scripts for alignment).

    Returns
    -------
    list of dicts with 'problem', 'level', 'hash' keys.
    """
    from datasets import load_dataset

    # Load MATH-500 directly — do NOT try other sources
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Loaded {len(ds)} problems from HuggingFaceH4/MATH-500")

    problems_by_level: dict[int, list] = {k: [] for k in range(1, 6)}

    for row in ds:
        level_raw = row.get("level", "")
        if not level_raw and level_raw != 0:
            continue
        try:
            if isinstance(level_raw, int):
                level = level_raw
            else:
                level = int(str(level_raw).replace("Level ", ""))
        except (ValueError, AttributeError):
            continue
        if 1 <= level <= 5:
            problems_by_level[level].append({
                "problem": row["problem"],
                "level": level,
                "hash": problem_hash(row["problem"]),
            })

    for lv in range(1, 6):
        print(f"  Level {lv}: {len(problems_by_level[lv])} problems available")

    if n_per_level is None:
        # Take ALL problems
        sampled = []
        for level in range(1, 6):
            sampled.extend(problems_by_level[level])
    else:
        # Sample with same RNG as extract_attention_weights.py
        rng = np.random.default_rng(seed)
        sampled = []
        for level in range(1, 6):
            pool = problems_by_level[level]
            n_sample = min(n_per_level, len(pool))
            if n_sample == 0:
                continue
            indices = rng.choice(len(pool), size=n_sample, replace=False)
            sampled.extend([pool[i] for i in indices])
            print(f"  Level {level}: sampled {n_sample}/{len(pool)}")

    print(f"Total: {len(sampled)} problems")
    return sampled


def verify_alignment(sampled_problems, attention_ph_path):
    """Verify that sampled problems match attention PH extraction."""
    if not os.path.exists(attention_ph_path):
        print("WARNING: No attention PH file to verify against")
        return False

    attn_data = np.load(attention_ph_path, allow_pickle=True)
    attn_levels = attn_data["difficulty_levels"]
    n_attn = len(attn_levels)

    # Check level distributions match
    sampled_levels = [p["level"] for p in sampled_problems]
    for lv in range(1, 6):
        n_sampled = sum(1 for l in sampled_levels if l == lv)
        n_attn_lv = int((attn_levels == lv).sum())
        if n_sampled < n_attn_lv:
            print(f"WARNING: Level {lv}: sampled {n_sampled} < attention {n_attn_lv}")
            return False

    print(f"Level distribution verified: sampled covers attention ({n_attn} problems)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract Qwen hidden states aligned with MATH-500"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output", default=os.path.join(DATA_DIR, "math500_hidden_states_aligned.npz"),
        help="Output path",
    )
    parser.add_argument(
        "--subset", choices=["attention", "full"], default="full",
        help="'attention' = 25/level matching attention PH, 'full' = all MATH-500",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load problems
    if args.subset == "attention":
        # Match attention extraction: 25/level with seed=42
        problems = load_math500(n_per_level=25, seed=args.seed)
    else:
        # All MATH-500 problems (no sampling)
        problems = load_math500(n_per_level=None, seed=args.seed)

    if not problems:
        print("No problems loaded. Exiting.")
        sys.exit(1)

    # Verify alignment with attention PH
    attn_path = os.path.join(DATA_DIR, "attention_ph_diagrams.npz")
    verify_alignment(problems, attn_path)

    # Load model
    print(f"\nLoading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    # Extraction loop
    all_last_hidden = []
    all_layer_hidden = []
    all_token_trajectories = []
    all_levels = []
    all_seq_lengths = []
    all_hashes = []
    skipped = []

    t0 = time.time()
    for idx, problem in enumerate(problems):
        if idx % 50 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(problems) - idx) / rate if rate > 0 else 0
            print(f"Processing {idx}/{len(problems)} "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]")

        prompt = PROMPT_TEMPLATE.format(problem=problem["problem"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=args.max_length,
        ).to(device)
        seq_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.hidden_states
            n_layers_plus_one = len(hidden_states)

            final_layer_all_tokens = hidden_states[-1][0]
            last_hidden = final_layer_all_tokens[-1].cpu().float().numpy()

            layer_states = (
                torch.stack([hidden_states[i][0, -1, :] for i in range(n_layers_plus_one)])
                .cpu().float().numpy()
            )
            token_traj = final_layer_all_tokens.cpu().float().numpy()

            all_last_hidden.append(last_hidden)
            all_layer_hidden.append(layer_states)
            all_token_trajectories.append(token_traj)
            all_levels.append(problem["level"])
            all_seq_lengths.append(seq_len)
            all_hashes.append(problem["hash"])

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on problem {idx} (seq_len={seq_len}), skipping")
            skipped.append(idx)
            torch.cuda.empty_cache()
            continue

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nDone. Extracted {len(all_last_hidden)}/{len(problems)} in {elapsed:.1f}s. "
          f"Skipped: {len(skipped)}")

    if len(all_last_hidden) == 0:
        print("ERROR: No hidden states extracted.")
        sys.exit(1)

    d = all_last_hidden[0].shape[0]
    n_layers = all_layer_hidden[0].shape[0]
    print(f"Hidden dim: {d}, Layers (incl. embedding): {n_layers}")

    level_counts = {}
    for lv in all_levels:
        level_counts[lv] = level_counts.get(lv, 0) + 1
    for lv in sorted(level_counts):
        print(f"  Level {lv}: {level_counts[lv]} problems")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        last_hidden_states=np.array(all_last_hidden),
        difficulty_levels=np.array(all_levels),
        layer_hidden_states=np.array(all_layer_hidden),
        token_trajectories=np.array(all_token_trajectories, dtype=object),
        seq_lengths=np.array(all_seq_lengths),
        problem_hashes=np.array(all_hashes),
        model_name=np.array(args.model),
        skipped_indices=np.array(skipped),
        hidden_dim=np.array(d),
        num_layers=np.array(n_layers),
        dataset_source=np.array("HuggingFaceH4/MATH-500"),
        subset_mode=np.array(args.subset),
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
