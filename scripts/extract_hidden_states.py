"""Extract LLM hidden states from MATH-500 for topological sparsity analysis.

Extracts last-hidden-state vectors, layer-wise profiles, and token trajectories
from a causal LM on MATH problems grouped by difficulty level (1-5).

Designed for NVIDIA RTX 2060 Super (8GB VRAM):
  - Default: Qwen2.5-1.5B-Instruct (~3GB in float16)
  - Alt: Llama-3.2-1B-Instruct (~3GB in float16)

Usage:
    python scripts/extract_hidden_states.py
    python scripts/extract_hidden_states.py --model meta-llama/Llama-3.2-1B-Instruct
    python scripts/extract_hidden_states.py --output data/transformer/llama_hidden_states.npz
"""

import argparse
import sys
import time

import numpy as np
import torch


def load_math_dataset(seed: int = 42):
    """Load MATH problems and sample 100 per difficulty level.

    Tries hendrycks/competition_math first, then lighteval/MATH.
    Returns list of dicts with 'problem' and 'level' (int 1-5) keys.
    """
    from datasets import load_dataset

    ds = None
    source = None

    # Attempt 1: hendrycks/competition_math
    try:
        ds = load_dataset("hendrycks/competition_math", split="test")
        source = "hendrycks/competition_math"
        print(f"Loaded {len(ds)} problems from {source}")
    except Exception as e:
        print(f"Could not load hendrycks/competition_math: {e}")

    # Attempt 2: lighteval/MATH
    if ds is None:
        try:
            ds = load_dataset("lighteval/MATH", split="test")
            source = "lighteval/MATH"
            print(f"Loaded {len(ds)} problems from {source}")
        except Exception as e:
            print(f"Could not load lighteval/MATH: {e}")

    if ds is None:
        print(
            "\nERROR: Could not load MATH dataset from HuggingFace.\n"
            "Please download manually and provide as JSON lines with fields:\n"
            '  {"problem": "...", "level": "Level 3", "solution": "...", "type": "..."}\n'
            "Expected: ~5000 problems with levels 'Level 1' through 'Level 5'."
        )
        sys.exit(1)

    # Parse levels and group by difficulty
    rng = np.random.default_rng(seed)
    problems_by_level: dict[int, list] = {k: [] for k in range(1, 6)}

    for row in ds:
        level_str = row.get("level", "")
        if not level_str:
            continue
        try:
            level = int(level_str.replace("Level ", ""))
        except (ValueError, AttributeError):
            continue
        if 1 <= level <= 5:
            problems_by_level[level].append(
                {"problem": row["problem"], "level": level}
            )

    # Sample 100 per level
    sampled = []
    for level in range(1, 6):
        pool = problems_by_level[level]
        n_available = len(pool)
        n_sample = min(100, n_available)
        if n_available == 0:
            print(f"WARNING: No problems found for Level {level}")
            continue
        indices = rng.choice(n_available, size=n_sample, replace=False)
        selected = [pool[i] for i in indices]
        sampled.extend(selected)
        print(f"Level {level}: sampled {n_sample}/{n_available} problems")

    print(f"Total sampled: {len(sampled)} problems from {source}")
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden states from LLM on MATH-500"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        type=str,
        help="HuggingFace model ID (must fit in 8GB VRAM at float16)",
    )
    parser.add_argument(
        "--output",
        default="data/transformer/math500_hidden_states.npz",
        type=str,
        help="Output path for compressed numpy archive",
    )
    parser.add_argument(
        "--max_length",
        default=512,
        type=int,
        help="Truncate prompts longer than this to manage VRAM",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for sampling"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load dataset
    problems = load_math_dataset(seed=args.seed)
    if not problems:
        print("No problems loaded. Exiting.")
        sys.exit(1)

    # Load model and tokenizer
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

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM after model load: {allocated:.2f} GB")

    # Extraction loop
    all_last_hidden = []
    all_layer_hidden = []
    all_token_trajectories = []
    all_levels = []
    all_seq_lengths = []
    skipped = []

    t0 = time.time()

    for idx, problem in enumerate(problems):
        if idx % 50 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(problems) - idx) / rate if rate > 0 else 0
            print(
                f"Processing {idx}/{len(problems)} "
                f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]"
            )

        level = problem["level"]
        prompt = (
            "You are a helpful math assistant. Provide the final answer.\n\n"
            f"{problem['problem']}\n\nPlease provide the final answer."
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        seq_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                outputs = model(**inputs)

            # outputs.hidden_states: tuple of (n_layers+1) tensors, each (1, T, d)
            # Index 0 = embedding layer output, index -1 = final transformer layer
            hidden_states = outputs.hidden_states
            n_layers_plus_one = len(hidden_states)

            # Last hidden state at final token, final layer → (d,)
            final_layer_all_tokens = hidden_states[-1][0]  # (T, d)
            last_hidden = final_layer_all_tokens[-1].cpu().float().numpy()

            # All layers at final token → (n_layers+1, d)
            layer_states = (
                torch.stack(
                    [hidden_states[i][0, -1, :] for i in range(n_layers_plus_one)]
                )
                .cpu()
                .float()
                .numpy()
            )

            # Token trajectory at final layer → (T, d)
            token_traj = final_layer_all_tokens.cpu().float().numpy()

            all_last_hidden.append(last_hidden)
            all_layer_hidden.append(layer_states)
            all_token_trajectories.append(token_traj)
            all_levels.append(level)
            all_seq_lengths.append(seq_len)

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on problem {idx} (seq_len={seq_len}), skipping")
            skipped.append(idx)
            torch.cuda.empty_cache()
            continue

        # Free GPU cache every problem to stay within budget
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(
        f"\nDone. Extracted {len(all_last_hidden)}/{len(problems)} problems "
        f"in {elapsed:.1f}s. Skipped: {len(skipped)}"
    )

    # Validate shapes
    if len(all_last_hidden) == 0:
        print("ERROR: No hidden states extracted. Exiting.")
        sys.exit(1)

    d = all_last_hidden[0].shape[0]
    n_layers = all_layer_hidden[0].shape[0]
    print(f"Hidden dim: {d}, Layers (incl. embedding): {n_layers}")

    # Level distribution
    level_counts = {}
    for lv in all_levels:
        level_counts[lv] = level_counts.get(lv, 0) + 1
    for lv in sorted(level_counts):
        print(f"  Level {lv}: {level_counts[lv]} problems")

    # Save — token_trajectories are variable-length, store as object array
    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    np.savez_compressed(
        args.output,
        last_hidden_states=np.array(all_last_hidden),  # (N, d)
        difficulty_levels=np.array(all_levels),  # (N,)
        layer_hidden_states=np.array(all_layer_hidden),  # (N, L+1, d)
        token_trajectories=np.array(all_token_trajectories, dtype=object),  # (N,) of (T_i, d)
        seq_lengths=np.array(all_seq_lengths),  # (N,)
        model_name=np.array(args.model),
        skipped_indices=np.array(skipped),
        hidden_dim=np.array(d),
        num_layers=np.array(n_layers),
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
