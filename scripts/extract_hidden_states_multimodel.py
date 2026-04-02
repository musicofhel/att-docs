"""Direction 6: Extract hidden states from multiple LLMs for cross-model replication.

Extracts hidden states from Phi-2, Pythia-1.4B, and StableLM-2-1.6B on a 250-problem
MATH subset (50/level) using the same prompt template as the Qwen baseline.

Usage:
    python scripts/extract_hidden_states_multimodel.py
    python scripts/extract_hidden_states_multimodel.py --models phi2 pythia stablelm
    python scripts/extract_hidden_states_multimodel.py --models phi2 --n-per-level 25
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "transformer")

MODEL_REGISTRY = {
    "phi2": {
        "hf_id": "microsoft/phi-2",
        "output": "phi2_hidden_states.npz",
        "trust_remote_code": True,
    },
    "pythia": {
        "hf_id": "EleutherAI/pythia-1.4b",
        "output": "pythia14b_hidden_states.npz",
        "trust_remote_code": False,
    },
    "stablelm": {
        "hf_id": "stabilityai/stablelm-2-1_6b",
        "output": "stablelm16b_hidden_states.npz",
        "trust_remote_code": False,
    },
}

PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Provide the final answer.\n\n"
    "{problem}\n\nPlease provide the final answer."
)


def load_math_subset(n_per_level: int = 50, seed: int = 42):
    """Load MATH problems and sample n_per_level per difficulty."""
    from datasets import load_dataset

    ds = None
    for name, loader in [
        ("hendrycks/competition_math", lambda: load_dataset("hendrycks/competition_math", split="test")),
        ("lighteval/MATH", lambda: load_dataset("lighteval/MATH", split="test")),
        ("HuggingFaceH4/MATH-500", lambda: load_dataset("HuggingFaceH4/MATH-500", split="test")),
    ]:
        try:
            ds = loader()
            print(f"Loaded {len(ds)} problems from {name}")
            break
        except Exception as e:
            print(f"Could not load {name}: {e}")

    if ds is None:
        print("ERROR: Could not load MATH dataset.")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    problems_by_level: dict[int, list] = {k: [] for k in range(1, 6)}

    for row in ds:
        level_raw = row.get("level", "")
        if not level_raw and level_raw != 0:
            continue
        try:
            level = int(level_raw) if isinstance(level_raw, int) else int(str(level_raw).replace("Level ", ""))
        except (ValueError, AttributeError):
            continue
        if 1 <= level <= 5:
            problems_by_level[level].append({"problem": row["problem"], "level": level})

    sampled = []
    for level in range(1, 6):
        pool = problems_by_level[level]
        n_sample = min(n_per_level, len(pool))
        if n_sample == 0:
            print(f"WARNING: No problems for Level {level}")
            continue
        indices = rng.choice(len(pool), size=n_sample, replace=False)
        sampled.extend([pool[i] for i in indices])
        print(f"Level {level}: sampled {n_sample}/{len(pool)}")

    print(f"Total: {len(sampled)} problems")
    return sampled


def extract_for_model(model_key: str, problems: list, max_length: int = 512, seed: int = 42):
    """Extract hidden states for a single model."""
    info = MODEL_REGISTRY[model_key]
    hf_id = info["hf_id"]
    output_path = os.path.join(DATA_DIR, info["output"])

    if os.path.exists(output_path):
        print(f"\n[{model_key}] Output already exists: {output_path} — skipping")
        return output_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"[{model_key}] Loading model: {hf_id}")
    print(f"[{model_key}] Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=info["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=info["trust_remote_code"],
    )
    model.eval()
    print(f"[{model_key}] Model loaded.")

    if device.type == "cuda":
        print(f"[{model_key}] VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    all_last_hidden = []
    all_layer_hidden = []
    all_token_trajectories = []
    all_levels = []
    all_seq_lengths = []
    skipped = []

    t0 = time.time()
    for idx, problem in enumerate(problems):
        if idx % 25 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(problems) - idx) / rate if rate > 0 else 0
            print(f"[{model_key}] {idx}/{len(problems)} [{elapsed:.0f}s, ~{remaining:.0f}s left]")

        prompt = PROMPT_TEMPLATE.format(problem=problem["problem"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        seq_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.hidden_states
            n_layers_plus_one = len(hidden_states)

            final_layer_all_tokens = hidden_states[-1][0]  # (T, d)
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

        except torch.cuda.OutOfMemoryError:
            print(f"[{model_key}] OOM on problem {idx} (seq_len={seq_len}), skipping")
            skipped.append(idx)
            torch.cuda.empty_cache()
            continue

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"[{model_key}] Done. {len(all_last_hidden)}/{len(problems)} in {elapsed:.1f}s. Skipped: {len(skipped)}")

    if len(all_last_hidden) == 0:
        print(f"[{model_key}] ERROR: No hidden states extracted.")
        return None

    d = all_last_hidden[0].shape[0]
    n_layers = all_layer_hidden[0].shape[0]
    print(f"[{model_key}] Hidden dim: {d}, Layers: {n_layers}")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(
        output_path,
        last_hidden_states=np.array(all_last_hidden),
        difficulty_levels=np.array(all_levels),
        layer_hidden_states=np.array(all_layer_hidden),
        token_trajectories=np.array(all_token_trajectories, dtype=object),
        seq_lengths=np.array(all_seq_lengths),
        model_name=np.array(hf_id),
        skipped_indices=np.array(skipped),
        hidden_dim=np.array(d),
        num_layers=np.array(n_layers),
    )
    print(f"[{model_key}] Saved: {output_path}")

    # Free GPU memory before next model
    del model, tokenizer
    torch.cuda.empty_cache()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Multi-model hidden state extraction")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to extract (default: all)",
    )
    parser.add_argument("--n-per-level", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Models: {args.models}")
    print(f"Problems per level: {args.n_per_level}")

    problems = load_math_subset(n_per_level=args.n_per_level, seed=args.seed)

    results = {}
    for model_key in args.models:
        path = extract_for_model(model_key, problems, max_length=args.max_length, seed=args.seed)
        results[model_key] = path

    print(f"\n{'='*60}")
    print("Summary:")
    for k, v in results.items():
        status = "OK" if v else "FAILED"
        print(f"  {k}: {status} — {v}")


if __name__ == "__main__":
    main()
