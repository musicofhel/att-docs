"""Direction 10 prerequisite: Extract attention-derived persistence diagrams.

Computes PH on attention distance matrices (1 - attention_weight) per problem
per layer. Stores persistence diagrams (not raw attention matrices — too large).

Focuses on terminal 5 layers where topological signal is strongest.

Usage:
    python scripts/extract_attention_weights.py
    python scripts/extract_attention_weights.py --model Qwen/Qwen2.5-1.5B-Instruct --n-per-level 25
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from att.topology.persistence import PersistenceAnalyzer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "transformer")

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
            continue
        indices = rng.choice(len(pool), size=n_sample, replace=False)
        sampled.extend([pool[i] for i in indices])
        print(f"Level {level}: sampled {n_sample}/{len(pool)}")

    print(f"Total: {len(sampled)} problems")
    return sampled


def attention_to_distance(attn_matrix: np.ndarray) -> np.ndarray:
    """Convert attention matrix to distance matrix: D = 1 - (A + A^T) / 2.

    Symmetrizes attention weights and converts to distance. Clips to [0, 1].
    """
    sym = (attn_matrix + attn_matrix.T) / 2.0
    dist = 1.0 - sym
    np.clip(dist, 0.0, 1.0, out=dist)
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_attention_ph(attn_matrix: np.ndarray, max_dim: int = 1, subsample: int = 64) -> dict:
    """Compute PH on attention distance matrix.

    Parameters
    ----------
    attn_matrix : (T, T) attention weight matrix (single head or head-averaged).
    max_dim : max homology dimension.
    subsample : max tokens to subsample (attention matrices can be large).

    Returns
    -------
    dict with 'diagrams' (list of (n, 2) arrays per dim) and 'entropy' (dict).
    """
    n = attn_matrix.shape[0]

    # Subsample if needed
    if n > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=subsample, replace=False)
        idx = np.sort(idx)
        attn_matrix = attn_matrix[np.ix_(idx, idx)]

    dist = attention_to_distance(attn_matrix)

    pa = PersistenceAnalyzer(max_dim=max_dim, backend="ripser", metric="precomputed")
    result = pa.fit_transform(dist)
    return {
        "diagrams": [dgm.tolist() for dgm in result["diagrams"]],
        "persistence_entropy": result["persistence_entropy"],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract attention-derived PH diagrams")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-per-level", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--n-terminal-layers", type=int, default=5)
    parser.add_argument("--max-dim", type=int, default=1)
    parser.add_argument("--subsample", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    problems = load_math_subset(n_per_level=args.n_per_level, seed=args.seed)

    print(f"\nLoading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # NOTE: Must use float32 + eager attention. float16 produces all-NaN
    # attention weights because the softmax underflows in half precision.
    # SDPA/flash attention also returns NaN for output_attentions.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto",
        output_attentions=True,
        output_hidden_states=False,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    # Determine which layers to extract attention from
    n_layers = model.config.num_hidden_layers
    terminal_layers = list(range(max(0, n_layers - args.n_terminal_layers), n_layers))
    print(f"Total layers: {n_layers}, extracting attention from: {terminal_layers}")

    # Results storage
    all_attention_ph = []  # list of dicts per problem
    all_levels = []
    skipped = []

    t0 = time.time()
    for idx, problem in enumerate(problems):
        if idx % 25 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(problems) - idx) / rate if rate > 0 else 0
            print(f"Processing {idx}/{len(problems)} [{elapsed:.0f}s, ~{remaining:.0f}s left]")

        prompt = PROMPT_TEMPLATE.format(problem=problem["problem"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)

        try:
            with torch.no_grad():
                outputs = model(**inputs)

            # outputs.attentions: tuple of n_layers tensors, each (1, n_heads, T, T)
            attentions = outputs.attentions

            problem_ph = {}
            for layer_idx in terminal_layers:
                # Average across attention heads → (T, T)
                attn = attentions[layer_idx][0].mean(dim=0).cpu().float().numpy()
                ph = compute_attention_ph(attn, max_dim=args.max_dim, subsample=args.subsample)
                problem_ph[layer_idx] = ph

            all_attention_ph.append(problem_ph)
            all_levels.append(problem["level"])

        except torch.cuda.OutOfMemoryError:
            print(f"OOM on problem {idx}, skipping")
            skipped.append(idx)
            torch.cuda.empty_cache()
            continue

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nDone. {len(all_attention_ph)}/{len(problems)} in {elapsed:.1f}s. Skipped: {len(skipped)}")

    if len(all_attention_ph) == 0:
        print("ERROR: No attention data extracted.")
        sys.exit(1)

    # Save as .npz
    output_path = args.output or os.path.join(DATA_DIR, "attention_ph_diagrams.npz")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    np.savez_compressed(
        output_path,
        attention_ph=np.array(all_attention_ph, dtype=object),
        difficulty_levels=np.array(all_levels),
        terminal_layers=np.array(terminal_layers),
        model_name=np.array(args.model),
        n_layers=np.array(n_layers),
        config=np.array({
            "max_dim": args.max_dim,
            "subsample": args.subsample,
            "n_terminal_layers": args.n_terminal_layers,
        }),
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
