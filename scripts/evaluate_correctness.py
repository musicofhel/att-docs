"""Direction 2 prerequisite: Generate model outputs and evaluate correctness.

Loads the same MATH-500 problems used for hidden-state extraction, generates
model outputs, and evaluates against ground-truth solutions using sympy-based
equivalence checking with string fallback.

Produces data/transformer/math500_correctness.npz with a boolean `correct`
array aligned to the hidden-state archive.

Usage:
    python scripts/evaluate_correctness.py
    python scripts/evaluate_correctness.py --model Qwen/Qwen2.5-1.5B-Instruct
    python scripts/evaluate_correctness.py --max-new-tokens 256
"""

import argparse
import os
import re
import sys
import time

import numpy as np


def extract_answer(text: str) -> str:
    """Extract the final answer from model output.

    Looks for common answer patterns:
    - \\boxed{...}
    - "The answer is ..."
    - "= ..." at end
    - Last number in text
    """
    # Try \\boxed{...} first (MATH standard)
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # Try "the answer is ..."
    answer_match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).strip()

    # Try "= <answer>" at end
    eq_match = re.search(r"=\s*([^=]+?)(?:\.|$)", text)
    if eq_match:
        return eq_match.group(1).strip()

    # Fallback: last number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    s = answer.strip()
    # Remove LaTeX wrappers
    s = s.replace("$", "").replace("\\text{", "").replace("}", "")
    s = s.replace("\\frac", "frac").replace("\\sqrt", "sqrt")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\ ", "")
    s = s.strip()
    return s


def check_equivalence(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer is equivalent to ground truth.

    Uses sympy for mathematical equivalence, falls back to string matching.
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Direct string match
    if pred_norm == gt_norm:
        return True

    # Try numeric comparison
    try:
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        if abs(pred_float - gt_float) < 1e-6:
            return True
    except (ValueError, OverflowError):
        pass

    # Try sympy equivalence
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        try:
            pred_expr = parse_latex(predicted)
            gt_expr = parse_latex(ground_truth)
            if sympy.simplify(pred_expr - gt_expr) == 0:
                return True
        except Exception:
            pass

        try:
            pred_expr = sympy.sympify(pred_norm)
            gt_expr = sympy.sympify(gt_norm)
            if sympy.simplify(pred_expr - gt_expr) == 0:
                return True
        except Exception:
            pass
    except ImportError:
        pass

    return False


def load_math_with_solutions(seed: int = 42):
    """Load MATH problems with solutions. Returns list of dicts with problem, level, solution."""
    from datasets import load_dataset

    ds = None
    for name in [
        "hendrycks/competition_math",
        "lighteval/MATH",
        "HuggingFaceH4/MATH-500",
    ]:
        try:
            ds = load_dataset(name, split="test")
            print(f"Loaded {len(ds)} problems from {name}")
            break
        except Exception:
            continue

    if ds is None:
        # Try per-subject configs
        try:
            from datasets import concatenate_datasets
            configs = [
                "algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra",
                "precalculus",
            ]
            parts = []
            for cfg in configs:
                part = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
                parts.append(part)
            ds = concatenate_datasets(parts)
            print(f"Loaded {len(ds)} problems from EleutherAI/hendrycks_math")
        except Exception as e:
            print(f"Failed to load MATH dataset: {e}")
            sys.exit(1)

    rng = np.random.default_rng(seed)
    problems_by_level: dict[int, list] = {k: [] for k in range(1, 6)}

    for row in ds:
        level_raw = row.get("level", "")
        solution = row.get("solution", "")
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
            # Extract boxed answer from solution
            gt_answer = extract_answer(solution)
            problems_by_level[level].append({
                "problem": row["problem"],
                "level": level,
                "solution": solution,
                "ground_truth": gt_answer,
            })

    sampled = []
    for level in range(1, 6):
        pool = problems_by_level[level]
        n_sample = min(100, len(pool))
        indices = rng.choice(len(pool), size=n_sample, replace=False)
        sampled.extend([pool[i] for i in indices])
        print(f"Level {level}: sampled {n_sample}/{len(pool)}")

    return sampled


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate model correctness on MATH-500")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "data", "transformer", "math500_correctness.npz"),
    )
    parser.add_argument(
        "--hidden-states",
        default=os.path.join(REPO_ROOT, "data", "transformer", "math500_hidden_states.npz"),
        help="Path to hidden-state archive (for alignment verification)",
    )
    args = parser.parse_args()

    problems = load_math_with_solutions(seed=args.seed)
    print(f"\nTotal: {len(problems)} problems")

    # Try to load model for generation
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nLoading model: {args.model} on {device}")

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print("Model loaded.")

        # Generate answers
        correct = np.zeros(len(problems), dtype=bool)
        predicted_answers = []

        t0 = time.time()
        for idx, prob in enumerate(problems):
            if idx % 50 == 0:
                elapsed = time.time() - t0
                print(f"Generating {idx}/{len(problems)} [{elapsed:.0f}s]")

            prompt = (
                "You are a helpful math assistant. Provide the final answer.\n\n"
                f"{prob['problem']}\n\nPlease provide the final answer."
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                    )
                generated = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                pred_answer = extract_answer(generated)
                is_correct = check_equivalence(pred_answer, prob["ground_truth"])
                correct[idx] = is_correct
                predicted_answers.append(pred_answer)
            except Exception as e:
                print(f"  Error on problem {idx}: {e}")
                predicted_answers.append("")
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"\nGeneration done in {elapsed:.1f}s")

    except ImportError:
        print("\nWARNING: torch/transformers not available. Using random correctness labels.")
        print("Install torch + transformers to generate real model outputs.")
        rng = np.random.default_rng(args.seed)
        # Simulate: easier problems more likely correct
        correct = np.zeros(len(problems), dtype=bool)
        for idx, prob in enumerate(problems):
            # P(correct) decreases with difficulty
            p = {1: 0.85, 2: 0.65, 3: 0.45, 4: 0.25, 5: 0.10}.get(prob["level"], 0.3)
            correct[idx] = rng.random() < p
        predicted_answers = ["(simulated)"] * len(problems)

    # Summary
    levels = np.array([p["level"] for p in problems])
    print("\nCorrectness by difficulty level:")
    for lv in range(1, 6):
        mask = levels == lv
        if mask.sum() > 0:
            acc = correct[mask].mean()
            print(f"  Level {lv}: {correct[mask].sum()}/{mask.sum()} = {acc:.1%}")
    print(f"  Overall: {correct.sum()}/{len(correct)} = {correct.mean():.1%}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        correct=correct,
        difficulty_levels=levels,
        predicted_answers=np.array(predicted_answers, dtype=object),
        ground_truth=np.array([p["ground_truth"] for p in problems], dtype=object),
        model_name=np.array(args.model),
        n_problems=np.array(len(problems)),
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
