"""Direction 8: Token-position-resolved topological analysis.

Partitions token positions into functional regions (instruction, problem,
closing instruction, answer) so PH can be computed per-region. This enables
testing whether specific parts of the input carry more topological signal
about difficulty.

The prompt template is fixed (from extract_hidden_states.py):
    "You are a helpful math assistant. Provide the final answer.\\n\\n"
    "{problem}\\n\\n"
    "Please provide the final answer."
"""

from __future__ import annotations

import re

import numpy as np


# Fixed prompt template components from extract_hidden_states.py
INSTRUCTION_PREFIX = "You are a helpful math assistant. Provide the final answer.\n\n"
INSTRUCTION_SUFFIX = "\n\nPlease provide the final answer."


class TokenPartitioner:
    """Partition token positions into functional regions.

    Regions:
        - instruction_prefix: system instruction before the problem
        - problem: the math problem text
        - instruction_suffix: closing instruction after the problem
        - operator: tokens within the problem that are math operators/symbols
        - numeric: tokens within the problem that are numbers

    Parameters
    ----------
    tokenizer : optional
        A HuggingFace tokenizer for accurate token-level partitioning.
        If None, uses character-length-based approximation.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def partition(
        self, problem_text: str, seq_length: int
    ) -> dict[str, np.ndarray]:
        """Partition token indices into functional regions.

        Parameters
        ----------
        problem_text : str
            The math problem text (without instruction wrapping).
        seq_length : int
            Total sequence length after tokenization.

        Returns
        -------
        dict mapping region name -> array of token indices.
        """
        if self.tokenizer is not None:
            return self._partition_with_tokenizer(problem_text, seq_length)
        return self._partition_by_char_ratio(problem_text, seq_length)

    def _partition_with_tokenizer(
        self, problem_text: str, seq_length: int
    ) -> dict[str, np.ndarray]:
        """Exact partition using tokenizer offsets."""
        full_prompt = INSTRUCTION_PREFIX + problem_text + INSTRUCTION_SUFFIX

        encoded = self.tokenizer(
            full_prompt, truncation=True, max_length=seq_length, return_offsets_mapping=True
        )
        offsets = encoded.get("offset_mapping", [])
        n_tokens = min(len(encoded["input_ids"]), seq_length)

        prefix_end = len(INSTRUCTION_PREFIX)
        problem_end = prefix_end + len(problem_text)

        regions: dict[str, list[int]] = {
            "instruction_prefix": [],
            "problem": [],
            "instruction_suffix": [],
        }

        for idx in range(n_tokens):
            if idx >= len(offsets):
                break
            start, end = offsets[idx]
            if start == 0 and end == 0:
                # Special tokens — assign to instruction_prefix
                regions["instruction_prefix"].append(idx)
            elif end <= prefix_end:
                regions["instruction_prefix"].append(idx)
            elif start >= problem_end:
                regions["instruction_suffix"].append(idx)
            else:
                regions["problem"].append(idx)

        # Sub-partition problem tokens into operator/numeric
        problem_indices = regions["problem"]
        operator_indices, numeric_indices = self._classify_problem_tokens(
            problem_text, problem_indices, offsets, prefix_end
        )
        regions["operator"] = operator_indices
        regions["numeric"] = numeric_indices

        return {k: np.array(v, dtype=np.intp) for k, v in regions.items()}

    def _partition_by_char_ratio(
        self, problem_text: str, seq_length: int
    ) -> dict[str, np.ndarray]:
        """Approximate partition using character-length ratios.

        Assumes roughly uniform characters-per-token ratio.
        """
        full_prompt = INSTRUCTION_PREFIX + problem_text + INSTRUCTION_SUFFIX
        total_chars = len(full_prompt)

        if total_chars == 0:
            return {
                "instruction_prefix": np.array([], dtype=np.intp),
                "problem": np.arange(seq_length, dtype=np.intp),
                "instruction_suffix": np.array([], dtype=np.intp),
                "operator": np.array([], dtype=np.intp),
                "numeric": np.array([], dtype=np.intp),
            }

        prefix_frac = len(INSTRUCTION_PREFIX) / total_chars
        problem_frac = len(problem_text) / total_chars

        prefix_tokens = max(1, int(round(prefix_frac * seq_length)))
        problem_tokens = max(1, int(round(problem_frac * seq_length)))
        # Clamp so we don't exceed seq_length
        if prefix_tokens + problem_tokens > seq_length:
            problem_tokens = seq_length - prefix_tokens
        suffix_tokens = seq_length - prefix_tokens - problem_tokens

        prefix_idx = np.arange(0, prefix_tokens, dtype=np.intp)
        problem_idx = np.arange(prefix_tokens, prefix_tokens + problem_tokens, dtype=np.intp)
        suffix_idx = np.arange(prefix_tokens + problem_tokens, seq_length, dtype=np.intp)

        # Sub-partition problem tokens using character analysis
        operator_idx, numeric_idx = self._classify_problem_tokens_approx(
            problem_text, problem_idx
        )

        return {
            "instruction_prefix": prefix_idx,
            "problem": problem_idx,
            "instruction_suffix": suffix_idx,
            "operator": operator_idx,
            "numeric": numeric_idx,
        }

    def _classify_problem_tokens(
        self,
        problem_text: str,
        problem_indices: list[int],
        offsets: list[tuple[int, int]],
        prefix_end: int,
    ) -> tuple[list[int], list[int]]:
        """Classify problem tokens as operator or numeric using token text."""
        operator_pattern = re.compile(r"^[\+\-\*/=<>\^%\(\)\[\]\{\}|\\]+$")
        numeric_pattern = re.compile(r"^[\d\.\,]+$")

        operator_indices = []
        numeric_indices = []

        for idx in problem_indices:
            if idx >= len(offsets):
                continue
            start, end = offsets[idx]
            # Get character span relative to problem text
            prob_start = start - prefix_end
            prob_end = end - prefix_end
            if prob_start < 0 or prob_end > len(problem_text):
                continue
            token_text = problem_text[prob_start:prob_end].strip()
            if not token_text:
                continue
            if operator_pattern.match(token_text):
                operator_indices.append(idx)
            elif numeric_pattern.match(token_text):
                numeric_indices.append(idx)

        return operator_indices, numeric_indices

    def _classify_problem_tokens_approx(
        self, problem_text: str, problem_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate operator/numeric classification by character ratios."""
        if len(problem_text) == 0 or len(problem_indices) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

        n_tokens = len(problem_indices)
        chars_per_token = max(1, len(problem_text) / n_tokens)

        operator_pattern = re.compile(r"[\+\-\*/=<>\^%\(\)\[\]\{\}|\\]")
        numeric_pattern = re.compile(r"\d")

        operator_idx = []
        numeric_idx = []

        for i, tok_idx in enumerate(problem_indices):
            char_start = int(i * chars_per_token)
            char_end = min(int((i + 1) * chars_per_token), len(problem_text))
            chunk = problem_text[char_start:char_end]
            if not chunk.strip():
                continue
            # Classify by majority character type
            n_op = len(operator_pattern.findall(chunk))
            n_num = len(numeric_pattern.findall(chunk))
            n_total = len(chunk.strip())
            if n_total > 0:
                if n_op / n_total > 0.5:
                    operator_idx.append(tok_idx)
                elif n_num / n_total > 0.5:
                    numeric_idx.append(tok_idx)

        return np.array(operator_idx, dtype=np.intp), np.array(numeric_idx, dtype=np.intp)

    def partition_batch(
        self,
        problem_texts: list[str],
        seq_lengths: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        """Partition a batch of problems.

        Parameters
        ----------
        problem_texts : list of str
            Problem texts (without instruction wrapping).
        seq_lengths : array of int
            Sequence lengths per problem.

        Returns
        -------
        list of partition dicts.
        """
        return [
            self.partition(text, int(sl))
            for text, sl in zip(problem_texts, seq_lengths)
        ]

    @staticmethod
    def validate_partition(
        partition: dict[str, np.ndarray], seq_length: int
    ) -> bool:
        """Check that instruction_prefix + problem + instruction_suffix covers all indices exactly once."""
        main_regions = ["instruction_prefix", "problem", "instruction_suffix"]
        all_indices = np.concatenate([partition[r] for r in main_regions if r in partition])
        all_indices = np.sort(all_indices)

        expected = np.arange(seq_length, dtype=np.intp)
        return np.array_equal(all_indices, expected)
