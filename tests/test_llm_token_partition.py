"""Tests for att.llm.token_partition — token position partitioning."""

import numpy as np
import pytest

from att.llm.token_partition import (
    INSTRUCTION_PREFIX,
    INSTRUCTION_SUFFIX,
    TokenPartitioner,
)


# --- Approximate (char-ratio) partition tests ---

class TestTokenPartitionerApprox:
    """Tests using character-ratio approximation (no tokenizer)."""

    def test_partition_covers_all_indices(self):
        tp = TokenPartitioner()
        problem = "What is 2 + 3?"
        seq_len = 20
        parts = tp.partition(problem, seq_len)
        assert TokenPartitioner.validate_partition(parts, seq_len)

    def test_main_regions_non_overlapping(self):
        tp = TokenPartitioner()
        problem = "Solve x^2 = 4"
        seq_len = 25
        parts = tp.partition(problem, seq_len)
        prefix = set(parts["instruction_prefix"].tolist())
        prob = set(parts["problem"].tolist())
        suffix = set(parts["instruction_suffix"].tolist())
        assert len(prefix & prob) == 0
        assert len(prefix & suffix) == 0
        assert len(prob & suffix) == 0

    def test_operator_numeric_subsets_of_problem(self):
        tp = TokenPartitioner()
        problem = "Calculate 123 + 456 * 789"
        seq_len = 30
        parts = tp.partition(problem, seq_len)
        problem_set = set(parts["problem"].tolist())
        operator_set = set(parts["operator"].tolist())
        numeric_set = set(parts["numeric"].tolist())
        assert operator_set.issubset(problem_set)
        assert numeric_set.issubset(problem_set)

    def test_operator_numeric_non_overlapping(self):
        tp = TokenPartitioner()
        problem = "2 + 3 * 5 = 25"
        seq_len = 25
        parts = tp.partition(problem, seq_len)
        operator_set = set(parts["operator"].tolist())
        numeric_set = set(parts["numeric"].tolist())
        assert len(operator_set & numeric_set) == 0

    def test_empty_problem(self):
        tp = TokenPartitioner()
        parts = tp.partition("", 10)
        # Should still cover all indices
        total = sum(len(parts[r]) for r in ["instruction_prefix", "problem", "instruction_suffix"])
        assert total == 10

    def test_very_short_sequence(self):
        tp = TokenPartitioner()
        parts = tp.partition("x=1", 3)
        assert TokenPartitioner.validate_partition(parts, 3)

    def test_long_problem_text(self):
        tp = TokenPartitioner()
        problem = "Find the value of x where " * 20 + "x = 42"
        seq_len = 150
        parts = tp.partition(problem, seq_len)
        assert TokenPartitioner.validate_partition(parts, seq_len)
        # Problem region should be largest
        assert len(parts["problem"]) > len(parts["instruction_prefix"])

    def test_regions_are_contiguous(self):
        tp = TokenPartitioner()
        problem = "What is the derivative of x^3?"
        seq_len = 30
        parts = tp.partition(problem, seq_len)
        # Each main region should be a contiguous range
        for region in ["instruction_prefix", "problem", "instruction_suffix"]:
            indices = parts[region]
            if len(indices) > 1:
                diffs = np.diff(indices)
                assert np.all(diffs == 1), f"{region} is not contiguous"


class TestTokenPartitionerBatch:
    def test_batch_length(self):
        tp = TokenPartitioner()
        problems = ["2+3", "x^2=4", "Find the area"]
        seq_lengths = np.array([15, 20, 25])
        results = tp.partition_batch(problems, seq_lengths)
        assert len(results) == 3

    def test_batch_individual_validity(self):
        tp = TokenPartitioner()
        problems = ["2+3", "x^2=4"]
        seq_lengths = np.array([12, 18])
        results = tp.partition_batch(problems, seq_lengths)
        for parts, sl in zip(results, seq_lengths):
            assert TokenPartitioner.validate_partition(parts, int(sl))


class TestValidatePartition:
    def test_valid_partition(self):
        parts = {
            "instruction_prefix": np.array([0, 1, 2]),
            "problem": np.array([3, 4, 5, 6]),
            "instruction_suffix": np.array([7, 8, 9]),
        }
        assert TokenPartitioner.validate_partition(parts, 10)

    def test_invalid_missing_indices(self):
        parts = {
            "instruction_prefix": np.array([0, 1]),
            "problem": np.array([3, 4]),  # missing 2
            "instruction_suffix": np.array([5]),
        }
        assert not TokenPartitioner.validate_partition(parts, 6)

    def test_invalid_duplicate_indices(self):
        parts = {
            "instruction_prefix": np.array([0, 1, 2]),
            "problem": np.array([2, 3, 4]),  # 2 is duplicated
            "instruction_suffix": np.array([5]),
        }
        assert not TokenPartitioner.validate_partition(parts, 6)


class TestPromptTemplate:
    """Verify the prompt template matches extract_hidden_states.py."""

    def test_prefix_content(self):
        assert INSTRUCTION_PREFIX == "You are a helpful math assistant. Provide the final answer.\n\n"

    def test_suffix_content(self):
        assert INSTRUCTION_SUFFIX == "\n\nPlease provide the final answer."

    def test_full_prompt_reconstruction(self):
        problem = "What is 2+2?"
        full = INSTRUCTION_PREFIX + problem + INSTRUCTION_SUFFIX
        expected = (
            "You are a helpful math assistant. Provide the final answer.\n\n"
            "What is 2+2?\n\n"
            "Please provide the final answer."
        )
        assert full == expected
