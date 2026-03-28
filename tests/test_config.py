"""Tests for att.config — seed management and YAML configs."""

import os
import tempfile
import numpy as np
import pytest

from att.config import set_seed, get_rng, load_config, save_config


class TestSetSeed:
    def test_deterministic_numpy(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_rng(self):
        set_seed(42)
        rng1 = get_rng()
        a = rng1.random(10)
        set_seed(42)
        rng2 = get_rng()
        b = rng2.random(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(99)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)

    def test_get_rng_with_explicit_seed(self):
        rng1 = get_rng(seed=123)
        a = rng1.random(5)
        rng2 = get_rng(seed=123)
        b = rng2.random(5)
        np.testing.assert_array_equal(a, b)


class TestConfig:
    def test_round_trip(self):
        config = {
            "seed": 42,
            "embedding": {"delay": "auto", "dimension": "auto"},
            "topology": {"max_dim": 2},
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            save_config(config, path)
            loaded = load_config(path)
            assert loaded == config
        finally:
            os.unlink(path)

    def test_load_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("")
            path = f.name

        try:
            config = load_config(path)
            assert config == {}
        finally:
            os.unlink(path)

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "nested", "config.yaml")
            save_config({"key": "value"}, path)
            loaded = load_config(path)
            assert loaded == {"key": "value"}
