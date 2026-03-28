"""YAML experiment configuration for reproducibility."""

from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    """Load experiment configuration from a YAML file.

    Supported keys: seed, embedding, topology, binding, benchmarks,
    transitions, surrogates, system, dataset, preprocessing.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        config = {}
    return config


def save_config(config: dict, path: str) -> None:
    """Save experiment parameters for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
