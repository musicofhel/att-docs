# Contributing to ATT

## Development Setup

```bash
git clone https://github.com/musicofhel/att-docs.git
cd att-docs
pip install -e ".[dev]"
```

## Running Tests

```bash
# Fast suite (~2 min, CI default)
pytest tests/ -k "not slow and not witness and not changepoints"

# Full suite including validation experiments (~30 min)
pytest tests/

# Single module
pytest tests/test_binding.py -v
```

## Code Style

- Formatter/linter: [ruff](https://docs.astral.sh/ruff/) (line-length 100, Python 3.10 target)
- Docstrings: NumPy style
- Check before committing: `ruff check .`

## Adding a New Coupling Method

Use the `register_method()` plugin interface:

```python
from att.benchmarks import CouplingBenchmark

def my_method(x, y, **kwargs):
    """Return a scalar coupling score."""
    ...
    return score

benchmark = CouplingBenchmark()
benchmark.register_method("my_method", my_method)
results = benchmark.sweep(generator, coupling_values=[0.0, 0.1, 0.5, 1.0])
```

See `att/benchmarks/` for existing implementations (TE, PAC, CRQA).

## Adding a New Synthetic System

Add a generator function to `att/synthetic/generators.py`:

```python
def my_system(n_steps=10000, dt=0.01, seed=None, **params):
    """Generate time series from my system.

    Parameters
    ----------
    n_steps : int
        Number of integration steps.
    dt : float
        Integration timestep.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Time series of shape (n_steps, n_vars).
    """
    ...
```

All generators must accept `seed` for reproducibility.

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Run `ruff check .` and `pytest tests/ -k "not slow"`
4. Open a PR against `master`
