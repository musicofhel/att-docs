# ATT Interactive Demo

A Streamlit app for exploring attractor topology, binding detection, and coupling sweeps using the Attractor Topology Toolkit.

## Setup

```bash
# From the repository root:
pip install -e ".[demo]"
```

## Run

```bash
streamlit run demo/app.py
```

The app opens at `http://localhost:8501` with three pages:

1. **Attractor Explorer** -- Generate Lorenz, Rossler, or coupled Lorenz attractors, view them in 3D, and inspect their persistence diagrams.

2. **Binding Detection** -- Adjust coupling between two Lorenz systems and see the binding score, persistence images (marginal X, marginal Y, joint, residual), and run a surrogate significance test.

3. **Coupling Sweep** -- Sweep coupling strength from 0 to 1 and plot the binding score curve, optionally compared with transfer entropy.

## Defaults

The demo uses conservative defaults for interactive speed:

- `n_steps = 5000` (time series length)
- `subsample = 300` (points sent to Ripser)
- `n_surrogates = 19` (for significance testing)
- `set_seed(42)` for reproducibility

Increase these for publication-quality results.
