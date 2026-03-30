"""2-column, 3-layer directed Aizawa attractor network.

Architecture:
    Layer 2 (source):     [C]
                         /   \\
    Layer 3 (receiver): [A3]  [B3]
                         |      |
    Layer 5 (receiver): [A5]  [B5]

Coupling is directed and asymmetric: C -> A3, C -> B3, A3 -> A5, B3 -> B5.
No upward or lateral coupling. Diffusive coupling on x,y components only.
Per-layer timescale separation via different dt values.
"""

import numpy as np

from att.config.seed import get_rng

NODE_NAMES = ("C", "A3", "B3", "A5", "B5")


def _aizawa_deriv(
    x: float,
    y: float,
    z: float,
    alpha: float = 0.95,
    beta: float = 0.7,
    gamma: float = 0.6,
    delta: float = 3.5,
    epsilon: float = 0.25,
    zeta: float = 0.1,
) -> tuple[float, float, float]:
    """Aizawa ODE right-hand side."""
    r2 = x * x + y * y
    dx = (z - beta) * x - delta * y
    dy = delta * x + (z - beta) * y
    dz = gamma + alpha * z - z**3 / 3 - r2 * (1 + epsilon * z) + zeta * z * x**3
    return dx, dy, dz


def layered_aizawa_network(
    n_steps: int = 80000,
    dt_layer2: float = 0.005,
    dt_layer3: float = 0.008,
    dt_layer5: float = 0.012,
    coupling_source: float = 0.15,
    coupling_down: float = 0.15,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Integrate a 2-column, 3-layer directed Aizawa network.

    Parameters
    ----------
    n_steps : number of integration steps
    dt_layer2 : timestep for source node C (fast, superficial)
    dt_layer3 : timestep for layer 3 receivers A3, B3
    dt_layer5 : timestep for layer 5 receivers A5, B5 (slow, deep)
    coupling_source : strength of C -> A3 and C -> B3 (xy diffusive)
    coupling_down : strength of A3 -> A5 and B3 -> B5 (xy diffusive)
    seed : random seed for initial conditions

    Returns
    -------
    dict mapping node name -> (n_steps, 3) trajectory array.
    Keys: 'C', 'A3', 'B3', 'A5', 'B5'.
    """
    rng = get_rng(seed)

    # Initialize each node with slightly different ICs
    states = {}
    for name in NODE_NAMES:
        states[name] = np.array([0.1, 0.0, 0.0]) + rng.normal(0, 0.01, 3)

    dt_map = {
        "C": dt_layer2,
        "A3": dt_layer3,
        "B3": dt_layer3,
        "A5": dt_layer5,
        "B5": dt_layer5,
    }

    # Pre-allocate output
    trajectories = {name: np.zeros((n_steps, 3)) for name in NODE_NAMES}

    for step in range(n_steps):
        # Record current state
        for name in NODE_NAMES:
            trajectories[name][step] = states[name]

        # Compute uncoupled derivatives
        derivs = {}
        for name in NODE_NAMES:
            derivs[name] = np.array(_aizawa_deriv(*states[name]))

        # Apply directed xy-coupling forces
        # C -> A3, C -> B3
        f_a3 = coupling_source * (states["C"][:2] - states["A3"][:2])
        f_b3 = coupling_source * (states["C"][:2] - states["B3"][:2])
        # A3 -> A5, B3 -> B5
        f_a5 = coupling_down * (states["A3"][:2] - states["A5"][:2])
        f_b5 = coupling_down * (states["B3"][:2] - states["B5"][:2])

        # Euler step with per-node dt + coupling on xy only
        # C: no coupling input (free-running source)
        states["C"] = states["C"] + derivs["C"] * dt_map["C"]

        for name, force in [("A3", f_a3), ("B3", f_b3), ("A5", f_a5), ("B5", f_b5)]:
            dt = dt_map[name]
            new_state = states[name] + derivs[name] * dt
            new_state[:2] += force * dt
            states[name] = new_state

    return trajectories


def layered_aizawa_network_symmetric(
    n_steps: int = 80000,
    dt_layer2: float = 0.005,
    dt_layer3: float = 0.008,
    dt_layer5: float = 0.012,
    coupling_source: float = 0.15,
    coupling_down: float = 0.15,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Symmetric (bidirectional, all-to-all) version for Experiment 5 control.

    Matches total coupling energy (Frobenius norm) with the directed version
    by scaling per-edge coupling: 4 directed edges -> 20 symmetric edges,
    so per-edge strength = original * sqrt(4/20).
    """
    rng = get_rng(seed)

    # Scale coupling to match Frobenius norm: directed has 4 edges, symmetric has 20
    scale = np.sqrt(4.0 / 20.0)
    sym_coupling = max(coupling_source, coupling_down) * scale

    states = {}
    for name in NODE_NAMES:
        states[name] = np.array([0.1, 0.0, 0.0]) + rng.normal(0, 0.01, 3)

    dt_map = {
        "C": dt_layer2,
        "A3": dt_layer3,
        "B3": dt_layer3,
        "A5": dt_layer5,
        "B5": dt_layer5,
    }

    trajectories = {name: np.zeros((n_steps, 3)) for name in NODE_NAMES}

    for step in range(n_steps):
        for name in NODE_NAMES:
            trajectories[name][step] = states[name]

        derivs = {}
        for name in NODE_NAMES:
            derivs[name] = np.array(_aizawa_deriv(*states[name]))

        # All-to-all symmetric xy coupling
        forces = {name: np.zeros(2) for name in NODE_NAMES}
        for i, ni in enumerate(NODE_NAMES):
            for j, nj in enumerate(NODE_NAMES):
                if i != j:
                    forces[ni] += sym_coupling * (states[nj][:2] - states[ni][:2])

        for name in NODE_NAMES:
            dt = dt_map[name]
            new_state = states[name] + derivs[name] * dt
            new_state[:2] += forces[name] * dt
            states[name] = new_state

    return trajectories
