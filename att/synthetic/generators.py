"""Chaotic system generators for validation and benchmarking.

All generators use RK4 integration and accept an optional seed parameter.
If seed is None, uses the global seed state from set_seed().
"""

import numpy as np
from scipy.integrate import solve_ivp

from att.config.seed import get_rng


def lorenz_system(
    n_steps: int = 10000,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    initial: np.ndarray | None = None,
    noise: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a Lorenz attractor trajectory.

    Returns: (n_steps, 3) array.
    """
    rng = get_rng(seed)
    if initial is None:
        initial = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)

    def deriv(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t_span = (0, n_steps * dt)
    t_eval = np.linspace(0, n_steps * dt, n_steps)
    sol = solve_ivp(deriv, t_span, initial, t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12)
    result = sol.y.T  # (n_steps, 3)

    if noise > 0:
        result += rng.normal(0, noise, result.shape)

    return result


def rossler_system(
    n_steps: int = 10000,
    dt: float = 0.01,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    initial: np.ndarray | None = None,
    noise: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a Rössler attractor trajectory.

    Returns: (n_steps, 3) array.
    """
    rng = get_rng(seed)
    if initial is None:
        initial = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)

    def deriv(t, state):
        x, y, z = state
        return [-(y + z), x + a * y, b + z * (x - c)]

    t_span = (0, n_steps * dt)
    t_eval = np.linspace(0, n_steps * dt, n_steps)
    sol = solve_ivp(deriv, t_span, initial, t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12)
    result = sol.y.T

    if noise > 0:
        result += rng.normal(0, noise, result.shape)

    return result


def coupled_lorenz(
    n_steps: int = 10000,
    dt: float = 0.01,
    coupling: float = 0.1,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two Lorenz systems with diffusive coupling on the x-variable.

    coupling=0 → independent. coupling→1 → synchronization.
    Returns: (ts_x, ts_y), each (n_steps, 3).
    """
    rng = get_rng(seed)
    init_x = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)
    init_y = np.array([-1.0, -1.0, 1.0]) + rng.normal(0, 0.01, 3)
    initial = np.concatenate([init_x, init_y])

    def deriv(t, state):
        x1, y1, z1, x2, y2, z2 = state
        dx1 = sigma * (y1 - x1) + coupling * (x2 - x1)
        dy1 = x1 * (rho - z1) - y1
        dz1 = x1 * y1 - beta * z1
        dx2 = sigma * (y2 - x2) + coupling * (x1 - x2)
        dy2 = x2 * (rho - z2) - y2
        dz2 = x2 * y2 - beta * z2
        return [dx1, dy1, dz1, dx2, dy2, dz2]

    t_span = (0, n_steps * dt)
    t_eval = np.linspace(0, n_steps * dt, n_steps)
    sol = solve_ivp(deriv, t_span, initial, t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12)
    result = sol.y.T  # (n_steps, 6)

    return result[:, :3], result[:, 3:]


def coupled_rossler_lorenz(
    n_steps: int = 10000,
    dt: float = 0.01,
    coupling: float = 0.1,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rössler coupled to Lorenz — different intrinsic timescales.

    Tests per-channel delay handling in JointEmbedder.
    Returns: (ts_rossler, ts_lorenz).
    """
    rng = get_rng(seed)
    init_r = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)
    init_l = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)
    initial = np.concatenate([init_r, init_l])

    def deriv(t, state):
        rx, ry, rz, lx, ly, lz = state
        # Rössler with coupling from Lorenz x
        drx = -(ry + rz) + coupling * (lx - rx)
        dry = rx + a * ry
        drz = b + rz * (rx - c)
        # Lorenz with coupling from Rössler x
        dlx = sigma * (ly - lx) + coupling * (rx - lx)
        dly = lx * (rho - lz) - ly
        dlz = lx * ly - beta * lz
        return [drx, dry, drz, dlx, dly, dlz]

    t_span = (0, n_steps * dt)
    t_eval = np.linspace(0, n_steps * dt, n_steps)
    sol = solve_ivp(deriv, t_span, initial, t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12)
    result = sol.y.T

    return result[:, :3], result[:, 3:]


def switching_rossler(
    n_steps: int = 20000,
    dt: float = 0.01,
    switch_every: int = 5000,
    seed: int | None = None,
) -> np.ndarray:
    """Rössler with parameter switches at known intervals.

    Alternates between two parameter regimes (c=5.7 and c=18.0) to create
    ground truth attractor transitions.

    Returns: (n_steps, 3) array.
    """
    rng = get_rng(seed)
    initial = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)

    a, b = 0.2, 0.2
    c_values = [5.7, 18.0]

    result = np.zeros((n_steps, 3))
    state = initial.copy()

    for i in range(n_steps):
        regime = (i // switch_every) % 2
        c = c_values[regime]

        x, y, z = state
        dx = -(y + z)
        dy = x + a * y
        dz = b + z * (x - c)

        # RK4 step
        k1 = np.array([dx, dy, dz])

        x2, y2, z2 = state + 0.5 * dt * k1
        k2 = np.array([-(y2 + z2), x2 + a * y2, b + z2 * (x2 - c)])

        x3, y3, z3 = state + 0.5 * dt * k2
        k3 = np.array([-(y3 + z3), x3 + a * y3, b + z3 * (x3 - c)])

        x4, y4, z4 = state + dt * k3
        k4 = np.array([-(y4 + z4), x4 + a * y4, b + z4 * (x4 - c)])

        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        result[i] = state

    return result


def coupled_oscillators(
    n_oscillators: int = 3,
    coupling_matrix: np.ndarray | None = None,
    n_steps: int = 10000,
    dt: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """Multiple coupled Rössler oscillators with configurable coupling.

    Returns: (n_steps, n_oscillators, 3) array.
    """
    rng = get_rng(seed)

    if coupling_matrix is None:
        coupling_matrix = 0.1 * np.ones((n_oscillators, n_oscillators))
        np.fill_diagonal(coupling_matrix, 0)

    # Slightly different parameters per oscillator for heterogeneity
    c_values = 5.7 + rng.normal(0, 0.3, n_oscillators)
    a, b = 0.2, 0.2

    state = rng.normal(0, 1, (n_oscillators, 3))
    result = np.zeros((n_steps, n_oscillators, 3))

    for step in range(n_steps):
        result[step] = state

        derivs = np.zeros_like(state)
        for i in range(n_oscillators):
            x, y, z = state[i]
            derivs[i, 0] = -(y + z)
            derivs[i, 1] = x + a * y
            derivs[i, 2] = b + z * (x - c_values[i])

            # Diffusive coupling on x
            for j in range(n_oscillators):
                if i != j:
                    derivs[i, 0] += coupling_matrix[i, j] * (state[j, 0] - state[i, 0])

        state = state + dt * derivs

    return result


def kuramoto_oscillators(
    n_oscillators: int = 2,
    n_steps: int = 10000,
    dt: float = 0.01,
    coupling: float = 0.5,
    omega_spread: float = 1.0,
    noise: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Kuramoto coupled phase oscillators.

    Simulates the Kuramoto model of phase-coupled oscillators.
    Useful for testing topological binding on oscillatory (non-chaotic) dynamics.

    The classic Kuramoto model: dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j − θ_i) + η(t)

    Parameters
    ----------
    n_oscillators : number of oscillators
    n_steps : number of integration steps
    dt : integration timestep
    coupling : coupling strength K
    omega_spread : std of natural frequency distribution (mean=1.0)
    noise : Gaussian noise amplitude
    seed : random seed

    Returns
    -------
    phases : (n_steps, n_oscillators) array of phase values
    signals : (n_steps, n_oscillators) array with signal_i = sin(phase_i)
    """
    rng = get_rng(seed)

    # Natural frequencies drawn from normal distribution around 1.0
    omega = 1.0 + rng.normal(0, omega_spread, n_oscillators)

    # Random initial phases
    theta = rng.uniform(0, 2 * np.pi, n_oscillators)

    phases = np.zeros((n_steps, n_oscillators))
    signals = np.zeros((n_steps, n_oscillators))

    for step in range(n_steps):
        phases[step] = theta
        signals[step] = np.sin(theta)

        # Coupling term: (K/N) * sum_j sin(theta_j - theta_i) for each i
        diff = theta[None, :] - theta[:, None]  # (N, N): diff[i, j] = theta_j - theta_i
        coupling_term = (coupling / n_oscillators) * np.sum(np.sin(diff), axis=1)

        # Noise term
        noise_term = rng.normal(0, noise, n_oscillators) if noise > 0 else 0.0

        # Euler integration (Kuramoto is smooth, no need for RK4)
        theta = theta + dt * (omega + coupling_term + noise_term)

    return phases, signals
