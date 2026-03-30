"""Visualization functions for cone detection results.

Plot types:
  - Availability profile: Betti_1 vs depth along projection axis
  - Coupling sweep: family of availability profiles across coupling strengths
  - Cross-section topology: persistence diagrams at each depth slice
  - CCA subspace comparison: full vs coupling-influence profiles
  - Cascade verification: lagged cross-correlations for Experiment 0
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_availability_profile(
    profile: dict,
    ax: plt.Axes | None = None,
    title: str = "Availability Profile",
    show_betti_0: bool = False,
) -> plt.Figure:
    """Plot Betti numbers as a function of depth along the projection axis.

    Parameters
    ----------
    profile : dict from ConeDetector.availability_profile()
        Must contain 'depths', 'betti_0', 'betti_1'
    ax : optional matplotlib axes
    title : plot title
    show_betti_0 : also plot Betti_0 (components)

    Returns figure.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    depths = profile["depths"]
    ax.plot(depths, profile["betti_1"], "o-", color="steelblue", label=r"$\beta_1$")
    if show_betti_0:
        ax.plot(depths, profile["betti_0"], "s--", color="coral", label=r"$\beta_0$")
    ax.set_xlabel("Depth along projection axis")
    ax.set_ylabel("Betti number")
    ax.set_title(title)
    if profile.get("trend_slope") is not None:
        ax.annotate(
            f"slope = {profile['trend_slope']:.3f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
        )
    ax.legend()
    return fig


def plot_coupling_sweep(
    coupling_values: np.ndarray,
    profiles: list[dict],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot family of availability profiles across coupling strengths.

    Parameters
    ----------
    coupling_values : 1D array of coupling strengths
    profiles : list of dicts from availability_profile(), one per coupling

    Returns figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: heatmap of Betti_1(coupling, depth)
    betti_matrix = np.array([p["betti_1"] for p in profiles])
    im = axes[0].imshow(
        betti_matrix,
        aspect="auto",
        origin="lower",
        extent=[
            profiles[0]["depths"][0],
            profiles[0]["depths"][-1],
            coupling_values[0],
            coupling_values[-1],
        ],
    )
    axes[0].set_xlabel("Depth")
    axes[0].set_ylabel("Coupling strength")
    axes[0].set_title(r"$\beta_1$ landscape")
    fig.colorbar(im, ax=axes[0], label=r"$\beta_1$")

    # Right panel: trend slope vs coupling
    slopes = [p["trend_slope"] for p in profiles]
    axes[1].plot(coupling_values, slopes, "o-")
    axes[1].set_xlabel("Coupling strength")
    axes[1].set_ylabel(r"$\beta_1$ trend slope")
    axes[1].set_title("Cone detectability")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_cross_sections(
    slices: list[np.ndarray],
    diagrams: list,
    depths: np.ndarray,
    max_cols: int = 5,
) -> plt.Figure:
    """Plot persistence diagrams for each depth slice side-by-side.

    Parameters
    ----------
    slices : list of point clouds, one per depth bin
    diagrams : list of persistence diagram lists, one per depth bin
    depths : bin center values
    max_cols : max columns in subplot grid
    """
    n = len(slices)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # Top row: PCA -> 2D projection of cross-section
        cloud = slices[i]
        if cloud.shape[1] > 2:
            centered = cloud - cloud.mean(0)
            U, S, _ = np.linalg.svd(centered, full_matrices=False)
            cloud_2d = U[:, :2] * S[:2]
        else:
            cloud_2d = cloud
        axes[0, i].scatter(cloud_2d[:, 0], cloud_2d[:, 1], s=1, alpha=0.3)
        axes[0, i].set_title(f"d={depths[i]:.2f}")
        axes[0, i].set_aspect("equal")

        # Bottom row: persistence diagram (birth-death plot)
        colors = ["tab:blue", "tab:orange", "tab:green"]
        max_val = 0.0
        for dim, dgm in enumerate(diagrams[i]):
            if len(dgm) > 0:
                axes[1, i].scatter(
                    dgm[:, 0], dgm[:, 1], s=10, alpha=0.6,
                    color=colors[dim % len(colors)], label=f"H{dim}",
                )
                max_val = max(max_val, dgm[:, 1].max())
        if max_val > 0:
            axes[1, i].plot([0, max_val], [0, max_val], "k--", alpha=0.3)
        axes[1, i].set_xlabel("Birth")
        if i == 0:
            axes[1, i].set_ylabel("Death")
        axes[1, i].legend(fontsize=7)

    fig.suptitle("Cross-section topology by depth", y=1.02)
    fig.tight_layout()
    return fig


def plot_subspace_comparison(
    profile_full: dict,
    profile_cca: dict,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Compare availability profiles in full embedding vs CCA subspace.

    Parameters
    ----------
    profile_full : availability profile in full Takens embedding
    profile_cca : availability profile in CCA subspace
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(
        profile_full["depths"], profile_full["betti_1"],
        "o-", label="Full embedding",
    )
    ax.plot(
        profile_cca["depths"], profile_cca["betti_1"],
        "s--", label="CCA subspace",
    )
    ax.set_xlabel("Depth")
    ax.set_ylabel(r"$\beta_1$")
    ax.legend()
    ax.set_title("Full vs coupling-influence subspace")
    return fig


def plot_cascade_verification(
    trajectories: dict[str, np.ndarray],
    max_lag: int = 500,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot lagged cross-correlations to verify directed cascade (Exp 0).

    Parameters
    ----------
    trajectories : dict from layered_aizawa_network()
    max_lag : maximum lag to compute
    """
    pairs = [
        ("C", "A3"), ("C", "A5"), ("A3", "A5"),
        ("C", "B3"), ("C", "B5"), ("B3", "B5"),
        ("A3", "B3"), ("A5", "B5"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (n1, n2) in enumerate(pairs):
        ax = axes.flat[idx]
        x1 = trajectories[n1][:, 0]
        x2 = trajectories[n2][:, 0]
        # Use a window for efficiency
        win = min(2 * max_lag, len(x1))
        x1_w = x1[:win]
        x2_w = x2[:win]
        x1_norm = (x1_w - x1_w.mean()) / (x1_w.std() + 1e-10)
        x2_norm = (x2_w - x2_w.mean()) / (x2_w.std() + 1e-10)
        corr = np.correlate(x1_norm, x2_norm, "full")
        corr /= len(x1_norm)
        lags = np.arange(-win + 1, win)
        # Trim to [-max_lag, max_lag]
        center = win - 1
        lo = max(0, center - max_lag)
        hi = min(len(lags), center + max_lag + 1)
        ax.plot(lags[lo:hi], corr[lo:hi])
        peak_lag = lags[lo:hi][np.argmax(np.abs(corr[lo:hi]))]
        ax.axvline(peak_lag, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"{n1}->{n2} (lag={peak_lag})")
        ax.set_xlabel("Lag")

    fig.suptitle("Cascade Verification: Lagged Cross-Correlations")
    fig.tight_layout()
    return fig


def plot_directed_vs_symmetric(
    profile_directed: dict,
    profile_symmetric: dict,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Compare availability profiles for directed vs symmetric coupling (Exp 5).

    Parameters
    ----------
    profile_directed : availability profile from directed network
    profile_symmetric : availability profile from symmetric network
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(
        profile_directed["depths"], profile_directed["betti_1"],
        "o-", color="steelblue",
        label=f"Directed (slope={profile_directed['trend_slope']:.3f})",
    )
    ax.plot(
        profile_symmetric["depths"], profile_symmetric["betti_1"],
        "s--", color="coral",
        label=f"Symmetric (slope={profile_symmetric['trend_slope']:.3f})",
    )
    ax.set_xlabel("Depth along projection axis")
    ax.set_ylabel(r"$\beta_1$")
    ax.legend()
    ax.set_title("Directed vs Symmetric: Availability Profile")
    return fig
