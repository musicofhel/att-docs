"""Publication-quality plotting utilities."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_persistence_diagram(
    diagrams: list[np.ndarray],
    ax: matplotlib.axes.Axes | None = None,
    colormap: str = "viridis",
) -> matplotlib.figure.Figure:
    """Plot persistence diagrams for all homology dimensions."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / max(len(diagrams) - 1, 1)) for i in range(len(diagrams))]

    all_vals = []
    for dgm in diagrams:
        if len(dgm) > 0:
            all_vals.extend(dgm.ravel())

    if all_vals:
        vmin, vmax = min(all_vals), max(all_vals)
    else:
        vmin, vmax = 0, 1

    # Diagonal
    ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.3, linewidth=1)

    for dim, dgm in enumerate(diagrams):
        if len(dgm) > 0:
            ax.scatter(
                dgm[:, 0], dgm[:, 1],
                c=[colors[dim]] * len(dgm),
                label=f"H{dim}",
                s=20, alpha=0.7, edgecolors="k", linewidths=0.3,
            )

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("Persistence Diagram")
    ax.legend()
    ax.set_aspect("equal")

    return fig


def plot_persistence_image(
    images: list[np.ndarray],
    ax: matplotlib.axes.Axes | None = None,
    colormap: str = "hot",
) -> matplotlib.figure.Figure:
    """Plot persistence images for all homology dimensions."""
    n = len(images)
    if ax is not None:
        fig = ax.get_figure()
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

    for i, (img, ax_) in enumerate(zip(images, axes)):
        im = ax_.imshow(img, cmap=colormap, origin="lower", aspect="auto")
        ax_.set_title(f"H{i} Persistence Image")
        fig.colorbar(im, ax=ax_, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def plot_barcode(
    diagrams: list[np.ndarray],
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot persistence barcodes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    y_offset = 0

    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        # Sort by persistence (longest first)
        lifetimes = dgm[:, 1] - dgm[:, 0]
        order = np.argsort(-lifetimes)
        color = colors[dim % len(colors)]

        for idx in order:
            birth, death = dgm[idx]
            ax.plot([birth, death], [y_offset, y_offset], color=color, linewidth=1.5)
            y_offset += 1

    ax.set_xlabel("Filtration Parameter")
    ax.set_ylabel("Feature")
    ax.set_title("Persistence Barcode")

    # Legend
    handles = []
    for dim in range(len(diagrams)):
        if len(diagrams[dim]) > 0:
            handles.append(plt.Line2D([0], [0], color=colors[dim % len(colors)], label=f"H{dim}"))
    ax.legend(handles=handles)

    return fig


def plot_betti_curve(
    betti_curves: list[np.ndarray],
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot Betti curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for dim, curve in enumerate(betti_curves):
        ax.plot(curve, color=colors[dim % len(colors)], label=f"β{dim}")

    ax.set_xlabel("Filtration Index")
    ax.set_ylabel("Betti Number")
    ax.set_title("Betti Curves")
    ax.legend()

    return fig


def plot_attractor_3d(
    cloud: np.ndarray,
    color_by: str = "time",
    backend: str = "plotly",
):
    """3D scatter/line plot of an attractor point cloud.

    Parameters
    ----------
    cloud : (n_points, 3+) array — uses first 3 columns
    color_by : "time" (color by index)
    backend : "plotly" or "matplotlib"
    """
    cloud = np.asarray(cloud)[:, :3]

    if backend == "plotly":
        import plotly.graph_objects as go

        colors = np.arange(len(cloud))
        fig = go.Figure(
            data=[go.Scatter3d(
                x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2],
                mode="lines",
                line=dict(color=colors, colorscale="Viridis", width=2),
            )]
        )
        fig.update_layout(
            title="Attractor",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
            width=700, height=600,
        )
        return fig
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        colors = np.arange(len(cloud))
        ax.scatter(
            cloud[:, 0], cloud[:, 1], cloud[:, 2],
            c=colors, cmap="viridis", s=0.5, alpha=0.5,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Attractor")
        return fig


def plot_surrogate_distribution(
    observed: float,
    surrogates: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Histogram of surrogate scores with observed score marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    ax.hist(surrogates, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(observed, color="red", linewidth=2, label=f"Observed = {observed:.4f}")
    p95 = np.percentile(surrogates, 95)
    ax.axvline(p95, color="orange", linewidth=1.5, linestyle="--", label=f"95th pctile = {p95:.4f}")
    ax.set_xlabel("Binding Score")
    ax.set_ylabel("Count")
    ax.set_title("Surrogate Distribution")
    ax.legend()

    return fig


def plot_benchmark_sweep(results, ax=None) -> matplotlib.figure.Figure:
    """Plot benchmark sweep with all methods overlaid.

    Parameters
    ----------
    results : pd.DataFrame with columns coupling, method, score, score_normalized
    """
    import pandas as pd

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    methods = results["method"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for method, color in zip(methods, colors):
        subset = results[results["method"] == method].sort_values("coupling")
        col = "score_normalized" if "score_normalized" in results.columns else "score"
        ax.plot(subset["coupling"], subset[col], "o-", color=color, label=method, markersize=4)

    ax.set_xlabel("Coupling Strength")
    ax.set_ylabel("Score (normalized)")
    ax.set_title("Coupling Benchmark Sweep")
    ax.legend()

    return fig


def plot_binding_comparison(detector) -> matplotlib.figure.Figure:
    """3-panel comparison: marginal X | joint (excess highlighted) | marginal Y.

    Parameters
    ----------
    detector : BindingDetector with fitted state

    Returns
    -------
    matplotlib Figure
    """
    diagrams_x = detector._result_x["diagrams"]
    diagrams_joint = detector._result_joint["diagrams"]
    diagrams_y = detector._result_y["diagrams"]
    residuals = detector._residual_images

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = ["Marginal X", "Joint (excess highlighted)", "Marginal Y"]
    all_diagrams = [diagrams_x, diagrams_joint, diagrams_y]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Find global axis range
    all_vals = []
    for diag_set in all_diagrams:
        for dgm in diag_set:
            if len(dgm) > 0:
                all_vals.extend(dgm.ravel())
    if all_vals:
        vmin, vmax = min(all_vals), max(all_vals)
    else:
        vmin, vmax = 0, 1

    for panel_idx, (diags, ax, title) in enumerate(zip(all_diagrams, axes, titles)):
        ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.3, linewidth=1)

        for dim, dgm in enumerate(diags):
            if len(dgm) > 0:
                ax.scatter(
                    dgm[:, 0], dgm[:, 1],
                    c=colors[dim % len(colors)],
                    label=f"H{dim}", s=20, alpha=0.7,
                    edgecolors="k", linewidths=0.3,
                )

        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(title)
        ax.legend()
        ax.set_aspect("equal")
        ax.set_xlim(vmin - 0.5, vmax + 0.5)
        ax.set_ylim(vmin - 0.5, vmax + 0.5)

    fig.tight_layout()
    return fig


def plot_binding_image(
    images: list[np.ndarray],
    colormap: str = "RdBu_r",
) -> matplotlib.figure.Figure:
    """Heatmap of residual persistence images.

    Parameters
    ----------
    images : list of (resolution, resolution) residual images, one per dimension
    colormap : diverging colormap (red=emergent, blue=deficit)

    Returns
    -------
    matplotlib Figure
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (img, ax) in enumerate(zip(images, axes)):
        vmax = max(abs(img.min()), abs(img.max())) or 1.0
        im = ax.imshow(
            img, cmap=colormap, origin="lower", aspect="auto",
            vmin=-vmax, vmax=vmax,
        )
        ax.set_title(f"H{i} Binding Image")
        ax.set_xlabel("Birth")
        ax.set_ylabel("Persistence")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def export_to_json(results: dict, path: str) -> None:
    """Export computed results as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(results), f, indent=2)


def load_from_json(path: str) -> dict:
    """Load results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def plot_transition_timeline(detector, ground_truth=None, figsize=(12, 6)):
    """Plot topology transition timeline from a fitted TransitionDetector.

    Parameters
    ----------
    detector : TransitionDetector
        Must have been fit_transform()'d.
    ground_truth : list of int or None
        True transition sample indices (plotted as green dotted lines).
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib Figure
    """
    result = detector._result
    if result is None:
        raise RuntimeError("TransitionDetector must be fitted first.")

    window_centers = result["window_centers"]
    image_distances = result["image_distances"]
    # image_distances has len = len(window_centers) - 1
    # Use midpoints between consecutive window centers
    dist_x = (window_centers[:-1] + window_centers[1:]) / 2

    # Compute H1 persistence entropy per window
    h1_entropy = []
    for topo in result["topology_timeseries"]:
        # persistence_entropy is a list per dim
        if len(topo["persistence_entropy"]) > 1:
            h1_entropy.append(topo["persistence_entropy"][1])
        else:
            h1_entropy.append(0.0)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top panel: image distances + changepoints
    ax = axes[0]
    ax.plot(dist_x, image_distances, 'k-', linewidth=1.5, label='PI distance')
    ax.set_ylabel('Image distance (L2)')
    ax.set_title('Topological Transition Timeline')

    # Detected changepoints
    try:
        changepoints = detector.detect_changepoints()
        for cp in changepoints:
            if cp < len(dist_x):
                ax.axvline(dist_x[cp], color='red', linestyle='--', alpha=0.8,
                          label='Detected' if cp == changepoints[0] else None)
    except Exception:
        pass

    # Ground truth
    if ground_truth is not None:
        for i, gt in enumerate(ground_truth):
            ax.axvline(gt, color='green', linestyle=':', alpha=0.7, linewidth=2,
                      label='Ground truth' if i == 0 else None)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Bottom panel: H1 persistence entropy
    ax = axes[1]
    ax.plot(window_centers, h1_entropy, 'b-', linewidth=1.5)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('H1 persistence entropy')
    ax.set_title('Loop Complexity Over Time')

    if ground_truth is not None:
        for gt in ground_truth:
            ax.axvline(gt, color='green', linestyle=':', alpha=0.7, linewidth=2)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
