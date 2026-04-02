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


# ============================================================
# LLM hidden-state analysis plots (Wave 1)
# ============================================================


def plot_zscore_profile(
    z_scores: np.ndarray,
    p_values: np.ndarray | None = None,
    per_dim_z_scores: dict | None = None,
    ax: matplotlib.axes.Axes | None = None,
    significance_threshold: float = 0.05,
) -> matplotlib.figure.Figure:
    """Layer-indexed z-score profile with significance shading.

    Parameters
    ----------
    z_scores : (n_layers,) aggregate z-scores.
    p_values : (n_layers,) p-values for significance shading.
    per_dim_z_scores : dict mapping dim (int) -> (n_layers,) z-scores.
    ax : optional axes.
    significance_threshold : p-value below which to shade.
    """
    if per_dim_z_scores is not None and ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    elif ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    layers = np.arange(len(z_scores))

    if per_dim_z_scores is not None:
        dim_colors = {0: "tab:blue", 1: "tab:red", 2: "tab:green"}
        dim_labels = {0: "H0", 1: "H1", 2: "H2"}
        for dim, zs in per_dim_z_scores.items():
            ax.plot(
                layers, zs, "o-",
                color=dim_colors.get(dim, "gray"),
                linewidth=1.5, markersize=3, alpha=0.7,
                label=dim_labels.get(dim, f"H{dim}"),
            )

    ax.plot(
        layers, z_scores, "s-",
        color="black", linewidth=2.5, markersize=5,
        label="Aggregate", zorder=10,
    )

    # Significance shading
    if p_values is not None:
        sig_mask = p_values < significance_threshold
        for i in range(len(layers)):
            if sig_mask[i]:
                ax.axvspan(
                    layers[i] - 0.5, layers[i] + 0.5,
                    alpha=0.08, color="gold", zorder=0,
                )

    # Terminal-layer shading
    n = len(z_scores)
    if n > 6:
        ax.axvspan(n - 5.5, n - 0.5, alpha=0.08, color="gray", label="Terminal 5 layers")

    ax.axhline(1.96, color="gray", linestyle="--", alpha=0.5, label="z=1.96")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("z-score")
    ax.set_title("Per-Layer Topological Discriminability Profile")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_crocker(
    betti_matrix: np.ndarray,
    parameter_labels: list[str] | None = None,
    filtration_range: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    colormap: str = "viridis",
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """2D heatmap of Betti numbers (filtration scale × parameter).

    Parameters
    ----------
    betti_matrix : (n_filtration_steps, n_parameters) Betti number matrix.
    parameter_labels : labels for the parameter axis.
    filtration_range : (min, max) of filtration values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, betti_matrix.shape[1] * 0.5), 6))
    else:
        fig = ax.get_figure()

    extent = None
    if filtration_range is not None:
        extent = [
            -0.5, betti_matrix.shape[1] - 0.5,
            filtration_range[0], filtration_range[1],
        ]

    im = ax.imshow(
        betti_matrix, aspect="auto", origin="lower",
        cmap=colormap, extent=extent, interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Betti number", fraction=0.046, pad=0.04)

    if parameter_labels is not None:
        ax.set_xticks(range(len(parameter_labels)))
        ax.set_xticklabels(parameter_labels, rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("Filtration scale (ε)")
    ax.set_xlabel("Parameter")
    ax.set_title(title or "CROCKER Plot")
    fig.tight_layout()
    return fig


def plot_compression_decomposition(
    levels: list[int],
    total_persistence: list[float],
    n_features: list[float],
    mean_lifetime: list[float],
    ax: matplotlib.axes.Axes | None = None,
    title: str = "H1 Persistence Decomposition",
) -> matplotlib.figure.Figure:
    """Dual-axis plot of feature count vs mean lifetime by difficulty.

    Parameters
    ----------
    levels : difficulty level labels.
    total_persistence : total persistence per level.
    n_features : feature count per level.
    mean_lifetime : mean lifetime per level.
    """
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(8, 5))
    else:
        ax1 = ax
        fig = ax.get_figure()

    ax2 = ax1.twinx()

    x = np.arange(len(levels))
    width = 0.3

    bars1 = ax1.bar(
        x - width / 2, n_features, width,
        color="steelblue", alpha=0.8, label="Feature Count",
    )
    bars2 = ax2.bar(
        x + width / 2, mean_lifetime, width,
        color="coral", alpha=0.8, label="Mean Lifetime",
    )

    ax1.set_xlabel("Difficulty Level")
    ax1.set_ylabel("Feature Count", color="steelblue")
    ax2.set_ylabel("Mean Lifetime", color="coral")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(lv) for lv in levels])
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ============================================================
# LLM hidden-state analysis plots (Wave 2)
# ============================================================


def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    ax: matplotlib.axes.Axes | None = None,
    title: str = "ROC Curves: Correctness Prediction",
) -> matplotlib.figure.Figure:
    """Plot ROC curves for correctness prediction.

    Parameters
    ----------
    roc_data : dict mapping label -> (fpr, tpr, auroc).
    ax : optional axes.
    title : plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    for label, (fpr, tpr, auroc) in roc_data.items():
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    return fig


def plot_id_profile(
    profiles: dict[int, np.ndarray],
    ax: matplotlib.axes.Axes | None = None,
    title: str = "Intrinsic Dimension by Layer",
    method_label: str = "TwoNN",
) -> matplotlib.figure.Figure:
    """Plot intrinsic dimension profiles across layers by difficulty level.

    Parameters
    ----------
    profiles : dict mapping level -> (n_layers,) array of ID estimates.
    ax : optional axes.
    title : plot title.
    method_label : label for the ID method.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.get_figure()

    levels = sorted(profiles.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))

    for level, color in zip(levels, colors):
        ids = profiles[level]
        layers = np.arange(len(ids))
        ax.plot(layers, ids, "-o", color=color, label=f"Level {level}",
                markersize=3, linewidth=1.5)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(f"Intrinsic Dimension ({method_label})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shade terminal layers
    n_layers = len(next(iter(profiles.values())))
    terminal_start = max(0, n_layers - 5)
    ax.axvspan(terminal_start, n_layers - 1, alpha=0.08, color="red")

    fig.tight_layout()
    return fig


def plot_spectral_comparison(
    euclidean_entropy: dict[int, list[float]],
    spectral_entropy: dict[int, list[float]],
    layer_indices: list[int],
    ax: matplotlib.axes.Axes | None = None,
    title: str = "Euclidean vs Spectral PH Entropy",
) -> matplotlib.figure.Figure:
    """Side-by-side comparison of Euclidean and spectral persistence entropy.

    Parameters
    ----------
    euclidean_entropy : dict mapping H-dim -> list of entropies per layer.
    spectral_entropy : dict mapping H-dim -> list of entropies per layer.
    layer_indices : list of layer index labels.
    ax : optional axes.
    title : plot title.
    """
    n_dims = len(euclidean_entropy)
    if ax is None:
        fig, axes = plt.subplots(1, n_dims, figsize=(7 * n_dims, 5))
        if n_dims == 1:
            axes = [axes]
    else:
        fig = ax.get_figure()
        axes = [ax]

    for dim_idx, dim in enumerate(sorted(euclidean_entropy.keys())):
        if dim_idx >= len(axes):
            break
        a = axes[dim_idx]
        a.plot(layer_indices, euclidean_entropy[dim], "b-o", label="Euclidean",
               markersize=4, linewidth=1.5)
        a.plot(layer_indices, spectral_entropy[dim], "r-s", label="Spectral",
               markersize=4, linewidth=1.5)
        a.set_xlabel("Layer")
        a.set_ylabel("Persistence Entropy")
        a.set_title(f"H{dim}")
        a.legend()
        a.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Wave 3: Zigzag persistence + token-region plots
# ---------------------------------------------------------------------------


def plot_zigzag_barcode(
    barcodes: np.ndarray,
    dim: int = 1,
    level: int | None = None,
    ax=None,
    title: str | None = None,
    colormap: str = "viridis",
):
    """Plot zigzag persistence barcode as horizontal lifetime bars.

    Parameters
    ----------
    barcodes : (n, 2) array of (birth_layer, death_layer).
    dim : homology dimension (for labeling).
    level : difficulty level (for labeling).
    ax : optional matplotlib axes.
    title : plot title override.
    colormap : matplotlib colormap name.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.get_figure()

    if len(barcodes) == 0:
        ax.text(0.5, 0.5, f"No H{dim} features", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        if title:
            ax.set_title(title)
        return fig

    order = np.argsort(barcodes[:, 0])
    barcodes = barcodes[order]
    lifetimes = barcodes[:, 1] - barcodes[:, 0]
    max_lt = max(lifetimes.max(), 1e-10)

    cmap = plt.get_cmap(colormap)
    colors = cmap(lifetimes / max_lt)

    for i, (bar, color) in enumerate(zip(barcodes, colors)):
        ax.barh(i, bar[1] - bar[0], left=bar[0], height=0.8, color=color, alpha=0.8)

    ax.set_xlabel("Layer (zigzag time)")
    ax.set_ylabel("Feature index")
    default_title = f"H{dim} Zigzag Barcode"
    if level is not None:
        default_title += f" — Level {level}"
    ax.set_title(title or default_title)
    ax.grid(True, alpha=0.2, axis="x")
    return fig


def plot_zigzag_comparison(
    results: dict,
    dim: int = 1,
    metric: str = "mean_lifetime",
    ax=None,
    title: str = "Zigzag Feature Statistics by Difficulty",
):
    """Bar chart comparing zigzag statistics across difficulty levels.

    Parameters
    ----------
    results : dict mapping level (int) -> stats dict from zigzag_feature_lifetime_stats.
    dim : homology dimension (for labeling).
    metric : which stat to plot ('mean_lifetime', 'n_features', 'max_lifetime', 'n_long_lived').
    ax : optional axes.
    title : plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    levels = sorted(results.keys())
    values = [results[l].get(metric, 0) for l in levels]

    colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    ax.bar(levels, values, color=colors, alpha=0.8)
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def plot_token_partition_topology(
    region_entropy: dict,
    levels: list[int] | None = None,
    ax=None,
    title: str = "Persistence Entropy by Token Region",
):
    """Grouped bar chart of persistence entropy across token regions and difficulty.

    Parameters
    ----------
    region_entropy : dict mapping region_name -> {level: list_of_entropy_values}.
    levels : which levels to include (default: all).
    ax : optional axes.
    title : plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    regions = sorted(region_entropy.keys())
    if levels is None:
        all_levels = set()
        for r_data in region_entropy.values():
            all_levels.update(r_data.keys())
        levels = sorted(all_levels)

    n_regions = len(regions)
    n_levels = len(levels)
    x = np.arange(n_regions)
    width = 0.8 / max(n_levels, 1)

    colors = plt.cm.viridis(np.linspace(0, 1, n_levels))

    for i, (level, color) in enumerate(zip(levels, colors)):
        means = []
        errs = []
        for region in regions:
            vals = region_entropy[region].get(level, [])
            if vals:
                means.append(np.mean(vals))
                errs.append(np.std(vals))
            else:
                means.append(0)
                errs.append(0)

        offset = (i - n_levels / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=errs, label=f"Level {level}",
               color=color, alpha=0.8, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in regions], fontsize=9)
    ax.set_ylabel("Persistence Entropy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def plot_cross_model_zscore(
    zscore_results: dict,
    model_labels: dict | None = None,
    model_colors: dict | None = None,
    ax=None,
    title: str = "Cross-Model Z-Score Profiles",
):
    """Overlay z-score profiles from multiple models.

    Parameters
    ----------
    zscore_results : dict mapping model_key -> {"z_scores": array, "n_layers": int}.
    model_labels : optional mapping model_key -> display name.
    model_colors : optional mapping model_key -> color.
    ax : optional axes.
    title : plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    default_colors = plt.cm.tab10(np.linspace(0, 1, len(zscore_results)))

    for i, (key, data) in enumerate(zscore_results.items()):
        z = np.asarray(data["z_scores"])
        x = np.linspace(0, 1, len(z))
        label = (model_labels or {}).get(key, key)
        color = (model_colors or {}).get(key, default_colors[i])
        ax.plot(x, z, label=label, color=color, linewidth=2, alpha=0.8)

    ax.axhline(y=1.96, color="gray", linestyle="--", alpha=0.5, label="p<0.05")
    ax.axhline(y=2.58, color="gray", linestyle=":", alpha=0.5, label="p<0.01")
    ax.set_xlabel("Normalized Layer Position (0=embedding, 1=final)")
    ax.set_ylabel("Z-Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_attention_binding_heatmap(
    scores: dict,
    levels: list[int],
    layer_indices: list[int],
    ax=None,
    title: str = "Attention-Hidden Binding Score",
    cmap: str = "RdYlBu_r",
):
    """Heatmap of binding scores (difficulty × layer).

    Parameters
    ----------
    scores : dict mapping (level, layer) -> float or {"mean": float}.
    levels : list of difficulty levels.
    layer_indices : list of layer indices.
    ax : optional axes.
    title : plot title.
    cmap : colormap name.
    """
    n_levels = len(levels)
    n_layers = len(layer_indices)
    matrix = np.zeros((n_levels, n_layers))

    for i, level in enumerate(levels):
        for j, layer in enumerate(layer_indices):
            val = scores.get((level, layer), 0.0)
            if isinstance(val, dict):
                val = val.get("mean", 0.0)
            matrix[i, j] = val

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, n_layers), max(4, n_levels * 0.8)))
    else:
        fig = ax.get_figure()

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(l) for l in layer_indices])
    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([f"Level {l}" for l in levels])
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Difficulty Level")
    ax.set_title(title)

    for i in range(n_levels):
        for j in range(n_layers):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i, j] > matrix.mean() else "black")

    fig.colorbar(im, ax=ax, label="Binding Score")
    return fig
