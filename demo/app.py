"""ATT Interactive Demo -- Streamlit app for exploring attractor topology.

Run with:
    pip install -e ".[demo]"
    streamlit run demo/app.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from att.config.seed import set_seed
from att.synthetic.generators import lorenz_system, rossler_system, coupled_lorenz
from att.embedding.takens import TakensEmbedder
from att.topology.persistence import PersistenceAnalyzer
from att.binding.detector import BindingDetector
from att.viz.plotting import (
    plot_persistence_diagram,
    plot_binding_image,
    plot_persistence_image,
    plot_surrogate_distribution,
)
from att.benchmarks.methods import transfer_entropy

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ATT Demo",
    page_icon="🌀",
    layout="wide",
)

set_seed(42)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Attractor Topology Toolkit")
st.sidebar.markdown(
    """
**ATT** detects *topological binding* between coupled dynamical systems.

**Core idea:** embed two time series (X, Y) individually and jointly via
Takens delay embedding, compute persistent homology on each point cloud,
convert diagrams to persistence images, then measure the *excess* topology
in the joint embedding that is absent from both marginals.

A positive **binding score** means the joint system has topological
features (loops, voids) that neither subsystem alone explains --
evidence of emergent coupling structure.

**Significance** is assessed by comparing the observed score against a
null distribution of phase-randomised surrogates.

---
*Built on: Ripser, persim, scikit-learn, SciPy*
"""
)

page = st.sidebar.radio(
    "Page",
    ["Attractor Explorer", "Binding Detection", "Coupling Sweep"],
)


# ---------------------------------------------------------------------------
# Cached computations
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Generating attractor...")
def generate_attractor(system: str, n_steps: int, coupling: float = 0.1):
    """Generate attractor data.  Returns dict with keys depending on system."""
    set_seed(42)
    if system == "Lorenz":
        data = lorenz_system(n_steps=n_steps)
        return {"type": "single", "data": data, "label": "Lorenz"}
    elif system == "Rossler":
        data = rossler_system(n_steps=n_steps)
        return {"type": "single", "data": data, "label": "Rossler"}
    else:  # Coupled Lorenz
        ts_x, ts_y = coupled_lorenz(n_steps=n_steps, coupling=coupling)
        return {
            "type": "coupled",
            "ts_x": ts_x,
            "ts_y": ts_y,
            "label": f"Coupled Lorenz (c={coupling:.2f})",
        }


@st.cache_data(show_spinner="Computing persistence...")
def compute_persistence(cloud: np.ndarray, subsample: int, max_dim: int = 2):
    """Run PersistenceAnalyzer on a point cloud."""
    set_seed(42)
    pa = PersistenceAnalyzer(max_dim=max_dim)
    result = pa.fit_transform(cloud, subsample=subsample, seed=42)
    return result


@st.cache_data(show_spinner="Computing binding score...")
def compute_binding(coupling: float, n_steps: int, subsample: int):
    """Run BindingDetector on coupled Lorenz at given coupling."""
    set_seed(42)
    ts_x, ts_y = coupled_lorenz(n_steps=n_steps, coupling=coupling, seed=42)
    x_scalar = ts_x[:, 0]
    y_scalar = ts_y[:, 0]

    det = BindingDetector(max_dim=1, image_resolution=50, image_sigma=0.1)
    det.fit(x_scalar, y_scalar, subsample=subsample, seed=42)

    score = det.binding_score()
    residuals = det.binding_image()
    images_x = det._images_x
    images_y = det._images_y
    images_joint = det._images_joint

    return {
        "score": score,
        "residuals": residuals,
        "images_x": images_x,
        "images_y": images_y,
        "images_joint": images_joint,
        "features": det.binding_features(),
        # Store raw data for significance test
        "x_scalar": x_scalar,
        "y_scalar": y_scalar,
    }


@st.cache_data(show_spinner="Running significance test (this may take a minute)...")
def run_significance(
    x_scalar: np.ndarray, y_scalar: np.ndarray, n_surrogates: int, subsample: int
):
    """Run surrogate significance test."""
    set_seed(42)
    det = BindingDetector(max_dim=1, image_resolution=50, image_sigma=0.1)
    det.fit(x_scalar, y_scalar, subsample=subsample, seed=42)
    sig = det.test_significance(n_surrogates=n_surrogates, method="phase_randomize", seed=42)
    return sig


@st.cache_data(show_spinner="Running coupling sweep...")
def run_coupling_sweep(
    coupling_values: tuple, n_steps: int, subsample: int, include_te: bool
):
    """Run binding score (and optionally TE) across coupling values."""
    set_seed(42)
    scores = []
    te_scores = [] if include_te else None

    for c in coupling_values:
        ts_x, ts_y = coupled_lorenz(n_steps=n_steps, coupling=c, seed=42)
        x_scalar = ts_x[:, 0]
        y_scalar = ts_y[:, 0]

        det = BindingDetector(max_dim=1, image_resolution=50, image_sigma=0.1)
        det.fit(x_scalar, y_scalar, subsample=subsample, seed=42)
        scores.append(det.binding_score())

        if include_te:
            te_val = transfer_entropy(x_scalar, y_scalar, k=1, bins=8)
            te_scores.append(te_val)

    return scores, te_scores


# ---------------------------------------------------------------------------
# Page: Attractor Explorer
# ---------------------------------------------------------------------------
def page_attractor_explorer():
    st.header("Attractor Explorer")
    st.markdown(
        "Generate a chaotic attractor, visualise it in 3D, and inspect its "
        "persistent homology."
    )

    col_ctrl, col_viz = st.columns([1, 3])

    with col_ctrl:
        system = st.selectbox("System", ["Lorenz", "Rossler", "Coupled Lorenz"])
        n_steps = st.slider("Time steps", 1000, 20000, 5000, step=1000)
        subsample = st.slider("PH subsample", 100, 1000, 300, step=50)
        coupling = 0.1
        if system == "Coupled Lorenz":
            coupling = st.slider(
                "Coupling strength", 0.0, 1.0, 0.1, step=0.05, key="explorer_coupling"
            )

    result = generate_attractor(system, n_steps, coupling)

    with col_viz:
        if result["type"] == "single":
            cloud = result["data"]
            _show_attractor_and_persistence(cloud, result["label"], subsample)
        else:
            tab_x, tab_y = st.tabs(["System X", "System Y"])
            with tab_x:
                _show_attractor_and_persistence(
                    result["ts_x"], f'{result["label"]} -- X', subsample
                )
            with tab_y:
                _show_attractor_and_persistence(
                    result["ts_y"], f'{result["label"]} -- Y', subsample
                )


def _show_attractor_and_persistence(cloud: np.ndarray, label: str, subsample: int):
    """Render 3D attractor + persistence diagram side-by-side."""
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader(f"3D Attractor: {label}")
        fig_3d = plt.figure(figsize=(6, 5))
        ax = fig_3d.add_subplot(111, projection="3d")
        n = len(cloud)
        step = max(1, n // 2000)  # thin for plotting speed
        c = cloud[::step]
        colors = np.arange(len(c))
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=colors, cmap="viridis", s=0.5, alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(label)
        st.pyplot(fig_3d)
        plt.close(fig_3d)

    with col_b:
        st.subheader("Persistence Diagram")
        ph = compute_persistence(cloud, subsample=subsample, max_dim=2)
        fig_pd = plot_persistence_diagram(ph["diagrams"])
        st.pyplot(fig_pd)
        plt.close(fig_pd)

        # Summary metrics
        st.markdown("**Persistence entropy** (per dimension):")
        for dim, ent in enumerate(ph["persistence_entropy"]):
            st.write(f"  H{dim}: {ent:.4f}")
        st.markdown("**Bottleneck norms:**")
        for dim, bn in enumerate(ph["bottleneck_norms"]):
            st.write(f"  H{dim}: {bn:.4f}")


# ---------------------------------------------------------------------------
# Page: Binding Detection
# ---------------------------------------------------------------------------
def page_binding_detection():
    st.header("Binding Detection")
    st.markdown(
        "Two coupled Lorenz systems. As coupling increases, the **joint** "
        "embedding develops topological features absent from either marginal. "
        "The binding score quantifies this excess topology."
    )

    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        coupling = st.slider(
            "Coupling strength", 0.0, 1.0, 0.3, step=0.05, key="binding_coupling"
        )
        n_steps = st.slider(
            "Time steps", 1000, 20000, 5000, step=1000, key="binding_nsteps"
        )
        subsample = st.slider(
            "PH subsample", 100, 1000, 300, step=50, key="binding_sub"
        )

    bd = compute_binding(coupling, n_steps, subsample)

    st.metric("Binding Score", f"{bd['score']:.4f}")

    features = bd["features"]
    fcols = st.columns(len(features))
    for i, (dim, feat) in enumerate(features.items()):
        with fcols[i]:
            st.markdown(f"**H{dim}**")
            st.write(f"Excess pixels: {feat['n_excess']}")
            st.write(f"Total persistence: {feat['total_persistence']:.4f}")
            st.write(f"Max persistence: {feat['max_persistence']:.4f}")

    st.subheader("Persistence Images")
    img_cols = st.columns(4)

    with img_cols[0]:
        st.markdown("**Marginal X (H1)**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        _show_single_image(ax, bd["images_x"][1], "Marginal X -- H1", "hot")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with img_cols[1]:
        st.markdown("**Marginal Y (H1)**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        _show_single_image(ax, bd["images_y"][1], "Marginal Y -- H1", "hot")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with img_cols[2]:
        st.markdown("**Joint (H1)**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        _show_single_image(ax, bd["images_joint"][1], "Joint -- H1", "hot")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with img_cols[3]:
        st.markdown("**Residual / Binding Image (H1)**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        img = bd["residuals"][1]
        vmax = max(abs(img.min()), abs(img.max())) or 1.0
        im = ax.imshow(img, cmap="RdBu_r", origin="lower", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title("Binding Image -- H1")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Significance test
    st.subheader("Significance Test")
    st.markdown(
        "Compare observed binding score against phase-randomised surrogates. "
        "Uses a small number of surrogates for interactive speed."
    )
    n_surr = st.number_input("Number of surrogates", min_value=9, max_value=99, value=19, step=10)

    if st.button("Run significance test"):
        sig = run_significance(bd["x_scalar"], bd["y_scalar"], n_surr, subsample)
        st.metric("p-value", f"{sig['p_value']:.4f}")
        if sig["significant"]:
            st.success(f"Significant at p < 0.05 (p = {sig['p_value']:.4f})")
        else:
            st.warning(f"Not significant (p = {sig['p_value']:.4f})")

        fig_surr = plot_surrogate_distribution(sig["observed_score"], sig["surrogate_scores"])
        st.pyplot(fig_surr)
        plt.close(fig_surr)


def _show_single_image(ax, img: np.ndarray, title: str, cmap: str):
    """Helper to render a single persistence image on an axis."""
    im = ax.imshow(img, cmap=cmap, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Page: Coupling Sweep
# ---------------------------------------------------------------------------
def page_coupling_sweep():
    st.header("Coupling Sweep")
    st.markdown(
        "Sweep coupling strength from 0 (independent) to 1 (synchronised) and "
        "plot the binding score curve. Optionally overlay transfer entropy for "
        "comparison."
    )

    col_ctrl, col_plot = st.columns([1, 3])

    with col_ctrl:
        n_points = st.slider("Number of coupling values", 3, 10, 6, key="sweep_n")
        n_steps = st.slider(
            "Time steps per run", 1000, 10000, 5000, step=1000, key="sweep_nsteps"
        )
        subsample = st.slider(
            "PH subsample", 100, 1000, 300, step=50, key="sweep_sub"
        )
        include_te = st.checkbox("Compare with Transfer Entropy", value=False)

    coupling_values = tuple(np.linspace(0.0, 1.0, n_points).round(3))

    scores, te_scores = run_coupling_sweep(
        coupling_values, n_steps, subsample, include_te
    )

    with col_plot:
        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_bind = "tab:blue"
        ax1.plot(coupling_values, scores, "o-", color=color_bind, linewidth=2, label="Binding Score")
        ax1.set_xlabel("Coupling Strength")
        ax1.set_ylabel("Binding Score", color=color_bind)
        ax1.tick_params(axis="y", labelcolor=color_bind)
        ax1.set_title("Binding Score vs Coupling Strength")

        if include_te and te_scores is not None:
            ax2 = ax1.twinx()
            color_te = "tab:orange"
            ax2.plot(
                coupling_values, te_scores, "s--", color=color_te, linewidth=2, label="TE(X->Y)"
            )
            ax2.set_ylabel("Transfer Entropy (nats)", color=color_te)
            ax2.tick_params(axis="y", labelcolor=color_te)

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        else:
            ax1.legend(loc="upper left")

        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Data table
        st.subheader("Raw Data")
        import pandas as pd

        data = {"Coupling": list(coupling_values), "Binding Score": scores}
        if include_te and te_scores is not None:
            data["Transfer Entropy"] = te_scores
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Attractor Explorer":
    page_attractor_explorer()
elif page == "Binding Detection":
    page_binding_detection()
elif page == "Coupling Sweep":
    page_coupling_sweep()
