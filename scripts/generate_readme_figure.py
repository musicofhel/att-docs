"""Generate the 4-panel README demo figure for ATT.

Panels:
  A) 3D Lorenz attractor point cloud
  B) Binding image (residual PI heatmap) for coupled Rossler-Lorenz
  C) Coupling sweep curve (binding score vs coupling strength)
  D) Benchmark comparison (4 methods, rank-normalized)

Usage:
    cd att-docs && python scripts/generate_readme_figure.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from att.config import set_seed
from att.synthetic import lorenz_system, coupled_rossler_lorenz
from att.embedding import TakensEmbedder, JointEmbedder
from att.topology import PersistenceAnalyzer
from att.binding import BindingDetector


def main():
    set_seed(42)
    fig = plt.figure(figsize=(12, 5.5), dpi=150)
    gs = GridSpec(1, 4, figure=fig, wspace=0.35, left=0.05, right=0.97, top=0.88, bottom=0.12)

    # --- Panel A: 3D Lorenz attractor ---
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ts = lorenz_system(n_steps=8000, dt=0.01)
    ax_a.plot(ts[1000:, 0], ts[1000:, 1], ts[1000:, 2],
              lw=0.3, alpha=0.8, color="#1565C0")
    ax_a.set_xlabel("x", fontsize=7, labelpad=-2)
    ax_a.set_ylabel("y", fontsize=7, labelpad=-2)
    ax_a.set_zlabel("z", fontsize=7, labelpad=-2)
    ax_a.tick_params(labelsize=5, pad=-2)
    ax_a.set_title("(a) Lorenz Attractor", fontsize=9, fontweight="bold", pad=8)
    ax_a.view_init(elev=25, azim=135)

    # --- Panel B: Binding image ---
    ax_b = fig.add_subplot(gs[0, 1])
    ts_x, ts_y = coupled_rossler_lorenz(coupling=0.3, n_steps=5000)
    detector = BindingDetector(max_dim=1, method="persistence_image")
    detector.fit(ts_x[:, 0], ts_y[:, 0], subsample=800, seed=42)
    if hasattr(detector, '_residual_images') and detector._residual_images is not None:
        # Show H0 residual
        img = detector._residual_images[0]
        im = ax_b.imshow(img, cmap="RdBu_r", origin="lower", aspect="auto",
                         vmin=-np.abs(img).max(), vmax=np.abs(img).max())
        plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    ax_b.set_xlabel("Birth", fontsize=7)
    ax_b.set_ylabel("Persistence", fontsize=7)
    ax_b.tick_params(labelsize=5)
    ax_b.set_title("(b) Binding Image\n(joint - marginal)", fontsize=9, fontweight="bold")

    # --- Panel C: Coupling sweep ---
    ax_c = fig.add_subplot(gs[0, 2])
    csv_path = os.path.join(os.path.dirname(__file__), "..", "figures", "fig3_benchmark_data.csv")
    df = pd.read_csv(csv_path)
    binding = df[df["method"] == "binding_score"].sort_values("coupling")
    ax_c.plot(binding["coupling"], binding["score"], "o-", color="#1565C0",
              lw=2, markersize=4, label="Binding score")
    ax_c.set_xlabel("Coupling strength", fontsize=8)
    ax_c.set_ylabel("Binding score", fontsize=8)
    ax_c.tick_params(labelsize=6)
    ax_c.set_title("(c) Coupling Sweep", fontsize=9, fontweight="bold")
    ax_c.axhline(y=binding["score"].iloc[0], color="gray", ls="--", lw=0.8, alpha=0.6)
    ax_c.text(0.5, binding["score"].iloc[0] + 5, "baseline", fontsize=6, color="gray",
              ha="center")

    # --- Panel D: Benchmark comparison (rank-normalized) ---
    ax_d = fig.add_subplot(gs[0, 3])
    colors = {"binding_score": "#1565C0", "transfer_entropy": "#E65100",
              "pac": "#2E7D32", "crqa": "#6A1B9A"}
    labels = {"binding_score": "Binding", "transfer_entropy": "TE",
              "pac": "PAC", "crqa": "CRQA"}
    for method in ["binding_score", "transfer_entropy", "pac", "crqa"]:
        sub = df[df["method"] == method].sort_values("coupling")
        ax_d.plot(sub["coupling"], sub["score_normalized"], "o-",
                  color=colors[method], lw=1.5, markersize=3, label=labels[method])
    ax_d.set_xlabel("Coupling strength", fontsize=8)
    ax_d.set_ylabel("Rank-normalized score", fontsize=8)
    ax_d.tick_params(labelsize=6)
    ax_d.set_title("(d) Method Comparison", fontsize=9, fontweight="bold")
    ax_d.legend(fontsize=6, loc="upper left", framealpha=0.8)

    out_path = os.path.join(os.path.dirname(__file__), "..", "figures", "readme_demo.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
