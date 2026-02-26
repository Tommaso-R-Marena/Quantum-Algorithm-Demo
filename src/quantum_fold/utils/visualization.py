"""
visualization.py
Robust, publication-quality 3D visualisations for protein structures.

Provides functions to plot 3D Cα traces (with B-spline smoothing),
Ramachandran plots, contact maps, and distance matrices.

Dependencies:
  - matplotlib
  - scipy (for spline interpolation)
  - plotly (optional, for interactive plots)
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev

# Set global matplotlib settings for publication quality
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def _set_axes_equal_3d(ax):
    """Make axes of 3D plot have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_3d_structure(
    coords: np.ndarray,
    native_coords: Optional[np.ndarray] = None,
    sequence: Optional[str] = None,
    title: str = "3D Protein Structure",
    filename: Optional[str] = None,
    show_atoms: bool = True,
    smooth_spline: bool = True,
    backbone: Optional[np.ndarray] = None,
):
    """
    Plot a highly detailed 3D Cα trace or full backbone.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    n_res = len(coords)
    # Colour gradient: Blue (N-term) to Red (C-term)
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(np.linspace(0, 1, n_res))

    def draw_chain(ax, c, clrs, alpha=1.0, is_native=False):
        if is_native:
            line_color = "gray"
            scatter_color = "gray"
            line_alpha = 0.5 * alpha
            scatter_alpha = 0.3 * alpha
            lw = 1.5
            zorder = 1
        else:
            line_color = "black"
            scatter_color = clrs
            line_alpha = 0.8 * alpha
            scatter_alpha = 1.0 * alpha
            lw = 2.5
            zorder = 5

        x, y, z = c[:, 0], c[:, 1], c[:, 2]

        if smooth_spline and len(c) > 3:
            try:
                tck, u = splprep([x, y, z], s=0, k=3)
                u_new = np.linspace(0, 1, len(c) * 10)
                x_new, y_new, z_new = splev(u_new, tck)

                if is_native:
                    ax.plot(x_new, y_new, z_new, color=line_color, alpha=line_alpha,
                            linewidth=lw, zorder=zorder)
                else:
                    points = np.array([x_new, y_new, z_new]).T.reshape(-1, 1, 3)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    line_colors = cmap(u_new[:-1])
                    for i in range(len(segments)):
                        ax.plot(
                            segments[i, :, 0], segments[i, :, 1], segments[i, :, 2],
                            color=line_colors[i], linewidth=lw+1, alpha=line_alpha, zorder=zorder
                        )
            except Exception:
                ax.plot(x, y, z, color=line_color, alpha=line_alpha, linewidth=lw, zorder=zorder)
        else:
            ax.plot(x, y, z, color=line_color, alpha=line_alpha, linewidth=lw, zorder=zorder)

        if show_atoms:
            ax.scatter(x, y, z, c=scatter_color, s=80, alpha=scatter_alpha,
                       edgecolor='black' if not is_native else 'none',
                       linewidth=0.5, zorder=zorder + 1)

            if not is_native:
                ax.text(x[0], y[0], z[0] + 1.5, "N", color="blue", fontweight="bold", zorder=zorder+2)
                ax.text(x[-1], y[-1], z[-1] + 1.5, "C", color="red", fontweight="bold", zorder=zorder+2)

    # Align native to predicted if both provided
    if native_coords is not None and len(native_coords) == n_res:
        from ..core.backbone import kabsch_rmsd
        rmsd, aligned_native = kabsch_rmsd(native_coords, coords)
        draw_chain(ax, aligned_native, colors, is_native=True)
        title += f"\nRMSD to Native: {rmsd:.2f} Å"

    # Draw predicted structure
    draw_chain(ax, coords, colors, is_native=False)

    # If full backbone is available, draw it as sticks
    if backbone is not None:
        # backbone is (3*N, 3) -> [N, CA, C]
        for i in range(n_res):
            n = backbone[3 * i]
            ca = backbone[3 * i + 1]
            c = backbone[3 * i + 2]
            ax.plot([n[0], ca[0]], [n[1], ca[1]], [n[2], ca[2]], color="gray", alpha=0.4, linewidth=1)
            ax.plot([ca[0], c[0]], [ca[1], c[1]], [ca[2], c[2]], color="gray", alpha=0.4, linewidth=1)
            if i < n_res - 1:
                n_next = backbone[3 * (i + 1)]
                ax.plot([c[0], n_next[0]], [c[1], n_next[1]], [c[2], n_next[2]], color="gray", alpha=0.4, linewidth=1)

    ax.set_title(title, pad=20)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)
    _set_axes_equal_3d(ax)

    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        plt.savefig(filename, bbox_inches="tight", transparent=False)
        plt.close(fig)
    else:
        plt.show()


def plot_distance_matrix(
    coords: np.ndarray,
    native_coords: Optional[np.ndarray] = None,
    title: str = "Distance Matrix",
    filename: Optional[str] = None,
    cmap: str = "viridis_r",
):
    """Plot a 2D distance matrix."""
    n_res = len(coords)
    D_pred = np.zeros((n_res, n_res))
    for i in range(n_res):
        for j in range(n_res):
            D_pred[i, j] = np.linalg.norm(coords[i] - coords[j])

    fig, ax = plt.subplots(figsize=(8, 7))

    if native_coords is not None and len(native_coords) == n_res:
        D_nat = np.zeros((n_res, n_res))
        for i in range(n_res):
            for j in range(n_res):
                D_nat[i, j] = np.linalg.norm(native_coords[i] - native_coords[j])
        D_combined = np.zeros((n_res, n_res))
        i_upper = np.triu_indices(n_res, k=1)
        i_lower = np.tril_indices(n_res, k=-1)
        D_combined[i_upper] = D_pred[i_upper]
        D_combined[i_lower] = (D_pred - D_nat)[i_lower]
        im = ax.imshow(D_combined, cmap="seismic", origin="lower", vmin=-10, vmax=10)
        D_upper_only = np.tril(np.full_like(D_pred, np.nan), k=0)
        D_upper_only[i_upper] = D_pred[i_upper]
        im_dist = ax.imshow(D_upper_only, cmap=cmap, origin="lower", vmin=0, vmax=max(30, np.max(D_pred)))
        cbar1 = fig.colorbar(im_dist, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label("Distance (Å) [Upper Tri]")
        cbar2 = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.15)
        cbar2.set_label("Distance Error (Pred - Nat) (Å) [Lower Tri]")
        ax.plot([0, n_res - 1], [0, n_res - 1], "k--", alpha=0.5)
        ax.set_title(title + "\nUpper: Predicted | Lower: Error")
    else:
        im = ax.imshow(D_pred, cmap=cmap, origin="lower", vmin=0, vmax=max(30, np.max(D_pred)))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Distance (Å)")
        ax.set_title(title)

    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Residue Index")
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_ramachandran(
    phi: np.ndarray,
    psi: np.ndarray,
    sequence: Optional[str] = None,
    title: str = "Ramachandran Plot",
    filename: Optional[str] = None,
):
    """Plot a Ramachandran plot."""
    fig, ax = plt.subplots(figsize=(6, 6))
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-180, 60), 120, 120, color="blue", alpha=0.1, zorder=0))
    ax.add_patch(Rectangle((-180, -180), 120, 60, color="blue", alpha=0.1, zorder=0))
    ax.add_patch(Rectangle((-120, -90), 80, 80, color="green", alpha=0.1, zorder=0))
    ax.add_patch(Rectangle((30, 0), 80, 80, color="red", alpha=0.1, zorder=0))
    n = len(phi)
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(np.linspace(0, 1, n))
    for i in range(1, n - 1):
        res = sequence[i] if sequence else "X"
        ax.scatter(phi_deg[i], psi_deg[i], c=[colors[i]], s=60, edgecolor="black", zorder=2)
        if sequence and len(sequence) < 20:
            ax.text(phi_deg[i]+3, psi_deg[i]+3, f"{res}{i+1}", fontsize=8, zorder=3)
    ax.plot(phi_deg[1:-1], psi_deg[1:-1], "k-", alpha=0.3, linewidth=1, zorder=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"$\phi$ (degrees)")
    ax.set_ylabel(r"$\psi$ (degrees)")
    ax.set_title(title)
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_pipeline_dashboard(
    results: Dict,
    output_dir: str = "visualizations",
    prefix: str = "protein",
):
    """Generate a comprehensive multi-panel dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    coords = results["predicted_coords"]
    native = results.get("raw_coords", None)
    seq = results.get("sequence", "")
    backbone = results.get("predicted_backbone", None)

    if native is not None and np.array_equal(coords, native):
        native = None

    if "aligned_coords" in results:
         coords_to_plot = results["aligned_coords"]
         native_to_plot = results.get("raw_coords", native)
    else:
         coords_to_plot = coords
         native_to_plot = native

    plot_3d_structure(
        coords_to_plot, native_coords=native_to_plot, sequence=seq,
        title=f"3D Structure: {prefix}",
        filename=os.path.join(output_dir, f"{prefix}_3d_structure.png"),
        backbone=backbone
    )
    plot_distance_matrix(
        coords, native_coords=native,
        title=f"Distance Matrix: {prefix}",
        filename=os.path.join(output_dir, f"{prefix}_distance_matrix.png")
    )

    # Composite summary
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[0], projection="3d")
    n_res = len(coords_to_plot)
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(np.linspace(0, 1, n_res))
    ax1.plot(coords_to_plot[:,0], coords_to_plot[:,1], coords_to_plot[:,2], "k-", alpha=0.5)
    ax1.scatter(coords_to_plot[:,0], coords_to_plot[:,1], coords_to_plot[:,2], c=colors, s=50, edgecolor="black")
    ax1.set_title(f"{prefix} Structure")
    _set_axes_equal_3d(ax1)
    ax2 = fig.add_subplot(gs[1])
    D_pred = np.zeros((n_res, n_res))
    for i in range(n_res):
        for j in range(n_res):
            D_pred[i, j] = np.linalg.norm(coords[i] - coords[j])
    im = ax2.imshow(D_pred, cmap="viridis_r", origin="lower")
    plt.colorbar(im, ax=ax2, label="Distance (Å)")
    ax2.set_title("Cα Distance Map")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_dashboard.png"))
    plt.close(fig)
    print(f"  Saved visualisations to {output_dir}/")


def plot_interactive_3d(
    coords: np.ndarray,
    sequence: str,
    confidence_scores: np.ndarray,
    title: str = "Interactive 3D Structure (pLDDT)",
    filename: Optional[str] = None,
):
    """Generate an AlphaFold-style interactive 3D plot using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  Plotly not installed, skipping interactive 3D export.")
        return

    n_res = len(coords)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    hover_text = []
    for i in range(n_res):
        res = sequence[i] if i < len(sequence) else "X"
        score = confidence_scores[i]
        hover_text.append(f"Residue: {res}{i+1}<br>Confidence (pLDDT): {score:.1f}")

    # AlphaFold pLDDT colorscale
    colorscale = [
        [0.0, "rgb(255, 125, 69)"],    # Orange: < 50
        [0.5, "rgb(255, 125, 69)"],
        [0.5, "rgb(255, 219, 19)"],    # Yellow: 50 - 70
        [0.7, "rgb(255, 219, 19)"],
        [0.7, "rgb(101, 203, 243)"],   # Light Blue: 70 - 90
        [0.9, "rgb(101, 203, 243)"],
        [0.9, "rgb(0, 83, 214)"],      # Dark Blue: > 90
        [1.0, "rgb(0, 83, 214)"]
    ]

    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+lines",
        name="Backbone Trace",
        marker=dict(
            size=8,
            color=confidence_scores,
            colorscale=colorscale,
            cmin=0, cmax=100,
            colorbar=dict(
                title="pLDDT",
                tickvals=[25, 60, 80, 95],
                ticktext=["Very Low (<50)", "Low (50-70)", "Confident (70-90)", "Very High (>90)"]
            ),
            line=dict(width=1, color="black"),
        ),
        line=dict(color="rgba(100, 100, 100, 0.5)", width=5),
        text=hover_text,
        hoverinfo="text"
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        template="plotly_white"
    )

    if filename:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        fig.write_html(filename)
        print(f"  Exported Interactive HTML: {filename}")
    else:
        fig.show()
