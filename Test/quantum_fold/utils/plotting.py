"""
plotting.py
Publication-quality visualisations for quantum protein folding experiments.

All plots use Matplotlib with consistent styling and LaTeX-ready fonts.
Figures are saved at 300 DPI in PNG format by default.

Provides:
  • 3D fold structure plots (with H/P colouring and HH contact lines)
  • Energy convergence plots with confidence bands
  • Contact maps (heatmaps)
  • Algorithm comparison bar charts
  • Energy distribution histograms
  • Approximation ratio comparison plots
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Dict, List, Optional, Tuple


# ─── Global style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Colour palette
_H_COLOR = "#D32F2F"   # hydrophobic: deep red
_P_COLOR = "#1976D2"   # polar: blue
_BACKBONE_COLOR = "#616161"
_CONTACT_COLOR = "#FF9800"  # H-H contacts: orange dashed


def plot_fold(
    coords: List,
    sequence: str,
    title: str = "Protein Fold",
    filename: Optional[str] = None,
    show_contacts: bool = True,
    figsize: Tuple[float, float] = (8, 8),
):
    """
    Publication-quality 3D structure plot.

    Parameters
    ----------
    coords : list of array-like
        3D coordinates for each bead.
    sequence : str
        Residue sequence (H/P).
    title : str
        Plot title.
    filename : str, optional
        Save to this path. If None, plt.show().
    show_contacts : bool
        Draw dashed lines for H-H contacts.
    figsize : tuple
        Figure size.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    xs = [float(c[0]) for c in coords]
    ys = [float(c[1]) for c in coords]
    zs = [float(c[2]) for c in coords]

    # Backbone
    ax.plot(
        xs, ys, zs,
        color=_BACKBONE_COLOR, linestyle="-", linewidth=2.5,
        alpha=0.7, zorder=1,
    )

    # Beads
    for i, (x, y, z, res) in enumerate(zip(xs, ys, zs, sequence)):
        color = _H_COLOR if res == "H" else _P_COLOR
        size = 200 if res == "H" else 120
        marker = "o" if res == "H" else "s"
        ax.scatter(
            [x], [y], [z],
            c=color, s=size, marker=marker,
            edgecolors="k", linewidth=0.8, zorder=2,
            depthshade=True,
        )

    # H-H contacts
    if show_contacts:
        n = len(coords)
        for i in range(n):
            for j in range(i + 2, n):
                d2 = sum((float(coords[i][k]) - float(coords[j][k])) ** 2 for k in range(3))
                if abs(d2 - 1.0) < 0.01 and sequence[i] == "H" and sequence[j] == "H":
                    ax.plot(
                        [xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                        color=_CONTACT_COLOR, linestyle="--", linewidth=1.5,
                        alpha=0.8, zorder=0,
                    )

    # N/C-terminal labels
    ax.text(xs[0], ys[0], zs[0], " N", fontsize=9, fontweight="bold", color="green")
    ax.text(xs[-1], ys[-1], zs[-1], " C", fontsize=9, fontweight="bold", color="purple")

    # Equal aspect ratio
    max_range = max(
        max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0
    ) / 2.0 + 0.5
    mid_x = (max(xs) + min(xs)) * 0.5
    mid_y = (max(ys) + min(ys)) * 0.5
    mid_z = (max(zs) + min(zs)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X (lattice units)")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, pad=15)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_H_COLOR,
               markersize=10, label="H (hydrophobic)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_P_COLOR,
               markersize=8, label="P (polar)"),
    ]
    if show_contacts:
        legend_elements.append(
            Line2D([0], [0], color=_CONTACT_COLOR, linestyle="--",
                   linewidth=1.5, label="H-H contact"),
        )
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

    if filename:
        plt.savefig(filename)
        print(f"  Saved: {filename}")
    else:
        plt.show()
    plt.close()


def plot_convergence(
    histories: Dict[str, List[float]],
    title: str = "Energy Convergence",
    filename: Optional[str] = None,
    ylabel: str = "CVaR Energy",
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Plot energy convergence curves for multiple algorithms.

    Parameters
    ----------
    histories : dict
        Keys are algorithm names, values are lists of energy per iteration.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(histories), 1)))
    for (name, hist), color in zip(histories.items(), colors):
        iters = np.arange(len(hist))
        ax.plot(iters, hist, label=name, color=color, linewidth=1.8)

        # Running minimum
        running_min = np.minimum.accumulate(hist)
        ax.plot(
            iters, running_min,
            color=color, linestyle="--", linewidth=1.0, alpha=0.5,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if filename:
        plt.savefig(filename)
        print(f"  Saved: {filename}")
    else:
        plt.show()
    plt.close()


def plot_contact_map(
    coords: List,
    sequence: str,
    title: str = "Contact Map",
    filename: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 7),
):
    """
    Plot a contact map heatmap showing pairwise bead distances.
    """
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d2 = sum((float(coords[i][k]) - float(coords[j][k])) ** 2 for k in range(3))
            dist_matrix[i, j] = np.sqrt(d2)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        dist_matrix, cmap="RdYlBu_r", origin="lower",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Distance (lattice units)")

    # Mark H-H contacts
    for i in range(n):
        for j in range(i + 2, n):
            if dist_matrix[i, j] <= 1.01 and sequence[i] == "H" and sequence[j] == "H":
                ax.plot(j, i, "k*", markersize=8)
                ax.plot(i, j, "k*", markersize=8)

    # Residue labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"{sequence[i]}{i}" for i in range(n)], fontsize=8)
    ax.set_yticklabels([f"{sequence[i]}{i}" for i in range(n)], fontsize=8)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Residue")
    ax.set_title(title)

    if filename:
        plt.savefig(filename)
        print(f"  Saved: {filename}")
    else:
        plt.show()
    plt.close()


def plot_energy_distribution(
    energies: List[float],
    optimal_energy: Optional[float] = None,
    title: str = "Sampled Energy Distribution",
    filename: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
):
    """
    Histogram of sampled energies from the quantum circuit.
    """
    fig, ax = plt.subplots(figsize=figsize)

    energies = np.array(energies)
    # Filter out extremely high penalty values for cleaner plot
    plot_mask = energies < np.percentile(energies, 95) + 10
    plot_energies = energies[plot_mask]

    ax.hist(
        plot_energies, bins=min(50, len(np.unique(plot_energies))),
        color="#5C6BC0", edgecolor="white", alpha=0.85, density=True,
    )

    if optimal_energy is not None:
        ax.axvline(
            optimal_energy, color=_H_COLOR, linestyle="--",
            linewidth=2, label=f"Optimal E*={optimal_energy:.1f}",
        )
        ax.legend()

    ax.set_xlabel("Energy")
    ax.set_ylabel("Density")
    ax.set_title(title)

    if filename:
        plt.savefig(filename)
        print(f"  Saved: {filename}")
    else:
        plt.show()
    plt.close()


def plot_algorithm_comparison(
    results: Dict[str, Dict],
    metric: str = "energy",
    title: str = "Algorithm Comparison",
    filename: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Bar chart comparing algorithms on a chosen metric.

    Parameters
    ----------
    results : dict
        Keys = algorithm names, values = dicts with metric keys.
    metric : str
        "energy", "time", "success_rate", "approx_ratio".
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor="k", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")

    if filename:
        plt.savefig(filename)
        print(f"  Saved: {filename}")
    else:
        plt.show()
    plt.close()
