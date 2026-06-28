"""Visualization tools for Voronoi-based AGI systems.

Provides matplotlib-based plotting (optionally with plotly) for 2D intuition
and higher-dimensional projection views.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .population import VoronoiPopulation
from .agents import MultiAgentCoverage


def plot_voronoi_2d(
    seeds: NDArray,
    ax: Any = None,
    show_vertices: bool = True,
    title: str = "Voronoi Tessellation",
    cmap: str = "viridis",
    alpha: float = 0.3,
) -> Any:
    """Plot a 2D Voronoi diagram with coloured cells.

    Parameters
    ----------
    seeds : ndarray of shape (n, 2)
        Generator points.
    ax : matplotlib Axes, optional
    show_vertices : bool
        Mark vertex positions.
    title, cmap, alpha : misc formatting.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from scipy.spatial import Voronoi, voronoi_plot_2d

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    vor = Voronoi(seeds)
    voronoi_plot_2d(vor, ax=ax, show_vertices=show_vertices, show_points=True)

    # Colour each region
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        polygon = vor.vertices[region]
        colour = plt.cm.get_cmap(cmap)(i / max(len(seeds) - 1, 1))
        ax.fill(polygon[:, 0], polygon[:, 1], color=colour, alpha=alpha)

    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def plot_population_diversity(
    population: VoronoiPopulation,
    ax: Any = None,
    colour_by: str = "fitness",
    title: str = "Population Diversity via Voronoi",
) -> Any:
    """Scatter plot of seeds, coloured by fitness or cell area.

    Parameters
    ----------
    population : VoronoiPopulation
    ax : matplotlib Axes, optional
    colour_by : "fitness" or "area"
    title : str

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    seeds = population.seeds
    if colour_by == "fitness":
        values = population.fitness
        cmap = "plasma"
    else:
        values = population.cell_areas()
        cmap = "viridis"

    scatter = ax.scatter(seeds[:, 0], seeds[:, 1], c=values, cmap=cmap, s=50, edgecolors="k", alpha=0.8)
    plt.colorbar(scatter, ax=ax, label=colour_by)
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    return ax


def plot_territorial_coverage(
    coverage: MultiAgentCoverage,
    ax: Any = None,
    title: str = "Agent Territorial Coverage",
) -> Any:
    """Plot agent territories for a MultiAgentCoverage system."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    colours = plt.cm.tab10(np.linspace(0, 1, len(coverage.agents)))

    for i, (agent, territory) in enumerate(zip(coverage.agents, coverage.territories)):
        ax.plot(agent.position[0], agent.position[1], "o", color=colours[i], markersize=8)
        if len(territory.vertices) >= 3:
            poly = MplPolygon(territory.vertices, closed=True, color=colours[i], alpha=0.15, ec="k", lw=0.5)
            ax.add_patch(poly)

    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    return ax


def voronoi_heatmap(
    seeds: NDArray,
    values: NDArray,
    resolution: int = 200,
    ax: Any = None,
    title: str = "Voronoi Heatmap",
) -> Any:
    """Rasterise a 2D Voronoi diagram coloured by per-cell values.

    Parameters
    ----------
    seeds : ndarray (n, 2)
    values : ndarray (n,)
    resolution : grid size (resolution x resolution)
    """
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    tree = cKDTree(seeds)

    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    _, idx = tree.query(grid_points)
    z = values[idx].reshape(resolution, resolution)

    ax.imshow(
        z,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="viridis",
        alpha=0.8,
        aspect="equal",
    )
    ax.scatter(seeds[:, 0], seeds[:, 1], c="k", s=20)
    ax.set_title(title)
    return ax


def animate_evolution(
    seed_history: list[NDArray],
    fitness_history: list[NDArray],
    interval: int = 200,
    save_path: Optional[str] = None,
) -> Any:
    """Animate a population evolving through Voronoi space.

    Parameters
    ----------
    seed_history : list of (n, 2) arrays, one per generation.
    fitness_history : list of (n,) arrays.
    interval : ms between frames.
    save_path : optional path to save GIF.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from scipy.spatial import Voronoi, voronoi_plot_2d

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        seeds = seed_history[frame]
        fitness = fitness_history[frame]

        vor = Voronoi(seeds)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False)

        scatter = ax.scatter(seeds[:, 0], seeds[:, 1], c=fitness, cmap="plasma", s=50, edgecolors="k")
        ax.set_title(f"Generation {frame}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        return [scatter]

    ani = animation.FuncAnimation(fig, update, frames=len(seed_history), interval=interval, blit=False)

    if save_path:
        ani.save(save_path, writer="pillow", fps=1000 // interval)

    return ani


def plot_high_dim_projection(
    seeds: NDArray,
    values: Optional[NDArray] = None,
    method: str = "pca",
    title: str = "High-D Seed Projection",
    ax: Any = None,
) -> Any:
    """Project high-dimensional seeds to 2D via PCA or t-SNE for visualisation.

    Parameters
    ----------
    seeds : ndarray (n, d) with d > 2
    values : ndarray (n,) or None — colours the points.
    method : "pca" or "tsne"
    title, ax : formatting.

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if seeds.shape[1] > 2:
        if method == "tsne":
            try:
                from sklearn.manifold import TSNE

                proj = TSNE(n_components=2, random_state=42, perplexity=min(30, seeds.shape[0] - 1)).fit_transform(
                    seeds
                )
            except ImportError:
                method = "pca"

        if method == "pca":
            from sklearn.decomposition import PCA

            proj = PCA(n_components=2).fit_transform(seeds)
    else:
        proj = seeds

    if values is not None:
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=values, cmap="plasma", s=50, edgecolors="k", alpha=0.8)
        plt.colorbar(sc, ax=ax)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=50, edgecolors="k", alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    return ax
