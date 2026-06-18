"""voronoi_agi — Voronoi Intelligence for AGI Research.

Voronoi-based seed management for evolutionary algorithms, agent populations,
and spatial reasoning in artificial general intelligence systems.
"""

from ._version import __version__
from .seeds import (
    SeedSampler,
    UniformSeedSampler,
    PoissonDiskSeedSampler,
    GaussianSeedSampler,
    SobolSeedSampler,
    seed_region_area,
    seed_region_vertices,
)
from .population import (
    VoronoiPopulation,
    init_population_from_seeds,
    diversity_metrics,
    territorial_niching,
)
from .evolution import (
    VoronoiGA,
    voronoi_selection,
    voronoi_mutation,
    novelty_search,
)
from .agents import (
    AgentTerritory,
    TerritorialAgent,
    MultiAgentCoverage,
    territorial_assignment,
)
from .visualization import (
    plot_voronoi_2d,
    plot_population_diversity,
    plot_territorial_coverage,
    animate_evolution,
    voronoi_heatmap,
)

__all__ = [
    "__version__",
    "SeedSampler",
    "UniformSeedSampler",
    "PoissonDiskSeedSampler",
    "GaussianSeedSampler",
    "SobolSeedSampler",
    "seed_region_area",
    "seed_region_vertices",
    "VoronoiPopulation",
    "init_population_from_seeds",
    "diversity_metrics",
    "territorial_niching",
    "VoronoiGA",
    "voronoi_selection",
    "voronoi_mutation",
    "novelty_search",
    "AgentTerritory",
    "TerritorialAgent",
    "MultiAgentCoverage",
    "territorial_assignment",
    "plot_voronoi_2d",
    "plot_population_diversity",
    "plot_territorial_coverage",
    "animate_evolution",
    "voronoi_heatmap",
]
