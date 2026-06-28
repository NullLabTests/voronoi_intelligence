#!/usr/bin/env python3
"""Example 3: Multi-agent coverage control with Voronoi territories.

Agents dynamically partition a 2D domain. A density field attracts agents
to the centre — they adjust positions to improve coverage.
"""

import matplotlib.pyplot as plt
import numpy as np

from voronoi_agi.agents import TerritorialAgent, MultiAgentCoverage
from voronoi_agi.visualization import plot_territorial_coverage

N_AGENTS = 8
N_STEPS = 50


# Density field: Gaussian bump at centre
def density_field(x: np.ndarray) -> float:
    centre = np.array([0.5, 0.5])
    return np.exp(-5 * np.linalg.norm(x - centre) ** 2)


# Policy: move toward centroid of own territory
def centroid_policy(position: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    if len(vertices) < 3:
        return position
    centroid = vertices.mean(axis=0)
    direction = centroid - position
    step = 0.05 * direction / (np.linalg.norm(direction) + 1e-12)
    return position + step


# Initialise agents
rng = np.random.default_rng(42)
agents = [TerritorialAgent(agent_id=i, position=rng.uniform(0, 1, 2)) for i in range(N_AGENTS)]

coverage = MultiAgentCoverage(agents=agents)

# Run coverage control
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
snapshots = [0, 5, 10, 20, 35, 49]

for step in range(N_STEPS):
    territories = coverage.step_coverage(centroid_policy, density_field)
    if step in snapshots:
        idx = snapshots.index(step)
        ax = axes.flat[idx]
        plot_territorial_coverage(
            coverage,
            ax=ax,
            title=f"Step {step}  (imbalance: {coverage.imbalance():.3f})",
        )

plt.tight_layout()
plt.savefig("docs/coverage_evolution.png", dpi=150, bbox_inches="tight")
print("Saved coverage evolution to docs/coverage_evolution.png")
print(f"Final imbalance: {coverage.imbalance():.4f}")
plt.show()
