#!/usr/bin/env python3
"""Example 2: Territory-aware GA benchmark on continuous optimisation problems.

Compares VoronoiGA (territory-aware mutation + neighbor-restricted crossover)
against a standard GA and a fitness-sharing GA on 2D benchmark functions.
"""

import matplotlib.pyplot as plt
import numpy as np

from voronoi_agi.evolution import VoronoiGA, StandardGA, FitnessSharingGA
from voronoi_agi.benchmarks import (
    FUNCTIONS,
)

POP_SIZE = 50
DIM = 2
N_GENS = 100
N_TRIALS = 5
SEED = 42


def run_trial(algo_class, fn, **kwargs) -> np.ndarray:
    histories = []
    for t in range(N_TRIALS):
        ga = algo_class(pop_size=POP_SIZE, dim=DIM, fitness_fn=fn, rng=np.random.default_rng(SEED + t), **kwargs)
        h = ga.run(n_generations=N_GENS, verbose=False)
        histories.append(h)
    return np.median(np.array(histories), axis=0)


# Run comparison
results = {}
for fn_name, fn in FUNCTIONS.items():
    print(f"\n=== {fn_name.upper()} ===")
    std = run_trial(StandardGA, fn)
    shr = run_trial(FitnessSharingGA, fn)
    vor = run_trial(VoronoiGA, fn)
    results[fn_name] = {"standard": std, "sharing": shr, "voronoi": vor}

    print(f"  Standard GA:      {std[-1]:.4f}")
    print(f"  Fitness Sharing:  {shr[-1]:.4f}")
    print(f"  VoronoiGA:        {vor[-1]:.4f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {"standard": "#4C72B0", "sharing": "#DD8452", "voronoi": "#55A868"}

for ax, (fn_name, fn) in zip(axes.flat, FUNCTIONS.items()):
    for algo_name, history in results[fn_name].items():
        ax.plot(history, label=algo_name, color=colors[algo_name], linewidth=2)
    ax.set_title(fn_name.capitalize(), fontsize=14)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
import os

save_path = os.path.join(os.path.dirname(__file__), "..", "docs", "benchmark_comparison.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nSaved benchmark plot to {save_path}")

# Print summary table
print(f"\n{'Function':<12} {'Standard':<12} {'Sharing':<12} {'Voronoi':<12} {'Improvement':<12}")
print("-" * 60)
for fn_name in FUNCTIONS:
    s = results[fn_name]["standard"][-1]
    h = results[fn_name]["sharing"][-1]
    v = results[fn_name]["voronoi"][-1]
    imp = (v - s) / abs(s) * 100 if s != 0 else 0
    print(f"{fn_name:<12} {s:<12.4f} {h:<12.4f} {v:<12.4f} {imp:<+11.1f}%")
