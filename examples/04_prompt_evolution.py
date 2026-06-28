#!/usr/bin/env python3
"""Example 4: Prompt evolution using Voronoi seeds.

Demonstrates using Voronoi seeds to structure a population of prompts
for LLM-based optimisation. Each seed encodes a prompt embedding; the
Voronoi cell structure maintains diversity across prompt space.
"""

import numpy as np

from voronoi_agi.seeds import UniformSeedSampler, PoissonDiskSeedSampler

# In a real system, these would be LLM prompt embeddings.
# Here we use random vectors to demonstrate the structure.
DIM = 8
N_PROMPTS = 30


def prompt_factory(seed: np.ndarray) -> dict:
    """Create a 'prompt' from a seed position.

    In practice, you'd decode the seed to a prompt string or embed it
    via the LLM. Here we just store the seed as metadata.
    """
    return {
        "embedding": seed,
        "template": "Solve this problem using step-by-step reasoning.",
        "temperature": float(0.5 + seed[0] * 0.5),
        "top_p": float(0.8 + seed[1] * 0.2),
    }


def prompt_quality(prompt: dict) -> float:
    """Mock fitness function — higher = better prompt.

    Replace with actual LLM evaluation (e.g., MMLU accuracy, instruction
    following score, etc.).
    """
    emb = prompt["embedding"]
    # Favour prompts near the centre of embedding space (simulates
    # a well-balanced instruction-following prompt)
    centre_quality = -np.sum((emb - 0.5) ** 2)
    # Penalise extreme temperature/p_p
    temp_quality = -abs(prompt["temperature"] - 0.7)
    return centre_quality + temp_quality


# Compare diversity between uniform and Poisson-disk seed distributions
for name, sampler_cls in [("Uniform", UniformSeedSampler), ("Poisson Disk", PoissonDiskSeedSampler)]:
    kwargs = {"radius": 0.12} if name == "Poisson Disk" else {}
    sampler = sampler_cls(n_seeds=N_PROMPTS, dim=DIM, **kwargs)
    seeds = sampler.sample()

    prompts = [prompt_factory(s) for s in seeds]
    fitness = np.array([prompt_quality(p) for p in prompts])

    print(f"\n=== {name} Seed Distribution ===")
    print(f"  Number of prompts: {N_PROMPTS}")
    print(f"  Best fitness: {fitness.max():.4f}")
    print(f"  Mean fitness: {fitness.mean():.4f}")
    print(f"  Seed span (per-dim range): {seeds.max(axis=0).mean() - seeds.min(axis=0).mean():.4f}")
    print(f"  Fitness diversity (CV): {np.std(fitness) / (np.mean(fitness) + 1e-12):.4f}")

print("\n--- Next Steps for Real Prompt Evolution ---")
print("1. Replace `prompt_factory` with an LLM prompt encoder/decoder")
print("2. Replace `prompt_quality` with actual LLM evaluation")
print("3. Use VoronoiGA to evolve prompt embeddings over generations")
print("4. Track prompt diversity via Voronoi cell areas")
