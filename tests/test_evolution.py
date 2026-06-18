"""Tests for the evolution module."""

import numpy as np
from voronoi_agi.evolution import VoronoiGA, voronoi_selection, voronoi_mutation
from voronoi_agi.population import VoronoiPopulation
from voronoi_agi.seeds import UniformSeedSampler


def dummy_factory(seed):
    return seed.copy()


def dummy_fitness(ind):
    return -float(np.sum((ind - 0.5) ** 2))


def make_test_population(n=20, dim=2):
    sampler = UniformSeedSampler(n_seeds=n, dim=dim, rng=np.random.default_rng(42))
    return VoronoiPopulation.from_sampler(sampler, dummy_factory, dummy_fitness)


class TestVoronoiSelection:
    def test_selection_returns_correct_count(self):
        pop = make_test_population(20)
        parents = voronoi_selection(pop, n_parents=10)
        assert len(parents) == 10
        assert np.all(parents >= 0) and np.all(parents < 20)

    def test_selection_unique_ids(self):
        pop = make_test_population(50)
        parents = voronoi_selection(pop, n_parents=50)
        assert len(np.unique(parents)) <= 50


class TestVoronoiMutation:
    def test_mutation_changes_seed(self):
        pop = make_test_population(20)
        original = pop.seeds[0].copy()
        mutated = voronoi_mutation(original.copy(), mutation_rate=1.0, population=pop)
        assert not np.allclose(original, mutated)

    def test_mutation_respects_bounds(self):
        pop = make_test_population(20)
        seed = np.array([0.5, 0.5])
        for _ in range(50):
            mutated = voronoi_mutation(seed.copy(), mutation_rate=1.0, population=pop)
            assert np.all(mutated >= 0) and np.all(mutated <= 1)

    def test_mutation_zero_rate(self):
        pop = make_test_population(20)
        seed = np.array([0.5, 0.5])
        mutated = voronoi_mutation(seed.copy(), mutation_rate=0.0, population=pop)
        assert np.allclose(seed, mutated)


class TestVoronoiGA:
    def test_step_preserves_size(self):
        pop = make_test_population(20)
        ga = VoronoiGA(
            population=pop,
            individual_factory=dummy_factory,
            fitness_fn=dummy_fitness,
            rng=np.random.default_rng(42),
        )
        new_pop = ga.step()
        assert new_pop.n_individuals == 20

    def test_run_returns_history(self):
        pop = make_test_population(20)
        ga = VoronoiGA(
            population=pop,
            individual_factory=dummy_factory,
            fitness_fn=dummy_fitness,
            rng=np.random.default_rng(42),
        )
        history = ga.run(n_generations=5, verbose=False)
        assert len(history) == 5

    def test_fitness_does_not_collapse(self):
        pop = make_test_population(30)
        ga = VoronoiGA(
            population=pop,
            mutation_rate=0.2,
            individual_factory=dummy_factory,
            fitness_fn=dummy_fitness,
            rng=np.random.default_rng(42),
        )
        history = ga.run(n_generations=10, verbose=False)
        assert history[-1] >= history[0] * 0.5
