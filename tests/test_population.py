"""Tests for the population module."""

import numpy as np
from voronoi_agi.population import VoronoiPopulation, diversity_metrics, territorial_niching
from voronoi_agi.seeds import UniformSeedSampler


def dummy_factory(seed):
    return seed.copy()


def dummy_fitness(ind):
    return -float(np.sum((ind - 0.5) ** 2))


def make_test_population(n=20, dim=2):
    sampler = UniformSeedSampler(n_seeds=n, dim=dim)
    return VoronoiPopulation.from_sampler(sampler, dummy_factory, dummy_fitness)


class TestVoronoiPopulation:
    def test_from_sampler(self):
        pop = make_test_population(20)
        assert pop.n_individuals == 20
        assert pop.seeds.shape == (20, 2)

    def test_cell_areas(self):
        pop = make_test_population(20)
        areas = pop.cell_areas()
        assert areas.shape == (20,)
        assert np.all(areas >= 0)

    def test_cell_density(self):
        pop = make_test_population(20)
        density = pop.cell_density()
        assert density.shape == (20,)
        assert np.all(density > 0)

    def test_territorial_diversity(self):
        pop = make_test_population(20)
        div = pop.territorial_diversity()
        assert div > 0

    def test_single_individual_diversity(self):
        pop = make_test_population(1)
        assert pop.territorial_diversity() == 0.0


class TestDiversityMetrics:
    def test_metrics_return_dict(self):
        pop = make_test_population(20)
        metrics = diversity_metrics(pop)
        assert "avg_pairwise_distance" in metrics
        assert "territorial_diversity" in metrics
        assert "mean_cell_area" in metrics
        assert metrics["avg_pairwise_distance"] > 0

    def test_single_individual(self):
        pop = make_test_population(1)
        metrics = diversity_metrics(pop)
        assert metrics["avg_pairwise_distance"] == 0.0


class TestTerritorialNiching:
    def test_shared_fitness_shape(self):
        pop = make_test_population(20)
        shared = territorial_niching(pop)
        assert shared.shape == (20,)

    def test_shared_fitness_less_than_raw(self):
        pop = make_test_population(20)
        shared = territorial_niching(pop)
        assert np.all(shared <= pop.fitness)
