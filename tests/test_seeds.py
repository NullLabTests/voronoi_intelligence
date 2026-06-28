"""Tests for the seeds module — including new samplers."""

import pytest
import numpy as np
from voronoi_agi.seeds import (
    UniformSeedSampler,
    PoissonDiskSeedSampler,
    GaussianSeedSampler,
    SobolSeedSampler,
    SphericalSeedSampler,
)
from voronoi_agi.agents import centroidal_voronoi_tessellation
from voronoi_agi.visualization import plot_high_dim_projection


class TestSeedSamplers:
    def test_uniform_shape(self):
        sampler = UniformSeedSampler(n_seeds=100, dim=2)
        seeds = sampler.sample()
        assert seeds.shape == (100, 2)

    def test_uniform_bounds(self):
        sampler = UniformSeedSampler(n_seeds=50, dim=2, bounds=np.array([[0, 1], [0, 1]]))
        seeds = sampler.sample()
        assert seeds.min() >= 0 and seeds.max() <= 1

    def test_uniform_custom_bounds(self):
        sampler = UniformSeedSampler(
            n_seeds=50,
            dim=2,
            bounds=np.array([[-1, 1], [-2, 2]]),
        )
        seeds = sampler.sample()
        assert seeds[:, 0].min() >= -1 and seeds[:, 0].max() <= 1
        assert seeds[:, 1].min() >= -2 and seeds[:, 1].max() <= 2

    def test_poisson_disk(self):
        sampler = PoissonDiskSeedSampler(n_seeds=30, dim=2, radius=0.15)
        seeds = sampler.sample()
        assert seeds.shape == (30, 2)

    def test_gaussian(self):
        sampler = GaussianSeedSampler(n_seeds=50, dim=2, n_centers=3)
        seeds = sampler.sample()
        assert seeds.shape == (50, 2)

    def test_sobol(self):
        sampler = SobolSeedSampler(n_seeds=64, dim=3)
        seeds = sampler.sample()
        assert seeds.shape == (64, 3)

    def test_spherical(self):
        sampler = SphericalSeedSampler(n_seeds=50, dim=3, radius=1.0)
        seeds = sampler.sample()
        assert seeds.shape == (50, 3)
        assert np.all(seeds >= 0) and np.all(seeds <= 1)

    def test_high_dim(self):
        sampler = UniformSeedSampler(n_seeds=20, dim=10)
        seeds = sampler.sample()
        assert seeds.shape == (20, 10)

    def test_reproducible(self):
        sampler1 = UniformSeedSampler(n_seeds=50, dim=2, rng=np.random.default_rng(42))
        sampler2 = UniformSeedSampler(n_seeds=50, dim=2, rng=np.random.default_rng(42))
        assert np.allclose(sampler1.sample(), sampler2.sample())

    def test_bounds_validation(self):
        try:
            UniformSeedSampler(n_seeds=5, dim=2, bounds=np.array([[0, 1]]))
            assert False, "Should have raised"
        except ValueError:
            pass


class TestLloyd:
    def test_centroidal_output_shape(self):
        seeds = np.random.uniform(0, 1, (20, 2))
        result = centroidal_voronoi_tessellation(seeds, n_iterations=5)
        assert result.shape == (20, 2)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_centroidal_reduces_imbalance(self):
        from voronoi_agi.agents import MultiAgentCoverage, TerritorialAgent

        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 1, (10, 2))
        agents = [TerritorialAgent(agent_id=i, position=p) for i, p in enumerate(positions)]
        cov = MultiAgentCoverage(agents=agents)
        imb_before = cov.imbalance()

        new_seeds = centroidal_voronoi_tessellation(cov.seeds, n_iterations=20)
        agents2 = [TerritorialAgent(agent_id=i, position=new_seeds[i]) for i in range(len(new_seeds))]
        cov2 = MultiAgentCoverage(agents=agents2)
        imb_after = cov2.imbalance()
        assert imb_after <= imb_before + 0.01


class TestHighDimProjection:
    def test_pca_projection(self):
        pytest.importorskip("sklearn")
        seeds = np.random.uniform(0, 1, (30, 10))
        import matplotlib

        matplotlib.use("Agg")
        ax = plot_high_dim_projection(seeds, method="pca")
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")
