"""Tests for the agents module."""

import numpy as np
from voronoi_agi.agents import (
    TerritorialAgent,
    MultiAgentCoverage,
    territorial_assignment,
    AgentTerritory,
)


def null_policy(position, vertices):
    return position


class TestTerritorialAgent:
    def test_act_within_territory(self):
        agent = TerritorialAgent(agent_id=0, position=np.array([0.5, 0.5]))
        territory = AgentTerritory(
            agent_id=0,
            seed=np.array([0.5, 0.5]),
            vertices=np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
            area=1.0,
            neighbours=[1],
        )
        new_pos = agent.act_within_territory(territory, null_policy)
        assert np.allclose(new_pos, [0.5, 0.5])


class TestMultiAgentCoverage:
    def _make_agents(self, n=5):
        rng = np.random.default_rng(42)
        return [
            TerritorialAgent(agent_id=i, position=rng.uniform(0.1, 0.9, 2))
            for i in range(n)
        ]

    def test_init_creates_territories(self):
        agents = self._make_agents(5)
        coverage = MultiAgentCoverage(agents=agents)
        assert len(coverage.territories) == 5
        assert all(hasattr(t, "area") for t in coverage.territories)

    def test_seeds_property(self):
        agents = self._make_agents(5)
        coverage = MultiAgentCoverage(agents=agents)
        assert coverage.seeds.shape == (5, 2)

    def test_step_updates_positions(self):
        agents = self._make_agents(5)
        coverage = MultiAgentCoverage(agents=agents)
        territories_after = coverage.step_coverage(null_policy)
        assert len(territories_after) == 5

    def test_imbalance(self):
        agents = self._make_agents(5)
        coverage = MultiAgentCoverage(agents=agents)
        imb = coverage.imbalance()
        assert imb >= 0

    def test_coverage_gap(self):
        agents = self._make_agents(5)
        coverage = MultiAgentCoverage(agents=agents)
        gap = coverage.coverage_gap()
        assert gap >= 0

    def test_two_agents(self):
        agents = self._make_agents(2)
        coverage = MultiAgentCoverage(agents=agents)
        assert len(coverage.territories) == 2


class TestTerritorialAssignment:
    def test_assignment_returns_correct_types(self):
        positions = np.random.uniform(0.1, 0.9, (10, 2))
        vor, territories = territorial_assignment(positions)
        assert len(territories) == 10
        assert all(isinstance(t, AgentTerritory) for t in territories)

    def test_area_positive(self):
        positions = np.random.uniform(0.1, 0.9, (10, 2))
        _, territories = territorial_assignment(positions)
        assert all(t.area > 0 for t in territories)
