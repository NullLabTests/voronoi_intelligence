# Contributing to Voronoi Intelligence

We welcome contributions! Here's how you can help:

## Getting Started

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Install in dev mode: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Lint: `ruff check .`
6. Open a PR

## Guidelines

- Keep the API simple and composable
- Add tests for new functionality
- Document public functions with NumPy-style docstrings
- Use type hints everywhere
- Prefer numpy/scipy over custom implementations for numerical work
- 2D visualisation is for intuition; the core logic should work in n-dimensions

## Research Contributions

We especially welcome:
- New seed sampling strategies (learned, adaptive, etc.)
- Real-world benchmarks using Voronoi-enhanced EAs
- Multi-agent coverage control examples
- Integration demos with LLM-based agent frameworks
- Papers / citations to add to the README

## Questions?

Open an issue or start a discussion!
