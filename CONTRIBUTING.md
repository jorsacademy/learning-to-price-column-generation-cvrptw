# Contributing

Changes should improve correctness, auditability, reproducibility, or experimental quality.

## Local quality gate

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest --cov=learning_to_price --cov-report=term-missing
```

## Requirements

- Add regression tests for pricing, dominance, master, feasibility, learning, or reporting changes.
- Keep full exact pricing available independently of learned components.
- Never allow a heuristic or learned model to certify column-generation convergence.
- Compare new pricing methods with exact and matched-density random controls.
- Separate arc-classification metrics from optimization metrics.
- Do not report restricted integer-master solutions as global CVRPTW optima.
- Keep training, validation, and test problem instances disjoint.
- Record random seeds and retain unfavorable outcomes.
- Update the exactness and methodology documentation when changing solver semantics.

Contributions are provided under the repository's PolyForm Noncommercial 1.0.0 terms unless agreed otherwise in writing.
