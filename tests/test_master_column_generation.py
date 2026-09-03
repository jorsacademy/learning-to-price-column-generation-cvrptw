from __future__ import annotations

import pytest

from learning_to_price.column_generation import (
    ColumnGenerationConfig,
    PricingMode,
    run_column_generation,
)
from learning_to_price.domain import CVRPTWInstance
from learning_to_price.generator import generate_solomon_like
from learning_to_price.master import solve_integer_master, solve_restricted_master
from learning_to_price.routes import enumerate_feasible_routes, singleton_routes


def test_singleton_master_is_feasible(tiny_instance: CVRPTWInstance) -> None:
    master = solve_restricted_master(tiny_instance, singleton_routes(tiny_instance))
    assert master.success
    assert master.objective == pytest.approx(25.0)
    assert len(master.customer_duals) == 3


def test_exact_column_generation_matches_full_route_lp() -> None:
    instance = generate_solomon_like(7, seed=91, window_regime="tight").instance
    all_routes = enumerate_feasible_routes(instance)
    full_lp = solve_restricted_master(instance, all_routes)
    result = run_column_generation(instance, mode=PricingMode.EXACT)
    assert result.converged and result.globally_certified
    assert result.final_lp_objective == pytest.approx(full_lp.objective, abs=1e-7)


def test_hybrid_heuristic_remains_exact_with_fallback() -> None:
    instance = generate_solomon_like(7, seed=92).instance
    exact = run_column_generation(instance, mode=PricingMode.EXACT)
    hybrid = run_column_generation(
        instance,
        mode=PricingMode.HEURISTIC,
        config=ColumnGenerationConfig(keep_ratio=0.25, min_outgoing=1),
    )
    assert hybrid.globally_certified
    assert hybrid.exact_fallback_calls >= 1  # final exact no-column certificate is mandatory
    assert hybrid.final_lp_objective == pytest.approx(exact.final_lp_objective, abs=1e-7)


def test_integer_master_matches_full_enumeration_on_tiny_instance(
    tiny_instance: CVRPTWInstance,
) -> None:
    routes = enumerate_feasible_routes(tiny_instance)
    solution = solve_integer_master(tiny_instance, routes)
    assert solution.success
    assert solution.objective == pytest.approx(18.0)
