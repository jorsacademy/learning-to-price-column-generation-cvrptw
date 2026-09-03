from __future__ import annotations

from learning_to_price.domain import CVRPTWInstance
from learning_to_price.routes import Route, enumerate_feasible_routes, evaluate_route


def test_capacity_and_time_window_feasibility(
    tiny_instance: CVRPTWInstance,
    ordered_window_instance: CVRPTWInstance,
) -> None:
    assert evaluate_route(tiny_instance, Route((1, 2))).feasible
    assert not evaluate_route(tiny_instance, Route((1, 2, 3))).feasible
    assert evaluate_route(ordered_window_instance, Route((1, 2))).feasible
    reverse = evaluate_route(ordered_window_instance, Route((2, 1)))
    assert not reverse.feasible
    assert "time window" in str(reverse.failure_reason)


def test_full_enumeration_returns_unique_feasible_routes(tiny_instance: CVRPTWInstance) -> None:
    routes = enumerate_feasible_routes(tiny_instance)
    assert len(routes) == len(set(routes))
    assert all(evaluate_route(tiny_instance, route).feasible for route in routes)
    assert Route((1, 2)) in routes
    assert Route((1, 2, 3)) not in routes
