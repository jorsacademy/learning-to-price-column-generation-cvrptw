from __future__ import annotations

from pathlib import Path

import pytest

from learning_to_price.domain import Customer, CVRPTWInstance, load_instance, save_instance
from learning_to_price.generator import generate_solomon_like
from learning_to_price.routes import Route, evaluate_route


def test_generator_is_deterministic_and_has_feasible_witness() -> None:
    first = generate_solomon_like(8, seed=17, distribution="clustered", window_regime="tight")
    second = generate_solomon_like(8, seed=17, distribution="clustered", window_regime="tight")
    assert first == second
    assert all(
        evaluate_route(first.instance, Route(route)).feasible for route in first.witness_routes
    )
    visited = sorted(node for route in first.witness_routes for node in route)
    assert visited == list(range(1, 9))


def test_instance_json_round_trip(tmp_path: Path) -> None:
    instance = generate_solomon_like(5, seed=3).instance
    path = tmp_path / "instance.json"
    save_instance(instance, path)
    assert load_instance(path) == instance


def test_instance_rejects_noncontiguous_customer_ids() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        CVRPTWInstance(
            name="bad",
            capacity=4.0,
            customers=(Customer(2, 0.0, 0.0, 1.0, 0.0, 10.0),),
        )
