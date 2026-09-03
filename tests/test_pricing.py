from __future__ import annotations

import numpy as np
import pytest

from learning_to_price.domain import CVRPTWInstance
from learning_to_price.generator import generate_solomon_like
from learning_to_price.pricing import brute_force_price, price_routes, reduced_cost
from learning_to_price.routes import Route


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_label_setting_matches_full_enumeration(seed: int) -> None:
    instance = generate_solomon_like(6, seed=seed, window_regime="tight").instance
    duals = np.random.default_rng(seed).normal(45.0, 20.0, size=instance.customer_count)
    exact = price_routes(instance, duals, top_k=5)
    brute = brute_force_price(instance, duals, top_k=5)
    assert exact.globally_certified
    assert exact.best is not None and brute.best is not None
    assert exact.best.reduced_cost == pytest.approx(brute.best.reduced_cost, abs=1e-8)
    assert [candidate.reduced_cost for candidate in exact.candidates] == pytest.approx(
        [candidate.reduced_cost for candidate in brute.candidates],
        abs=1e-8,
    )


def test_reduced_cost_is_route_cost_minus_customer_duals(tiny_instance: CVRPTWInstance) -> None:
    route = Route((1, 2))
    assert reduced_cost(tiny_instance, route, (3.0, 4.0, 5.0)) == pytest.approx(2.0)


def test_pruned_graph_is_not_globally_certified(tiny_instance: CVRPTWInstance) -> None:
    allowed = np.zeros((4, 4), dtype=bool)
    allowed[0, 1:] = True
    allowed[1:, 0] = True
    result = price_routes(tiny_instance, (8.0, 8.0, 8.0), allowed_arcs=allowed)
    assert not result.globally_certified
    assert result.stats.allowed_arc_fraction < 1.0
