from __future__ import annotations

import pytest

from learning_to_price.domain import Customer, CVRPTWInstance


@pytest.fixture
def tiny_instance() -> CVRPTWInstance:
    return CVRPTWInstance(
        name="tiny",
        capacity=4.0,
        depot_x=0.0,
        depot_y=0.0,
        depot_due_time=100.0,
        fixed_vehicle_cost=5.0,
        customers=(
            Customer(1, 1.0, 0.0, 2.0, 0.0, 100.0, 0.0),
            Customer(2, 2.0, 0.0, 2.0, 0.0, 100.0, 0.0),
            Customer(3, 0.0, 2.0, 2.0, 0.0, 100.0, 0.0),
        ),
    )


@pytest.fixture
def ordered_window_instance() -> CVRPTWInstance:
    return CVRPTWInstance(
        name="ordered-windows",
        capacity=10.0,
        depot_x=0.0,
        depot_y=0.0,
        depot_due_time=30.0,
        fixed_vehicle_cost=0.0,
        customers=(
            Customer(1, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0),
            Customer(2, 2.0, 0.0, 1.0, 5.0, 10.0, 0.0),
        ),
    )
