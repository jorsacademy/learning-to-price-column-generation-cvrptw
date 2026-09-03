"""Solomon-like synthetic CVRPTW instance generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from learning_to_price.domain import CVRPTWInstance, Customer, Distribution, WindowRegime
from learning_to_price.geometry import euclidean_distance


@dataclass(frozen=True, slots=True)
class GeneratedInstance:
    instance: CVRPTWInstance
    witness_routes: tuple[tuple[int, ...], ...]


def _coordinates(
    rng: np.random.Generator,
    customer_count: int,
    distribution: Distribution,
) -> np.ndarray:
    if distribution == "uniform":
        return rng.uniform(5.0, 95.0, size=(customer_count, 2))
    if distribution == "clustered":
        cluster_count = max(2, min(4, round(math.sqrt(customer_count))))
        centers = rng.uniform(15.0, 85.0, size=(cluster_count, 2))
        assignments = rng.integers(0, cluster_count, size=customer_count)
        noise = rng.normal(0.0, 8.0, size=(customer_count, 2))
        return np.clip(centers[assignments] + noise, 2.0, 98.0)
    raise ValueError(f"unsupported distribution: {distribution}")


def generate_solomon_like(
    customer_count: int,
    *,
    seed: int,
    distribution: Distribution = "uniform",
    window_regime: WindowRegime = "wide",
    fixed_vehicle_cost: float = 35.0,
) -> GeneratedInstance:
    """Generate a reproducible feasible CVRPTW instance.

    A hidden constructive route partition is used only to place time windows so
    that at least one complete route set is feasible. It is returned as a witness
    for tests and diagnostics, not used by the optimization algorithms.
    """

    if customer_count < 2:
        raise ValueError("customer_count must be at least 2")
    rng = np.random.default_rng(seed)
    depot = (50.0, 50.0)
    coordinates = _coordinates(rng, customer_count, distribution)
    demands = rng.integers(1, 10, size=customer_count).astype(float)

    target_routes = max(2, round(customer_count / 3.5))
    capacity = max(float(demands.max()), math.ceil(float(demands.sum()) / target_routes * 1.15))
    permutation = [int(value) + 1 for value in rng.permutation(customer_count)]

    groups: list[list[int]] = []
    current: list[int] = []
    current_load = 0.0
    for node in permutation:
        demand = float(demands[node - 1])
        if current and current_load + demand > capacity:
            groups.append(current)
            current = []
            current_load = 0.0
        current.append(node)
        current_load += demand
    if current:
        groups.append(current)

    base_width = 105.0 if window_regime == "wide" else 42.0
    ready = np.zeros(customer_count, dtype=float)
    due = np.zeros(customer_count, dtype=float)
    service = rng.uniform(3.0, 7.0, size=customer_count)
    max_return = 0.0

    for route in groups:
        time = 0.0
        previous_xy = depot
        for node in route:
            node_xy = tuple(float(value) for value in coordinates[node - 1])
            arrival = time + euclidean_distance(previous_xy, node_xy)
            width = base_width * float(rng.uniform(0.85, 1.15))
            left = width * float(rng.uniform(0.15, 0.35))
            right = width - left
            ready[node - 1] = max(0.0, arrival - left)
            due[node - 1] = arrival + right
            time = max(arrival, ready[node - 1]) + float(service[node - 1])
            previous_xy = node_xy
        max_return = max(max_return, time + euclidean_distance(previous_xy, depot))

    depot_due = max(220.0, max_return + base_width * 0.75)
    customers = tuple(
        Customer(
            id=index + 1,
            x=float(coordinates[index, 0]),
            y=float(coordinates[index, 1]),
            demand=float(demands[index]),
            ready_time=float(ready[index]),
            due_time=float(min(due[index], depot_due - 1e-6)),
            service_time=float(service[index]),
        )
        for index in range(customer_count)
    )
    instance = CVRPTWInstance(
        name=f"synthetic-{distribution}-{window_regime}-n{customer_count}-s{seed}",
        customers=customers,
        capacity=float(capacity),
        depot_x=depot[0],
        depot_y=depot[1],
        depot_due_time=float(depot_due),
        fixed_vehicle_cost=fixed_vehicle_cost,
    )
    return GeneratedInstance(
        instance=instance,
        witness_routes=tuple(tuple(route) for route in groups),
    )
