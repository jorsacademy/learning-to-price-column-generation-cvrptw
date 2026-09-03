"""Route feasibility, evaluation, and finite enumeration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from learning_to_price.domain import CVRPTWInstance
from learning_to_price.geometry import distance_matrix


@dataclass(frozen=True, slots=True, order=True)
class Route:
    """An elementary depot-to-depot customer sequence."""

    customers: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.customers:
            raise ValueError("route must visit at least one customer")
        if any(node <= 0 for node in self.customers):
            raise ValueError("route customer IDs must be positive")
        if len(set(self.customers)) != len(self.customers):
            raise ValueError("route must be elementary")

    def arcs(self) -> tuple[tuple[int, int], ...]:
        nodes = (0, *self.customers, 0)
        return tuple(pairwise(nodes))


@dataclass(frozen=True, slots=True)
class RouteEvaluation:
    feasible: bool
    distance: float
    cost: float
    load: float
    service_start_times: tuple[float, ...]
    return_time: float
    failure_reason: str | None = None


def evaluate_route(instance: CVRPTWInstance, route: Route) -> RouteEvaluation:
    matrix = distance_matrix(instance)
    load = 0.0
    time = instance.depot_ready_time
    distance = 0.0
    previous = 0
    starts: list[float] = []

    for node in route.customers:
        if node > instance.customer_count:
            return RouteEvaluation(
                feasible=False,
                distance=distance,
                cost=instance.fixed_vehicle_cost + distance,
                load=load,
                service_start_times=tuple(starts),
                return_time=time,
                failure_reason=f"unknown customer {node}",
            )
        customer = instance.customer(node)
        load += customer.demand
        if load > instance.capacity + 1e-9:
            return RouteEvaluation(
                feasible=False,
                distance=distance,
                cost=instance.fixed_vehicle_cost + distance,
                load=load,
                service_start_times=tuple(starts),
                return_time=time,
                failure_reason="vehicle capacity exceeded",
            )
        travel = float(matrix[previous, node])
        distance += travel
        arrival = time + travel
        start = max(arrival, customer.ready_time)
        if start > customer.due_time + 1e-9:
            return RouteEvaluation(
                feasible=False,
                distance=distance,
                cost=instance.fixed_vehicle_cost + distance,
                load=load,
                service_start_times=tuple(starts),
                return_time=start,
                failure_reason=f"customer {node} time window violated",
            )
        starts.append(start)
        time = start + customer.service_time
        previous = node

    return_leg = float(matrix[previous, 0])
    distance += return_leg
    return_time = time + return_leg
    if return_time > instance.depot_due_time + 1e-9:
        return RouteEvaluation(
            feasible=False,
            distance=distance,
            cost=instance.fixed_vehicle_cost + distance,
            load=load,
            service_start_times=tuple(starts),
            return_time=return_time,
            failure_reason="depot horizon violated",
        )
    return RouteEvaluation(
        feasible=True,
        distance=distance,
        cost=instance.fixed_vehicle_cost + distance,
        load=load,
        service_start_times=tuple(starts),
        return_time=return_time,
    )


def route_cost(instance: CVRPTWInstance, route: Route) -> float:
    evaluation = evaluate_route(instance, route)
    if not evaluation.feasible:
        raise ValueError(f"route is infeasible: {evaluation.failure_reason}")
    return evaluation.cost


def singleton_routes(instance: CVRPTWInstance) -> tuple[Route, ...]:
    routes = tuple(Route((node,)) for node in range(1, instance.customer_count + 1))
    infeasible = [
        route.customers[0]
        for route in routes
        if not evaluate_route(instance, route).feasible
    ]
    if infeasible:
        raise ValueError(f"singleton routes are infeasible for customers {infeasible}")
    return routes


def enumerate_feasible_routes(
    instance: CVRPTWInstance,
    *,
    max_customers: int = 11,
    max_routes: int = 500_000,
) -> tuple[Route, ...]:
    """Enumerate every feasible elementary route for small instances.

    This is an independent finite oracle used in tests and tiny benchmarks. It is
    deliberately capped because the number of elementary routes is exponential.
    """

    if instance.customer_count > max_customers:
        raise ValueError(
            f"full route enumeration is capped at {max_customers} customers; "
            f"received {instance.customer_count}"
        )
    matrix = distance_matrix(instance)
    routes: list[Route] = []

    def extend(
        path: tuple[int, ...],
        visited_mask: int,
        node: int,
        load: float,
        time: float,
    ) -> None:
        for next_node in range(1, instance.customer_count + 1):
            bit = 1 << (next_node - 1)
            if visited_mask & bit:
                continue
            customer = instance.customer(next_node)
            new_load = load + customer.demand
            if new_load > instance.capacity + 1e-9:
                continue
            arrival = time + float(matrix[node, next_node])
            start = max(arrival, customer.ready_time)
            if start > customer.due_time + 1e-9:
                continue
            departure = start + customer.service_time
            if departure + float(matrix[next_node, 0]) > instance.depot_due_time + 1e-9:
                continue
            new_path = (*path, next_node)
            routes.append(Route(new_path))
            if len(routes) > max_routes:
                raise RuntimeError(
                    f"route enumeration exceeded max_routes={max_routes}; "
                    "reduce the instance size or tighten the cap"
                )
            extend(new_path, visited_mask | bit, next_node, new_load, departure)

    extend((), 0, 0, 0.0, instance.depot_ready_time)
    return tuple(routes)


def coverage_vector(instance: CVRPTWInstance, route: Route) -> tuple[int, ...]:
    covered = set(route.customers)
    return tuple(1 if node in covered else 0 for node in range(1, instance.customer_count + 1))
