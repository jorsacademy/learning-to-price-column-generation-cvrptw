"""Exact and graph-restricted ESPPRC pricing for CVRPTW column generation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from learning_to_price.domain import CVRPTWInstance
from learning_to_price.geometry import distance_matrix
from learning_to_price.routes import Route, enumerate_feasible_routes, route_cost


@dataclass(frozen=True, slots=True)
class PricedRoute:
    route: Route
    route_cost: float
    reduced_cost: float


@dataclass(frozen=True, slots=True)
class PricingStats:
    labels_created: int
    labels_expanded: int
    labels_dominated: int
    feasible_extensions: int
    arc_checks: int
    complete_routes_considered: int
    allowed_arc_fraction: float
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class PricingResult:
    candidates: tuple[PricedRoute, ...]
    globally_certified: bool
    stats: PricingStats

    @property
    def best(self) -> PricedRoute | None:
        return self.candidates[0] if self.candidates else None


@dataclass(slots=True)
class _Label:
    node: int
    mask: int
    load: float
    departure_time: float
    reduced_cost_open: float
    path: tuple[int, ...]


def _validate_duals(instance: CVRPTWInstance, duals: tuple[float, ...] | np.ndarray) -> np.ndarray:
    values = np.asarray(duals, dtype=float)
    if values.shape != (instance.customer_count,):
        raise ValueError(
            f"expected {instance.customer_count} customer duals, received shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("customer duals must be finite")
    return values


def all_arcs(instance: CVRPTWInstance) -> np.ndarray:
    count = instance.customer_count + 1
    allowed = np.ones((count, count), dtype=bool)
    np.fill_diagonal(allowed, False)
    return allowed


def allowed_arc_fraction(allowed: np.ndarray) -> float:
    if allowed.ndim != 2 or allowed.shape[0] != allowed.shape[1]:
        raise ValueError("allowed arc matrix must be square")
    count = allowed.shape[0]
    denominator = count * (count - 1)
    return float(np.count_nonzero(allowed) / denominator) if denominator else 0.0


def reduced_cost(
    instance: CVRPTWInstance,
    route: Route,
    customer_duals: tuple[float, ...] | np.ndarray,
) -> float:
    duals = _validate_duals(instance, customer_duals)
    return route_cost(instance, route) - sum(duals[node - 1] for node in route.customers)


def price_routes(
    instance: CVRPTWInstance,
    customer_duals: tuple[float, ...] | np.ndarray,
    *,
    allowed_arcs: np.ndarray | None = None,
    top_k: int = 1,
    excluded_routes: set[Route] | None = None,
    dominance_tolerance: float = 1e-10,
    max_labels: int = 2_000_000,
) -> PricingResult:
    """Solve an elementary shortest path pricing problem by label setting.

    With ``allowed_arcs=None`` the result is globally exact for the declared
    elementary pricing problem. Supplying a pruned graph makes the solve exact
    only on that graph; callers must use full-graph fallback before declaring
    column-generation convergence.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_labels <= 0:
        raise ValueError("max_labels must be positive")
    duals = _validate_duals(instance, customer_duals)
    full_graph = all_arcs(instance)
    if allowed_arcs is None:
        allowed = full_graph
        globally_certified = True
    else:
        allowed = np.asarray(allowed_arcs, dtype=bool)
        if allowed.shape != full_graph.shape:
            raise ValueError(f"allowed_arcs must have shape {full_graph.shape}")
        allowed = allowed.copy()
        np.fill_diagonal(allowed, False)
        globally_certified = bool(np.array_equal(allowed, full_graph))

    excluded = excluded_routes or set()
    matrix = distance_matrix(instance)
    started = time.perf_counter()

    start = _Label(
        node=0,
        mask=0,
        load=0.0,
        departure_time=instance.depot_ready_time,
        reduced_cost_open=instance.fixed_vehicle_cost,
        path=(),
    )
    stack: list[_Label] = [start]
    pareto: dict[tuple[int, int], list[tuple[float, float]]] = {
        (0, 0): [(0.0, instance.fixed_vehicle_cost)]
    }
    best_by_route: dict[Route, PricedRoute] = {}

    labels_created = 1
    labels_expanded = 0
    labels_dominated = 0
    feasible_extensions = 0
    arc_checks = 0
    complete_routes_considered = 0

    def consider(candidate: PricedRoute) -> None:
        existing = best_by_route.get(candidate.route)
        if existing is None or candidate.reduced_cost < existing.reduced_cost:
            best_by_route[candidate.route] = candidate
        if len(best_by_route) > max(top_k * 4, top_k + 8):
            retained = sorted(best_by_route.values(), key=lambda item: item.reduced_cost)[:top_k]
            best_by_route.clear()
            best_by_route.update((item.route, item) for item in retained)

    while stack:
        label = stack.pop()
        labels_expanded += 1
        for next_node in range(1, instance.customer_count + 1):
            arc_checks += 1
            if not allowed[label.node, next_node]:
                continue
            bit = 1 << (next_node - 1)
            if label.mask & bit:
                continue
            customer = instance.customer(next_node)
            new_load = label.load + customer.demand
            if new_load > instance.capacity + 1e-9:
                continue
            arrival = label.departure_time + float(matrix[label.node, next_node])
            service_start = max(arrival, customer.ready_time)
            if service_start > customer.due_time + 1e-9:
                continue
            departure = service_start + customer.service_time
            if departure + float(matrix[next_node, 0]) > instance.depot_due_time + 1e-9:
                continue

            feasible_extensions += 1
            new_mask = label.mask | bit
            new_reduced_open = (
                label.reduced_cost_open
                + float(matrix[label.node, next_node])
                - float(duals[next_node - 1])
            )
            key = (new_mask, next_node)
            frontier = pareto.setdefault(key, [])
            dominated = any(
                old_time <= departure + dominance_tolerance
                and old_cost <= new_reduced_open + dominance_tolerance
                for old_time, old_cost in frontier
            )
            if dominated:
                labels_dominated += 1
                continue
            frontier[:] = [
                (old_time, old_cost)
                for old_time, old_cost in frontier
                if not (
                    departure <= old_time + dominance_tolerance
                    and new_reduced_open <= old_cost + dominance_tolerance
                )
            ]
            frontier.append((departure, new_reduced_open))
            new_path = (*label.path, next_node)
            new_label = _Label(
                node=next_node,
                mask=new_mask,
                load=new_load,
                departure_time=departure,
                reduced_cost_open=new_reduced_open,
                path=new_path,
            )
            stack.append(new_label)
            labels_created += 1
            if labels_created > max_labels:
                raise RuntimeError(
                    f"pricing exceeded max_labels={max_labels}; reduce instance size or use pruning"
                )

            if allowed[next_node, 0]:
                complete_routes_considered += 1
                route = Route(new_path)
                if route not in excluded:
                    total_reduced = new_reduced_open + float(matrix[next_node, 0])
                    total_cost = (
                        instance.fixed_vehicle_cost
                        + sum(float(matrix[i, j]) for i, j in route.arcs())
                    )
                    consider(
                        PricedRoute(
                            route=route,
                            route_cost=total_cost,
                            reduced_cost=total_reduced,
                        )
                    )

    candidates = tuple(
        sorted(best_by_route.values(), key=lambda item: (item.reduced_cost, item.route.customers))[
            :top_k
        ]
    )
    runtime = time.perf_counter() - started
    return PricingResult(
        candidates=candidates,
        globally_certified=globally_certified,
        stats=PricingStats(
            labels_created=labels_created,
            labels_expanded=labels_expanded,
            labels_dominated=labels_dominated,
            feasible_extensions=feasible_extensions,
            arc_checks=arc_checks,
            complete_routes_considered=complete_routes_considered,
            allowed_arc_fraction=allowed_arc_fraction(allowed),
            runtime_seconds=runtime,
        ),
    )


def brute_force_price(
    instance: CVRPTWInstance,
    customer_duals: tuple[float, ...] | np.ndarray,
    *,
    top_k: int = 1,
    excluded_routes: set[Route] | None = None,
) -> PricingResult:
    """Independent full-enumeration pricing oracle for tiny regression cases."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    duals = _validate_duals(instance, customer_duals)
    excluded = excluded_routes or set()
    started = time.perf_counter()
    priced = [
        PricedRoute(
            route=route,
            route_cost=route_cost(instance, route),
            reduced_cost=reduced_cost(instance, route, duals),
        )
        for route in enumerate_feasible_routes(instance)
        if route not in excluded
    ]
    priced.sort(key=lambda item: (item.reduced_cost, item.route.customers))
    runtime = time.perf_counter() - started
    return PricingResult(
        candidates=tuple(priced[:top_k]),
        globally_certified=True,
        stats=PricingStats(
            labels_created=0,
            labels_expanded=0,
            labels_dominated=0,
            feasible_extensions=0,
            arc_checks=0,
            complete_routes_considered=len(priced),
            allowed_arc_fraction=1.0,
            runtime_seconds=runtime,
        ),
    )
