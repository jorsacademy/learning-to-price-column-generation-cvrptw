"""Restricted master LP and integer set-partitioning solves."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix

from learning_to_price.domain import CVRPTWInstance
from learning_to_price.routes import Route, coverage_vector, evaluate_route


@dataclass(frozen=True, slots=True)
class MasterSolution:
    success: bool
    objective: float | None
    route_values: tuple[float, ...]
    customer_duals: tuple[float, ...]
    status_code: int
    message: str
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class IntegerMasterSolution:
    success: bool
    objective: float | None
    selected_route_indices: tuple[int, ...]
    route_values: tuple[float, ...]
    status_code: int
    message: str
    runtime_seconds: float


def _master_arrays(
    instance: CVRPTWInstance,
    routes: tuple[Route, ...] | list[Route],
) -> tuple[np.ndarray, np.ndarray]:
    if not routes:
        raise ValueError("route pool must be nonempty")
    costs = np.empty(len(routes), dtype=float)
    coverage = np.empty((instance.customer_count, len(routes)), dtype=float)
    for column, route in enumerate(routes):
        evaluation = evaluate_route(instance, route)
        if not evaluation.feasible:
            raise ValueError(f"route {route.customers} is infeasible: {evaluation.failure_reason}")
        costs[column] = evaluation.cost
        coverage[:, column] = np.asarray(coverage_vector(instance, route), dtype=float)
    if np.any(np.sum(coverage, axis=1) == 0):
        raise ValueError("route pool does not cover every customer")
    return costs, coverage


def solve_restricted_master(
    instance: CVRPTWInstance,
    routes: tuple[Route, ...] | list[Route],
) -> MasterSolution:
    """Solve the route-based set-partitioning LP relaxation."""

    costs, coverage = _master_arrays(instance, routes)
    started = time.perf_counter()
    result = linprog(
        costs,
        A_eq=csc_matrix(coverage),
        b_eq=np.ones(instance.customer_count, dtype=float),
        bounds=(0.0, None),
        method="highs",
    )
    runtime = time.perf_counter() - started
    if not result.success or result.fun is None or result.x is None:
        return MasterSolution(
            success=False,
            objective=None,
            route_values=(),
            customer_duals=(),
            status_code=int(result.status),
            message=str(result.message),
            runtime_seconds=runtime,
        )
    duals = tuple(float(value) for value in result.eqlin.marginals)
    return MasterSolution(
        success=True,
        objective=float(result.fun),
        route_values=tuple(float(value) for value in result.x),
        customer_duals=duals,
        status_code=int(result.status),
        message=str(result.message),
        runtime_seconds=runtime,
    )


def solve_integer_master(
    instance: CVRPTWInstance,
    routes: tuple[Route, ...] | list[Route],
    *,
    time_limit: float | None = None,
) -> IntegerMasterSolution:
    """Solve the binary master over a fixed route pool."""

    costs, coverage = _master_arrays(instance, routes)
    options: dict[str, float] = {}
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError("time_limit must be positive")
        options["time_limit"] = time_limit
    started = time.perf_counter()
    result = milp(
        c=costs,
        integrality=np.ones(len(routes), dtype=int),
        bounds=Bounds(np.zeros(len(routes)), np.ones(len(routes))),
        constraints=LinearConstraint(
            csc_matrix(coverage),
            np.ones(instance.customer_count),
            np.ones(instance.customer_count),
        ),
        options=options or None,
    )
    runtime = time.perf_counter() - started
    if not result.success or result.fun is None or result.x is None:
        return IntegerMasterSolution(
            success=False,
            objective=None,
            selected_route_indices=(),
            route_values=(),
            status_code=int(result.status),
            message=str(result.message),
            runtime_seconds=runtime,
        )
    values = tuple(float(value) for value in result.x)
    selected = tuple(index for index, value in enumerate(values) if value > 0.5)
    return IntegerMasterSolution(
        success=True,
        objective=float(result.fun),
        selected_route_indices=selected,
        route_values=values,
        status_code=int(result.status),
        message=str(result.message),
        runtime_seconds=runtime,
    )
