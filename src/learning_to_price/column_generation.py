"""Exact and hybrid learned-pricing column-generation orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

from learning_to_price.domain import CVRPTWInstance
from learning_to_price.features import (
    ArcScorer,
    heuristic_arc_selection,
    learned_arc_selection,
    random_arc_selection,
)
from learning_to_price.master import (
    IntegerMasterSolution,
    MasterSolution,
    solve_integer_master,
    solve_restricted_master,
)
from learning_to_price.pricing import PricedRoute, PricingResult, price_routes
from learning_to_price.routes import Route, singleton_routes


class PricingMode(StrEnum):
    EXACT = "exact"
    HEURISTIC = "heuristic"
    LEARNED = "learned"
    RANDOM = "random"


@dataclass(frozen=True, slots=True)
class ColumnGenerationConfig:
    max_iterations: int = 200
    reduced_cost_tolerance: float = 1e-8
    keep_ratio: float = 0.35
    min_outgoing: int = 2
    pricing_top_k: int = 1
    random_seed: int = 0
    integer_time_limit: float | None = 60.0

    def __post_init__(self) -> None:
        if self.max_iterations <= 0 or self.pricing_top_k <= 0:
            raise ValueError("max_iterations and pricing_top_k must be positive")
        if self.reduced_cost_tolerance < 0:
            raise ValueError("reduced_cost_tolerance must be nonnegative")
        if not 0 < self.keep_ratio <= 1 or self.min_outgoing <= 0:
            raise ValueError("invalid arc-selection configuration")


@dataclass(frozen=True, slots=True)
class IterationRecord:
    iteration: int
    master_objective: float
    route_pool_size: int
    selected_source: str
    selected_route: tuple[int, ...] | None
    selected_reduced_cost: float | None
    restricted_arc_fraction: float | None
    heuristic_pricing_seconds: float
    exact_pricing_seconds: float
    exact_fallback_used: bool
    exact_certification_iteration: bool


@dataclass(frozen=True, slots=True)
class ColumnGenerationResult:
    instance_name: str
    mode: PricingMode
    converged: bool
    globally_certified: bool
    iterations: int
    final_lp_objective: float | None
    route_pool: tuple[Route, ...]
    final_master: MasterSolution
    integer_master: IntegerMasterSolution
    records: tuple[IterationRecord, ...]
    heuristic_successes: int
    exact_fallback_calls: int
    exact_pricing_calls: int
    total_master_seconds: float
    total_heuristic_pricing_seconds: float
    total_exact_pricing_seconds: float
    total_runtime_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_name": self.instance_name,
            "mode": self.mode.value,
            "converged": self.converged,
            "globally_certified": self.globally_certified,
            "iterations": self.iterations,
            "final_lp_objective": self.final_lp_objective,
            "route_pool_size": len(self.route_pool),
            "integer_objective": self.integer_master.objective,
            "selected_integer_routes": [
                list(self.route_pool[index].customers)
                for index in self.integer_master.selected_route_indices
            ],
            "heuristic_successes": self.heuristic_successes,
            "exact_fallback_calls": self.exact_fallback_calls,
            "exact_pricing_calls": self.exact_pricing_calls,
            "total_master_seconds": self.total_master_seconds,
            "total_heuristic_pricing_seconds": self.total_heuristic_pricing_seconds,
            "total_exact_pricing_seconds": self.total_exact_pricing_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "records": [
                {
                    "iteration": record.iteration,
                    "master_objective": record.master_objective,
                    "route_pool_size": record.route_pool_size,
                    "selected_source": record.selected_source,
                    "selected_route": (
                        list(record.selected_route) if record.selected_route else None
                    ),
                    "selected_reduced_cost": record.selected_reduced_cost,
                    "restricted_arc_fraction": record.restricted_arc_fraction,
                    "heuristic_pricing_seconds": record.heuristic_pricing_seconds,
                    "exact_pricing_seconds": record.exact_pricing_seconds,
                    "exact_fallback_used": record.exact_fallback_used,
                    "exact_certification_iteration": record.exact_certification_iteration,
                }
                for record in self.records
            ],
        }


def _restricted_pricing(
    instance: CVRPTWInstance,
    duals: tuple[float, ...],
    mode: PricingMode,
    config: ColumnGenerationConfig,
    *,
    scorer: ArcScorer | None,
    iteration: int,
    excluded_routes: set[Route],
) -> tuple[PricingResult, float]:
    if mode is PricingMode.LEARNED:
        if scorer is None:
            raise ValueError("learned pricing requires an arc scorer")
        selection = learned_arc_selection(
            instance,
            duals,
            scorer,
            keep_ratio=config.keep_ratio,
            min_outgoing=config.min_outgoing,
        )
    elif mode is PricingMode.HEURISTIC:
        selection = heuristic_arc_selection(
            instance,
            duals,
            keep_ratio=config.keep_ratio,
            min_outgoing=config.min_outgoing,
        )
    elif mode is PricingMode.RANDOM:
        selection = random_arc_selection(
            instance,
            duals,
            seed=config.random_seed + iteration,
            keep_ratio=config.keep_ratio,
            min_outgoing=config.min_outgoing,
        )
    else:
        raise ValueError(f"restricted pricing is not defined for mode {mode}")
    pricing = price_routes(
        instance,
        duals,
        allowed_arcs=selection.allowed_arcs,
        top_k=config.pricing_top_k,
        excluded_routes=excluded_routes,
    )
    return pricing, selection.retained_fraction


def _negative_candidate(
    pricing: PricingResult,
    tolerance: float,
) -> PricedRoute | None:
    best = pricing.best
    if best is not None and best.reduced_cost < -tolerance:
        return best
    return None


def run_column_generation(
    instance: CVRPTWInstance,
    *,
    mode: PricingMode | str = PricingMode.EXACT,
    scorer: ArcScorer | None = None,
    config: ColumnGenerationConfig | None = None,
) -> ColumnGenerationResult:
    """Run route-based LP column generation with safe exact fallback.

    In heuristic, random, and learned modes, a restricted-graph pricing solve is
    attempted first. A full exact pricing solve is mandatory whenever the
    restricted solve finds no improving route. Therefore convergence is declared
    only after a globally exact no-negative-reduced-cost certificate.
    """

    mode = PricingMode(mode)
    config = config or ColumnGenerationConfig()
    if mode is PricingMode.LEARNED and scorer is None:
        raise ValueError("learned pricing requires an arc scorer")

    started = time.perf_counter()
    routes = list(singleton_routes(instance))
    route_set: set[Route] = set(routes)
    records: list[IterationRecord] = []
    converged = False
    globally_certified = False
    heuristic_successes = 0
    exact_fallback_calls = 0
    exact_pricing_calls = 0
    total_master_seconds = 0.0
    total_heuristic_seconds = 0.0
    total_exact_seconds = 0.0

    for iteration in range(config.max_iterations):
        master = solve_restricted_master(instance, routes)
        total_master_seconds += master.runtime_seconds
        if not master.success or master.objective is None:
            raise RuntimeError(f"restricted master failed: {master.message}")

        chosen: PricedRoute | None = None
        selected_source = "none"
        restricted_fraction: float | None = None
        heuristic_seconds = 0.0
        exact_seconds = 0.0
        fallback = False
        certification_iteration = False

        if mode is PricingMode.EXACT:
            exact = price_routes(
                instance,
                master.customer_duals,
                top_k=config.pricing_top_k,
                excluded_routes=route_set,
            )
            exact_pricing_calls += 1
            exact_seconds = exact.stats.runtime_seconds
            total_exact_seconds += exact_seconds
            chosen = _negative_candidate(exact, config.reduced_cost_tolerance)
            if chosen is None:
                converged = True
                globally_certified = exact.globally_certified
                certification_iteration = True
                selected_source = "exact-certificate"
            else:
                selected_source = "exact"
        else:
            restricted, restricted_fraction = _restricted_pricing(
                instance,
                master.customer_duals,
                mode,
                config,
                scorer=scorer,
                iteration=iteration,
                excluded_routes=route_set,
            )
            heuristic_seconds = restricted.stats.runtime_seconds
            total_heuristic_seconds += heuristic_seconds
            chosen = _negative_candidate(restricted, config.reduced_cost_tolerance)
            if chosen is not None:
                heuristic_successes += 1
                selected_source = mode.value
            else:
                fallback = True
                exact_fallback_calls += 1
                exact = price_routes(
                    instance,
                    master.customer_duals,
                    top_k=config.pricing_top_k,
                    excluded_routes=route_set,
                )
                exact_pricing_calls += 1
                exact_seconds = exact.stats.runtime_seconds
                total_exact_seconds += exact_seconds
                chosen = _negative_candidate(exact, config.reduced_cost_tolerance)
                if chosen is None:
                    converged = True
                    globally_certified = exact.globally_certified
                    certification_iteration = True
                    selected_source = "exact-certificate"
                else:
                    selected_source = "exact-fallback"

        records.append(
            IterationRecord(
                iteration=iteration,
                master_objective=master.objective,
                route_pool_size=len(routes),
                selected_source=selected_source,
                selected_route=chosen.route.customers if chosen else None,
                selected_reduced_cost=chosen.reduced_cost if chosen else None,
                restricted_arc_fraction=restricted_fraction,
                heuristic_pricing_seconds=heuristic_seconds,
                exact_pricing_seconds=exact_seconds,
                exact_fallback_used=fallback,
                exact_certification_iteration=certification_iteration,
            )
        )
        if chosen is None:
            break
        routes.append(chosen.route)
        route_set.add(chosen.route)

    final_master = solve_restricted_master(instance, routes)
    total_master_seconds += final_master.runtime_seconds
    integer_master = solve_integer_master(
        instance,
        routes,
        time_limit=config.integer_time_limit,
    )
    return ColumnGenerationResult(
        instance_name=instance.name,
        mode=mode,
        converged=converged,
        globally_certified=globally_certified,
        iterations=len(records),
        final_lp_objective=final_master.objective,
        route_pool=tuple(routes),
        final_master=final_master,
        integer_master=integer_master,
        records=tuple(records),
        heuristic_successes=heuristic_successes,
        exact_fallback_calls=exact_fallback_calls,
        exact_pricing_calls=exact_pricing_calls,
        total_master_seconds=total_master_seconds,
        total_heuristic_pricing_seconds=total_heuristic_seconds,
        total_exact_pricing_seconds=total_exact_seconds,
        total_runtime_seconds=time.perf_counter() - started,
    )
