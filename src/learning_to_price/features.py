"""Dual-aware arc features and graph-pruning policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from learning_to_price.domain import CVRPTWInstance
from learning_to_price.geometry import distance_matrix
from learning_to_price.pricing import all_arcs, allowed_arc_fraction
from learning_to_price.routes import Route

ARC_FEATURE_NAMES = (
    "distance",
    "origin_depot_distance",
    "target_depot_distance",
    "clarke_wright_saving",
    "origin_dual",
    "target_dual",
    "target_dual_minus_distance",
    "origin_demand_fraction",
    "target_demand_fraction",
    "origin_due_fraction",
    "target_ready_fraction",
    "target_due_fraction",
    "target_window_width_fraction",
    "target_service_fraction",
    "origin_is_depot",
    "target_is_depot",
    "static_time_feasible",
)


class ArcScorer(Protocol):
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return one score in [0, 1] per row."""


@dataclass(frozen=True, slots=True)
class ArcSelection:
    allowed_arcs: np.ndarray
    scores: np.ndarray
    arcs: tuple[tuple[int, int], ...]
    retained_fraction: float


def arc_feature_matrix(
    instance: CVRPTWInstance,
    customer_duals: tuple[float, ...] | np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    duals = np.asarray(customer_duals, dtype=float)
    if duals.shape != (instance.customer_count,):
        raise ValueError("customer_duals have the wrong shape")
    matrix = distance_matrix(instance)
    max_distance = max(float(np.max(matrix)), 1.0)
    horizon = max(instance.depot_due_time - instance.depot_ready_time, 1.0)
    cost_scale = max(instance.fixed_vehicle_cost, float(np.max(np.abs(duals))), max_distance, 1.0)

    arcs: list[tuple[int, int]] = []
    rows: list[list[float]] = []
    for origin in range(instance.customer_count + 1):
        origin_dual = 0.0 if origin == 0 else float(duals[origin - 1])
        origin_demand = instance.node_demand(origin)
        origin_due = instance.node_due_time(origin)
        earliest_departure = instance.node_ready_time(origin) + instance.node_service_time(origin)
        for target in range(instance.customer_count + 1):
            if origin == target:
                continue
            target_dual = 0.0 if target == 0 else float(duals[target - 1])
            target_demand = instance.node_demand(target)
            target_ready = instance.node_ready_time(target)
            target_due = instance.node_due_time(target)
            target_service = instance.node_service_time(target)
            distance = float(matrix[origin, target])
            saving = float(matrix[origin, 0] + matrix[0, target] - distance)
            static_feasible = float(
                earliest_departure + distance <= target_due + 1e-9
                and origin_demand + target_demand <= instance.capacity + 1e-9
            )
            arcs.append((origin, target))
            rows.append(
                [
                    distance / max_distance,
                    float(matrix[origin, 0]) / max_distance,
                    float(matrix[target, 0]) / max_distance,
                    saving / max_distance,
                    origin_dual / cost_scale,
                    target_dual / cost_scale,
                    (target_dual - distance) / cost_scale,
                    origin_demand / instance.capacity,
                    target_demand / instance.capacity,
                    origin_due / horizon,
                    target_ready / horizon,
                    target_due / horizon,
                    (target_due - target_ready) / horizon,
                    target_service / horizon,
                    float(origin == 0),
                    float(target == 0),
                    static_feasible,
                ]
            )
    return np.asarray(rows, dtype=float), tuple(arcs)


def route_arc_labels(
    arcs: tuple[tuple[int, int], ...],
    routes: tuple[Route, ...] | list[Route],
) -> np.ndarray:
    positive = {arc for route in routes for arc in route.arcs()}
    return np.asarray([1.0 if arc in positive else 0.0 for arc in arcs], dtype=float)


def _selection_from_scores(
    instance: CVRPTWInstance,
    scores: np.ndarray,
    arcs: tuple[tuple[int, int], ...],
    *,
    keep_ratio: float,
    min_outgoing: int,
) -> ArcSelection:
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if min_outgoing <= 0:
        raise ValueError("min_outgoing must be positive")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(arcs),):
        raise ValueError("scores have the wrong shape")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")

    count = instance.customer_count + 1
    allowed = np.zeros((count, count), dtype=bool)
    score_matrix = np.full((count, count), -math.inf, dtype=float)
    for score, (origin, target) in zip(scores, arcs, strict=True):
        score_matrix[origin, target] = float(score)

    # Depot arcs remain available. This prevents the selector from deleting all
    # route starts or all return arcs and isolates learning to customer transitions.
    allowed[0, 1:] = True
    allowed[1:, 0] = True

    customer_successors = max(
        1,
        min(
            instance.customer_count - 1,
            max(min_outgoing, math.ceil(keep_ratio * max(instance.customer_count - 1, 1))),
        ),
    )
    for origin in range(1, count):
        candidates = [target for target in range(1, count) if target != origin]
        candidates.sort(key=lambda target: (-score_matrix[origin, target], target))
        for target in candidates[:customer_successors]:
            allowed[origin, target] = True

    np.fill_diagonal(allowed, False)
    return ArcSelection(
        allowed_arcs=allowed,
        scores=scores,
        arcs=arcs,
        retained_fraction=allowed_arc_fraction(allowed),
    )


def learned_arc_selection(
    instance: CVRPTWInstance,
    customer_duals: tuple[float, ...] | np.ndarray,
    scorer: ArcScorer,
    *,
    keep_ratio: float = 0.35,
    min_outgoing: int = 2,
) -> ArcSelection:
    features, arcs = arc_feature_matrix(instance, customer_duals)
    scores = np.asarray(scorer.predict_proba(features), dtype=float)
    return _selection_from_scores(
        instance,
        scores,
        arcs,
        keep_ratio=keep_ratio,
        min_outgoing=min_outgoing,
    )


def heuristic_arc_selection(
    instance: CVRPTWInstance,
    customer_duals: tuple[float, ...] | np.ndarray,
    *,
    keep_ratio: float = 0.35,
    min_outgoing: int = 2,
) -> ArcSelection:
    features, arcs = arc_feature_matrix(instance, customer_duals)
    distance = features[:, 0]
    target_dual_minus_distance = features[:, 6]
    saving = features[:, 3]
    static_feasible = features[:, 16]
    scores = target_dual_minus_distance + 0.25 * saving - 0.15 * distance + 2.0 * static_feasible
    return _selection_from_scores(
        instance,
        scores,
        arcs,
        keep_ratio=keep_ratio,
        min_outgoing=min_outgoing,
    )


def random_arc_selection(
    instance: CVRPTWInstance,
    customer_duals: tuple[float, ...] | np.ndarray,
    *,
    seed: int,
    keep_ratio: float = 0.35,
    min_outgoing: int = 2,
) -> ArcSelection:
    features, arcs = arc_feature_matrix(instance, customer_duals)
    del features
    rng = np.random.default_rng(seed)
    scores = rng.random(len(arcs))
    return _selection_from_scores(
        instance,
        scores,
        arcs,
        keep_ratio=keep_ratio,
        min_outgoing=min_outgoing,
    )


def full_arc_selection(instance: CVRPTWInstance) -> ArcSelection:
    allowed = all_arcs(instance)
    arcs = tuple(
        (origin, target)
        for origin in range(instance.customer_count + 1)
        for target in range(instance.customer_count + 1)
        if origin != target
    )
    return ArcSelection(
        allowed_arcs=allowed,
        scores=np.ones(len(arcs), dtype=float),
        arcs=arcs,
        retained_fraction=1.0,
    )
