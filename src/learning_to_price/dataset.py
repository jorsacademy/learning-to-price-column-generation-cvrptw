"""Exact column-generation trajectory collection for supervised pricing data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from learning_to_price.domain import CVRPTWInstance
from learning_to_price.features import ARC_FEATURE_NAMES, arc_feature_matrix, route_arc_labels
from learning_to_price.master import solve_restricted_master
from learning_to_price.pricing import price_routes
from learning_to_price.routes import Route, singleton_routes


@dataclass(frozen=True, slots=True)
class PricingDataset:
    features: np.ndarray
    labels: np.ndarray
    instance_ids: np.ndarray
    iteration_ids: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be a matrix")
        row_count = self.features.shape[0]
        if self.labels.shape != (row_count,):
            raise ValueError("labels have the wrong shape")
        if self.instance_ids.shape != (row_count,) or self.iteration_ids.shape != (row_count,):
            raise ValueError("dataset IDs have the wrong shape")
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError("feature_names do not match the feature matrix")

    @property
    def positive_rate(self) -> float:
        return float(np.mean(self.labels)) if len(self.labels) else 0.0

    def summary(self) -> dict[str, object]:
        return {
            "rows": int(self.features.shape[0]),
            "features": int(self.features.shape[1]),
            "instances": len(np.unique(self.instance_ids)),
            "iterations": len(
                np.unique(
                    np.column_stack((self.instance_ids, self.iteration_ids)),
                    axis=0,
                )
            ),
            "positive_rate": self.positive_rate,
            "feature_names": list(self.feature_names),
            "metadata": self.metadata,
        }


def collect_pricing_dataset(
    instances: list[CVRPTWInstance] | tuple[CVRPTWInstance, ...],
    *,
    top_k_routes: int = 3,
    max_iterations: int = 100,
    reduced_cost_tolerance: float = 1e-8,
) -> PricingDataset:
    """Collect dual-aware arc labels from exact pricing trajectories.

    Each RMP iteration is labeled with the union of arcs appearing in the exact
    top-k priced routes. The gold routes are generated after solving the master;
    they are never exposed to the pricing model at inference time.
    """

    if not instances:
        raise ValueError("instances must be nonempty")
    if top_k_routes <= 0 or max_iterations <= 0:
        raise ValueError("top_k_routes and max_iterations must be positive")

    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    instance_batches: list[np.ndarray] = []
    iteration_batches: list[np.ndarray] = []
    trajectory_lengths: list[int] = []

    for instance_index, instance in enumerate(instances):
        routes = list(singleton_routes(instance))
        route_set: set[Route] = set(routes)
        iterations = 0
        for iteration in range(max_iterations):
            master = solve_restricted_master(instance, routes)
            if not master.success:
                raise RuntimeError(
                    f"restricted master failed for {instance.name}: {master.message}"
                )
            pricing = price_routes(
                instance,
                master.customer_duals,
                top_k=top_k_routes,
                excluded_routes=route_set,
            )
            features, arcs = arc_feature_matrix(instance, master.customer_duals)
            labeled_routes = tuple(candidate.route for candidate in pricing.candidates)
            labels = route_arc_labels(arcs, labeled_routes)
            feature_batches.append(features)
            label_batches.append(labels)
            instance_batches.append(np.full(len(labels), instance_index, dtype=int))
            iteration_batches.append(np.full(len(labels), iteration, dtype=int))
            iterations += 1

            best = pricing.best
            if best is None or best.reduced_cost >= -reduced_cost_tolerance:
                break
            routes.append(best.route)
            route_set.add(best.route)
        else:
            raise RuntimeError(
                f"exact column generation did not converge within {max_iterations} iterations "
                f"for {instance.name}"
            )
        trajectory_lengths.append(iterations)

    features = np.concatenate(feature_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    instance_ids = np.concatenate(instance_batches, axis=0)
    iteration_ids = np.concatenate(iteration_batches, axis=0)
    return PricingDataset(
        features=features,
        labels=labels,
        instance_ids=instance_ids,
        iteration_ids=iteration_ids,
        feature_names=ARC_FEATURE_NAMES,
        metadata={
            "top_k_routes": top_k_routes,
            "reduced_cost_tolerance": reduced_cost_tolerance,
            "trajectory_lengths": trajectory_lengths,
            "instance_names": [instance.name for instance in instances],
        },
    )


def save_dataset(dataset: PricingDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=dataset.features,
        labels=dataset.labels,
        instance_ids=dataset.instance_ids,
        iteration_ids=dataset.iteration_ids,
        feature_names=np.asarray(dataset.feature_names),
        metadata_json=np.asarray([json.dumps(dataset.metadata, sort_keys=True)]),
    )


def load_dataset(path: str | Path) -> PricingDataset:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = cast(
            dict[str, object],
            json.loads(str(payload["metadata_json"][0])),
        )
        return PricingDataset(
            features=np.asarray(payload["features"], dtype=float),
            labels=np.asarray(payload["labels"], dtype=float),
            instance_ids=np.asarray(payload["instance_ids"], dtype=int),
            iteration_ids=np.asarray(payload["iteration_ids"], dtype=int),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            metadata=metadata,
        )


def split_by_instance(
    dataset: PricingDataset,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[PricingDataset, PricingDataset]:
    """Split complete instance trajectories, preventing iteration leakage."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    unique_ids = np.unique(dataset.instance_ids)
    if len(unique_ids) < 2:
        raise ValueError("at least two instances are required for a grouped split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    validation_count = max(
        1,
        min(len(unique_ids) - 1, round(len(unique_ids) * validation_fraction)),
    )
    validation_ids = set(int(value) for value in shuffled[:validation_count])
    validation_mask = np.asarray([int(value) in validation_ids for value in dataset.instance_ids])

    def subset(mask: np.ndarray, label: str) -> PricingDataset:
        metadata = dict(dataset.metadata)
        metadata["split"] = label
        return PricingDataset(
            features=dataset.features[mask],
            labels=dataset.labels[mask],
            instance_ids=dataset.instance_ids[mask],
            iteration_ids=dataset.iteration_ids[mask],
            feature_names=dataset.feature_names,
            metadata=metadata,
        )

    return subset(~validation_mask, "train"), subset(validation_mask, "validation")
