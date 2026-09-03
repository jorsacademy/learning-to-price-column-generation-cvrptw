"""Geometry and travel-time helpers."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from learning_to_price.domain import CVRPTWInstance


@lru_cache(maxsize=512)
def distance_matrix(instance: CVRPTWInstance) -> np.ndarray:
    """Return a read-only Euclidean distance matrix for depot and customers."""

    coordinates = np.asarray(
        [instance.node_xy(node) for node in range(instance.customer_count + 1)],
        dtype=float,
    )
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    matrix = np.sqrt(np.sum(delta * delta, axis=2))
    matrix.setflags(write=False)
    return matrix


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
