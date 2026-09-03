"""Typed CVRPTW domain objects and deterministic JSON serialization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Distribution = Literal["uniform", "clustered"]
WindowRegime = Literal["wide", "tight"]


@dataclass(frozen=True, slots=True)
class Customer:
    """A single CVRPTW customer.

    Customer IDs are one-based. Node zero is reserved for the depot.
    """

    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float = 0.0

    def __post_init__(self) -> None:
        numeric = (
            self.x,
            self.y,
            self.demand,
            self.ready_time,
            self.due_time,
            self.service_time,
        )
        if self.id <= 0:
            raise ValueError("customer id must be positive")
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("customer values must be finite")
        if self.demand <= 0:
            raise ValueError("customer demand must be positive")
        if self.ready_time < 0 or self.due_time < self.ready_time:
            raise ValueError("customer time window is invalid")
        if self.service_time < 0:
            raise ValueError("service_time must be nonnegative")


@dataclass(frozen=True, slots=True)
class CVRPTWInstance:
    """A finite homogeneous-fleet CVRPTW instance.

    The master formulation uses an unlimited homogeneous fleet. Every route pays
    ``fixed_vehicle_cost`` and starts and ends at the single depot.
    """

    name: str
    customers: tuple[Customer, ...]
    capacity: float
    depot_x: float = 50.0
    depot_y: float = 50.0
    depot_ready_time: float = 0.0
    depot_due_time: float = 500.0
    fixed_vehicle_cost: float = 50.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("instance name must be nonempty")
        numeric = (
            self.capacity,
            self.depot_x,
            self.depot_y,
            self.depot_ready_time,
            self.depot_due_time,
            self.fixed_vehicle_cost,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("instance values must be finite")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.depot_ready_time < 0 or self.depot_due_time <= self.depot_ready_time:
            raise ValueError("depot time window is invalid")
        if self.fixed_vehicle_cost < 0:
            raise ValueError("fixed_vehicle_cost must be nonnegative")
        expected_ids = tuple(range(1, len(self.customers) + 1))
        actual_ids = tuple(customer.id for customer in self.customers)
        if actual_ids != expected_ids:
            raise ValueError("customer ids must be contiguous and ordered from 1")
        for customer in self.customers:
            if customer.demand > self.capacity:
                raise ValueError(f"customer {customer.id} demand exceeds vehicle capacity")
            if customer.due_time > self.depot_due_time:
                raise ValueError(f"customer {customer.id} due time exceeds the depot horizon")

    @property
    def customer_count(self) -> int:
        return len(self.customers)

    def customer(self, node: int) -> Customer:
        if node <= 0 or node > self.customer_count:
            raise IndexError(f"customer node must be in 1..{self.customer_count}")
        return self.customers[node - 1]

    def node_xy(self, node: int) -> tuple[float, float]:
        if node == 0:
            return self.depot_x, self.depot_y
        customer = self.customer(node)
        return customer.x, customer.y

    def node_demand(self, node: int) -> float:
        return 0.0 if node == 0 else self.customer(node).demand

    def node_ready_time(self, node: int) -> float:
        return self.depot_ready_time if node == 0 else self.customer(node).ready_time

    def node_due_time(self, node: int) -> float:
        return self.depot_due_time if node == 0 else self.customer(node).due_time

    def node_service_time(self, node: int) -> float:
        return 0.0 if node == 0 else self.customer(node).service_time

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "capacity": self.capacity,
            "depot": {
                "x": self.depot_x,
                "y": self.depot_y,
                "ready_time": self.depot_ready_time,
                "due_time": self.depot_due_time,
            },
            "fixed_vehicle_cost": self.fixed_vehicle_cost,
            "customers": [asdict(customer) for customer in self.customers],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CVRPTWInstance:
        depot = payload.get("depot")
        if not isinstance(depot, dict):
            raise ValueError("depot must be an object")
        raw_customers = payload.get("customers")
        if not isinstance(raw_customers, list):
            raise ValueError("customers must be a list")
        parsed_customers: list[Customer] = []
        for item in raw_customers:
            if not isinstance(item, dict):
                raise ValueError("every customer must be an object")
            parsed_customers.append(
                Customer(
                    id=int(item["id"]),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    demand=float(item["demand"]),
                    ready_time=float(item["ready_time"]),
                    due_time=float(item["due_time"]),
                    service_time=float(item.get("service_time", 0.0)),
                )
            )
        customers = tuple(parsed_customers)
        return cls(
            name=str(payload["name"]),
            customers=customers,
            capacity=float(payload["capacity"]),
            depot_x=float(depot["x"]),
            depot_y=float(depot["y"]),
            depot_ready_time=float(depot.get("ready_time", 0.0)),
            depot_due_time=float(depot["due_time"]),
            fixed_vehicle_cost=float(payload.get("fixed_vehicle_cost", 0.0)),
        )


def save_instance(instance: CVRPTWInstance, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(instance.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_instance(path: str | Path) -> CVRPTWInstance:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("instance JSON must be an object")
    return CVRPTWInstance.from_dict(payload)
