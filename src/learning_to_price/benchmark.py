"""Repeated-instance benchmarks for exact and hybrid pricing policies."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, fields
from pathlib import Path

from learning_to_price.column_generation import (
    ColumnGenerationConfig,
    ColumnGenerationResult,
    PricingMode,
    run_column_generation,
)
from learning_to_price.domain import CVRPTWInstance
from learning_to_price.features import ArcScorer
from learning_to_price.master import solve_integer_master, solve_restricted_master
from learning_to_price.routes import enumerate_feasible_routes


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    instance: str
    customer_count: int
    mode: str
    lp_objective: float | None
    full_lp_objective: float | None
    lp_gap_percent: float | None
    restricted_integer_objective: float | None
    full_integer_objective: float | None
    integer_gap_percent: float | None
    iterations: int
    route_pool_size: int
    exact_fallback_calls: int
    heuristic_successes: int
    exact_pricing_calls: int
    master_seconds: float
    heuristic_pricing_seconds: float
    exact_pricing_seconds: float
    total_seconds: float
    globally_certified: bool

    def to_dict(self) -> dict[str, object]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    rows: tuple[BenchmarkRow, ...]
    summary: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "summary": self.summary,
        }


def _gap(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return 100.0 * (value - reference) / max(abs(reference), 1e-12)


def _row_from_result(
    instance: CVRPTWInstance,
    result: ColumnGenerationResult,
    *,
    full_lp: float | None,
    full_integer: float | None,
) -> BenchmarkRow:
    return BenchmarkRow(
        instance=instance.name,
        customer_count=instance.customer_count,
        mode=result.mode.value,
        lp_objective=result.final_lp_objective,
        full_lp_objective=full_lp,
        lp_gap_percent=_gap(result.final_lp_objective, full_lp),
        restricted_integer_objective=result.integer_master.objective,
        full_integer_objective=full_integer,
        integer_gap_percent=_gap(result.integer_master.objective, full_integer),
        iterations=result.iterations,
        route_pool_size=len(result.route_pool),
        exact_fallback_calls=result.exact_fallback_calls,
        heuristic_successes=result.heuristic_successes,
        exact_pricing_calls=result.exact_pricing_calls,
        master_seconds=result.total_master_seconds,
        heuristic_pricing_seconds=result.total_heuristic_pricing_seconds,
        exact_pricing_seconds=result.total_exact_pricing_seconds,
        total_seconds=result.total_runtime_seconds,
        globally_certified=result.globally_certified,
    )


def run_benchmark(
    instances: list[CVRPTWInstance] | tuple[CVRPTWInstance, ...],
    *,
    scorer: ArcScorer | None = None,
    modes: tuple[PricingMode, ...] = (
        PricingMode.EXACT,
        PricingMode.HEURISTIC,
        PricingMode.RANDOM,
    ),
    config: ColumnGenerationConfig | None = None,
    full_enumeration_limit: int = 8,
) -> BenchmarkReport:
    if not instances:
        raise ValueError("instances must be nonempty")
    config = config or ColumnGenerationConfig()
    rows: list[BenchmarkRow] = []

    for instance in instances:
        full_lp: float | None = None
        full_integer: float | None = None
        if instance.customer_count <= full_enumeration_limit:
            all_routes = enumerate_feasible_routes(
                instance,
                max_customers=full_enumeration_limit,
            )
            full_lp_solution = solve_restricted_master(instance, all_routes)
            full_integer_solution = solve_integer_master(instance, all_routes)
            full_lp = full_lp_solution.objective
            full_integer = full_integer_solution.objective

        for mode in modes:
            if mode is PricingMode.LEARNED and scorer is None:
                raise ValueError("learned benchmark mode requires an arc scorer")
            result = run_column_generation(
                instance,
                mode=mode,
                scorer=scorer,
                config=config,
            )
            rows.append(
                _row_from_result(
                    instance,
                    result,
                    full_lp=full_lp,
                    full_integer=full_integer,
                )
            )

    summary: dict[str, dict[str, float]] = {}
    for mode_name in sorted({row.mode for row in rows}):
        selected = [row for row in rows if row.mode == mode_name]

        def mean(attribute: str, selected_rows: list[BenchmarkRow] = selected) -> float:
            values = [
                float(value)
                for row in selected_rows
                if (value := getattr(row, attribute)) is not None
            ]
            return statistics.fmean(values) if values else float("nan")

        summary[mode_name] = {
            "instances": float(len(selected)),
            "certified_rate": statistics.fmean(float(row.globally_certified) for row in selected),
            "mean_lp_gap_percent": mean("lp_gap_percent"),
            "mean_integer_gap_percent": mean("integer_gap_percent"),
            "mean_iterations": mean("iterations"),
            "mean_route_pool_size": mean("route_pool_size"),
            "mean_exact_fallback_calls": mean("exact_fallback_calls"),
            "mean_total_seconds": mean("total_seconds"),
            "mean_exact_pricing_seconds": mean("exact_pricing_seconds"),
        }
    return BenchmarkReport(rows=tuple(rows), summary=summary)


def save_report_json(report: BenchmarkReport, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_report_csv(report: BenchmarkReport, path: str | Path) -> None:
    rows = [row.to_dict() for row in report.rows]
    if not rows:
        raise ValueError("report has no rows")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
