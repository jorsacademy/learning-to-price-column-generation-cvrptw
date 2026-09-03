"""End-to-end research protocol with in-distribution and shifted test scenarios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from learning_to_price.benchmark import BenchmarkReport, run_benchmark
from learning_to_price.column_generation import ColumnGenerationConfig, PricingMode
from learning_to_price.dataset import collect_pricing_dataset, split_by_instance
from learning_to_price.domain import CVRPTWInstance, Distribution, WindowRegime
from learning_to_price.generator import generate_solomon_like
from learning_to_price.learning import (
    NumpyMLPArcScorer,
    TrainingConfig,
    binary_classification_metrics,
)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    customers: int
    distribution: Distribution
    window_regime: WindowRegime
    seed_start: int
    instances: int


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    training_dataset: dict[str, object]
    training_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    scenarios: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "training_dataset": self.training_dataset,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "scenarios": self.scenarios,
        }


def _instances(
    count: int,
    customers: int,
    seed_start: int,
    distribution: Distribution,
    window_regime: WindowRegime,
) -> list[CVRPTWInstance]:
    return [
        generate_solomon_like(
            customers,
            seed=seed_start + index,
            distribution=distribution,
            window_regime=window_regime,
        ).instance
        for index in range(count)
    ]


def run_research_experiment(
    *,
    train_instances: int = 24,
    train_customers: int = 8,
    train_seed_start: int = 10_000,
    hidden_dim: int = 32,
    epochs: int = 80,
    benchmark_instances: int = 8,
    keep_ratio: float = 0.35,
) -> tuple[NumpyMLPArcScorer, ExperimentResult]:
    """Run a frozen train/validation/test protocol.

    Test seeds do not overlap with training seeds. The scenarios separate
    in-distribution evaluation, size shift, spatial distribution shift, and
    tighter time windows.
    """

    training_instances = _instances(
        train_instances,
        train_customers,
        train_seed_start,
        "uniform",
        "wide",
    )
    dataset = collect_pricing_dataset(training_instances, top_k_routes=3)
    train, validation = split_by_instance(dataset, validation_fraction=0.2, seed=2026)
    scorer = NumpyMLPArcScorer(train.features.shape[1], hidden_dim=hidden_dim, seed=2026)
    scorer.fit(
        train.features,
        train.labels,
        TrainingConfig(
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=512,
            seed=2026,
        ),
    )
    training_metrics = binary_classification_metrics(
        train.labels,
        scorer.predict_proba(train.features),
    )
    validation_metrics = binary_classification_metrics(
        validation.labels,
        scorer.predict_proba(validation.features),
    )

    scenarios = (
        Scenario(
            name="in_distribution",
            customers=train_customers,
            distribution="uniform",
            window_regime="wide",
            seed_start=30_000,
            instances=benchmark_instances,
        ),
        Scenario(
            name="size_shift",
            customers=train_customers + 2,
            distribution="uniform",
            window_regime="wide",
            seed_start=40_000,
            instances=benchmark_instances,
        ),
        Scenario(
            name="clustered_shift",
            customers=train_customers,
            distribution="clustered",
            window_regime="wide",
            seed_start=50_000,
            instances=benchmark_instances,
        ),
        Scenario(
            name="tight_windows_shift",
            customers=train_customers,
            distribution="uniform",
            window_regime="tight",
            seed_start=60_000,
            instances=benchmark_instances,
        ),
    )
    scenario_reports: dict[str, dict[str, object]] = {}
    for scenario in scenarios:
        instances = _instances(
            scenario.instances,
            scenario.customers,
            scenario.seed_start,
            scenario.distribution,
            scenario.window_regime,
        )
        report: BenchmarkReport = run_benchmark(
            instances,
            scorer=scorer,
            modes=(
                PricingMode.EXACT,
                PricingMode.HEURISTIC,
                PricingMode.RANDOM,
                PricingMode.LEARNED,
            ),
            config=ColumnGenerationConfig(
                keep_ratio=keep_ratio,
                min_outgoing=2,
                random_seed=scenario.seed_start,
            ),
            full_enumeration_limit=8,
        )
        scenario_reports[scenario.name] = report.to_dict()

    result = ExperimentResult(
        training_dataset=dataset.summary(),
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        scenarios=scenario_reports,
    )
    return scorer, result


def save_experiment_result(result: ExperimentResult, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
