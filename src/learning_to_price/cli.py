"""Command-line interface for data collection, training, and benchmarking."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from learning_to_price import __version__
from learning_to_price.benchmark import run_benchmark, save_report_csv, save_report_json
from learning_to_price.column_generation import (
    ColumnGenerationConfig,
    PricingMode,
    run_column_generation,
)
from learning_to_price.dataset import (
    collect_pricing_dataset,
    load_dataset,
    save_dataset,
    split_by_instance,
)
from learning_to_price.domain import CVRPTWInstance
from learning_to_price.generator import generate_solomon_like
from learning_to_price.learning import (
    NumpyMLPArcScorer,
    TrainingConfig,
    binary_classification_metrics,
)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _generated_instances(
    args: argparse.Namespace, *, seed_offset: int = 0
) -> list[CVRPTWInstance]:
    return [
        generate_solomon_like(
            args.customers,
            seed=args.seed + seed_offset + index,
            distribution=args.distribution,
            window_regime=args.window_regime,
            fixed_vehicle_cost=args.fixed_vehicle_cost,
        ).instance
        for index in range(args.instances)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learning-to-price",
        description="Safe learned pricing for CVRPTW column generation.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run one generated column-generation instance.")
    demo.add_argument("--customers", type=int, default=8)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--distribution", choices=["uniform", "clustered"], default="uniform")
    demo.add_argument("--window-regime", choices=["wide", "tight"], default="wide")
    demo.add_argument("--fixed-vehicle-cost", type=float, default=35.0)
    demo.add_argument("--mode", choices=[mode.value for mode in PricingMode], default="exact")
    demo.add_argument("--checkpoint")
    demo.add_argument("--keep-ratio", type=float, default=0.35)
    demo.add_argument("--min-outgoing", type=int, default=2)
    demo.set_defaults(handler=_handle_demo)

    collect = subparsers.add_parser(
        "collect",
        help="Collect exact-pricing arc labels from generated training trajectories.",
    )
    collect.add_argument("--instances", type=int, default=24)
    collect.add_argument("--customers", type=int, default=8)
    collect.add_argument("--seed", type=int, default=1000)
    collect.add_argument("--distribution", choices=["uniform", "clustered"], default="uniform")
    collect.add_argument("--window-regime", choices=["wide", "tight"], default="wide")
    collect.add_argument("--fixed-vehicle-cost", type=float, default=35.0)
    collect.add_argument("--top-k-routes", type=int, default=3)
    collect.add_argument("--output", "-o", default="artifacts/pricing-dataset.npz")
    collect.set_defaults(handler=_handle_collect)

    train = subparsers.add_parser("train", help="Train the NumPy MLP arc scorer.")
    train.add_argument("dataset")
    train.add_argument("--checkpoint", "-o", default="artifacts/arc-scorer.npz")
    train.add_argument("--hidden-dim", type=int, default=32)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(handler=_handle_train)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Compare exact, heuristic, random, and optionally learned pricing.",
    )
    benchmark.add_argument("--checkpoint")
    benchmark.add_argument("--instances", type=int, default=12)
    benchmark.add_argument("--customers", type=int, default=9)
    benchmark.add_argument("--seed", type=int, default=20000)
    benchmark.add_argument("--distribution", choices=["uniform", "clustered"], default="uniform")
    benchmark.add_argument("--window-regime", choices=["wide", "tight"], default="wide")
    benchmark.add_argument("--fixed-vehicle-cost", type=float, default=35.0)
    benchmark.add_argument("--keep-ratio", type=float, default=0.35)
    benchmark.add_argument("--min-outgoing", type=int, default=2)
    benchmark.add_argument("--full-enumeration-limit", type=int, default=8)
    benchmark.add_argument("--output-json", default="artifacts/benchmark.json")
    benchmark.add_argument("--output-csv", default="artifacts/benchmark.csv")
    benchmark.set_defaults(handler=_handle_benchmark)

    return parser


def _handle_demo(args: argparse.Namespace) -> int:
    generated = generate_solomon_like(
        args.customers,
        seed=args.seed,
        distribution=args.distribution,
        window_regime=args.window_regime,
        fixed_vehicle_cost=args.fixed_vehicle_cost,
    )
    scorer = NumpyMLPArcScorer.load(args.checkpoint) if args.checkpoint else None
    result = run_column_generation(
        generated.instance,
        mode=PricingMode(args.mode),
        scorer=scorer,
        config=ColumnGenerationConfig(
            keep_ratio=args.keep_ratio,
            min_outgoing=args.min_outgoing,
            random_seed=args.seed,
        ),
    )
    payload = result.to_dict()
    payload["witness_routes"] = [list(route) for route in generated.witness_routes]
    _print_json(payload)
    return 0 if result.globally_certified else 3


def _handle_collect(args: argparse.Namespace) -> int:
    instances = _generated_instances(args)
    dataset = collect_pricing_dataset(instances, top_k_routes=args.top_k_routes)
    save_dataset(dataset, args.output)
    _print_json({"output": args.output, **dataset.summary()})
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    train, validation = split_by_instance(
        dataset,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    model = NumpyMLPArcScorer(
        input_dim=train.features.shape[1],
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    history = model.fit(
        train.features,
        train.labels,
        TrainingConfig(
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
    )
    train_metrics = binary_classification_metrics(train.labels, model.predict_proba(train.features))
    validation_metrics = binary_classification_metrics(
        validation.labels,
        model.predict_proba(validation.features),
    )
    model.save(
        args.checkpoint,
        metadata={
            "version": __version__,
            "dataset_metadata": dataset.metadata,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        },
    )
    _print_json(
        {
            "checkpoint": args.checkpoint,
            "initial_loss": history.losses[0],
            "final_loss": history.losses[-1],
            "positive_weight": history.positive_weight,
            "train": train_metrics,
            "validation": validation_metrics,
        }
    )
    return 0


def _handle_benchmark(args: argparse.Namespace) -> int:
    instances = _generated_instances(args)
    scorer = NumpyMLPArcScorer.load(args.checkpoint) if args.checkpoint else None
    modes = [PricingMode.EXACT, PricingMode.HEURISTIC, PricingMode.RANDOM]
    if scorer is not None:
        modes.append(PricingMode.LEARNED)
    report = run_benchmark(
        instances,
        scorer=scorer,
        modes=tuple(modes),
        config=ColumnGenerationConfig(
            keep_ratio=args.keep_ratio,
            min_outgoing=args.min_outgoing,
            random_seed=args.seed,
        ),
        full_enumeration_limit=args.full_enumeration_limit,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    save_report_json(report, args.output_json)
    save_report_csv(report, args.output_csv)
    _print_json(
        {
            "output_json": args.output_json,
            "output_csv": args.output_csv,
            "summary": report.summary,
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
