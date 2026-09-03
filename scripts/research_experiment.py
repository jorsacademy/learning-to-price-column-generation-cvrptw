from __future__ import annotations

import argparse
from pathlib import Path

from learning_to_price.experiment import run_research_experiment, save_experiment_result

parser = argparse.ArgumentParser()
parser.add_argument("--train-instances", type=int, default=24)
parser.add_argument("--train-customers", type=int, default=8)
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--benchmark-instances", type=int, default=8)
parser.add_argument("--keep-ratio", type=float, default=0.35)
parser.add_argument("--checkpoint", default="artifacts/research-arc-scorer.npz")
parser.add_argument("--output", default="artifacts/research-experiment.json")
args = parser.parse_args()

scorer, result = run_research_experiment(
    train_instances=args.train_instances,
    train_customers=args.train_customers,
    epochs=args.epochs,
    benchmark_instances=args.benchmark_instances,
    keep_ratio=args.keep_ratio,
)
Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
scorer.save(args.checkpoint, metadata={"protocol": "research_experiment_v1"})
save_experiment_result(result, args.output)
print(args.output)
