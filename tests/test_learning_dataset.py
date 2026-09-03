from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
from learning_to_price.generator import generate_solomon_like
from learning_to_price.learning import NumpyMLPArcScorer, TrainingConfig


def _instances(count: int = 4) -> list:
    return [generate_solomon_like(5, seed=300 + index).instance for index in range(count)]


def test_dataset_round_trip_and_grouped_split(tmp_path: Path) -> None:
    dataset = collect_pricing_dataset(_instances(), top_k_routes=2)
    assert dataset.features.shape[1] == len(dataset.feature_names)
    assert 0.0 < dataset.positive_rate < 1.0
    path = tmp_path / "pricing.npz"
    save_dataset(dataset, path)
    loaded = load_dataset(path)
    assert np.array_equal(loaded.features, dataset.features)
    train, validation = split_by_instance(loaded, validation_fraction=0.25, seed=4)
    assert set(train.instance_ids).isdisjoint(set(validation.instance_ids))


def test_numpy_mlp_training_and_checkpoint_round_trip(tmp_path: Path) -> None:
    dataset = collect_pricing_dataset(_instances(), top_k_routes=2)
    model = NumpyMLPArcScorer(dataset.features.shape[1], hidden_dim=12, seed=5)
    history = model.fit(
        dataset.features,
        dataset.labels,
        TrainingConfig(hidden_dim=12, epochs=18, batch_size=128, seed=5),
    )
    assert history.losses[-1] < history.losses[0]
    before = model.predict_proba(dataset.features[:20])
    checkpoint = tmp_path / "model.npz"
    model.save(checkpoint, metadata={"purpose": "test"})
    restored = NumpyMLPArcScorer.load(checkpoint)
    after = restored.predict_proba(dataset.features[:20])
    assert np.allclose(before, after)
    assert restored.metadata["purpose"] == "test"


def test_learned_hybrid_matches_exact_lp() -> None:
    training = collect_pricing_dataset(_instances(3), top_k_routes=2)
    scorer = NumpyMLPArcScorer(training.features.shape[1], hidden_dim=12, seed=9)
    scorer.fit(
        training.features,
        training.labels,
        TrainingConfig(hidden_dim=12, epochs=10, batch_size=128, seed=9),
    )
    instance = generate_solomon_like(6, seed=999).instance
    exact = run_column_generation(instance, mode=PricingMode.EXACT)
    learned = run_column_generation(
        instance,
        mode=PricingMode.LEARNED,
        scorer=scorer,
        config=ColumnGenerationConfig(keep_ratio=0.4, min_outgoing=2),
    )
    assert learned.globally_certified
    assert learned.final_lp_objective == pytest.approx(exact.final_lp_objective, abs=1e-7)
