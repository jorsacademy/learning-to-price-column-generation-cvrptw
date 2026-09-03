from __future__ import annotations

from learning_to_price.experiment import run_research_experiment


def test_research_experiment_smoke() -> None:
    _, result = run_research_experiment(
        train_instances=3,
        train_customers=4,
        hidden_dim=8,
        epochs=3,
        benchmark_instances=1,
        keep_ratio=0.5,
    )
    assert set(result.scenarios) == {
        "in_distribution",
        "size_shift",
        "clustered_shift",
        "tight_windows_shift",
    }
    for payload in result.scenarios.values():
        assert payload["summary"]["learned"]["certified_rate"] == 1.0
