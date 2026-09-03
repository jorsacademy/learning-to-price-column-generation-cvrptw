from __future__ import annotations

import json
from pathlib import Path

import pytest

from learning_to_price.benchmark import run_benchmark, save_report_csv, save_report_json
from learning_to_price.cli import main
from learning_to_price.column_generation import ColumnGenerationConfig, PricingMode
from learning_to_price.generator import generate_solomon_like


def test_benchmark_reports_zero_lp_gap_for_certified_modes(tmp_path: Path) -> None:
    instance = generate_solomon_like(6, seed=41).instance
    report = run_benchmark(
        [instance],
        modes=(PricingMode.EXACT, PricingMode.HEURISTIC),
        config=ColumnGenerationConfig(keep_ratio=0.4),
        full_enumeration_limit=6,
    )
    assert all(row.globally_certified for row in report.rows)
    assert all(row.lp_gap_percent == pytest.approx(0.0, abs=1e-7) for row in report.rows)
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    save_report_json(report, json_path)
    save_report_csv(report, csv_path)
    assert json.loads(json_path.read_text())["summary"]["exact"]["certified_rate"] == 1.0
    assert csv_path.read_text().startswith("instance,customer_count,mode")


def test_demo_cli_outputs_certified_result(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["demo", "--customers", "4", "--seed", "7", "--mode", "exact"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["globally_certified"] is True
    assert payload["mode"] == "exact"
