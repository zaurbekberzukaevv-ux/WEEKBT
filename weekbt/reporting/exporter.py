"""Export cycle reports and row-level predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def export_cycle_results(results: list[dict[str, object]], reports_dir: Path) -> list[tuple[Path, Path]]:
    """Write per-cycle JSON reports and CSV predictions."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    exported: list[tuple[Path, Path]] = []

    for result in results:
        cycle = result["cycle"]
        if not isinstance(cycle, dict):
            raise TypeError("cycle must be a dictionary")

        cycle_start = _date_slug(str(cycle["test_start"]))
        cycle_end = _date_slug(str(cycle["test_end"]))
        report_path = reports_dir / f"cycle_report_{cycle_start}_{cycle_end}.json"
        predictions_path = reports_dir / f"cycle_predictions_{cycle_start}_{cycle_end}.csv"

        predictions = result["predictions"]
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError("predictions must be a DataFrame")
        predictions.to_csv(predictions_path, index=False)

        report_payload = {
            "cycle_number": result["cycle_number"],
            "cycle": result["cycle"],
            "metrics": result["metrics"],
            "config": _jsonable(result["config"]),
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
        exported.append((report_path, predictions_path))

    return exported


def _date_slug(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
