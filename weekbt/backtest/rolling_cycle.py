"""Rolling half-year train/test backtest logic."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from weekbt.config import BacktestConfig
from weekbt.evaluation.metrics import compute_cycle_metrics
from weekbt.features.feature_engineer import compute_window_features
from weekbt.features.scaling import fit_transform_cycle
from weekbt.features.windowing import create_windows
from weekbt.similarity.cosine_similarity import find_top_analogs
from weekbt.targets.target_engineer import compute_targets
from weekbt.time_utils import ensure_utc


def generate_cycles(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int = 6,
    test_months: int = 6,
) -> list[dict[str, pd.Timestamp]]:
    """Generate calendar-month train/test cycles."""

    cycles: list[dict[str, pd.Timestamp]] = []
    train_start = ensure_utc(start)
    end = ensure_utc(end)

    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        cycles.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        train_start = test_end

    return cycles


def select_cycle_windows(
    windows: pd.DataFrame,
    cycle: dict[str, pd.Timestamp],
) -> tuple[pd.Index, pd.Index]:
    """Select train and test window ids by window end timestamp."""

    train_mask = (windows["window_end"] >= cycle["train_start"]) & (windows["window_end"] < cycle["train_end"])
    test_mask = (windows["window_end"] >= cycle["test_start"]) & (windows["window_end"] < cycle["test_end"])
    return windows.index[train_mask], windows.index[test_mask]


def run_backtest(ohlcv: pd.DataFrame, config: BacktestConfig) -> list[dict[str, object]]:
    """Run all rolling cycles and return metrics plus row-level predictions."""

    strategy = config.strategy
    windows = create_windows(ohlcv, strategy.window_size, step=1)
    features = compute_window_features(ohlcv, windows)
    targets = compute_targets(ohlcv, windows, strategy.window_size, strategy.horizon)

    if windows.empty:
        return []

    start = ensure_utc(config.market.start_date)
    end = ensure_utc(config.market.end_date)
    cycles = generate_cycles(start, end, strategy.train_months, strategy.test_months)
    results: list[dict[str, object]] = []

    for cycle_number, cycle in enumerate(cycles, start=1):
        train_ids, test_ids = select_cycle_windows(windows, cycle)
        train_ids = train_ids.intersection(features.index).intersection(targets.index)
        test_ids = test_ids.intersection(features.index).intersection(targets.index)

        if len(train_ids) == 0 or len(test_ids) == 0:
            predictions = _empty_predictions()
            metrics = compute_cycle_metrics(predictions)
            results.append(_cycle_result(cycle_number, cycle, metrics, predictions, config))
            continue

        train_features = features.loc[train_ids]
        test_features = features.loc[test_ids]
        train_scaled, test_scaled, _ = fit_transform_cycle(train_features, test_features)

        rows: list[dict[str, object]] = []
        for test_window_id, test_vector in test_scaled.iterrows():
            analog_ids = find_top_analogs(
                train_scaled,
                test_vector,
                int(test_window_id),
                strategy.top_n,
                strategy.exclusion_radius,
            )
            if not analog_ids:
                continue

            analog_targets = targets.loc[analog_ids]
            actual_target = targets.loc[test_window_id]
            rows.append(
                {
                    "test_window_id": int(test_window_id),
                    "timestamp_of_window_end": windows.loc[test_window_id, "window_end"],
                    "predicted_future_return": float(analog_targets["future_return"].mean()),
                    "actual_future_return": float(actual_target["future_return"]),
                    "predicted_mfe": float(analog_targets["mfe"].mean()),
                    "actual_mfe": float(actual_target["mfe"]),
                    "predicted_mae": float(analog_targets["mae"].mean()),
                    "actual_mae": float(actual_target["mae"]),
                    "analog_window_ids": ",".join(str(window_id) for window_id in analog_ids),
                }
            )

        predictions = pd.DataFrame(rows)
        if not predictions.empty:
            predictions["timestamp_of_window_end"] = pd.to_datetime(predictions["timestamp_of_window_end"], utc=True)
        else:
            predictions = _empty_predictions()

        metrics = compute_cycle_metrics(predictions)
        results.append(_cycle_result(cycle_number, cycle, metrics, predictions, config))

    return results


def _cycle_result(
    cycle_number: int,
    cycle: dict[str, pd.Timestamp],
    metrics: dict[str, float | int],
    predictions: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, object]:
    return {
        "cycle_number": cycle_number,
        "cycle": {key: value.isoformat() for key, value in cycle.items()},
        "metrics": metrics,
        "predictions": predictions,
        "config": asdict(config),
    }


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "test_window_id",
            "timestamp_of_window_end",
            "predicted_future_return",
            "actual_future_return",
            "predicted_mfe",
            "actual_mfe",
            "predicted_mae",
            "actual_mae",
            "analog_window_ids",
        ]
    )
