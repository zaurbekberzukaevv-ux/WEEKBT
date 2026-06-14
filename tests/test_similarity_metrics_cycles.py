from __future__ import annotations

import numpy as np
import pandas as pd

from weekbt.backtest.rolling_cycle import generate_cycles, select_cycle_windows
from weekbt.evaluation.metrics import compute_cycle_metrics
from weekbt.features.windowing import create_windows
from weekbt.similarity.cosine_similarity import cosine_similarity_scores, find_top_analogs


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    train = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 0.0]}, index=[10, 11])
    test_vector = pd.Series({"a": 0.0, "b": 0.0})

    scores = cosine_similarity_scores(train, test_vector)

    assert scores.to_dict() == {10: 0.0, 11: 0.0}


def test_find_top_analogs_applies_exclusion_radius() -> None:
    train = pd.DataFrame({"a": [1.0, 1.0, 0.0], "b": [0.0, 0.0, 1.0]}, index=[8, 10, 20])
    test_vector = pd.Series({"a": 1.0, "b": 0.0})

    analogs = find_top_analogs(train, test_vector, test_window_id=10, top_n=2, exclusion_radius=2)

    assert analogs == [20]


def test_compute_cycle_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "predicted_future_return": [0.1, -0.1],
            "actual_future_return": [0.2, 0.1],
            "predicted_mfe": [0.3, 0.0],
            "actual_mfe": [0.1, 0.1],
            "predicted_mae": [-0.1, -0.2],
            "actual_mae": [-0.2, -0.1],
        }
    )

    metrics = compute_cycle_metrics(predictions)

    assert metrics["n_predictions"] == 2
    assert np.isclose(metrics["MAE_error_return"], 0.15)
    assert np.isclose(metrics["RMSE_return"], np.sqrt((0.1**2 + 0.2**2) / 2))
    assert metrics["directional_accuracy"] == 0.5


def test_generate_cycles_uses_calendar_months() -> None:
    cycles = generate_cycles(
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"),
    )

    assert len(cycles) == 2
    assert cycles[0]["train_start"] == pd.Timestamp("2020-01-01", tz="UTC")
    assert cycles[0]["test_end"] == pd.Timestamp("2021-01-01", tz="UTC")
    assert cycles[1]["train_start"] == pd.Timestamp("2021-01-01", tz="UTC")


def test_select_cycle_windows_by_window_end() -> None:
    ohlcv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC"),
            "open": range(10),
            "high": range(10),
            "low": range(10),
            "close": range(10),
            "volume": range(10),
        }
    )
    windows = create_windows(ohlcv, window_size=2)
    cycle = {
        "train_start": pd.Timestamp("2020-01-02", tz="UTC"),
        "train_end": pd.Timestamp("2020-01-05", tz="UTC"),
        "test_start": pd.Timestamp("2020-01-05", tz="UTC"),
        "test_end": pd.Timestamp("2020-01-08", tz="UTC"),
    }

    train_ids, test_ids = select_cycle_windows(windows, cycle)

    assert list(train_ids) == [0, 1, 2]
    assert list(test_ids) == [3, 4, 5]
