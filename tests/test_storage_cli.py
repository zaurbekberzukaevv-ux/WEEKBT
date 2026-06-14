from __future__ import annotations

import pandas as pd

from weekbt.cli import choose_automatic_start
from weekbt.data.ohlcv_storage import load_ohlcv_csv, save_ohlcv_csv


def test_save_and_load_ohlcv_csv_roundtrip(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [10.0, 20.0],
        }
    )
    path = tmp_path / "ohlcv.csv"

    save_ohlcv_csv(df, path)
    loaded = load_ohlcv_csv(path)

    pd.testing.assert_frame_equal(loaded, df)


def test_choose_automatic_start_prefers_five_years_when_available() -> None:
    ohlcv = pd.DataFrame({"timestamp": pd.date_range("2018-01-01", periods=10, freq="YS", tz="UTC")})
    end = pd.Timestamp("2025-01-01", tz="UTC")

    assert choose_automatic_start(ohlcv, end) == pd.Timestamp("2020-01-01", tz="UTC")


def test_choose_automatic_start_falls_back_to_earliest_when_less_than_three_years() -> None:
    ohlcv = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC")})
    end = pd.Timestamp("2025-01-01", tz="UTC")

    assert choose_automatic_start(ohlcv, end) == pd.Timestamp("2024-01-01", tz="UTC")
