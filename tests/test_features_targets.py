from __future__ import annotations

import numpy as np
import pandas as pd

from weekbt.features.feature_engineer import compute_window_features
from weekbt.features.windowing import create_windows
from weekbt.targets.target_engineer import compute_targets


def sample_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC"),
            "open": [10, 11, 12, 11, 13, 14, 15, 16],
            "high": [10, 11, 12, 11, 13, 14, 15, 16],
            "low": [10, 11, 12, 11, 13, 14, 15, 16],
            "close": [10, 11, 12, 11, 13, 14, 15, 16],
            "volume": [100, 120, 140, 160, 180, 200, 220, 240],
        }
    )


def test_create_windows_uses_raw_index_as_window_id() -> None:
    windows = create_windows(sample_ohlcv(), window_size=3)

    assert list(windows.index) == [0, 1, 2, 3, 4, 5]
    assert windows.loc[0, "end_idx"] == 2
    assert windows.loc[5, "window_end"] == pd.Timestamp("2024-01-08", tz="UTC")


def test_compute_window_features_matches_formulas() -> None:
    ohlcv = sample_ohlcv()
    windows = create_windows(ohlcv, window_size=3)

    features = compute_window_features(ohlcv, windows)

    closes = np.array([10.0, 11.0, 12.0])
    expected_volatility = np.std(np.diff(np.log(closes)), ddof=0)
    expected_slope = np.polyfit(np.arange(3, dtype=float), closes, deg=1)[0]

    assert np.isclose(features.loc[0, "cumulative_return"], 0.2)
    assert np.isclose(features.loc[0, "volatility"], expected_volatility)
    assert features.loc[0, "average_volume"] == 120.0
    assert np.isclose(features.loc[0, "trend_strength"], expected_slope / 10.0)
    assert features.loc[0, "max_drawdown"] == 0.0


def test_compute_targets_drops_windows_without_future_horizon() -> None:
    ohlcv = sample_ohlcv()
    windows = create_windows(ohlcv, window_size=3)

    targets = compute_targets(ohlcv, windows, window_size=3, horizon=2)

    assert list(targets.index) == [0, 1, 2, 3]
    assert np.isclose(targets.loc[0, "future_return"], (13 / 11) - 1)
    assert np.isclose(targets.loc[0, "mfe"], (13 / 12) - 1)
    assert np.isclose(targets.loc[0, "mae"], (11 / 12) - 1)
