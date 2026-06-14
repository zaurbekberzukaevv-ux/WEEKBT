"""Deterministic feature computation for OHLCV windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "cumulative_return",
    "volatility",
    "average_volume",
    "trend_strength",
    "max_drawdown",
]


def compute_window_features(ohlcv: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Compute the formal feature matrix for each window."""

    records: list[dict[str, float | int]] = []
    closes_all = ohlcv["close"].to_numpy(dtype=float)
    volumes_all = ohlcv["volume"].to_numpy(dtype=float)

    for window_id, window in windows.iterrows():
        start_idx = int(window["start_idx"])
        end_idx = int(window["end_idx"])
        closes = closes_all[start_idx : end_idx + 1]
        volumes = volumes_all[start_idx : end_idx + 1]

        close_first = closes[0]
        close_last = closes[-1]
        cumulative_return = (close_last / close_first) - 1 if close_first != 0 else 0.0

        if len(closes) > 1:
            log_returns = np.diff(np.log(closes))
            volatility = float(np.std(log_returns, ddof=0))
        else:
            volatility = 0.0

        average_volume = float(np.mean(volumes))

        if close_first != 0:
            x = np.arange(len(closes), dtype=float)
            slope = float(np.polyfit(x, closes, deg=1)[0])
            trend_strength = slope / close_first
        else:
            trend_strength = 0.0

        running_max = np.maximum.accumulate(closes)
        drawdowns = closes / running_max - 1
        max_drawdown = float(np.min(drawdowns))

        records.append(
            {
                "window_id": int(window_id),
                "cumulative_return": float(cumulative_return),
                "volatility": volatility,
                "average_volume": average_volume,
                "trend_strength": float(trend_strength),
                "max_drawdown": max_drawdown,
            }
        )

    if not records:
        return pd.DataFrame(columns=FEATURE_COLUMNS).rename_axis("window_id")

    return pd.DataFrame.from_records(records).set_index("window_id").loc[:, FEATURE_COLUMNS]
