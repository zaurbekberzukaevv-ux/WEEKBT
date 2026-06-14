"""Sliding-window construction over OHLCV rows."""

from __future__ import annotations

import pandas as pd


def create_windows(ohlcv: pd.DataFrame, window_size: int, step: int = 1) -> pd.DataFrame:
    """Create sliding windows indexed by raw OHLCV row index."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if len(ohlcv) < window_size:
        return pd.DataFrame(
            columns=["window_id", "start_idx", "end_idx", "window_start", "window_end"]
        ).set_index("window_id")

    records = []
    for start_idx in range(0, len(ohlcv) - window_size + 1, step):
        end_idx = start_idx + window_size - 1
        records.append(
            {
                "window_id": start_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "window_start": ohlcv.loc[start_idx, "timestamp"],
                "window_end": ohlcv.loc[end_idx, "timestamp"],
            }
        )

    return pd.DataFrame.from_records(records).set_index("window_id")
