"""Future return, MFE, and MAE computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["future_return", "mfe", "mae"]


def compute_targets(ohlcv: pd.DataFrame, windows: pd.DataFrame, window_size: int, horizon: int) -> pd.DataFrame:
    """Compute future targets for windows that have enough future candles."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    closes = ohlcv["close"].to_numpy(dtype=float)
    records: list[dict[str, float | int]] = []

    for window_id in windows.index:
        window_id_int = int(window_id)
        window_end_idx = window_id_int + window_size - 1
        future_start_idx = window_id_int + window_size
        future_end_idx = future_start_idx + horizon - 1

        if future_end_idx >= len(closes):
            continue

        future_closes = closes[future_start_idx : future_end_idx + 1]
        close_future_first = future_closes[0]
        close_future_last = future_closes[-1]
        entry = closes[window_end_idx]

        future_return = (close_future_last / close_future_first) - 1 if close_future_first != 0 else 0.0
        mfe = (np.max(future_closes) / entry) - 1 if entry != 0 else 0.0
        mae = (np.min(future_closes) / entry) - 1 if entry != 0 else 0.0

        records.append(
            {
                "window_id": window_id_int,
                "future_return": float(future_return),
                "mfe": float(mfe),
                "mae": float(mae),
            }
        )

    if not records:
        return pd.DataFrame(columns=TARGET_COLUMNS).rename_axis("window_id")

    return pd.DataFrame.from_records(records).set_index("window_id").loc[:, TARGET_COLUMNS]
