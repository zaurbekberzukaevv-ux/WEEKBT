"""CSV storage helpers for OHLCV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def save_ohlcv_csv(df: pd.DataFrame, path: Path) -> None:
    """Save OHLCV data to CSV, creating parent directories when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, OHLCV_COLUMNS].to_csv(path, index=False)


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Load OHLCV data from CSV with UTC timestamps."""

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.loc[:, OHLCV_COLUMNS]
