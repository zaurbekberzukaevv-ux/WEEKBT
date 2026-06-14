"""UTC timestamp helpers."""

from __future__ import annotations

import pandas as pd


def ensure_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-aware UTC pandas Timestamp."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def inclusive_end_of_day(value: str | pd.Timestamp) -> pd.Timestamp:
    """Treat date-only CLI end dates as inclusive through the end of that UTC day."""

    timestamp = ensure_utc(value)
    return timestamp + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
