"""Leakage-safe StandardScaler helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(train_features: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on training features only."""

    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


def transform_features(features: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Transform features while preserving index and columns."""

    transformed = scaler.transform(features)
    return pd.DataFrame(transformed, index=features.index, columns=features.columns)


def fit_transform_cycle(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit on train features and transform train/test features."""

    scaler = fit_scaler(train_features)
    return transform_features(train_features, scaler), transform_features(test_features, scaler), scaler
