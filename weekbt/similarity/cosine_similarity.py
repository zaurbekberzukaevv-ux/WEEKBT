"""Cosine similarity search with window-id exclusion."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cosine_similarity_scores(train_features: pd.DataFrame, test_vector: pd.Series) -> pd.Series:
    """Compute cosine similarity, returning 0 for zero-norm vectors."""

    train_values = train_features.to_numpy(dtype=float)
    test_values = test_vector.to_numpy(dtype=float)
    train_norms = np.linalg.norm(train_values, axis=1)
    test_norm = float(np.linalg.norm(test_values))

    if test_norm == 0.0:
        return pd.Series(0.0, index=train_features.index)

    denom = train_norms * test_norm
    scores = np.divide(
        train_values @ test_values,
        denom,
        out=np.zeros(len(train_features), dtype=float),
        where=denom != 0,
    )
    return pd.Series(scores, index=train_features.index)


def find_top_analogs(
    train_features: pd.DataFrame,
    test_vector: pd.Series,
    test_window_id: int,
    top_n: int,
    exclusion_radius: int,
) -> list[int]:
    """Return up to top_n most similar train window ids after exclusion."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if train_features.empty:
        return []

    candidate_features = train_features.copy()
    if exclusion_radius >= 0:
        candidate_features = candidate_features[
            np.abs(candidate_features.index.to_numpy(dtype=int) - int(test_window_id)) > exclusion_radius
        ]
    if candidate_features.empty:
        return []

    scores = cosine_similarity_scores(candidate_features, test_vector)
    ranked = scores.sort_values(ascending=False, kind="mergesort")
    return [int(window_id) for window_id in ranked.head(top_n).index]
