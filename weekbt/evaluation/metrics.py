"""Cycle-level backtest metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cycle_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Compute aggregate metrics for one test cycle."""

    n_predictions = int(len(predictions))
    if n_predictions == 0:
        return {
            "n_predictions": 0,
            "MAE_error_return": 0.0,
            "RMSE_return": 0.0,
            "directional_accuracy": 0.0,
            "MFE_MAE_error": 0.0,
            "MAE_MAE_error": 0.0,
        }

    return_error = predictions["predicted_future_return"] - predictions["actual_future_return"]
    mfe_error = predictions["predicted_mfe"] - predictions["actual_mfe"]
    mae_error = predictions["predicted_mae"] - predictions["actual_mae"]

    predicted_sign = np.sign(predictions["predicted_future_return"].to_numpy(dtype=float))
    actual_sign = np.sign(predictions["actual_future_return"].to_numpy(dtype=float))

    return {
        "n_predictions": n_predictions,
        "MAE_error_return": float(np.mean(np.abs(return_error))),
        "RMSE_return": float(np.sqrt(np.mean(np.square(return_error)))),
        "directional_accuracy": float(np.mean(predicted_sign == actual_sign)),
        "MFE_MAE_error": float(np.mean(np.abs(mfe_error))),
        "MAE_MAE_error": float(np.mean(np.abs(mae_error))),
    }
