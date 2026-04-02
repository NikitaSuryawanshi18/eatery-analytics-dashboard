"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted length mismatch")
    if len(actual) == 0:
        raise ValueError("actual/predicted arrays are empty")

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    non_zero_mask = actual != 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(error[non_zero_mask] / actual[non_zero_mask])) * 100.0)
    else:
        mape = float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape}

