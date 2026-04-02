import math

import numpy as np

from milk_dashboard.models.metrics import calculate_metrics


def test_calculate_metrics_values():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([8.0, 21.0, 29.0])

    metrics = calculate_metrics(actual, predicted)

    assert round(metrics["mae"], 6) == round((2 + 1 + 1) / 3, 6)
    assert round(metrics["rmse"], 6) == round(math.sqrt((4 + 1 + 1) / 3), 6)
    assert metrics["mape"] > 0

