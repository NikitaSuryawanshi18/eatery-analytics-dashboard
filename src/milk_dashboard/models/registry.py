"""Model registry map."""

from __future__ import annotations

from typing import Any

from .base import ModelRunner
from .moving_average_runner import MovingAverageRunner
from .prophet_runner import ProphetRunner
from .seasonal_naive_runner import SeasonalNaiveRunner


MODEL_REGISTRY: dict[str, type[ModelRunner]] = {
    "prophet": ProphetRunner,
    "seasonal_naive": SeasonalNaiveRunner,
    "moving_average": MovingAverageRunner,
}


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def create_model(model_name: str, **kwargs: Any) -> ModelRunner:
    key = model_name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list_models()}")
    return MODEL_REGISTRY[key](**kwargs)

