"""Shared model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ModelRunner(ABC):
    """Interface implemented by all candidate models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model key used in registry and MLflow logs."""

    @property
    @abstractmethod
    def versioned_params(self) -> dict[str, object]:
        """Serializable hyperparameters stored in experiment logs."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit model on historical daily data."""

    @abstractmethod
    def predict(self, horizon_days: int) -> pd.DataFrame:
        """Forecast future gallons for `horizon_days`."""

    @abstractmethod
    def clone(self) -> "ModelRunner":
        """Return a fresh untrained instance with the same params."""

