"""MLflow PyFunc wrapper for fitted runner objects."""

from __future__ import annotations

import cloudpickle
import pandas as pd
import mlflow.pyfunc


class RunnerPyFuncModel(mlflow.pyfunc.PythonModel):
    """Generic pyfunc wrapper that delegates to saved ModelRunner."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        with open(context.artifacts["runner"], "rb") as f:
            self._runner = cloudpickle.load(f)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("model_input must be a pandas DataFrame")
        if "horizon_days" not in model_input.columns:
            raise ValueError("model_input must include horizon_days")

        horizon = int(model_input["horizon_days"].iloc[0])
        if horizon <= 0:
            raise ValueError("horizon_days must be > 0")
        return self._runner.predict(horizon_days=horizon)

