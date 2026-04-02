"""MLflow model loading for inference."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from milk_dashboard.constants import DEFAULT_MODEL_ALIAS, DEFAULT_MODEL_NAME


@dataclass
class LoadedModel:
    pyfunc_model: object
    model_name: str
    model_version: str | None
    run_id: str | None


def load_model_from_registry(
    *,
    tracking_uri: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    model_alias: str = DEFAULT_MODEL_ALIAS,
) -> LoadedModel:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, model_alias)
    model_uri = f"models:/{model_name}@{model_alias}"
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    return LoadedModel(
        pyfunc_model=pyfunc_model,
        model_name=model_name,
        model_version=str(model_version.version),
        run_id=model_version.run_id,
    )

