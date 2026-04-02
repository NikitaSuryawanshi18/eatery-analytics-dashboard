"""MLflow experiment logging utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import cloudpickle
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import pandas as pd

from milk_dashboard.constants import DEFAULT_EXPERIMENT_NAME, DEFAULT_MODEL_NAME
from milk_dashboard.models.base import ModelRunner
from milk_dashboard.models.evaluation import EvaluationBundle

from .pyfunc_model import RunnerPyFuncModel


@dataclass
class LoggedRunInfo:
    run_id: str
    model_uri: str
    registered_model_name: str | None
    registered_model_version: str | None


def _flatten_dict(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_dict(flat_key, value))
        else:
            out[flat_key] = value
    return out


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_actual_vs_forecast_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    required = {"date", "actual_gallons", "forecast_gallons"}
    if not required.issubset(df.columns):
        return

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")
    if d.empty:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(d["date"], d["actual_gallons"], label="Actual", linewidth=2)
    plt.plot(d["date"], d["forecast_gallons"], label="Forecast", linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Gallons")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _configure_tracking_uri(tracking_uri: str | None) -> str:
    uri = tracking_uri or "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(uri)
    return uri


def _get_or_create_experiment(name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name)


def _extract_registered_version(model_name: str, run_id: str) -> str | None:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    for version in versions:
        if version.run_id == run_id:
            return str(version.version)
    return None


def promote_model_alias(model_name: str, version: str, alias: str = "production") -> None:
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)


def log_experiment_run(
    *,
    fitted_model: ModelRunner,
    evaluation: EvaluationBundle,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    source_config: Mapping[str, Any],
    preprocess_config: Mapping[str, Any],
    split_config: Mapping[str, Any],
    split_details: Mapping[str, Any] | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    registered_model_name: str | None = DEFAULT_MODEL_NAME,
    tracking_uri: str | None = None,
) -> LoggedRunInfo:
    _configure_tracking_uri(tracking_uri)
    experiment_id = _get_or_create_experiment(experiment_name)

    with tempfile.TemporaryDirectory(prefix="milk_run_") as tmp_dir:
        tmp = Path(tmp_dir)

        train_path = tmp / "train.parquet"
        test_path = tmp / "test.parquet"
        holdout_pred_path = tmp / "test_predictions.parquet"
        cv_pred_path = tmp / "cv_predictions.parquet"
        holdout_plot_path = tmp / "holdout_actual_vs_forecast.png"
        cv_plot_path = tmp / "cv_actual_vs_forecast.png"
        schema_path = tmp / "schema.json"
        split_meta_path = tmp / "split_metadata.json"
        summary_path = tmp / "run_summary.json"
        runner_path = tmp / "runner.pkl"

        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        evaluation.holdout_result.predictions.to_parquet(holdout_pred_path, index=False)
        cv_pred_df = evaluation.cv_predictions_table()
        cv_pred_df.to_parquet(cv_pred_path, index=False)

        _write_actual_vs_forecast_plot(
            evaluation.holdout_result.predictions,
            holdout_plot_path,
            "Holdout: Actual vs Forecast",
        )
        if not cv_pred_df.empty:
            # Aggregate CV predictions by date for a quick global trend view.
            cv_plot_df = (
                cv_pred_df.groupby("date", as_index=False)[["actual_gallons", "forecast_gallons"]]
                .mean()
                .sort_values("date")
            )
            _write_actual_vs_forecast_plot(
                cv_plot_df,
                cv_plot_path,
                "CV (mean by date): Actual vs Forecast",
            )

        schema_payload = {
            "train_columns": train_df.dtypes.astype(str).to_dict(),
            "test_columns": test_df.dtypes.astype(str).to_dict(),
        }
        split_payload = {
            "cv_splits": [result.split_name for result in evaluation.cv_results],
            "holdout_split": evaluation.holdout_result.split_name,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_date_min": str(pd.to_datetime(train_df["date"], errors="coerce").min()) if "date" in train_df.columns else None,
            "train_date_max": str(pd.to_datetime(train_df["date"], errors="coerce").max()) if "date" in train_df.columns else None,
            "test_date_min": str(pd.to_datetime(test_df["date"], errors="coerce").min()) if "date" in test_df.columns else None,
            "test_date_max": str(pd.to_datetime(test_df["date"], errors="coerce").max()) if "date" in test_df.columns else None,
        }
        if split_details is not None:
            split_payload["split_details"] = dict(split_details)
        summary_payload = {
            "model_name": fitted_model.name,
            "metrics": evaluation.metric_summary(),
        }
        _write_json(schema_path, schema_payload)
        _write_json(split_meta_path, split_payload)
        _write_json(summary_path, summary_payload)

        with open(runner_path, "wb") as f:
            cloudpickle.dump(fitted_model, f)

        input_example = pd.DataFrame({"horizon_days": [7]})
        output_example = fitted_model.predict(horizon_days=7)
        signature = infer_signature(input_example, output_example)

        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params(_flatten_dict("model", fitted_model.versioned_params))
            mlflow.log_params(_flatten_dict("source", source_config))
            mlflow.log_params(_flatten_dict("preprocess", preprocess_config))
            mlflow.log_params(_flatten_dict("split", split_config))
            mlflow.log_metrics(evaluation.metric_summary())

            mlflow.log_artifact(str(train_path), artifact_path="datasets")
            mlflow.log_artifact(str(test_path), artifact_path="datasets")
            mlflow.log_artifact(str(holdout_pred_path), artifact_path="predictions")
            mlflow.log_artifact(str(cv_pred_path), artifact_path="predictions")
            mlflow.log_artifact(str(schema_path), artifact_path="metadata")
            mlflow.log_artifact(str(split_meta_path), artifact_path="metadata")
            mlflow.log_artifact(str(summary_path), artifact_path="metadata")
            if holdout_plot_path.exists():
                mlflow.log_artifact(str(holdout_plot_path), artifact_path="charts")
            if cv_plot_path.exists():
                mlflow.log_artifact(str(cv_plot_path), artifact_path="charts")

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=RunnerPyFuncModel(),
                artifacts={"runner": str(runner_path)},
                input_example=input_example,
                signature=signature,
                registered_model_name=registered_model_name,
            )

            run = mlflow.active_run()
            if run is None:
                raise RuntimeError("MLflow run unexpectedly missing")
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"

    model_version = None
    if registered_model_name:
        model_version = _extract_registered_version(registered_model_name, run_id)
    return LoggedRunInfo(
        run_id=run_id,
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        registered_model_version=model_version,
    )
