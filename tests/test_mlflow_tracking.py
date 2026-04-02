from pathlib import Path

import mlflow
import pandas as pd

from milk_dashboard.data.splits import generate_time_splits
from milk_dashboard.mlflow.tracking import log_experiment_run
from milk_dashboard.models.evaluation import evaluate_model
from milk_dashboard.models.registry import create_model


def test_log_experiment_run_writes_metrics_and_artifacts(tmp_path: Path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    mlflow.set_tracking_uri(tracking_uri)

    daily_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=80, freq="D"),
            "gallons": [float((i % 10) + 5) for i in range(80)],
            "total_milk_oz": [float(((i % 10) + 5) * 128) for i in range(80)],
        }
    )
    cv_splits, holdout_split = generate_time_splits(
        daily_df,
        holdout_days=14,
        min_train_days=28,
        horizon_days=7,
        step_days=7,
        max_splits=3,
    )

    model_template = create_model("moving_average", window_size=7)
    evaluation = evaluate_model(model_template, cv_splits=cv_splits, holdout_split=holdout_split)

    fitted_model = create_model("moving_average", window_size=7)
    fitted_model.fit(daily_df)

    run_info = log_experiment_run(
        fitted_model=fitted_model,
        evaluation=evaluation,
        train_df=holdout_split.train,
        test_df=holdout_split.test,
        source_config={"source_id": "test"},
        preprocess_config={"fill_missing_days": True},
        split_config={"holdout_days": 14},
        experiment_name="milk_test_experiment",
        registered_model_name=None,
        tracking_uri=tracking_uri,
    )

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_info.run_id)

    assert "holdout_mae" in run.data.metrics
    artifacts = client.list_artifacts(run_info.run_id, path="datasets")
    names = sorted([a.path.split("/")[-1] for a in artifacts])
    assert "train.parquet" in names
    assert "test.parquet" in names

