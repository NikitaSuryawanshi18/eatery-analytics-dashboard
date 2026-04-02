# Architecture and Code Walkthrough

This document explains how the codebase is wired and where each responsibility lives.

## High-level flow

```text
Raw data files
   -> DataSource adapter
   -> Cleaning pipeline
   -> Daily aggregated dataset
   -> Time splits (rolling CV + holdout)
   -> Model runner fit/predict
   -> Evaluation metrics
   -> MLflow run logging + pyfunc packaging
   -> Manual model promotion (alias: production)
   -> FastAPI loads promoted model and serves forecasts
```

## Module map

### `src/milk_dashboard/data`

- `interfaces.py`
  - `DataSource` contract (`load_sales`, `load_ingredients`, `source_id`)
- `file_source.py`
  - `FileDataSource` for CSV/XLSX-backed data
- `cleaning.py`
  - Reusable transforms:
    - `standardize_column_names`
    - `parse_money_columns`
    - `attach_datetime_columns`
    - `remove_voided_items`
    - `prepare_ingredient_table`
    - `merge_sales_and_ingredients`
    - `aggregate_daily_milk_usage`
    - `ensure_daily_frequency`
    - `prepare_daily_data`
- `splits.py`
  - Train/validation split logic:
    - final holdout split
    - rolling time-series CV splits

### `src/milk_dashboard/models`

- `base.py`
  - `ModelRunner` interface (required for all models):
    - `name`
    - `versioned_params`
    - `fit(train_df)`
    - `predict(horizon_days)`
    - `clone()`
- `prophet_runner.py`
  - Prophet implementation of `ModelRunner`
- `seasonal_naive_runner.py`
  - weekday baseline runner
- `moving_average_runner.py`
  - moving average baseline runner
- `registry.py`
  - central model registry map and model factory
- `metrics.py`
  - MAE, RMSE, MAPE calculation
- `evaluation.py`
  - split-level evaluation and summary aggregation

### `src/milk_dashboard/mlflow`

- `tracking.py`
  - `log_experiment_run` logs:
    - params (model/source/preprocess/split)
    - metrics (CV + holdout summary)
    - artifacts:
      - `datasets/train.parquet`
      - `datasets/test.parquet`
      - `predictions/cv_predictions.parquet`
      - `predictions/test_predictions.parquet`
      - metadata JSON files
    - model as MLflow pyfunc
  - `promote_model_alias` sets registry alias (for production selection)
- `pyfunc_model.py`
  - generic pyfunc wrapper that loads serialized runner and predicts from `horizon_days`

### `src/milk_dashboard/service`

- `model_loader.py`
  - loads model from MLflow by name + alias (default alias `production`)
- `schemas.py`
  - request/response contracts for inference
- `inference.py`
  - `ForecastService`:
    - calls pyfunc model
    - validates output shape (`date`, `forecast_gallons`)
    - computes totals and order quantity with safety buffer

### `src/milk_dashboard/api`

- `app.py`
  - FastAPI app with:
    - `GET /health`
    - `GET /ready`
    - `POST /v1/forecast`
  - lazy service loading from MLflow via env-configurable settings

### `src/milk_dashboard/pipeline.py`

High-level orchestration used by notebooks and CLI:

- `prepare_from_source`
- `evaluate_registered_model`
- `fit_final_model`
- `evaluate_fit_and_log`

## Notebooks

- `notebooks/01_data_cleaning.ipynb`
  - runs modular cleaning pipeline and exports reproducible data artifacts.
- `notebooks/02_model_lab.ipynb`
  - compares declared models, logs runs to MLflow, and supports manual promotion.

## CLI entrypoints

- `python -m milk_dashboard.cli.run_data_prep`
- `python -m milk_dashboard.cli.run_experiment`

Use these for scriptable/non-notebook execution.

## Inference contract

Input (`POST /v1/forecast`):

- `horizon_days` (1-30)
- `lookback_days` (1-365)
- `safety_buffer_pct` (0-100)
- optional `source_id`

Output:

- `daily_forecast[]` with `date`, `forecast_gallons`
- `expected_total_gallons`
- `order_gallons`
- `safety_buffer_pct`
- model metadata (`model_name`, `model_version`, `run_id`)

## Notes and current constraints

- Production selection is manual via MLflow alias promotion.
- The serving layer expects pyfunc model output with `date` and `forecast_gallons`.
- `lookback_days` is part of API contract for future parity needs; current production scoring is horizon-driven from the promoted model artifact.
