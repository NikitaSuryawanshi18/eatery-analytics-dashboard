# Ingredient Forecasting

This document describes the automated ingredient forecasting pipeline added on top of the existing milk forecasting code.

## Current Project Findings

The implementation reuses the current project structure instead of introducing a new system.

### Existing data sources

- Raw sales input: `JItters data.csv`
- Ingredient mapping sheet: `Ingredient Measure.xlsx`
- Existing cleaned artifact: `artifacts/data/cleaned_merged_with_ingredients.csv`

### Existing model code reused

- `src/milk_dashboard/models/prophet_runner.py`
- `src/milk_dashboard/models/moving_average_runner.py`
- `src/milk_dashboard/models/seasonal_naive_runner.py`
- Shared runner contract: `src/milk_dashboard/models/base.py`
- Model factory: `src/milk_dashboard/models/registry.py`

### Existing cleaning code reused as reference

- `src/milk_dashboard/data/cleaning.py`
- `src/milk_dashboard/pipeline.py`

### Existing storage style preserved

The project already uses files and SQLite.

For this feature, the pipeline stays file-based:

- CSV logs
- CSV config
- CSV usage history
- Pickle model artifacts
- JSON model metadata

No new database layer, orchestration tool, or external model registry was added.

## Important Constraint From Current Data

The current cleaned artifact only preserves milk.

The live ingredient mapping sheet contains additional ingredient columns:

- `Milk (oz)`
- `espresso/Coffee (grams)`
- `Tea`
- `flavor oz`

Because of that, the new pipeline can do one of two things:

1. Use an existing `ingredient_usage_history.csv` if present.
2. If that file is missing, derive ingredient usage history directly from:
   - `JItters data.csv`
   - `Ingredient Measure.xlsx`

This keeps the pipeline repeatable while still using the current Square sales + ingredient mapping workflow.

## New Files

### Core pipeline

- `src/milk_dashboard/ingredient_forecasting.py`

Main functions:

- `load_ingredient_config(...)`
- `build_ingredient_usage_history_from_sources(...)`
- `load_or_build_ingredient_usage_history(...)`
- `forecast_ingredient_usage(...)`
- `evaluate_completed_forecasts(...)`
- `run_daily_forecast_pipeline(...)`
- `monthly_retrain_all_models(...)`

### Run scripts

- `scripts/run_daily_ingredient_forecast.py`
- `scripts/run_monthly_ingredient_retrain.py`

### Tests

- `tests/test_ingredient_forecasting.py`

## File Layout

Default paths used by the new pipeline:

```text
artifacts/data/
  ingredient_forecast_config.csv
  ingredient_usage_history.csv

artifacts/ingredient_models/
  milk/
    model_latest.pkl
    metadata_latest.json
    model_vYYYY_MM.pkl
    metadata_vYYYY_MM.json
  coffee_beans/
    ...

logs/
  forecast_log.csv
  retraining_log.csv
  pipeline_run_log.csv
```

## Ingredient Config

Default config path:

- `artifacts/data/ingredient_forecast_config.csv`

Current starter schema:

```csv
ingredient_id,ingredient_name,unit,order_frequency_days,current_stock,safety_stock,model_name,source_column,active,apply_qty_multiplier
milk,Milk,oz,7,,,prophet,Milk (oz),true,true
coffee_beans,Coffee Beans,grams,14,,,moving_average,espresso/Coffee (grams),true,true
tea,Tea,bags,14,,,seasonal_naive,Tea,false,true
flavor_syrup,Flavor Syrup,oz,10,,,moving_average,flavor oz,false,true
```

### Notes

- `order_frequency_days` controls both:
  - forecast window length
  - reorder cadence
- `source_column` is the ingredient sheet column used when deriving usage history from raw sales + ingredient mappings.
- `active=true` means the ingredient is included in daily forecasting and monthly retraining.
- `apply_qty_multiplier=true` means the mapped ingredient amount is multiplied by transaction `Qty`.

## Usage History

Default path:

- `artifacts/data/ingredient_usage_history.csv`

Schema:

```csv
date,ingredient_id,ingredient_name,usage_qty,unit
```

Rules:

- If the file already exists, the pipeline uses it directly.
- If it does not exist, the pipeline derives it from:
  - `JItters data.csv`
  - `Ingredient Measure.xlsx`
  - the configured `source_column` values

## Daily Forecast Pipeline

Function:

```python
run_daily_forecast_pipeline(run_date=None)
```

Execution order:

1. Load ingredient config.
2. Load or build ingredient usage history.
3. Evaluate completed pending forecasts first.
4. For each active ingredient:
   - skip if there is already a pending forecast covering the current run date
   - load latest model
   - if no model exists, train one immediately from history
   - forecast total usage for the next `order_frequency_days`
   - compute suggested order if `current_stock` and `safety_stock` are available
   - append a new forecast row to `forecast_log.csv`
5. Append a run record to `pipeline_run_log.csv`

### Why duplicate forecasts are avoided

The pipeline does not create overlapping daily duplicate forecasts for an ingredient that already has an active pending forecast window.

Example:

- Milk forecast created on `2026-05-01`
- Window is `2026-05-01` to `2026-05-07`
- A run on `2026-05-03` will not create another milk forecast
- A run on `2026-05-08` will create the next milk forecast window

This matches the intended ordering cadence better than logging a new overlapping 7-day forecast every day.

## Forecast Log

Default path:

- `logs/forecast_log.csv`

Schema:

```csv
forecast_id,forecast_created_date,ingredient_id,ingredient_name,window_start_date,window_end_date,order_frequency_days,predicted_usage,current_stock,safety_stock,suggested_order_qty,model_name,model_version,actual_usage,forecast_error,absolute_error,percentage_error,bias,evaluated_date,status
```

Suggested order formula:

```text
suggested_order_qty = max(predicted_usage + safety_stock - current_stock, 0)
```

If stock data is missing, `suggested_order_qty` remains blank.

## Forecast Evaluation

Function:

```python
evaluate_completed_forecasts(run_date=None, forecast_log_path=..., usage_history=...)
```

Evaluation rule:

```text
run_date > window_end_date
```

For each pending forecast whose window is complete:

- `actual_usage = sum(usage_qty between window_start_date and window_end_date)`
- `forecast_error = predicted_usage - actual_usage`
- `absolute_error = abs(forecast_error)`
- `percentage_error = absolute_error / actual_usage`
- `bias = forecast_error`

If `actual_usage = 0`, percentage error is left blank.

The original `forecast_log.csv` row is updated in place and status changes from:

- `pending_evaluation`

to:

- `evaluated`

## Monthly Retraining

Function:

```python
monthly_retrain_all_models(run_date=None)
```

Behavior:

1. Load active ingredients from config.
2. Load or build ingredient usage history.
3. For each active ingredient:
   - build the daily training series using all history before `run_date`
   - create the configured model from the existing runner registry
   - fit the model
   - save:
     - `model_latest.pkl`
     - `metadata_latest.json`
     - versioned month-stamped copies
4. Log each ingredient retrain result.
5. Continue even if one ingredient fails.

Model version format:

```text
vYYYY_MM
```

## Retraining Log

Default path:

- `logs/retraining_log.csv`

Schema:

```csv
run_date,ingredient_id,ingredient_name,model_name,model_version,train_start_date,train_end_date,status,error_message
```

## Pipeline Run Log

Default path:

- `logs/pipeline_run_log.csv`

Schema:

```csv
run_date,run_type,status,ingredients_processed,ingredients_failed,error_message
```

Current run types:

- `daily_forecast`
- `monthly_retrain`
- `daily_forecast:<ingredient_id>` for per-ingredient failures

## Operational Runbook

### First-time setup

1. Review or edit:
   - `artifacts/data/ingredient_forecast_config.csv`
2. If you already maintain a curated ingredient usage history:
   - place it at `artifacts/data/ingredient_usage_history.csv`
3. Otherwise let the pipeline derive it from:
   - `JItters data.csv`
   - `Ingredient Measure.xlsx`

### Daily run

```bash
python scripts/run_daily_ingredient_forecast.py
```

### Monthly retrain

```bash
python scripts/run_monthly_ingredient_retrain.py
```

## Design Choices

### Why the existing model runners were reused

The current model runners already support:

- `fit(train_df)`
- `predict(horizon_days)`

They were originally written against a milk `gallons` target, but the pipeline now feeds each ingredient’s daily `usage_qty` series into that same interface under the existing `date`/`gallons` shape.

That keeps the model code unchanged and avoids a broader model-layer refactor.

### Why no MLflow was added to this pipeline

The project already uses MLflow for notebook experiments, but the requested operational ingredient pipeline needed to stay simple and file-based.

So this feature uses:

- pickle for models
- JSON for metadata
- CSV for logs

The experiment code remains available for future benchmarking, but it is not required for daily operational forecasting.

## Test Coverage Added

Automated tests now cover:

1. Multi-ingredient usage history derivation from raw sales + ingredient sheet.
2. Daily forecast creation across at least two ingredients.
3. Automatic evaluation of completed forecasts.
4. Monthly retraining across multiple ingredients.
5. Per-ingredient model artifact creation.

## Known Limits

- The source ingredient sheet contains mixed human-entered values such as text with numbers embedded, especially in `espresso/Coffee (grams)`.
- The pipeline currently extracts the first numeric value from such cells.
- If a mapping cell contains no numeric value, it contributes zero usage.
- The current cleaned artifact still remains milk-specific; the generic ingredient pipeline derives broader ingredient usage from the raw sales CSV plus the ingredient sheet when needed.

## Summary

This change does not replace the existing milk forecasting system.

It adds a reusable operational ingredient forecasting pipeline that:

- works across multiple ingredients
- uses `order_frequency_days` from config
- logs each forecast
- automatically evaluates completed forecast windows
- stores forecast error metrics
- retrains per ingredient monthly
- saves per-ingredient model artifacts
- continues processing even if one ingredient fails
