# Model Development Guide

This is the practical guide for building new models in this repo.

## What you need to do to build models now

Follow this sequence every time:

1. Prepare clean daily dataset.
2. Train/evaluate candidate models (including your new model).
3. Log full reproducibility artifacts to MLflow.
4. Register model and promote selected version to `production`.
5. Serve through FastAPI using the promoted alias.

Everything below maps directly to implemented code.

## Step 0: Environment

Install once:

```bash
pip install -e .[dev]
```

Optional MLflow URI:

```bash
set MLFLOW_TRACKING_URI=sqlite:///mlruns_v2.db
```

## Step 1: Build cleaned dataset

Notebook path:

- `notebooks/01_data_cleaning.ipynb`

or CLI:

```bash
python -m milk_dashboard.cli.run_data_prep ^
  --sales "JItters data.csv" ^
  --ingredients "Ingredient Measure.xlsx" ^
  --out-dir "artifacts/data" ^
  --source-id "default_files"
```

This uses:

- `FileDataSource` from `data/file_source.py`
- `prepare_daily_data` from `data/cleaning.py`

Outputs:

- `daily.parquet` (feature-ready daily table)
- `schema.json` (column/dtype snapshot)

Default behavior is DataImport-compatible:

- no qty multiplier on mapped milk (`apply_qty_multiplier=False`)
- no automatic missing-day filling (`fill_missing_days=False`)

## Step 2: Run experiments

Notebook path:

- `notebooks/04_expanding_window_experiment_v2.ipynb`

or CLI:

```bash
python -m milk_dashboard.cli.run_experiment ^
  --input-csv "artifacts\\data\\cleaned_merged_with_ingredients.csv" ^
  --model prophet ^
  --tracking-uri "sqlite:///mlruns_v2.db" ^
  --experiment-name "lookforward_30_30_all_models" ^
  --registered-model-name "milk_forecast" ^
  --holdout-days 14 ^
  --min-train-days 30 ^
  --horizon-days 7 ^
  --step-days 7 ^
  --max-splits 5
```

If you want raw ingestion instead, skip `--input-csv` and pass both `--sales` and `--ingredients`.

Execution path:

- split generation: `data/splits.py`
- model creation: `models/registry.py`
- scoring: `models/evaluation.py` + `models/metrics.py`
- run logging: `mlflow/tracking.py`

Metrics logged:

- `cv_mae_mean`, `cv_rmse_mean`, `cv_mape_mean`
- `holdout_mae`, `holdout_rmse`, `holdout_mape`

## Step 3: Promote selected model to production

Use MLflow UI or code:

```python
from milk_dashboard.mlflow.tracking import promote_model_alias
promote_model_alias("milk_forecast", version="3", alias="production")
```

This alias is what FastAPI serves.

## Step 4: Serve inference

```bash
uvicorn milk_dashboard.api.app:app --reload
```

Production loading path:

- `service/model_loader.py` -> load `models:/milk_forecast@production`
- `service/inference.py` -> run forecasts + compute order recommendation

## How to add a new model

### 1) Create a new runner

Add file in `src/milk_dashboard/models/` (example: `my_model_runner.py`) implementing `ModelRunner`.

Minimum required methods:

- `name` property
- `versioned_params` property
- `fit(train_df)`
- `predict(horizon_days)` returning columns:
  - `date`
  - `forecast_gallons`
- `clone()`

Template:

```python
from dataclasses import dataclass
import pandas as pd
from .base import ModelRunner

@dataclass
class MyModelRunner(ModelRunner):
    my_param: int = 10

    @property
    def name(self) -> str:
        return "my_model"

    @property
    def versioned_params(self) -> dict[str, object]:
        return {"model": self.name, "my_param": self.my_param}

    def fit(self, train_df: pd.DataFrame) -> None:
        # train logic
        pass

    def predict(self, horizon_days: int) -> pd.DataFrame:
        # must return date + forecast_gallons
        return pd.DataFrame(...)

    def clone(self) -> "MyModelRunner":
        return MyModelRunner(my_param=self.my_param)
```

### 2) Register it

Update `models/registry.py`:

```python
from .my_model_runner import MyModelRunner

MODEL_REGISTRY = {
    ...
    "my_model": MyModelRunner,
}
```

### 3) Run notebook/CLI again

- Your model automatically becomes selectable via `list_models()`.
- It will run through the same CV + holdout evaluation and MLflow logging.

### 4) Add tests

Add at least:

- fit/predict shape test
- metric sanity test on sample data
- registry creation test

## Artifact contract per run

Each run should produce:

- `datasets/train.parquet`
- `datasets/test.parquet`
- `predictions/cv_predictions.parquet`
- `predictions/test_predictions.parquet`
- `metadata/schema.json`
- `metadata/split_metadata.json`
- `metadata/run_summary.json`
- `model` (MLflow pyfunc)

This ensures reproducibility and deployability.

## Common pitfalls

- Missing output columns from new model:
  - Fix `predict` to return `date` + `forecast_gallons`.
- Model not loading in API:
  - Ensure you promoted a valid version to alias `production`.
- Data merge failures:
  - Verify source files still include `category`, `item`, `price_point_name`.
- Bad metrics due to short history:
  - Increase training window or reduce split aggressiveness.

## Recommended daily operating loop

1. Update data files.
2. Run `01_data_cleaning.ipynb`.
3. Run `04_expanding_window_experiment_v2.ipynb`.
4. Compare leaderboard metrics.
5. Promote best version.
6. Restart/reload API and validate `/ready`, then `/v1/forecast`.
