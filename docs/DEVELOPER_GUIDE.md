# Developer Guide

## Repository layout
- `notebooks/`:
  - `01_data_cleaning.ipynb`: cleaning and aggregation.
  - `02_model_lab.ipynb`: older model lab.
  - `03_expanding_window_model_lab.ipynb`: previous expanding-window version.
  - `04_expanding_window_experiment_v2.ipynb`: current inspectable experiment baseline.
- `src/milk_dashboard/`:
  - `data/`: cleaning/splits/data source adapters.
  - `models/`: model interfaces and reusable runners used by src pipeline.
  - `mlflow/`: MLflow helper and pyfunc wrapper.
  - `service/`: model loading + inference schemas.
  - `api/`: FastAPI app and static dashboard assets.
- `artifacts/`:
  - `data/`: cleaned outputs.
  - `expanding_backtest_v2/`: notebook v2 outputs.
  - `old_notebooks/`: archived root notebooks.

## Dependencies
Base install:
```bash
pip install -e .[dev]
```

Notebook v2 runtime extras:
```bash
pip install scikit-learn xgboost prophet statsmodels mlflow
```

## Runbook
1. Run cleaning notebook.
2. Open [04_expanding_window_experiment_v2.ipynb](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/notebooks/04_expanding_window_experiment_v2.ipynb).
3. Configure only the top config block.
4. Run all cells.
5. Inspect:
   - `artifacts/expanding_backtest_v2/summary/model_comparison_summary.csv`
   - MLflow experiment `milk_models_expanding_v2`.

Smoke test setup:
```bash
set SMOKE_TEST=1
set RUN_OUTPUT_SUFFIX=smoke
```

## Data flow in notebook v2
1. Load `artifacts/data/cleaned_merged_with_ingredients.csv`.
2. Resolve date and target columns.
3. Build daily `gallons` series (`gallons` direct, else `total_milk_oz / 128`).
4. Generate expanding splits (`45 train / 30 predict / 30 step`).
5. Fit and score each model on the same split timeline.
6. Build ensemble with inverse historical RMSE weights.
7. Persist per-model dumps and global summaries.
8. Log parent + child + comparison runs to MLflow.

## Models in notebook v2
- `XGBoost`
- `LinearRegression`
- `Prophet`
- `ARIMA` (`order=(2,1,2)`)
- `Ensemble_InverseRMSE` (optional, currently disabled by default via `ENABLE_ENSEMBLE=False`)

## Artifact contracts
Per model:
- `metrics/{model}_window_metrics.csv`
- `predictions/{model}_predictions_all_splits.csv`
- `models/{model}/final_model.*`
- `models/{model}/model_config.json`
- `models/{model}/feature_schema.json`

Global:
- `metrics/all_models_metrics.csv`
- `predictions/all_models_predictions.csv`
- `summary/model_comparison_summary.csv`
- `summary/latest_vs_avg_rmse.csv`
- `summary/all_models_comparison.png`
- `summary/Ensemble_InverseRMSE_weights.csv`

## MLflow metrics glossary
- `test_rmse`, `test_mae`, `test_mape`: test-window errors.
- `test_accuracy`: `100 - test_mape` (regression proxy).
- `test_bias`: `mean(prediction - actual)`.
- `test_error_pct`: `(rmse / mean(actual)) * 100`.
- `test_error_std`: standard deviation of residuals.
- `train_*`: same metric family on train-side evaluation.
- `generalization_gap_rmse`: `test_rmse - train_rmse`.
- `avg_*`: parent aggregate over all window runs.
- `latest_*`: parent metric from most recent window run.
- `rmse_trend_slope`, `rmse_trend_delta`: degradation/improvement trend across expanding windows.

## Troubleshooting
- Missing package error: install reported dependency and rerun.
- No splits generated: reduce `PREDICTION_WINDOW_DAYS` or verify date coverage.
- Empty/NaN metrics for a model: inspect model-specific child run and split artifact residuals.
- MLflow schema/DB conflicts: use dedicated experiment names and avoid deleting DB tables manually.
