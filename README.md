# Milk Dashboard

Notebook-first milk forecasting stack with:
- data cleaning and EDA notebooks,
- expanding-window model lab with MLflow tracking,
- FastAPI + Streamlit UI surfaces for operational forecasting.

## Primary docs
- [Process Flow](docs/PROCESS_FLOW.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [UI Guide](docs/UI_GUIDE.md)
- Legacy architecture reference: `docs/ARCHITECTURE.md`

## Setup
```bash
pip install -e .[dev]
pip install scikit-learn xgboost prophet statsmodels mlflow
```

Optional env vars:
```bash
set MLFLOW_TRACKING_URI=sqlite:///mlruns_v2.db
set MLFLOW_MODEL_NAME=milk_forecast
set MLFLOW_MODEL_ALIAS=production
```

## Notebook workflow
1. Run `notebooks/01_data_cleaning.ipynb` to produce cleaned data artifacts.
2. Run `notebooks/04_expanding_window_experiment_v2.ipynb` for:
   - models: `XGBoost`, `LinearRegression`, `Prophet`, `ARIMA`, `SARIMA`,
   - expanding window: 45-day train, 30-day prediction, 30-day expansion step,
   - parent/child MLflow logging with stepped metric history.
3. Review artifacts in `artifacts/expanding_backtest_v2/`.

Current default:
- `ENABLE_ENSEMBLE = False` in notebook config (ensemble is temporarily disabled).

Smoke run:
```bash
set SMOKE_TEST=1
set RUN_OUTPUT_SUFFIX=smoke
```
Then run all notebook cells.

## MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlruns_v2.db --port 5001
```

Experiment:
- `lookforward_30_30_all_models`

Run layout:
- parent run per model,
- nested child run per split,
- nested `final_expanded` run,
- separate comparison run (`model_comparison_v2`).

## Serving
FastAPI:
```bash
uvicorn milk_dashboard.api.app:app --reload
```

Streamlit:
```bash
streamlit run app.py
```

## Cleanup note
Root legacy notebooks were archived to:
- `artifacts/old_notebooks/DataImport_EDA.ipynb`
- `artifacts/old_notebooks/Jitters_Forecast_Dashboard (2).ipynb`
