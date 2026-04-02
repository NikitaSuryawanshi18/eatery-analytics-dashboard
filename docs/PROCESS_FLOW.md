# Process Flow

## High-level pipeline
1. Raw inputs (`JItters data.csv`, `Ingredient Measure.xlsx`) are cleaned in [01_data_cleaning.ipynb](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/notebooks/01_data_cleaning.ipynb).
2. Cleaned merged output (`artifacts/data/cleaned_merged_with_ingredients.csv`) is consumed by [04_expanding_window_experiment_v2.ipynb](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/notebooks/04_expanding_window_experiment_v2.ipynb).
3. Notebook derives daily `gallons` target and runs expanding-window backtests.
4. Models (`XGBoost`, `XGBoost_Aggressive`, `LinearRegression`, `Prophet`, `ARIMA`, `SARIMA`) are evaluated on identical split windows. Ensemble is currently kept disabled by default.
5. Outputs are written to `artifacts/expanding_backtest_v2/` and logged to MLflow (`mlruns_v2.db` + `notebooks/mlruns/` artifacts).
6. Serving layer (`src/milk_dashboard/api/app.py`) loads promoted model aliases for operational forecast APIs.

## File map by stage
- Data prep:
  - [01_data_cleaning.ipynb](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/notebooks/01_data_cleaning.ipynb)
  - `src/milk_dashboard/data/*`
- Model experiment:
  - [04_expanding_window_experiment_v2.ipynb](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/notebooks/04_expanding_window_experiment_v2.ipynb)
  - `artifacts/expanding_backtest_v2/metrics`
  - `artifacts/expanding_backtest_v2/predictions`
  - `artifacts/expanding_backtest_v2/models`
  - `artifacts/expanding_backtest_v2/summary`
- Serving and UI:
  - [app.py](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/app.py)
  - [api app](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/src/milk_dashboard/api/app.py)
  - [dashboard.js](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/src/milk_dashboard/api/static/dashboard.js)
  - [dashboard.css](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/src/milk_dashboard/api/static/dashboard.css)

## Archived notebooks
Legacy root notebooks are moved to:
- `artifacts/old_notebooks/DataImport_EDA.ipynb`
- `artifacts/old_notebooks/Jitters_Forecast_Dashboard (2).ipynb`
