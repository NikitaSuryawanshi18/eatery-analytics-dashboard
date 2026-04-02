# DataImport_EDA Analysis and Production Mapping

This document captures the current scratchpad notebook behavior (`DataImport_EDA.ipynb`) and how it maps into production modules without breaking logic.

## What DataImport_EDA currently does

## 1) Cleaning and normalization

From notebook cells around 7/22/28/32:

- Standardizes sales column names to snake_case.
- Parses money fields (`gross_sales`, `discounts`, `net_sales`, `tax`) to numeric.
- Converts `qty` to numeric.
- Builds `datetime` from `date + time`.
- Drops rows with missing `item` or `qty`.
- Removes `item` values containing `(voided)`.
- Drops many non-essential sales columns.
- Loads ingredient file and keeps:
  - `category`
  - `item`
  - `price_point_name`
  - `Milk (oz)`
- Merges sales + ingredient mapping on those 3 keys.

## 2) Daily milk aggregation logic

Important behavior in `DataImport_EDA`:

- Daily aggregation sums mapped `Milk (oz)` per row.
- It does **not** multiply by `qty` in the default scratchpad flow.
- Daily gallons = `total_milk_oz / 128`.

## 3) Time series and Prophet usage

From later cells:

- Builds daily series for forecasting.
- Uses interpolation when converting to dense daily frequency for model work.
- Includes Prophet tuning example with:
  - grid over `changepoint_prior_scale`, `seasonality_prior_scale`
  - Prophet cross-validation + metrics
- Includes sample comparative evaluation cells (ARIMA/SARIMA/Prophet).

## Production mapping (implemented)

## Module mapping

- Scratchpad cleaning logic -> `src/milk_dashboard/data/cleaning.py`
- Split logic -> `src/milk_dashboard/data/splits.py`
- Prophet + baseline models -> `src/milk_dashboard/models/*`
- Experiment logging -> `src/milk_dashboard/mlflow/tracking.py`
- End-user serving -> `src/milk_dashboard/service/*` + `src/milk_dashboard/api/app.py`
- Notebook orchestration -> `src/milk_dashboard/pipeline.py`

## Behavior parity decisions

To keep scratchpad behavior intact by default:

- `prepare_daily_data(..., apply_qty_multiplier=False)` is default.
- `prepare_daily_data(..., fill_missing_days=False)` is default.
- Non-essential columns are dropped by default.

This means baseline output now follows DataImport_EDA assumptions unless you explicitly opt into stricter production behavior.

## Production hardening switches (opt-in)

You can opt into hardened behavior without changing default logic:

- `apply_qty_multiplier=True`
  - treats ingredient milk mapping as per-unit and multiplies by `qty`.
- `fill_missing_days=True`
  - creates dense daily series with missing dates filled to zero.

These are available in:

- `prepare_from_source(...)` in `pipeline.py`
- CLI flags:
  - `--apply-qty-multiplier`
  - `--fill-missing-days`
- notebook config cells in:
  - `notebooks/01_data_cleaning.ipynb`
  - `notebooks/02_model_lab.ipynb`

## Notebook reliability fix

Issue reported:

- notebook code cells had literal `\\n` in source strings, so execution failed.

Fix applied:

- both production notebooks were regenerated with valid line-based source arrays.
- verified no code cells contain literal `\\n` sequences now.

## Invariants to keep stable going forward

- Keep join keys unchanged:
  - `category`, `item`, `price_point_name`
- Keep output columns for service compatibility:
  - model forecast output must include `date`, `forecast_gallons`
- Keep MLflow run artifacts complete:
  - train/test datasets
  - predictions
  - run metadata
  - packaged pyfunc model
