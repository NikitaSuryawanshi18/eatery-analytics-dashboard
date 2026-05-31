from pathlib import Path

import pandas as pd

from milk_dashboard.ingredient_forecasting import (
    CONFIG_COLUMNS,
    FORECAST_LOG_COLUMNS,
    IngredientForecastPaths,
    RETRAINING_LOG_COLUMNS,
    build_ingredient_usage_history_from_sources,
    load_ingredient_config,
    monthly_retrain_all_models,
    run_daily_forecast_pipeline,
)


def test_build_ingredient_usage_history_from_sources(tmp_path: Path):
    sales_path = tmp_path / "sales.csv"
    ingredients_path = tmp_path / "ingredients.xlsx"
    config_path = tmp_path / "config.csv"

    pd.DataFrame(
        [
            {
                "Date": "2026-01-01",
                "Time": "09:00:00",
                "Category": "Coffee",
                "Item": "Latte",
                "Qty": 2,
                "Price Point Name": "Medium",
            },
            {
                "Date": "2026-01-01",
                "Time": "10:00:00",
                "Category": "Coffee",
                "Item": "Drip Coffee",
                "Qty": 1,
                "Price Point Name": "Small",
            },
        ]
    ).to_csv(sales_path, index=False)

    pd.DataFrame(
        [
            {
                "category": "Coffee",
                "item": "Latte",
                "price_point_name": "Medium",
                "Milk (oz)": 8,
                "espresso/Coffee (grams)": 18,
            },
            {
                "category": "Coffee",
                "item": "Drip Coffee",
                "price_point_name": "Small",
                "Milk (oz)": 0,
                "espresso/Coffee (grams)": "11 brewed coffee",
            },
        ]
    ).to_excel(ingredients_path, index=False)

    pd.DataFrame(
        [
            {
                "ingredient_id": "milk",
                "ingredient_name": "Milk",
                "unit": "oz",
                "order_frequency_days": 7,
                "current_stock": "",
                "safety_stock": "",
                "model_name": "moving_average",
                "source_column": "Milk (oz)",
                "active": True,
                "apply_qty_multiplier": True,
            },
            {
                "ingredient_id": "coffee_beans",
                "ingredient_name": "Coffee Beans",
                "unit": "grams",
                "order_frequency_days": 14,
                "current_stock": "",
                "safety_stock": "",
                "model_name": "moving_average",
                "source_column": "espresso/Coffee (grams)",
                "active": True,
                "apply_qty_multiplier": True,
            },
        ]
    ).to_csv(config_path, index=False)

    config_df = load_ingredient_config(config_path, ingredients_path=ingredients_path)
    history = build_ingredient_usage_history_from_sources(
        sales_path=sales_path,
        ingredients_path=ingredients_path,
        config_df=config_df,
    )

    milk_usage = history[history["ingredient_id"] == "milk"]["usage_qty"].sum()
    coffee_usage = history[history["ingredient_id"] == "coffee_beans"]["usage_qty"].sum()
    assert milk_usage == 16
    assert coffee_usage == 47


def test_daily_pipeline_evaluates_and_retrains_multiple_ingredients(tmp_path: Path):
    paths = IngredientForecastPaths(
        sales_path=tmp_path / "unused_sales.csv",
        ingredients_path=tmp_path / "unused_ingredients.xlsx",
        ingredient_config_path=tmp_path / "ingredient_config.csv",
        usage_history_path=tmp_path / "ingredient_usage_history.csv",
        forecast_log_path=tmp_path / "forecast_log.csv",
        retraining_log_path=tmp_path / "retraining_log.csv",
        pipeline_run_log_path=tmp_path / "pipeline_run_log.csv",
        models_dir=tmp_path / "models",
    )

    config_df = pd.DataFrame(
        [
            {
                "ingredient_id": "milk",
                "ingredient_name": "Milk",
                "unit": "oz",
                "order_frequency_days": 7,
                "current_stock": 20,
                "safety_stock": 5,
                "model_name": "moving_average",
                "source_column": "Milk (oz)",
                "active": True,
                "apply_qty_multiplier": True,
            },
            {
                "ingredient_id": "coffee_beans",
                "ingredient_name": "Coffee Beans",
                "unit": "grams",
                "order_frequency_days": 14,
                "current_stock": 100,
                "safety_stock": 20,
                "model_name": "moving_average",
                "source_column": "espresso/Coffee (grams)",
                "active": True,
                "apply_qty_multiplier": True,
            },
        ]
    )
    config_df.to_csv(paths.ingredient_config_path, index=False)

    dates = pd.date_range("2026-01-01", "2026-01-22", freq="D")
    history_rows = []
    for dt in dates:
        history_rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "ingredient_id": "milk",
                "ingredient_name": "Milk",
                "usage_qty": 10.0,
                "unit": "oz",
            }
        )
        history_rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "ingredient_id": "coffee_beans",
                "ingredient_name": "Coffee Beans",
                "usage_qty": 5.0,
                "unit": "grams",
            }
        )
    pd.DataFrame(history_rows).to_csv(paths.usage_history_path, index=False)

    first_run = run_daily_forecast_pipeline(run_date="2026-01-08", paths=paths)
    assert first_run["forecasts_created"] == 2

    forecast_log = pd.read_csv(paths.forecast_log_path)
    assert list(forecast_log.columns) == FORECAST_LOG_COLUMNS
    assert set(forecast_log["status"]) == {"pending_evaluation"}

    second_run = run_daily_forecast_pipeline(run_date="2026-01-23", paths=paths)
    assert second_run["forecasts_created"] == 2

    updated_log = pd.read_csv(paths.forecast_log_path)
    evaluated = updated_log[updated_log["status"] == "evaluated"].copy()
    assert len(evaluated) == 2
    assert set(evaluated["ingredient_id"]) == {"milk", "coffee_beans"}
    assert evaluated["absolute_error"].fillna(0.0).sum() == 0.0

    retrain_result = monthly_retrain_all_models(run_date="2026-02-01", paths=paths)
    assert retrain_result["ingredients_processed"] == 2
    retraining_log = pd.read_csv(paths.retraining_log_path)
    assert list(retraining_log.columns) == RETRAINING_LOG_COLUMNS
    assert set(retraining_log["status"]) == {"ok"}
    assert (paths.models_dir / "milk" / "model_latest.pkl").exists()
    assert (paths.models_dir / "coffee_beans" / "model_latest.pkl").exists()
