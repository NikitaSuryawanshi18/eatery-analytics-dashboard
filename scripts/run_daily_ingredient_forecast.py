"""Run the daily ingredient forecasting pipeline."""

from __future__ import annotations

import json

from milk_dashboard.ingredient_forecasting import run_daily_forecast_pipeline


if __name__ == "__main__":
    result = run_daily_forecast_pipeline()
    print(json.dumps(result, ensure_ascii=True, indent=2))
