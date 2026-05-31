"""Run the monthly ingredient model retraining pipeline."""

from __future__ import annotations

import json

from milk_dashboard.ingredient_forecasting import monthly_retrain_all_models


if __name__ == "__main__":
    result = monthly_retrain_all_models()
    print(json.dumps(result, ensure_ascii=True, indent=2))
