import pandas as pd

from milk_dashboard.models.registry import create_model, list_models


def _sample_daily(periods: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=periods, freq="D"),
            "gallons": [float((i % 7) + 10) for i in range(periods)],
        }
    )


def test_registry_contains_required_models():
    names = list_models()
    assert "prophet" in names
    assert "seasonal_naive" in names
    assert "moving_average" in names


def test_baseline_runners_fit_and_predict():
    train_df = _sample_daily()
    for model_name in ("seasonal_naive", "moving_average"):
        model = create_model(model_name)
        model.fit(train_df)
        pred = model.predict(7)
        assert len(pred) == 7
        assert {"date", "forecast_gallons"}.issubset(pred.columns)

