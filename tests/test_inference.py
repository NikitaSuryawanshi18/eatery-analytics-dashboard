import pandas as pd

from milk_dashboard.service.inference import ForecastService
from milk_dashboard.service.schemas import ForecastRequest


class _DummyModel:
    def predict(self, model_input):
        horizon = int(model_input["horizon_days"].iloc[0])
        dates = pd.date_range("2026-02-01", periods=horizon, freq="D")
        return pd.DataFrame({"date": dates, "forecast_gallons": [10.0] * horizon})


def test_forecast_service_response_shape():
    svc = ForecastService(
        pyfunc_model=_DummyModel(),
        model_name="dummy",
        model_version="1",
        run_id="abc123",
    )
    request = ForecastRequest(horizon_days=3, lookback_days=30, safety_buffer_pct=10.0)
    response = svc.forecast(request)

    assert len(response.daily_forecast) == 3
    assert response.expected_total_gallons == 30.0
    assert response.order_gallons == 33
    assert response.model_name == "dummy"

