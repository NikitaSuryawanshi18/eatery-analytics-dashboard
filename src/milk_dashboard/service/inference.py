"""Forecast inference service."""

from __future__ import annotations

import math

import pandas as pd

from .model_loader import load_model_from_registry
from .schemas import ForecastPoint, ForecastRequest, ForecastResponse


class ForecastService:
    def __init__(self, pyfunc_model: object, model_name: str, model_version: str | None, run_id: str | None) -> None:
        self._model = pyfunc_model
        self._model_name = model_name
        self._model_version = model_version
        self._run_id = run_id

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        model_input = pd.DataFrame({"horizon_days": [request.horizon_days]})
        raw = self._model.predict(model_input)
        df = pd.DataFrame(raw).copy()

        if "date" not in df.columns or "forecast_gallons" not in df.columns:
            raise ValueError("Production model output must include date and forecast_gallons")

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["forecast_gallons"] = pd.to_numeric(df["forecast_gallons"], errors="coerce").fillna(0.0).clip(lower=0.0)

        expected_total = float(df["forecast_gallons"].sum())
        expected_with_buffer = expected_total * (1.0 + request.safety_buffer_pct / 100.0)
        order_gallons = int(math.ceil(expected_with_buffer))

        daily = [
            ForecastPoint(date=row["date"], forecast_gallons=float(row["forecast_gallons"]))
            for _, row in df.iterrows()
        ]
        return ForecastResponse(
            daily_forecast=daily,
            expected_total_gallons=expected_total,
            order_gallons=order_gallons,
            safety_buffer_pct=request.safety_buffer_pct,
            model_name=self._model_name,
            model_version=self._model_version,
            run_id=self._run_id,
        )


def build_forecast_service(
    *,
    tracking_uri: str | None,
    model_name: str,
    model_alias: str,
) -> ForecastService:
    loaded = load_model_from_registry(
        tracking_uri=tracking_uri,
        model_name=model_name,
        model_alias=model_alias,
    )
    return ForecastService(
        pyfunc_model=loaded.pyfunc_model,
        model_name=loaded.model_name,
        model_version=loaded.model_version,
        run_id=loaded.run_id,
    )

