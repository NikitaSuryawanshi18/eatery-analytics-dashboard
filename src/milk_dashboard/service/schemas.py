"""Request/response schemas."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    horizon_days: int = Field(default=7, ge=1, le=30)
    lookback_days: int = Field(default=30, ge=1, le=365)
    safety_buffer_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    source_id: str | None = Field(default=None)


class ForecastPoint(BaseModel):
    date: date
    forecast_gallons: float


class ForecastResponse(BaseModel):
    daily_forecast: list[ForecastPoint]
    expected_total_gallons: float
    order_gallons: int
    safety_buffer_pct: float
    model_name: str
    model_version: str | None
    run_id: str | None

