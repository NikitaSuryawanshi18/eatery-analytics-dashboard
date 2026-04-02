"""Prophet runner implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from prophet import Prophet

from .base import ModelRunner


@dataclass
class ProphetRunner(ModelRunner):
    seasonality_mode: str = "multiplicative"
    weekly_seasonality: bool = True
    yearly_seasonality: bool = False
    daily_seasonality: bool = False
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 1.0
    _model: Prophet | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return "prophet"

    @property
    def versioned_params(self) -> dict[str, object]:
        return {
            "model": self.name,
            "seasonality_mode": self.seasonality_mode,
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_seasonality": self.yearly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
        }

    def fit(self, train_df: pd.DataFrame) -> None:
        d = train_df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date")
        prophet_df = d[["date", "gallons"]].rename(columns={"date": "ds", "gallons": "y"})
        prophet_df = prophet_df.set_index("ds").asfreq("D")
        prophet_df["y"] = prophet_df["y"].interpolate("time").fillna(0.0)
        prophet_df = prophet_df.reset_index()

        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )
        model.fit(prophet_df)
        self._model = model

    def predict(self, horizon_days: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        if horizon_days <= 0:
            raise ValueError("horizon_days must be > 0")

        future = self._model.make_future_dataframe(periods=horizon_days, freq="D")
        forecast = self._model.predict(future).tail(horizon_days)
        out = forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "forecast_gallons"})
        out["date"] = pd.to_datetime(out["date"])
        out["forecast_gallons"] = pd.to_numeric(out["forecast_gallons"], errors="coerce").fillna(0.0)
        out["forecast_gallons"] = out["forecast_gallons"].clip(lower=0.0)
        return out.reset_index(drop=True)

    def clone(self) -> "ProphetRunner":
        return ProphetRunner(
            seasonality_mode=self.seasonality_mode,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )

