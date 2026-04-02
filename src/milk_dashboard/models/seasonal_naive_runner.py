"""Seasonal-naive baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .base import ModelRunner


@dataclass
class SeasonalNaiveRunner(ModelRunner):
    _weekday_average: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _overall_avg: float = field(default=0.0, init=False, repr=False)
    _last_date: pd.Timestamp | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return "seasonal_naive"

    @property
    def versioned_params(self) -> dict[str, object]:
        return {"model": self.name}

    def fit(self, train_df: pd.DataFrame) -> None:
        d = train_df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date")
        d["weekday"] = d["date"].dt.day_name()

        grouped = d.groupby("weekday")["gallons"].mean()
        self._weekday_average = {k: float(v) for k, v in grouped.items()}
        self._overall_avg = float(d["gallons"].mean())
        self._last_date = d["date"].max()

    def predict(self, horizon_days: int) -> pd.DataFrame:
        if self._last_date is None:
            raise RuntimeError("Model is not fitted")
        if horizon_days <= 0:
            raise ValueError("horizon_days must be > 0")

        future_dates = pd.date_range(self._last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        preds = []
        for dt in future_dates:
            preds.append(float(self._weekday_average.get(dt.day_name(), self._overall_avg)))

        out = pd.DataFrame({"date": future_dates, "forecast_gallons": preds})
        out["forecast_gallons"] = out["forecast_gallons"].clip(lower=0.0)
        return out

    def clone(self) -> "SeasonalNaiveRunner":
        return SeasonalNaiveRunner()

