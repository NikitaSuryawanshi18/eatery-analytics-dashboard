"""Moving-average baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .base import ModelRunner


@dataclass
class MovingAverageRunner(ModelRunner):
    window_size: int = 7
    _history: list[float] = field(default_factory=list, init=False, repr=False)
    _last_date: pd.Timestamp | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return "moving_average"

    @property
    def versioned_params(self) -> dict[str, object]:
        return {"model": self.name, "window_size": self.window_size}

    def fit(self, train_df: pd.DataFrame) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        d = train_df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date")

        self._history = [float(v) for v in d["gallons"].tolist()]
        self._last_date = d["date"].max()

    def predict(self, horizon_days: int) -> pd.DataFrame:
        if not self._history or self._last_date is None:
            raise RuntimeError("Model is not fitted")
        if horizon_days <= 0:
            raise ValueError("horizon_days must be > 0")

        extended = self._history.copy()
        preds: list[float] = []
        for _ in range(horizon_days):
            window = extended[-self.window_size :]
            pred = float(sum(window) / len(window))
            preds.append(max(pred, 0.0))
            extended.append(pred)

        future_dates = pd.date_range(self._last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        return pd.DataFrame({"date": future_dates, "forecast_gallons": preds})

    def clone(self) -> "MovingAverageRunner":
        return MovingAverageRunner(window_size=self.window_size)

