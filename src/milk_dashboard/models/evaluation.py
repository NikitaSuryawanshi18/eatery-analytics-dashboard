"""Model evaluation against rolling CV and holdout splits."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from milk_dashboard.data.splits import TimeSplit

from .base import ModelRunner
from .metrics import calculate_metrics


@dataclass
class SplitEvaluation:
    split_name: str
    metrics: dict[str, float]
    predictions: pd.DataFrame


@dataclass
class EvaluationBundle:
    model_name: str
    cv_results: list[SplitEvaluation]
    holdout_result: SplitEvaluation

    def metric_summary(self) -> dict[str, float]:
        summary: dict[str, float] = {}
        if self.cv_results:
            for key in ("mae", "rmse", "mape"):
                vals = [r.metrics[key] for r in self.cv_results]
                summary[f"cv_{key}_mean"] = float(sum(vals) / len(vals))
        for key, value in self.holdout_result.metrics.items():
            summary[f"holdout_{key}"] = float(value)
        return summary

    def cv_predictions_table(self) -> pd.DataFrame:
        if not self.cv_results:
            return pd.DataFrame(columns=["split_name", "date", "actual_gallons", "forecast_gallons"])
        frames = []
        for result in self.cv_results:
            frame = result.predictions.copy()
            frame["split_name"] = result.split_name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def _evaluate_single_split(model_template: ModelRunner, split: TimeSplit) -> SplitEvaluation:
    model = model_template.clone()
    model.fit(split.train)

    preds = model.predict(horizon_days=len(split.test)).copy()
    preds["date"] = pd.to_datetime(preds["date"])

    actual = split.test[["date", "gallons"]].copy()
    actual["date"] = pd.to_datetime(actual["date"])
    actual = actual.rename(columns={"gallons": "actual_gallons"})

    joined = actual.merge(preds, on="date", how="left")
    joined["forecast_gallons"] = pd.to_numeric(joined["forecast_gallons"], errors="coerce")
    joined["forecast_gallons"] = joined["forecast_gallons"].fillna(0.0).clip(lower=0.0)

    metrics = calculate_metrics(
        actual=joined["actual_gallons"].to_numpy(),
        predicted=joined["forecast_gallons"].to_numpy(),
    )
    return SplitEvaluation(split_name=split.name, metrics=metrics, predictions=joined)


def evaluate_model(
    model_template: ModelRunner,
    cv_splits: list[TimeSplit],
    holdout_split: TimeSplit,
) -> EvaluationBundle:
    cv_results = [_evaluate_single_split(model_template, split) for split in cv_splits]
    holdout_result = _evaluate_single_split(model_template, holdout_split)
    return EvaluationBundle(model_name=model_template.name, cv_results=cv_results, holdout_result=holdout_result)

