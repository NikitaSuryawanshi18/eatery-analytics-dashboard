"""Time-series split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeSplit:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame


def _validate_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "gallons"}
    missing = required.difference(set(daily_df.columns))
    if missing:
        raise ValueError(f"daily_df missing required columns: {sorted(missing)}")

    out = daily_df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def build_train_holdout_split(daily_df: pd.DataFrame, holdout_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_days <= 0:
        raise ValueError("holdout_days must be > 0")

    d = _validate_daily(daily_df)
    if len(d) <= holdout_days:
        raise ValueError("Not enough rows for requested holdout_days")

    train = d.iloc[:-holdout_days].reset_index(drop=True)
    holdout = d.iloc[-holdout_days:].reset_index(drop=True)
    return train, holdout


def build_rolling_cv_splits(
    train_df: pd.DataFrame,
    min_train_days: int = 30,
    horizon_days: int = 7,
    step_days: int = 7,
    max_splits: int = 5,
) -> list[TimeSplit]:
    if min_train_days <= 0 or horizon_days <= 0 or step_days <= 0 or max_splits <= 0:
        raise ValueError("Split parameters must be positive")

    d = _validate_daily(train_df)
    splits: list[TimeSplit] = []

    end_idx = min_train_days
    split_idx = 1
    while end_idx + horizon_days <= len(d):
        split_train = d.iloc[:end_idx].reset_index(drop=True)
        split_test = d.iloc[end_idx : end_idx + horizon_days].reset_index(drop=True)
        splits.append(TimeSplit(name=f"cv_{split_idx}", train=split_train, test=split_test))
        split_idx += 1
        end_idx += step_days

    if not splits and len(d) > horizon_days + 1:
        fallback_train = d.iloc[:-horizon_days].reset_index(drop=True)
        fallback_test = d.iloc[-horizon_days:].reset_index(drop=True)
        splits = [TimeSplit(name="cv_1", train=fallback_train, test=fallback_test)]

    return splits[-max_splits:]


def generate_time_splits(
    daily_df: pd.DataFrame,
    holdout_days: int = 14,
    min_train_days: int = 30,
    horizon_days: int = 7,
    step_days: int = 7,
    max_splits: int = 5,
) -> tuple[list[TimeSplit], TimeSplit]:
    train, holdout = build_train_holdout_split(daily_df, holdout_days=holdout_days)
    cv_splits = build_rolling_cv_splits(
        train,
        min_train_days=min_train_days,
        horizon_days=horizon_days,
        step_days=step_days,
        max_splits=max_splits,
    )
    holdout_split = TimeSplit(name="holdout", train=train, test=holdout)
    return cv_splits, holdout_split

