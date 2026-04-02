"""High-level orchestration helpers used by notebooks and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from milk_dashboard.constants import OZ_PER_GALLON
from milk_dashboard.data.cleaning import build_schema_metadata, prepare_daily_data
from milk_dashboard.data.interfaces import DataSource
from milk_dashboard.data.splits import TimeSplit, generate_time_splits
from milk_dashboard.mlflow.tracking import LoggedRunInfo, log_experiment_run
from milk_dashboard.models.evaluation import EvaluationBundle, evaluate_model
from milk_dashboard.models.registry import create_model


@dataclass
class PreparedDataset:
    daily_df: pd.DataFrame
    schema: dict[str, dict[str, str]]
    source_id: str


def _read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format for cleaned file: {path}")


def build_daily_from_cleaned_dataframe(
    cleaned_df: pd.DataFrame,
    *,
    date_col: str = "date",
    datetime_col: str = "datetime",
    milk_total_col: str = "milk_oz_total",
    milk_col: str = "milk_oz",
    total_col: str = "total_milk_oz",
) -> pd.DataFrame:
    df = cleaned_df.copy()
    columns = set(df.columns)

    if {"date", "total_milk_oz", "gallons"}.issubset(columns):
        out = df[["date", "total_milk_oz", "gallons"]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["total_milk_oz"] = pd.to_numeric(out["total_milk_oz"], errors="coerce").fillna(0.0)
        out = out.dropna(subset=["date"])
        out = (
            out.groupby(out["date"].dt.date)["total_milk_oz"]
            .sum()
            .reset_index()
            .rename(columns={"date": "date"})
        )
        out["date"] = pd.to_datetime(out["date"])
        out["gallons"] = out["total_milk_oz"] / OZ_PER_GALLON
        return out.sort_values("date").reset_index(drop=True)

    if datetime_col in columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        value_col = milk_total_col if milk_total_col in columns else (milk_col if milk_col in columns else None)
        if value_col is None:
            raise ValueError(
                f"Cleaned input requires '{milk_total_col}' or '{milk_col}' when using '{datetime_col}'."
            )
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
        grouped = (
            df.dropna(subset=[datetime_col])
            .groupby(df[datetime_col].dt.date)[value_col]
            .sum()
            .reset_index()
            .rename(columns={datetime_col: "date", value_col: "total_milk_oz"})
        )
    elif date_col in columns:
        value_col = None
        for candidate in (total_col, milk_total_col, milk_col):
            if candidate in columns:
                value_col = candidate
                break
        if value_col is None:
            raise ValueError(
                f"Cleaned input requires one of: '{total_col}', '{milk_total_col}', '{milk_col}'."
            )
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
        grouped = (
            df.dropna(subset=[date_col])
            .groupby(df[date_col].dt.date)[value_col]
            .sum()
            .reset_index()
            .rename(columns={date_col: "date", value_col: "total_milk_oz"})
        )
    else:
        raise ValueError(
            "Cleaned input must include either 'datetime' with milk columns or "
            "'date' with total milk columns."
        )

    grouped["date"] = pd.to_datetime(grouped["date"])
    grouped["total_milk_oz"] = pd.to_numeric(grouped["total_milk_oz"], errors="coerce").fillna(0.0).clip(lower=0.0)
    grouped["gallons"] = grouped["total_milk_oz"] / OZ_PER_GALLON
    return grouped.sort_values("date").reset_index(drop=True)


def prepare_from_cleaned_file(
    cleaned_file_path: str,
    *,
    source_id: str = "cleaned_input",
) -> PreparedDataset:
    path = Path(cleaned_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned input file not found: {path}")

    cleaned_df = _read_tabular(path)
    daily = build_daily_from_cleaned_dataframe(cleaned_df)
    schema = build_schema_metadata(daily)
    return PreparedDataset(daily_df=daily, schema=schema, source_id=source_id)


def prepare_from_source(
    source: DataSource,
    fill_missing_days: bool = False,
    apply_qty_multiplier: bool = False,
) -> PreparedDataset:
    sales = source.load_sales()
    ingredients = source.load_ingredients()
    daily = prepare_daily_data(
        sales,
        ingredients,
        fill_missing_days=fill_missing_days,
        apply_qty_multiplier=apply_qty_multiplier,
    )
    schema = build_schema_metadata(daily)
    return PreparedDataset(daily_df=daily, schema=schema, source_id=source.source_id)


def evaluate_registered_model(
    daily_df: pd.DataFrame,
    model_name: str,
    model_kwargs: dict[str, Any] | None = None,
    split_kwargs: dict[str, Any] | None = None,
) -> tuple[EvaluationBundle, TimeSplit, list[TimeSplit]]:
    model_kwargs = model_kwargs or {}
    split_kwargs = split_kwargs or {}

    cv_splits, holdout_split = generate_time_splits(daily_df, **split_kwargs)
    model_template = create_model(model_name, **model_kwargs)
    evaluation = evaluate_model(model_template=model_template, cv_splits=cv_splits, holdout_split=holdout_split)
    return evaluation, holdout_split, cv_splits


def fit_final_model(
    daily_df: pd.DataFrame,
    model_name: str,
    model_kwargs: dict[str, Any] | None = None,
):
    model_kwargs = model_kwargs or {}
    model = create_model(model_name, **model_kwargs)
    model.fit(daily_df)
    return model


def evaluate_fit_and_log(
    *,
    daily_df: pd.DataFrame,
    model_name: str,
    source_config: dict[str, Any],
    preprocess_config: dict[str, Any],
    split_config: dict[str, Any],
    model_kwargs: dict[str, Any] | None = None,
    experiment_name: str = "milk_models",
    registered_model_name: str | None = "milk_forecast",
    tracking_uri: str | None = None,
) -> tuple[EvaluationBundle, LoggedRunInfo]:
    evaluation, holdout_split, _ = evaluate_registered_model(
        daily_df=daily_df,
        model_name=model_name,
        model_kwargs=model_kwargs,
        split_kwargs=split_config,
    )
    fitted_model = fit_final_model(daily_df=daily_df, model_name=model_name, model_kwargs=model_kwargs)
    run_info = log_experiment_run(
        fitted_model=fitted_model,
        evaluation=evaluation,
        train_df=holdout_split.train,
        test_df=holdout_split.test,
        source_config=source_config,
        preprocess_config=preprocess_config,
        split_config=split_config,
        experiment_name=experiment_name,
        registered_model_name=registered_model_name,
        tracking_uri=tracking_uri,
    )
    return evaluation, run_info


def evaluate_fit_and_log_from_cleaned_file(
    *,
    cleaned_file_path: str,
    model_name: str,
    split_config: dict[str, Any],
    source_id: str = "cleaned_input",
    model_kwargs: dict[str, Any] | None = None,
    experiment_name: str = "milk_models",
    registered_model_name: str | None = "milk_forecast",
    tracking_uri: str | None = None,
) -> tuple[EvaluationBundle, LoggedRunInfo]:
    prepared = prepare_from_cleaned_file(cleaned_file_path=cleaned_file_path, source_id=source_id)
    source_config = {
        "source_id": source_id,
        "input_file": cleaned_file_path,
        "source_mode": "cleaned_input",
    }
    preprocess_config = {"source_mode": "cleaned_input"}
    return evaluate_fit_and_log(
        daily_df=prepared.daily_df,
        model_name=model_name,
        source_config=source_config,
        preprocess_config=preprocess_config,
        split_config=split_config,
        model_kwargs=model_kwargs,
        experiment_name=experiment_name,
        registered_model_name=registered_model_name,
        tracking_uri=tracking_uri,
    )
