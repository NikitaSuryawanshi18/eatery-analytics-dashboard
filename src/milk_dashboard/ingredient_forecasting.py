"""File-based ingredient forecasting pipeline built on existing runners and cleaning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd

from milk_dashboard.data.cleaning import attach_datetime_columns, remove_voided_items, standardize_column_names
from milk_dashboard.models.registry import create_model


MERGE_KEYS = ("category", "item", "price_point_name")
USAGE_HISTORY_COLUMNS = ["date", "ingredient_id", "ingredient_name", "usage_qty", "unit"]
CONFIG_COLUMNS = [
    "ingredient_id",
    "ingredient_name",
    "unit",
    "order_frequency_days",
    "current_stock",
    "safety_stock",
    "model_name",
    "source_column",
    "active",
    "apply_qty_multiplier",
]
FORECAST_LOG_COLUMNS = [
    "forecast_id",
    "forecast_created_date",
    "ingredient_id",
    "ingredient_name",
    "window_start_date",
    "window_end_date",
    "order_frequency_days",
    "predicted_usage",
    "current_stock",
    "safety_stock",
    "suggested_order_qty",
    "model_name",
    "model_version",
    "actual_usage",
    "forecast_error",
    "absolute_error",
    "percentage_error",
    "bias",
    "evaluated_date",
    "status",
]
RETRAINING_LOG_COLUMNS = [
    "run_date",
    "ingredient_id",
    "ingredient_name",
    "model_name",
    "model_version",
    "train_start_date",
    "train_end_date",
    "status",
    "error_message",
]
PIPELINE_RUN_LOG_COLUMNS = [
    "run_date",
    "run_type",
    "status",
    "ingredients_processed",
    "ingredients_failed",
    "error_message",
]


@dataclass
class IngredientForecastPaths:
    sales_path: Path
    ingredients_path: Path
    ingredient_config_path: Path
    usage_history_path: Path
    forecast_log_path: Path
    retraining_log_path: Path
    pipeline_run_log_path: Path
    models_dir: Path


def default_ingredient_forecast_paths(repo_root: Path | None = None) -> IngredientForecastPaths:
    root = repo_root or Path(__file__).resolve().parents[2]
    return IngredientForecastPaths(
        sales_path=root / "JItters data.csv",
        ingredients_path=root / "Ingredient Measure.xlsx",
        ingredient_config_path=root / "artifacts" / "data" / "ingredient_forecast_config.csv",
        usage_history_path=root / "artifacts" / "data" / "ingredient_usage_history.csv",
        forecast_log_path=root / "logs" / "forecast_log.csv",
        retraining_log_path=root / "logs" / "retraining_log.csv",
        pipeline_run_log_path=root / "logs" / "pipeline_run_log.csv",
        models_dir=root / "artifacts" / "ingredient_models",
    )


def _canonicalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _coerce_optional_float(value: Any) -> float | None:
    if pd.isna(value) or value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_numeric_measurement(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    try:
        return float(match.group(0))
    except ValueError:
        return 0.0


def _normalize_run_date(run_date: str | date | datetime | pd.Timestamp | None) -> pd.Timestamp:
    if run_date is None:
        return pd.Timestamp(datetime.now(timezone.utc).date())
    return pd.Timestamp(run_date).normalize()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
    df = pd.read_csv(path)
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[columns].copy()


def _write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    _ensure_parent(path)
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    out = out[columns]
    out.to_csv(path, index=False)


def _append_row(df: pd.DataFrame, row: dict[str, Any], columns: list[str]) -> pd.DataFrame:
    row_df = pd.DataFrame([{column: row.get(column, pd.NA) for column in columns}])
    if df.empty or df.dropna(how="all").empty:
        return row_df[columns].copy()
    return pd.concat([df, row_df[columns]], ignore_index=True)


def initialize_default_ingredient_config(
    *,
    config_path: Path,
    ingredients_path: Path,
) -> Path:
    ingredients_df = pd.read_excel(ingredients_path)
    ingredient_columns = set(ingredients_df.columns.astype(str).tolist())
    starter_rows = [
        {
            "ingredient_id": "milk",
            "ingredient_name": "Milk",
            "unit": "oz",
            "order_frequency_days": 7,
            "current_stock": "",
            "safety_stock": "",
            "model_name": "prophet",
            "source_column": "Milk (oz)",
            "active": True,
            "apply_qty_multiplier": True,
        },
        {
            "ingredient_id": "coffee_beans",
            "ingredient_name": "Coffee Beans",
            "unit": "grams",
            "order_frequency_days": 14,
            "current_stock": "",
            "safety_stock": "",
            "model_name": "moving_average",
            "source_column": "espresso/Coffee (grams)",
            "active": True,
            "apply_qty_multiplier": True,
        },
        {
            "ingredient_id": "tea",
            "ingredient_name": "Tea",
            "unit": "bags",
            "order_frequency_days": 14,
            "current_stock": "",
            "safety_stock": "",
            "model_name": "seasonal_naive",
            "source_column": "Tea",
            "active": False,
            "apply_qty_multiplier": True,
        },
        {
            "ingredient_id": "flavor_syrup",
            "ingredient_name": "Flavor Syrup",
            "unit": "oz",
            "order_frequency_days": 10,
            "current_stock": "",
            "safety_stock": "",
            "model_name": "moving_average",
            "source_column": "flavor oz",
            "active": False,
            "apply_qty_multiplier": True,
        },
    ]
    rows = [row for row in starter_rows if row["source_column"] in ingredient_columns]
    _write_csv(pd.DataFrame(rows), config_path, CONFIG_COLUMNS)
    return config_path


def load_ingredient_config(
    config_path: Path,
    *,
    ingredients_path: Path | None = None,
) -> pd.DataFrame:
    if not config_path.exists():
        if ingredients_path is None or not ingredients_path.exists():
            raise FileNotFoundError(f"Ingredient config not found: {config_path}")
        initialize_default_ingredient_config(config_path=config_path, ingredients_path=ingredients_path)

    df = pd.read_csv(config_path)
    for column in CONFIG_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[CONFIG_COLUMNS].copy()
    df["ingredient_id"] = df["ingredient_id"].fillna("").astype(str).str.strip()
    df["ingredient_name"] = df["ingredient_name"].fillna("").astype(str).str.strip()
    df["unit"] = df["unit"].fillna("").astype(str).str.strip()
    df["model_name"] = df["model_name"].fillna("prophet").astype(str).str.strip().replace("", "prophet")
    df["source_column"] = df["source_column"].fillna("").astype(str).str.strip()
    df["order_frequency_days"] = pd.to_numeric(df["order_frequency_days"], errors="coerce").fillna(0).astype(int)
    df["current_stock"] = df["current_stock"].apply(_coerce_optional_float)
    df["safety_stock"] = df["safety_stock"].apply(_coerce_optional_float)
    df["active"] = df["active"].apply(lambda value: _coerce_bool(value, default=True))
    df["apply_qty_multiplier"] = df["apply_qty_multiplier"].apply(lambda value: _coerce_bool(value, default=True))
    df = df[(df["ingredient_id"] != "") & (df["ingredient_name"] != "") & (df["order_frequency_days"] > 0)].copy()
    if df.empty:
        raise ValueError("Ingredient config has no valid active or inactive ingredient rows.")
    return df.reset_index(drop=True)


def build_ingredient_usage_history_from_sources(
    *,
    sales_path: Path,
    ingredients_path: Path,
    config_df: pd.DataFrame,
) -> pd.DataFrame:
    sales_df = pd.read_csv(sales_path, encoding="ISO-8859-1", low_memory=False)
    sales_df = standardize_column_names(sales_df)
    sales_df = attach_datetime_columns(sales_df)
    sales_df = remove_voided_items(sales_df)
    sales_df = sales_df.dropna(subset=["datetime", "qty", "item"]).copy()
    sales_df["qty"] = pd.to_numeric(sales_df["qty"], errors="coerce").fillna(0.0)

    ingredients_df = pd.read_excel(ingredients_path)
    ingredients_df = standardize_column_names(ingredients_df)
    missing = [key for key in MERGE_KEYS if key not in ingredients_df.columns]
    if missing:
        raise ValueError(f"Ingredient mapping file missing merge keys: {missing}")
    extra_drop = [column for column in ingredients_df.columns if column not in set(MERGE_KEYS) and column == "qty"]
    if extra_drop:
        ingredients_df = ingredients_df.drop(columns=extra_drop, errors="ignore")
    merged = sales_df.merge(ingredients_df, on=list(MERGE_KEYS), how="left")
    if "qty" not in merged.columns:
        qty_candidates = [column for column in ("qty_x", "qty_sale", "sales_qty") if column in merged.columns]
        if not qty_candidates:
            raise ValueError("Could not resolve sales quantity column after merging ingredient mappings.")
        merged["qty"] = pd.to_numeric(merged[qty_candidates[0]], errors="coerce").fillna(0.0)
    merged["date"] = pd.to_datetime(merged["datetime"], errors="coerce").dt.normalize()

    history_frames: list[pd.DataFrame] = []
    for _, ingredient in config_df.iterrows():
        source_column = _canonicalize_column_name(ingredient["source_column"])
        if not source_column or source_column not in merged.columns:
            continue
        usage_values = merged[source_column].map(_parse_numeric_measurement).astype(float)
        if bool(ingredient["apply_qty_multiplier"]):
            usage_values = usage_values * merged["qty"].astype(float)
        ingredient_history = pd.DataFrame(
            {
                "date": merged["date"],
                "ingredient_id": ingredient["ingredient_id"],
                "ingredient_name": ingredient["ingredient_name"],
                "usage_qty": usage_values,
                "unit": ingredient["unit"],
            }
        )
        ingredient_history = ingredient_history.dropna(subset=["date"])
        ingredient_history = ingredient_history[ingredient_history["usage_qty"] > 0].copy()
        if ingredient_history.empty:
            continue
        ingredient_history = (
            ingredient_history.groupby(["date", "ingredient_id", "ingredient_name", "unit"], as_index=False)["usage_qty"]
            .sum()
            .sort_values("date")
            .reset_index(drop=True)
        )
        history_frames.append(ingredient_history)

    if not history_frames:
        return pd.DataFrame(columns=USAGE_HISTORY_COLUMNS)
    history = pd.concat(history_frames, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    history["usage_qty"] = pd.to_numeric(history["usage_qty"], errors="coerce").fillna(0.0)
    return history[USAGE_HISTORY_COLUMNS].sort_values(["ingredient_id", "date"]).reset_index(drop=True)


def load_or_build_ingredient_usage_history(
    *,
    paths: IngredientForecastPaths,
    config_df: pd.DataFrame,
) -> pd.DataFrame:
    if paths.usage_history_path.exists():
        history = pd.read_csv(paths.usage_history_path)
    else:
        history = build_ingredient_usage_history_from_sources(
            sales_path=paths.sales_path,
            ingredients_path=paths.ingredients_path,
            config_df=config_df,
        )
        _write_csv(history, paths.usage_history_path, USAGE_HISTORY_COLUMNS)

    for column in USAGE_HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = pd.NA
    history = history[USAGE_HISTORY_COLUMNS].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    history["ingredient_id"] = history["ingredient_id"].fillna("").astype(str).str.strip()
    history["ingredient_name"] = history["ingredient_name"].fillna("").astype(str).str.strip()
    history["unit"] = history["unit"].fillna("").astype(str).str.strip()
    history["usage_qty"] = pd.to_numeric(history["usage_qty"], errors="coerce").fillna(0.0)
    history = history.dropna(subset=["date"])
    return history.sort_values(["ingredient_id", "date"]).reset_index(drop=True)


def _prepare_training_series(
    usage_history: pd.DataFrame,
    *,
    ingredient_id: str,
    run_date: pd.Timestamp,
) -> pd.DataFrame:
    filtered = usage_history[usage_history["ingredient_id"] == ingredient_id].copy()
    filtered = filtered[filtered["date"] < run_date].copy()
    if filtered.empty:
        raise ValueError(f"No usage history available before {run_date.date()} for ingredient '{ingredient_id}'.")
    daily = filtered.groupby("date", as_index=False)["usage_qty"].sum().sort_values("date")
    full_range = pd.date_range(daily["date"].min(), run_date - pd.Timedelta(days=1), freq="D")
    daily = daily.set_index("date").reindex(full_range, fill_value=0.0).reset_index()
    daily = daily.rename(columns={"index": "date", "usage_qty": "gallons"})
    daily["gallons"] = pd.to_numeric(daily["gallons"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return daily[["date", "gallons"]]


def _model_version_for_date(run_date: pd.Timestamp) -> str:
    return f"v{run_date.strftime('%Y_%m')}"


def _ingredient_model_dir(models_dir: Path, ingredient_id: str) -> Path:
    return models_dir / ingredient_id


def _save_model_artifacts(
    *,
    models_dir: Path,
    ingredient_id: str,
    model: Any,
    metadata: dict[str, Any],
) -> None:
    ingredient_dir = _ingredient_model_dir(models_dir, ingredient_id)
    ingredient_dir.mkdir(parents=True, exist_ok=True)
    version = str(metadata["model_version"])
    versioned_model_path = ingredient_dir / f"model_{version}.pkl"
    latest_model_path = ingredient_dir / "model_latest.pkl"
    versioned_meta_path = ingredient_dir / f"metadata_{version}.json"
    latest_meta_path = ingredient_dir / "metadata_latest.json"
    with versioned_model_path.open("wb") as f:
        pickle.dump(model, f)
    with latest_model_path.open("wb") as f:
        pickle.dump(model, f)
    versioned_meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    latest_meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_latest_model(models_dir: Path, ingredient_id: str) -> tuple[Any, dict[str, Any]] | tuple[None, None]:
    ingredient_dir = _ingredient_model_dir(models_dir, ingredient_id)
    latest_model_path = ingredient_dir / "model_latest.pkl"
    latest_meta_path = ingredient_dir / "metadata_latest.json"
    if not latest_model_path.exists() or not latest_meta_path.exists():
        return None, None
    with latest_model_path.open("rb") as f:
        model = pickle.load(f)
    metadata = json.loads(latest_meta_path.read_text(encoding="utf-8"))
    return model, metadata


def train_and_save_ingredient_model(
    *,
    ingredient_row: pd.Series,
    usage_history: pd.DataFrame,
    run_date: pd.Timestamp,
    models_dir: Path,
) -> dict[str, Any]:
    train_df = _prepare_training_series(
        usage_history,
        ingredient_id=str(ingredient_row["ingredient_id"]),
        run_date=run_date,
    )
    model_name = str(ingredient_row["model_name"] or "prophet")
    model = create_model(model_name)
    model.fit(train_df)
    metadata = {
        "ingredient_id": str(ingredient_row["ingredient_id"]),
        "ingredient_name": str(ingredient_row["ingredient_name"]),
        "model_name": model_name,
        "model_version": _model_version_for_date(run_date),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_start_date": train_df["date"].min().strftime("%Y-%m-%d"),
        "train_end_date": train_df["date"].max().strftime("%Y-%m-%d"),
        "training_rows": int(len(train_df)),
        "unit": str(ingredient_row["unit"]),
    }
    _save_model_artifacts(
        models_dir=models_dir,
        ingredient_id=str(ingredient_row["ingredient_id"]),
        model=model,
        metadata=metadata,
    )
    return metadata


def forecast_ingredient_usage(
    ingredient_id: str,
    usage_history: pd.DataFrame,
    order_frequency_days: int,
    *,
    run_date: str | date | datetime | pd.Timestamp | None = None,
    model: Any | None = None,
    model_name: str = "prophet",
) -> dict[str, Any]:
    forecast_start = _normalize_run_date(run_date)
    if order_frequency_days <= 0:
        raise ValueError("order_frequency_days must be > 0.")
    train_df = _prepare_training_series(
        usage_history,
        ingredient_id=ingredient_id,
        run_date=forecast_start,
    )
    runner = model or create_model(model_name)
    if model is None:
        runner.fit(train_df)
    predictions = runner.predict(order_frequency_days).copy()
    predictions["date"] = pd.date_range(forecast_start, periods=order_frequency_days, freq="D")
    predictions["forecast_gallons"] = pd.to_numeric(predictions["forecast_gallons"], errors="coerce").fillna(0.0)
    predictions["forecast_gallons"] = predictions["forecast_gallons"].clip(lower=0.0)
    predicted_usage = float(predictions["forecast_gallons"].sum())
    return {
        "ingredient_id": ingredient_id,
        "forecast_start_date": forecast_start.strftime("%Y-%m-%d"),
        "forecast_end_date": (forecast_start + pd.Timedelta(days=order_frequency_days - 1)).strftime("%Y-%m-%d"),
        "order_frequency_days": int(order_frequency_days),
        "predicted_usage": predicted_usage,
        "daily_predictions": [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "predicted_usage": float(row["forecast_gallons"]),
            }
            for _, row in predictions.iterrows()
        ],
    }


def _next_forecast_id(forecast_log: pd.DataFrame, ingredient_id: str, run_date: pd.Timestamp) -> str:
    prefix = f"{ingredient_id}_{run_date.strftime('%Y%m%d')}"
    existing = forecast_log["forecast_id"].fillna("").astype(str)
    matches = existing[existing.str.startswith(prefix)]
    return f"{prefix}_{len(matches) + 1:03d}"


def _has_active_pending_forecast(
    forecast_log: pd.DataFrame,
    *,
    ingredient_id: str,
    run_date: pd.Timestamp,
) -> bool:
    if forecast_log.empty:
        return False
    pending = forecast_log[
        (forecast_log["ingredient_id"].astype(str) == ingredient_id)
        & (forecast_log["status"].astype(str) == "pending_evaluation")
    ].copy()
    if pending.empty:
        return False
    pending["window_start_date"] = pd.to_datetime(pending["window_start_date"], errors="coerce")
    pending["window_end_date"] = pd.to_datetime(pending["window_end_date"], errors="coerce")
    active = pending[
        (pending["window_start_date"] <= run_date)
        & (pending["window_end_date"] >= run_date)
    ]
    return not active.empty


def evaluate_completed_forecasts(
    *,
    run_date: str | date | datetime | pd.Timestamp | None = None,
    forecast_log_path: Path,
    usage_history: pd.DataFrame,
) -> pd.DataFrame:
    effective_run_date = _normalize_run_date(run_date)
    forecast_log = _read_csv_or_empty(forecast_log_path, FORECAST_LOG_COLUMNS)
    if forecast_log.empty:
        return forecast_log

    forecast_log["evaluated_date"] = forecast_log["evaluated_date"].astype("object")
    forecast_log["status"] = forecast_log["status"].astype("object")
    forecast_log["window_start_date"] = pd.to_datetime(forecast_log["window_start_date"], errors="coerce")
    forecast_log["window_end_date"] = pd.to_datetime(forecast_log["window_end_date"], errors="coerce")
    forecast_log["predicted_usage"] = pd.to_numeric(forecast_log["predicted_usage"], errors="coerce")

    for idx, row in forecast_log.iterrows():
        if str(row["status"]) != "pending_evaluation":
            continue
        window_end = pd.to_datetime(row["window_end_date"], errors="coerce")
        if pd.isna(window_end) or effective_run_date <= window_end:
            continue
        ingredient_id = str(row["ingredient_id"])
        start_date = pd.to_datetime(row["window_start_date"], errors="coerce")
        actual_usage = usage_history[
            (usage_history["ingredient_id"] == ingredient_id)
            & (usage_history["date"] >= start_date)
            & (usage_history["date"] <= window_end)
        ]["usage_qty"].sum()
        predicted = float(row["predicted_usage"] or 0.0)
        error = predicted - float(actual_usage)
        abs_error = abs(error)
        pct_error = None if float(actual_usage) == 0.0 else abs_error / float(actual_usage)
        forecast_log.at[idx, "actual_usage"] = float(actual_usage)
        forecast_log.at[idx, "forecast_error"] = float(error)
        forecast_log.at[idx, "absolute_error"] = float(abs_error)
        forecast_log.at[idx, "percentage_error"] = pct_error
        forecast_log.at[idx, "bias"] = float(error)
        forecast_log.at[idx, "evaluated_date"] = effective_run_date.strftime("%Y-%m-%d")
        forecast_log.at[idx, "status"] = "evaluated"

    _write_csv(forecast_log, forecast_log_path, FORECAST_LOG_COLUMNS)
    return forecast_log


def monthly_retrain_all_models(
    *,
    run_date: str | date | datetime | pd.Timestamp | None = None,
    paths: IngredientForecastPaths | None = None,
) -> dict[str, Any]:
    effective_run_date = _normalize_run_date(run_date)
    resolved_paths = paths or default_ingredient_forecast_paths()
    config_df = load_ingredient_config(
        resolved_paths.ingredient_config_path,
        ingredients_path=resolved_paths.ingredients_path,
    )
    usage_history = load_or_build_ingredient_usage_history(paths=resolved_paths, config_df=config_df)
    retraining_log = _read_csv_or_empty(resolved_paths.retraining_log_path, RETRAINING_LOG_COLUMNS)

    processed = 0
    failed = 0
    for _, ingredient_row in config_df[config_df["active"]].iterrows():
        processed += 1
        try:
            metadata = train_and_save_ingredient_model(
                ingredient_row=ingredient_row,
                usage_history=usage_history,
                run_date=effective_run_date,
                models_dir=resolved_paths.models_dir,
            )
            retraining_log = _append_row(retraining_log, {
                "run_date": effective_run_date.strftime("%Y-%m-%d"),
                "ingredient_id": ingredient_row["ingredient_id"],
                "ingredient_name": ingredient_row["ingredient_name"],
                "model_name": metadata["model_name"],
                "model_version": metadata["model_version"],
                "train_start_date": metadata["train_start_date"],
                "train_end_date": metadata["train_end_date"],
                "status": "ok",
                "error_message": "",
            }, RETRAINING_LOG_COLUMNS)
        except Exception as exc:
            failed += 1
            retraining_log = _append_row(retraining_log, {
                "run_date": effective_run_date.strftime("%Y-%m-%d"),
                "ingredient_id": ingredient_row["ingredient_id"],
                "ingredient_name": ingredient_row["ingredient_name"],
                "model_name": ingredient_row["model_name"],
                "model_version": _model_version_for_date(effective_run_date),
                "train_start_date": "",
                "train_end_date": "",
                "status": "failed",
                "error_message": str(exc),
            }, RETRAINING_LOG_COLUMNS)

    _write_csv(retraining_log, resolved_paths.retraining_log_path, RETRAINING_LOG_COLUMNS)
    pipeline_log = _read_csv_or_empty(resolved_paths.pipeline_run_log_path, PIPELINE_RUN_LOG_COLUMNS)
    pipeline_log = _append_row(pipeline_log, {
        "run_date": effective_run_date.strftime("%Y-%m-%d"),
        "run_type": "monthly_retrain",
        "status": "ok" if failed == 0 else "partial_failure",
        "ingredients_processed": processed,
        "ingredients_failed": failed,
        "error_message": "",
    }, PIPELINE_RUN_LOG_COLUMNS)
    _write_csv(pipeline_log, resolved_paths.pipeline_run_log_path, PIPELINE_RUN_LOG_COLUMNS)
    return {
        "run_date": effective_run_date.strftime("%Y-%m-%d"),
        "status": "ok" if failed == 0 else "partial_failure",
        "ingredients_processed": processed,
        "ingredients_failed": failed,
    }


def run_daily_forecast_pipeline(
    *,
    run_date: str | date | datetime | pd.Timestamp | None = None,
    paths: IngredientForecastPaths | None = None,
) -> dict[str, Any]:
    effective_run_date = _normalize_run_date(run_date)
    resolved_paths = paths or default_ingredient_forecast_paths()
    config_df = load_ingredient_config(
        resolved_paths.ingredient_config_path,
        ingredients_path=resolved_paths.ingredients_path,
    )
    usage_history = load_or_build_ingredient_usage_history(paths=resolved_paths, config_df=config_df)
    forecast_log = evaluate_completed_forecasts(
        run_date=effective_run_date,
        forecast_log_path=resolved_paths.forecast_log_path,
        usage_history=usage_history,
    )
    pipeline_log = _read_csv_or_empty(resolved_paths.pipeline_run_log_path, PIPELINE_RUN_LOG_COLUMNS)

    created = 0
    failed = 0
    active_df = config_df[config_df["active"]].copy()
    for _, ingredient_row in active_df.iterrows():
        ingredient_id = str(ingredient_row["ingredient_id"])
        if _has_active_pending_forecast(forecast_log, ingredient_id=ingredient_id, run_date=effective_run_date):
            continue
        try:
            model, metadata = _load_latest_model(resolved_paths.models_dir, ingredient_id)
            if model is None or metadata is None:
                metadata = train_and_save_ingredient_model(
                    ingredient_row=ingredient_row,
                    usage_history=usage_history,
                    run_date=effective_run_date,
                    models_dir=resolved_paths.models_dir,
                )
                model, metadata = _load_latest_model(resolved_paths.models_dir, ingredient_id)
            if model is None or metadata is None:
                raise RuntimeError(f"Could not load a latest model for ingredient '{ingredient_id}'.")

            forecast = forecast_ingredient_usage(
                ingredient_id=ingredient_id,
                usage_history=usage_history,
                order_frequency_days=int(ingredient_row["order_frequency_days"]),
                run_date=effective_run_date,
                model=model,
                model_name=str(metadata["model_name"]),
            )
            current_stock = _coerce_optional_float(ingredient_row["current_stock"])
            safety_stock = _coerce_optional_float(ingredient_row["safety_stock"])
            suggested_order_qty = None
            if current_stock is not None and safety_stock is not None:
                suggested_order_qty = max(forecast["predicted_usage"] + safety_stock - current_stock, 0.0)
            forecast_log = _append_row(forecast_log, {
                "forecast_id": _next_forecast_id(forecast_log, ingredient_id, effective_run_date),
                "forecast_created_date": effective_run_date.strftime("%Y-%m-%d"),
                "ingredient_id": ingredient_id,
                "ingredient_name": ingredient_row["ingredient_name"],
                "window_start_date": forecast["forecast_start_date"],
                "window_end_date": forecast["forecast_end_date"],
                "order_frequency_days": int(ingredient_row["order_frequency_days"]),
                "predicted_usage": float(forecast["predicted_usage"]),
                "current_stock": current_stock,
                "safety_stock": safety_stock,
                "suggested_order_qty": None if suggested_order_qty is None else float(suggested_order_qty),
                "model_name": metadata["model_name"],
                "model_version": metadata["model_version"],
                "actual_usage": pd.NA,
                "forecast_error": pd.NA,
                "absolute_error": pd.NA,
                "percentage_error": pd.NA,
                "bias": pd.NA,
                "evaluated_date": pd.NA,
                "status": "pending_evaluation",
            }, FORECAST_LOG_COLUMNS)
            created += 1
        except Exception as exc:
            failed += 1
            pipeline_log = _append_row(pipeline_log, {
                "run_date": effective_run_date.strftime("%Y-%m-%d"),
                "run_type": f"daily_forecast:{ingredient_id}",
                "status": "failed",
                "ingredients_processed": 0,
                "ingredients_failed": 1,
                "error_message": str(exc),
            }, PIPELINE_RUN_LOG_COLUMNS)

    _write_csv(forecast_log, resolved_paths.forecast_log_path, FORECAST_LOG_COLUMNS)
    pipeline_log = _append_row(pipeline_log, {
        "run_date": effective_run_date.strftime("%Y-%m-%d"),
        "run_type": "daily_forecast",
        "status": "ok" if failed == 0 else "partial_failure",
        "ingredients_processed": created,
        "ingredients_failed": failed,
        "error_message": "",
    }, PIPELINE_RUN_LOG_COLUMNS)
    _write_csv(pipeline_log, resolved_paths.pipeline_run_log_path, PIPELINE_RUN_LOG_COLUMNS)
    return {
        "run_date": effective_run_date.strftime("%Y-%m-%d"),
        "status": "ok" if failed == 0 else "partial_failure",
        "forecasts_created": created,
        "ingredients_failed": failed,
        "usage_history_path": str(resolved_paths.usage_history_path),
        "forecast_log_path": str(resolved_paths.forecast_log_path),
    }
