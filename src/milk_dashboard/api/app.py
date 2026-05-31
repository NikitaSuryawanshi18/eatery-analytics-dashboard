"""FastAPI API + UI app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
from urllib.parse import urlencode
from typing import Any, Literal

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from prophet import Prophet
from pydantic import BaseModel, Field
import requests

from milk_dashboard.auth import AuthStore, UserRecord
from milk_dashboard.constants import DEFAULT_MODEL_ALIAS, DEFAULT_MODEL_NAME
from milk_dashboard.data.cleaning import (
    aggregate_daily_milk_usage,
    attach_datetime_columns,
    drop_non_essential_columns,
    merge_sales_and_ingredients,
    parse_money_columns,
    remove_voided_items,
    standardize_column_names,
)
from milk_dashboard.service import (
    ForecastRequest as InferenceForecastRequest,
    ForecastResponse as InferenceForecastResponse,
    build_forecast_service,
)
from milk_dashboard.models.registry import create_model, list_models
from milk_dashboard.ops.eod_sync import (
    EODSyncConfig,
    load_latest_sync_run,
    run_eod_sync,
)


WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[2]
STATIC_DIR = APP_DIR / "static"
RUNTIME_DIR = REPO_ROOT / "artifacts" / "runtime"
LOGGER = logging.getLogger(__name__)
SESSION_COOKIE_NAME = "milk_dashboard_session"

app = FastAPI(title="Milk Forecast API", version="0.2.0")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@dataclass
class DashboardData:
    sales_clean: pd.DataFrame
    merged: pd.DataFrame
    daily: pd.DataFrame
    ingredient_edit: pd.DataFrame


@dataclass
class StartupModelChoice:
    model_key: str
    source: str
    metric_name: str | None
    metric_value: float | None
    run_id: str | None
    note: str | None = None


@dataclass
class StartupRetrainedModel:
    model_key: str
    runner: Any
    selection: StartupModelChoice
    trained_at_utc: str
    training_rows: int
    train_date_min: str | None
    train_date_max: str | None


class IngredientRow(BaseModel):
    category: str = Field(default="")
    item: str = Field(default="")
    price_point_name: str = Field(default="")
    milk_oz: float = Field(default=0.0)


class IngredientUpdateRequest(BaseModel):
    rows: list[IngredientRow]


class AdminModelConfigUpdateRequest(BaseModel):
    model_name: str = Field(min_length=1)
    model_alias: str = Field(min_length=1)


class EODSyncRequest(BaseModel):
    merchant_id: str | None = Field(default=None)
    dry_run: bool = Field(default=False)
    start_at_utc: str | None = Field(default=None)
    end_at_utc: str | None = Field(default=None)
    horizon_days: int = Field(default=7, ge=1, le=30)
    lookback_days: int = Field(default=30, ge=7, le=365)
    safety_buffer_pct: float = Field(default=10.0, ge=0.0, le=100.0)


class AuthCredentialsRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=8)


class SquareDisconnectRequest(BaseModel):
    revoke_in_square: bool = Field(default=True)


def _auth_db_path() -> Path:
    path_value = os.getenv("MILK_AUTH_DB_PATH", "artifacts/runtime/auth.db")
    return _resolve_data_path(path_value)


@lru_cache(maxsize=1)
def _auth_store() -> AuthStore:
    return AuthStore(_auth_db_path())


def _session_cookie_secure() -> bool:
    raw = os.getenv("MILK_SESSION_COOKIE_SECURE", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _set_session_cookie(response: Response, session_token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        samesite="lax",
        secure=_session_cookie_secure(),
        max_age=int(timedelta(days=30).total_seconds()),
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def _current_user(session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> UserRecord:
    user = _auth_store().get_user_by_session(session_token)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def _square_oauth_settings() -> dict[str, str]:
    client_id = os.getenv("SQUARE_CLIENT_ID", "").strip()
    client_secret = os.getenv("SQUARE_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv("SQUARE_REDIRECT_URI", "").strip()
    square_env = os.getenv("SQUARE_ENV", "production").strip().lower()
    scopes = os.getenv(
        "SQUARE_SCOPES",
        "MERCHANT_PROFILE_READ ORDERS_READ ITEMS_READ PAYMENTS_READ",
    ).strip()
    missing = [name for name, value in {
        "SQUARE_CLIENT_ID": client_id,
        "SQUARE_CLIENT_SECRET": client_secret,
        "SQUARE_REDIRECT_URI": redirect_uri,
    }.items() if not value]
    if missing:
        raise RuntimeError(f"Missing Square OAuth settings: {', '.join(missing)}")
    host = "connect.squareupsandbox.com" if square_env == "sandbox" else "connect.squareup.com"
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "square_env": square_env,
        "scopes": scopes,
        "base_url": f"https://{host}",
    }


def _runtime_model_config_path() -> Path:
    path_value = os.getenv("MILK_RUNTIME_MODEL_CONFIG_PATH", "artifacts/runtime/admin_model_config.json")
    return _resolve_data_path(path_value)


def _load_runtime_model_config() -> dict[str, str | None]:
    path = _runtime_model_config_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    model_name = str(payload.get("model_name") or "").strip() or None
    model_alias = str(payload.get("model_alias") or "").strip() or None
    return {"model_name": model_name, "model_alias": model_alias}


def _save_runtime_model_config(*, model_name: str, model_alias: str) -> Path:
    path = _runtime_model_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name.strip(),
        "model_alias": model_alias.strip(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def _admin_api_key() -> str | None:
    key = os.getenv("MILK_ADMIN_API_KEY", "").strip()
    return key or None


def _require_admin_key(x_admin_key: str | None) -> None:
    expected = _admin_api_key()
    if expected and x_admin_key != expected:
        raise HTTPException(status_code=401, detail="Invalid admin key")


def _ops_settings() -> dict[str, str]:
    return {
        "token_backup_path": os.getenv("SQUARE_TOKEN_BACKUP_FILE", "square_oauth_token_backup.txt"),
        "sync_db_path": os.getenv("MILK_SYNC_DB_PATH", "artifacts/data/eod_sync.db"),
        "square_env": os.getenv("SQUARE_ENV", "production"),
        "local_timezone": os.getenv("MILK_LOCAL_TIMEZONE", "America/New_York"),
        "default_lookback_days": os.getenv("MILK_EOD_DEFAULT_LOOKBACK_DAYS", "7"),
        "square_client_id": os.getenv("SQUARE_CLIENT_ID", ""),
        "square_client_secret": os.getenv("SQUARE_CLIENT_SECRET", ""),
    }


def _parse_optional_utc(value: str | None) -> datetime | None:
    if value is None or value.strip() == "":
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid datetime '{value}'. Use ISO-8601 format, e.g. 2026-04-24T18:30:00Z."
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _settings() -> dict[str, str | None]:
    runtime_cfg = _load_runtime_model_config()
    model_name = runtime_cfg.get("model_name") or os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
    model_alias = runtime_cfg.get("model_alias") or os.getenv("MLFLOW_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)
    return {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "model_name": model_name,
        "model_alias": model_alias,
    }


def _startup_training_settings() -> dict[str, str]:
    return {
        "experiment_name": os.getenv(
            "MILK_STARTUP_EXPERIMENT_NAME",
            os.getenv("MLFLOW_EXPERIMENT_NAME", "lookforward_30_30_all_models"),
        ),
        "selection_metric": os.getenv("MILK_STARTUP_SELECTION_METRIC", "holdout_rmse"),
        "summary_csv_path": os.getenv(
            "MILK_STARTUP_SUMMARY_CSV_PATH",
            "artifacts/expanding_backtest_v2/summary/model_comparison_summary.csv",
        ),
        "default_model": os.getenv("MILK_STARTUP_DEFAULT_MODEL", "prophet"),
        "enabled_models": os.getenv("MILK_STARTUP_ENABLED_MODELS", "prophet,seasonal_naive,moving_average"),
    }


def _enabled_startup_models(raw_enabled: str, default_model: str) -> list[str]:
    available = set(list_models())
    enabled = [tok.strip().lower() for tok in raw_enabled.split(",") if tok.strip()]
    filtered = [name for name in enabled if name in available]
    if filtered:
        return filtered
    fallback = default_model.strip().lower()
    if fallback in available:
        return [fallback]
    return sorted(available)


def _coerce_finite_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _map_model_label_to_registry(raw_label: str | None) -> str | None:
    if raw_label is None:
        return None
    normalized = (
        str(raw_label)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    aliases = {
        "prophet": "prophet",
        "seasonal_naive": "seasonal_naive",
        "seasonalnaive": "seasonal_naive",
        "moving_average": "moving_average",
        "movingaverage": "moving_average",
    }
    return aliases.get(normalized)


def _extract_run_model_label(run: Any) -> str | None:
    params = run.data.params
    tags = run.data.tags
    return (
        params.get("model.model")
        or params.get("model_name")
        or params.get("model")
        or tags.get("model_name")
    )


def _select_best_model_from_mlflow_runs(
    *,
    experiment_name: str,
    selection_metric: str,
    enabled_models: list[str],
) -> StartupModelChoice:
    tracking_uri = _settings()["tracking_uri"]
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{selection_metric} ASC", "attribute.start_time DESC"],
        max_results=500,
    )
    metric_priority = [selection_metric, "holdout_rmse", "cv_rmse_mean", "avg_test_rmse", "test_rmse"]

    best_choice: StartupModelChoice | None = None
    for run in runs:
        mapped_model = _map_model_label_to_registry(_extract_run_model_label(run))
        if mapped_model not in enabled_models:
            continue

        metric_name = None
        metric_value = None
        for key in metric_priority:
            metric_value = _coerce_finite_float(run.data.metrics.get(key))
            if metric_value is not None:
                metric_name = key
                break
        if metric_name is None or metric_value is None:
            continue

        if best_choice is None or metric_value < float(best_choice.metric_value):
            best_choice = StartupModelChoice(
                model_key=mapped_model,
                source="mlflow_tracking",
                metric_name=metric_name,
                metric_value=metric_value,
                run_id=run.info.run_id,
            )

    if best_choice is None:
        raise ValueError(
            "No completed MLflow runs with usable metrics for enabled startup models "
            f"{enabled_models} in experiment '{experiment_name}'."
        )
    return best_choice


def _select_best_model_from_summary_csv(
    *,
    summary_csv_path: Path,
    enabled_models: list[str],
) -> StartupModelChoice:
    if not summary_csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv_path}")

    df = pd.read_csv(summary_csv_path)
    if "model" not in df.columns:
        raise ValueError("Summary CSV must contain a 'model' column.")

    df = df.copy()
    df["model_key"] = df["model"].apply(_map_model_label_to_registry)
    df = df[df["model_key"].isin(enabled_models)].copy()
    if df.empty:
        raise ValueError(
            f"Summary CSV has no retrainable models for enabled set {enabled_models}. "
            "Only prophet/seasonal_naive/moving_average are currently retrainable in API."
        )

    metric_col = None
    for candidate in ("avg_RMSE", "latest_RMSE"):
        if candidate in df.columns:
            values = pd.to_numeric(df[candidate], errors="coerce")
            if values.notna().any():
                df[candidate] = values
                metric_col = candidate
                break
    if metric_col is None:
        raise ValueError("Summary CSV must include at least one usable metric: avg_RMSE or latest_RMSE.")

    selected = df.sort_values(metric_col, ascending=True).iloc[0]
    metric_value = _coerce_finite_float(selected[metric_col])
    if metric_value is None:
        raise ValueError(f"Selected metric '{metric_col}' is not finite in summary CSV.")

    return StartupModelChoice(
        model_key=str(selected["model_key"]),
        source="summary_csv",
        metric_name=metric_col,
        metric_value=metric_value,
        run_id=None,
        note=f"Selected from summary file '{summary_csv_path.name}'.",
    )


def _dashboard_settings() -> dict[str, str]:
    return {
        "sales_path": os.getenv("MILK_SALES_PATH", "JItters data.csv"),
        "ingredients_path": os.getenv("MILK_INGREDIENTS_PATH", "Ingredient Measure.xlsx"),
    }


def _resolve_data_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _normalize_ingredient_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(raw_df)

    required = ["category", "item", "price_point_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Ingredient file missing required columns: {missing}")

    milk_col = None
    for column in df.columns:
        if "milk" in column:
            milk_col = column
            break
    if milk_col is None:
        raise ValueError("Ingredient file must contain a milk oz column")

    out = df[required + [milk_col]].copy().rename(columns={milk_col: "milk_oz"})
    out["category"] = out["category"].fillna("").astype(str)
    out["item"] = out["item"].fillna("").astype(str)
    out["price_point_name"] = out["price_point_name"].fillna("").astype(str)
    out["milk_oz"] = pd.to_numeric(out["milk_oz"], errors="coerce").fillna(0.0)
    return out


@lru_cache(maxsize=1)
def _load_dashboard_data() -> DashboardData:
    settings = _dashboard_settings()
    sales_path = _resolve_data_path(settings["sales_path"])
    ingredients_path = _resolve_data_path(settings["ingredients_path"])

    if not sales_path.exists():
        raise FileNotFoundError(f"Sales file not found: {sales_path}")
    if not ingredients_path.exists():
        raise FileNotFoundError(f"Ingredients file not found: {ingredients_path}")

    sales_raw = pd.read_csv(sales_path, encoding="ISO-8859-1", low_memory=False)
    ingredients_raw = pd.read_excel(ingredients_path)

    sales_clean = standardize_column_names(sales_raw)
    sales_clean = parse_money_columns(sales_clean)
    sales_clean = attach_datetime_columns(sales_clean)
    sales_clean = remove_voided_items(sales_clean)
    sales_clean = drop_non_essential_columns(sales_clean)
    sales_clean = sales_clean.dropna(subset=["item", "qty", "datetime"]).copy()

    merged = merge_sales_and_ingredients(
        sales_clean,
        ingredients_raw,
        apply_qty_multiplier=True,
    )
    daily = aggregate_daily_milk_usage(merged)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    ingredient_edit = _normalize_ingredient_table(ingredients_raw)
    return DashboardData(
        sales_clean=sales_clean,
        merged=merged,
        daily=daily,
        ingredient_edit=ingredient_edit,
    )


def _select_startup_model_choice(enabled_models: list[str]) -> StartupModelChoice:
    cfg = _startup_training_settings()
    selection_errors: list[str] = []

    try:
        return _select_best_model_from_mlflow_runs(
            experiment_name=cfg["experiment_name"],
            selection_metric=cfg["selection_metric"],
            enabled_models=enabled_models,
        )
    except Exception as exc:
        selection_errors.append(f"MLflow run selection failed: {exc}")

    try:
        return _select_best_model_from_summary_csv(
            summary_csv_path=_resolve_data_path(cfg["summary_csv_path"]),
            enabled_models=enabled_models,
        )
    except Exception as exc:
        selection_errors.append(f"Summary CSV fallback failed: {exc}")

    default_model = cfg["default_model"].strip().lower()
    if default_model not in enabled_models:
        default_model = enabled_models[0]
    return StartupModelChoice(
        model_key=default_model,
        source="default_model",
        metric_name=None,
        metric_value=None,
        run_id=None,
        note=" ".join(selection_errors) if selection_errors else None,
    )


@lru_cache(maxsize=1)
def _startup_retrained_model() -> StartupRetrainedModel:
    data = _load_dashboard_data()
    daily = data.daily.copy()
    if daily.empty:
        raise ValueError("Cannot train startup model: daily dataset is empty.")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["gallons"] = pd.to_numeric(daily["gallons"], errors="coerce")
    daily = daily.dropna(subset=["date", "gallons"]).sort_values("date").reset_index(drop=True)
    if daily.empty:
        raise ValueError("Cannot train startup model: no valid date/gallons rows.")

    cfg = _startup_training_settings()
    enabled_models = _enabled_startup_models(cfg["enabled_models"], cfg["default_model"])
    selection = _select_startup_model_choice(enabled_models)

    runner = create_model(selection.model_key)
    runner.fit(daily[["date", "gallons"]].copy())

    return StartupRetrainedModel(
        model_key=selection.model_key,
        runner=runner,
        selection=selection,
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        training_rows=int(len(daily)),
        train_date_min=daily["date"].min().strftime("%Y-%m-%d"),
        train_date_max=daily["date"].max().strftime("%Y-%m-%d"),
    )


def _startup_retrained_forecast(horizon_days: int) -> tuple[pd.DataFrame, StartupRetrainedModel]:
    trained = _startup_retrained_model()
    out = trained.runner.predict(horizon_days).copy()
    if "date" not in out.columns or "forecast_gallons" not in out.columns:
        raise ValueError(
            f"Startup model '{trained.model_key}' must output date and forecast_gallons columns."
        )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["forecast_gallons"] = pd.to_numeric(out["forecast_gallons"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Startup model '{trained.model_key}' returned no forecast rows.")
    return out, trained


def _simple_weekday_forecast(daily_actual: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    d = daily_actual.sort_values("date").copy()
    last7 = d.tail(7).copy()
    if last7.empty:
        future_dates = pd.date_range(d["date"].max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        return pd.DataFrame(
            {
                "date": future_dates,
                "forecast_gallons": [0.0] * horizon_days,
                "lower_gallons": [None] * horizon_days,
                "upper_gallons": [None] * horizon_days,
            }
        )

    last7["weekday"] = last7["date"].dt.day_name()
    weekday_avg = last7.groupby("weekday")["gallons"].mean().to_dict()
    overall_avg = float(last7["gallons"].mean())

    future_dates = pd.date_range(d["date"].max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    preds = [float(weekday_avg.get(dt.day_name(), overall_avg)) for dt in future_dates]
    return pd.DataFrame(
        {
            "date": future_dates,
            "forecast_gallons": preds,
            "lower_gallons": [None] * horizon_days,
            "upper_gallons": [None] * horizon_days,
        }
    )


def _prophet_forecast(daily_actual: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    prophet_df = daily_actual[["date", "gallons"]].rename(columns={"date": "ds", "gallons": "y"}).copy()
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], errors="coerce")
    prophet_df = prophet_df.dropna(subset=["ds"]).set_index("ds").asfreq("D")
    prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce").interpolate("time").fillna(0.0)
    prophet_df = prophet_df.reset_index()

    model = Prophet(
        seasonality_mode="multiplicative",
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=1.0,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future).tail(horizon_days)
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy().rename(
        columns={
            "ds": "date",
            "yhat": "forecast_gallons",
            "yhat_lower": "lower_gallons",
            "yhat_upper": "upper_gallons",
        }
    )
    for col in ("forecast_gallons", "lower_gallons", "upper_gallons"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return out.reset_index(drop=True)


def _registry_model_forecast(daily_actual: pd.DataFrame, horizon_days: int, model_name: str) -> pd.DataFrame:
    runner = create_model(model_name)
    train = daily_actual[["date", "gallons"]].copy()
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    train["gallons"] = pd.to_numeric(train["gallons"], errors="coerce").fillna(0.0).clip(lower=0.0)
    train = train.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if train.empty:
        raise ValueError("No valid training rows available for selected model.")

    runner.fit(train)
    out = runner.predict(horizon_days).copy()
    if "date" not in out.columns or "forecast_gallons" not in out.columns:
        raise ValueError(f"Model '{model_name}' output must include date and forecast_gallons.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["forecast_gallons"] = pd.to_numeric(out["forecast_gallons"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["lower_gallons"] = pd.to_numeric(out.get("lower_gallons"), errors="coerce")
    out["upper_gallons"] = pd.to_numeric(out.get("upper_gallons"), errors="coerce")
    return out[["date", "forecast_gallons", "lower_gallons", "upper_gallons"]].dropna(subset=["date"]).reset_index(
        drop=True
    )


def _mlflow_registry_forecast(
    *,
    horizon_days: int,
    lookback_days: int,
    safety_buffer_pct: float,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    svc = _service()
    payload = InferenceForecastRequest(
        horizon_days=horizon_days,
        lookback_days=lookback_days,
        safety_buffer_pct=safety_buffer_pct,
    )
    resp = svc.forecast(payload)
    rows = [
        {
            "date": pt.date,
            "forecast_gallons": float(pt.forecast_gallons),
            "lower_gallons": None,
            "upper_gallons": None,
        }
        for pt in resp.daily_forecast
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("MLflow forecast returned no rows.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out, {
        "model_name": resp.model_name,
        "model_version": resp.model_version,
        "run_id": resp.run_id,
    }


def _candidate_local_model_dirs() -> list[Path]:
    raw = os.getenv("MLFLOW_LOCAL_MODEL_DIRS", "mlruns/1/models;notebooks/mlruns/1/models")
    pieces = [p.strip() for p in raw.split(";") if p.strip()]
    dirs: list[Path] = []
    for piece in pieces:
        path = _resolve_data_path(piece)
        if path.exists() and path.is_dir():
            dirs.append(path)
    return dirs


def _latest_local_model_artifact() -> Path:
    candidates: list[Path] = []
    for base in _candidate_local_model_dirs():
        for child in base.iterdir():
            if not child.is_dir():
                continue
            art = child / "artifacts"
            if (art / "MLmodel").exists():
                candidates.append(art)
    if not candidates:
        raise ValueError("No local MLflow model artifacts found in configured model directories.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@lru_cache(maxsize=4)
def _load_local_pyfunc_model(model_artifact_path: str):
    return mlflow.pyfunc.load_model(model_artifact_path)


def _mlflow_local_artifact_forecast(
    *,
    horizon_days: int,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    art_path = _latest_local_model_artifact()
    model = _load_local_pyfunc_model(str(art_path))
    raw = model.predict(pd.DataFrame({"horizon_days": [horizon_days]}))
    out = pd.DataFrame(raw).copy()
    if "date" not in out.columns or "forecast_gallons" not in out.columns:
        raise ValueError("Local MLflow model output must include date and forecast_gallons.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["forecast_gallons"] = pd.to_numeric(out["forecast_gallons"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["lower_gallons"] = pd.to_numeric(out.get("lower_gallons"), errors="coerce")
    out["upper_gallons"] = pd.to_numeric(out.get("upper_gallons"), errors="coerce")
    out = out[["date", "forecast_gallons", "lower_gallons", "upper_gallons"]].dropna(subset=["date"]).reset_index(
        drop=True
    )
    if out.empty:
        raise ValueError("Local MLflow model returned no rows.")
    model_id = art_path.parent.name
    return out, {
        "model_name": f"local_mlflow_artifact:{model_id}",
        "model_version": None,
        "run_id": None,
    }


def _has_complete_bounds(forecast_df: pd.DataFrame) -> bool:
    if forecast_df.empty:
        return False
    required = {"lower_gallons", "upper_gallons"}
    if not required.issubset(forecast_df.columns):
        return False
    lower = pd.to_numeric(forecast_df["lower_gallons"], errors="coerce")
    upper = pd.to_numeric(forecast_df["upper_gallons"], errors="coerce")
    return lower.notna().all() and upper.notna().all()


def _normalize_forecast_dates(
    forecast_df: pd.DataFrame,
    *,
    latest_date: pd.Timestamp,
    horizon_days: int,
) -> tuple[pd.DataFrame, bool]:
    out = forecast_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("Forecast output has no valid dates.")

    if len(out) > horizon_days:
        out = out.head(horizon_days).copy()
    elif len(out) < horizon_days:
        missing = horizon_days - len(out)
        last_row = out.iloc[-1:].copy()
        pad = pd.concat([last_row] * missing, ignore_index=True)
        out = pd.concat([out, pad], ignore_index=True)

    needs_reanchor = bool((out["date"] <= latest_date).any()) or not out["date"].is_monotonic_increasing
    if needs_reanchor:
        out["date"] = pd.date_range(latest_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    return out, needs_reanchor


def _estimate_weekday_abs_errors(daily_actual: pd.DataFrame, history_days: int = 7) -> list[float]:
    d = daily_actual.sort_values("date").copy()
    d["gallons"] = pd.to_numeric(d["gallons"], errors="coerce")
    d = d.dropna(subset=["date", "gallons"]).reset_index(drop=True)
    if len(d) <= history_days:
        return []

    abs_errors: list[float] = []
    for idx in range(history_days, len(d)):
        hist = d.iloc[max(0, idx - history_days) : idx].copy()
        if hist.empty:
            continue
        hist["weekday"] = hist["date"].dt.day_name()
        weekday_avg = hist.groupby("weekday")["gallons"].mean().to_dict()
        overall_avg = float(hist["gallons"].mean())
        target_day = d.iloc[idx]["date"].day_name()
        pred = float(weekday_avg.get(target_day, overall_avg))
        actual = float(d.iloc[idx]["gallons"])
        abs_errors.append(abs(actual - pred))
    return abs_errors


def _apply_empirical_bounds(
    forecast_df: pd.DataFrame,
    daily_actual: pd.DataFrame,
) -> tuple[pd.DataFrame, float, str]:
    out = forecast_df.copy()
    out["forecast_gallons"] = pd.to_numeric(out["forecast_gallons"], errors="coerce").fillna(0.0).clip(lower=0.0)

    abs_errors = _estimate_weekday_abs_errors(daily_actual)
    method = "empirical_p90_abs_error"
    if abs_errors:
        half_width = float(pd.Series(abs_errors, dtype=float).quantile(0.9))
    else:
        day_deltas = (
            pd.to_numeric(daily_actual.sort_values("date")["gallons"], errors="coerce").diff().abs().dropna()
        )
        if day_deltas.empty:
            half_width = 0.0
            method = "zero_width_fallback"
        else:
            half_width = float(day_deltas.quantile(0.9))
            method = "empirical_p90_daily_delta"

    out["lower_gallons"] = (out["forecast_gallons"] - half_width).clip(lower=0.0)
    out["upper_gallons"] = out["forecast_gallons"] + half_width
    return out, half_width, method


def _serialize_rows(df: pd.DataFrame, numeric_cols: tuple[str, ...]) -> list[dict[str, Any]]:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out.to_dict(orient="records")


def _weekday_stats(daily_df: pd.DataFrame) -> dict[str, Any]:
    d = daily_df.copy()
    d["weekday"] = d["date"].dt.day_name()
    means = d.groupby("weekday", as_index=False)["gallons"].mean().rename(columns={"gallons": "value"})
    medians = d.groupby("weekday", as_index=False)["gallons"].median().rename(columns={"gallons": "value"})

    means["weekday"] = pd.Categorical(means["weekday"], categories=WEEKDAY_ORDER, ordered=True)
    medians["weekday"] = pd.Categorical(medians["weekday"], categories=WEEKDAY_ORDER, ordered=True)
    means = means.sort_values("weekday")
    medians = medians.sort_values("weekday")

    boxes: list[dict[str, Any]] = []
    for weekday in WEEKDAY_ORDER:
        vals = d.loc[d["weekday"] == weekday, "gallons"].astype(float).tolist()
        boxes.append({"weekday": weekday, "values": vals})

    return {
        "mean": means.assign(value=lambda x: x["value"].round(4)).to_dict(orient="records"),
        "median": medians.assign(value=lambda x: x["value"].round(4)).to_dict(orient="records"),
        "boxplot": boxes,
    }


def _monthly_stats(daily_df: pd.DataFrame) -> list[dict[str, Any]]:
    d = daily_df.copy()
    d["month_num"] = d["date"].dt.month
    d["month"] = d["date"].dt.month_name()
    out = (
        d.groupby(["month_num", "month"], as_index=False)
        .agg(
            total_gallons=("gallons", "sum"),
            avg_gallons_per_day=("gallons", "mean"),
            days=("gallons", "count"),
        )
        .sort_values("month_num")
    )
    out["month"] = pd.Categorical(out["month"], categories=MONTH_ORDER, ordered=True)
    out = out.sort_values("month")
    out["total_gallons"] = out["total_gallons"].round(4)
    out["avg_gallons_per_day"] = out["avg_gallons_per_day"].round(4)
    return out.to_dict(orient="records")


def _build_forecast_payload(
    *,
    lookback_days: int,
    horizon_days: int,
    safety_buffer_pct: float,
    model_name: str,
) -> dict[str, Any]:
    data = _load_dashboard_data()
    daily = data.daily.copy()
    if daily.empty:
        raise ValueError("No daily milk data available")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["gallons"] = pd.to_numeric(daily["gallons"], errors="coerce")
    daily = daily.dropna(subset=["date", "gallons"]).sort_values("date").reset_index(drop=True)
    if daily.empty:
        raise ValueError("No valid daily rows available for forecasting.")

    latest_date = daily["date"].max()
    past_window = daily.copy()

    requested_model = model_name.strip().lower()
    allowed = {
        "mlflow_pretrained",
        "mlflow_production",
        "mlflow_active",
        "auto",
        "weekday_baseline",
        "prophet",
        "moving_average",
        "seasonal_naive",
    }
    if requested_model not in allowed:
        raise ValueError(
            f"Unknown forecast model '{requested_model}'. Allowed values: {sorted(allowed)}"
        )
    note_parts = [
        "Training window control is disabled: startup retraining always uses all historical daily data."
    ]
    if requested_model not in {"mlflow_pretrained", "mlflow_production", "mlflow_active", "auto"}:
        note_parts.append(f"Requested model '{requested_model}' ignored; serving startup best-model retrain.")

    forecast_df, startup_model = _startup_retrained_forecast(horizon_days=horizon_days)
    model_used = f"{startup_model.model_key}_startup_retrained"
    model_version: str | None = None
    run_id: str | None = startup_model.selection.run_id
    model_alias: str | None = "startup_best_mlflow"

    if startup_model.selection.source == "mlflow_tracking":
        note_parts.append(
            "Best retrainable model selected from MLflow runs: "
            f"{startup_model.model_key} ({startup_model.selection.metric_name}="
            f"{startup_model.selection.metric_value:.4f})."
        )
    elif startup_model.selection.source == "summary_csv":
        note_parts.append(
            "MLflow run query unavailable; selected from MLflow summary artifact: "
            f"{startup_model.model_key} ({startup_model.selection.metric_name}="
            f"{startup_model.selection.metric_value:.4f})."
        )
    else:
        note_parts.append(f"MLflow selection unavailable; using default startup model '{startup_model.model_key}'.")

    note_parts.append(
        "Startup retrain window: "
        f"{startup_model.train_date_min} to {startup_model.train_date_max} "
        f"({startup_model.training_rows} daily rows)."
    )
    if startup_model.selection.note:
        note_parts.append(startup_model.selection.note)

    forecast_df, date_reanchored = _normalize_forecast_dates(
        forecast_df,
        latest_date=latest_date,
        horizon_days=horizon_days,
    )
    if date_reanchored:
        reanchor_note = (
            "Model output dates were historical/non-future; forecasts were re-anchored to the next horizon window."
        )
        note_parts.append(reanchor_note)

    interval_method = "prophet_prediction_interval"
    interval_half_width = float("nan")
    if _has_complete_bounds(forecast_df):
        upper = pd.to_numeric(forecast_df["upper_gallons"], errors="coerce")
        lower = pd.to_numeric(forecast_df["lower_gallons"], errors="coerce")
        interval_half_width = float(((upper - lower) / 2.0).mean())
    else:
        forecast_df, interval_half_width, interval_method = _apply_empirical_bounds(forecast_df, past_window)
        fallback_note = (
            "Forecast range is an empirical uncertainty band from recent residual behavior."
        )
        note_parts.append(fallback_note)

    note = " ".join(note_parts).strip() or None

    forecast_df["forecast_gallons"] = pd.to_numeric(
        forecast_df["forecast_gallons"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    forecast_df["lower_gallons"] = pd.to_numeric(
        forecast_df["lower_gallons"], errors="coerce"
    ).fillna(forecast_df["forecast_gallons"]).clip(lower=0.0)
    forecast_df["upper_gallons"] = pd.to_numeric(
        forecast_df["upper_gallons"], errors="coerce"
    ).fillna(forecast_df["forecast_gallons"]).clip(lower=0.0)
    bounds = pd.concat(
        [
            forecast_df["lower_gallons"].rename("lower"),
            forecast_df["upper_gallons"].rename("upper"),
        ],
        axis=1,
    )
    forecast_df["lower_gallons"] = bounds.min(axis=1)
    forecast_df["upper_gallons"] = bounds.max(axis=1)

    expected_total = float(forecast_df["forecast_gallons"].sum())
    lower_total = float(forecast_df["lower_gallons"].sum())
    upper_total = float(forecast_df["upper_gallons"].sum())
    expected_pm = float(max(expected_total - lower_total, upper_total - expected_total))

    order_gallons = int(math.ceil(expected_total * (1.0 + safety_buffer_pct / 100.0)))
    order_lower = int(math.ceil(lower_total * (1.0 + safety_buffer_pct / 100.0)))
    order_upper = int(math.ceil(upper_total * (1.0 + safety_buffer_pct / 100.0)))
    order_pm = int(max(order_gallons - order_lower, order_upper - order_gallons))

    weekday = _weekday_stats(daily)
    monthly = _monthly_stats(daily)

    return {
        "summary": {
            "horizon_days": horizon_days,
            "lookback_days": int(len(past_window)),
            "safety_buffer_pct": safety_buffer_pct,
            "model_requested": requested_model,
            "expected_total_gallons": round(expected_total, 4),
            "expected_total_lower_gallons": round(lower_total, 4),
            "expected_total_upper_gallons": round(upper_total, 4),
            "expected_total_pm_gallons": round(expected_pm, 4),
            "order_gallons": order_gallons,
            "order_lower_gallons": order_lower,
            "order_upper_gallons": order_upper,
            "order_pm_gallons": order_pm,
            "model_used": model_used,
            "model_alias": model_alias,
            "model_version": model_version,
            "run_id": run_id,
            "note": note,
            "interval_method": interval_method,
            "interval_half_width_gallons": round(interval_half_width, 4)
            if pd.notna(interval_half_width)
            else None,
            "training_rows_used": startup_model.training_rows,
            "training_date_min": startup_model.train_date_min,
            "training_date_max": startup_model.train_date_max,
        },
        "series": {
            "past": _serialize_rows(past_window[["date", "gallons"]], numeric_cols=("gallons",)),
            "forecast": _serialize_rows(
                forecast_df[["date", "forecast_gallons", "lower_gallons", "upper_gallons"]],
                numeric_cols=("forecast_gallons", "lower_gallons", "upper_gallons"),
            ),
        },
        "patterns": {
            "weekday": weekday,
            "monthly": monthly,
        },
    }


def _resolve_sales_window(
    sales_df: pd.DataFrame,
    *,
    window: str,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if sales_df.empty:
        raise ValueError("No sales data available.")
    dmin = pd.to_datetime(sales_df["date"], errors="coerce").min()
    dmax = pd.to_datetime(sales_df["date"], errors="coerce").max()
    if pd.isna(dmin) or pd.isna(dmax):
        raise ValueError("Sales dataset has invalid date values.")

    if window == "all_time":
        return dmin, dmax
    if window == "past_month":
        return max(dmin, dmax - pd.Timedelta(days=30)), dmax
    if window == "past_quarter":
        return max(dmin, dmax - pd.Timedelta(days=90)), dmax
    if window == "custom":
        if not start_date or not end_date:
            raise ValueError("Custom window requires start_date and end_date.")
        sd = pd.to_datetime(start_date, errors="coerce")
        ed = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(sd) or pd.isna(ed):
            raise ValueError("Invalid custom date values.")
        if sd > ed:
            raise ValueError("start_date must be <= end_date.")
        return sd, ed
    raise ValueError(f"Unknown window '{window}'.")


def _sales_breakdown_payload(
    top_n: int,
    *,
    window: str,
    start_date: str | None,
    end_date: str | None,
    drill_category: str | None,
) -> dict[str, Any]:
    data = _load_dashboard_data()
    sales = data.sales_clean.copy()
    merged = data.merged.copy()
    if sales.empty:
        raise ValueError("No sales data available.")

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales = sales.dropna(subset=["date"]).copy()
    sales["category"] = sales["category"].fillna("Uncategorized").astype(str)
    sales["item"] = sales["item"].fillna("Unknown").astype(str)
    sales["net_sales"] = pd.to_numeric(sales.get("net_sales"), errors="coerce").fillna(0.0)

    window_start, window_end = _resolve_sales_window(
        sales,
        window=window,
        start_date=start_date,
        end_date=end_date,
    )
    sales = sales[(sales["date"] >= window_start) & (sales["date"] <= window_end)].copy()
    if sales.empty:
        raise ValueError("No rows in selected analysis window.")

    merged["datetime"] = pd.to_datetime(merged["datetime"], errors="coerce")
    merged = merged.dropna(subset=["datetime"]).copy()
    merged = merged[
        (merged["datetime"] >= window_start) & (merged["datetime"] <= (window_end + pd.Timedelta(days=1)))
    ].copy()

    category_dual = (
        sales.groupby("category", as_index=False)
        .agg(total_qty=("qty", "sum"), total_sales_amount=("net_sales", "sum"))
        .sort_values("total_qty", ascending=False)
    )
    top_items = (
        sales.groupby(["category", "item"], as_index=False)
        .agg(total_qty=("qty", "sum"), total_sales_amount=("net_sales", "sum"))
        .sort_values("total_qty", ascending=False)
        .head(top_n)
    )

    sales["month"] = sales["date"].dt.to_period("M").dt.to_timestamp()
    category_totals = sales.groupby("category", as_index=False)["qty"].sum().sort_values("qty", ascending=False)
    major_categories = category_totals.head(6)["category"].tolist()
    comp = sales.copy()
    comp["category_comp"] = comp["category"].where(comp["category"].isin(major_categories), "Other")
    comp_series = (
        comp.groupby(["month", "category_comp"], as_index=False)["qty"]
        .sum()
        .rename(columns={"category_comp": "category", "qty": "total_qty"})
    )
    month_totals = comp_series.groupby("month", as_index=False)["total_qty"].sum().rename(
        columns={"total_qty": "month_total_qty"}
    )
    comp_series = comp_series.merge(month_totals, on="month", how="left")
    comp_series["share_pct"] = (
        comp_series["total_qty"] / comp_series["month_total_qty"].replace(0, pd.NA) * 100.0
    ).fillna(0.0)

    categories = sorted(sales["category"].dropna().astype(str).unique().tolist())
    selected_category = drill_category if (drill_category is not None and drill_category in categories) else None

    item_series = pd.DataFrame(columns=["month", "item", "total_qty", "total_sales_amount"])
    if selected_category is not None:
        cat_df = sales[sales["category"] == selected_category].copy()
        if not cat_df.empty:
            top_items_in_cat = (
                cat_df.groupby("item", as_index=False)["qty"].sum().sort_values("qty", ascending=False).head(8)["item"].tolist()
            )
            cat_df["item_comp"] = cat_df["item"].where(cat_df["item"].isin(top_items_in_cat), "Other")
            item_series = (
                cat_df.groupby(["month", "item_comp"], as_index=False)
                .agg(total_qty=("qty", "sum"), total_sales_amount=("net_sales", "sum"))
                .rename(columns={"item_comp": "item"})
                .sort_values(["month", "total_qty"], ascending=[True, False])
            )

    category_milk = (
        merged.groupby("category", as_index=False)["milk_oz_total"]
        .sum()
        .rename(columns={"milk_oz_total": "total_milk_oz"})
        .sort_values("total_milk_oz", ascending=False)
    )
    category_milk["total_gallons"] = category_milk["total_milk_oz"] / 128.0

    date_min = pd.to_datetime(sales["date"], errors="coerce").min()
    date_max = pd.to_datetime(sales["date"], errors="coerce").max()
    mapped_rows = int((pd.to_numeric(merged["milk_oz_total"], errors="coerce").fillna(0.0) > 0).sum())

    item_series_payload: list[dict[str, Any]]
    if item_series.empty:
        item_series_payload = []
    else:
        item_series_payload = item_series.assign(
            month=lambda x: pd.to_datetime(x["month"], errors="coerce").dt.strftime("%Y-%m-01"),
            total_qty=lambda x: pd.to_numeric(x["total_qty"], errors="coerce").fillna(0.0).round(2),
            total_sales_amount=lambda x: pd.to_numeric(x["total_sales_amount"], errors="coerce").fillna(0.0).round(2),
        ).to_dict(orient="records")

    return {
        "window": {
            "mode": window,
            "start_date": None if pd.isna(window_start) else window_start.strftime("%Y-%m-%d"),
            "end_date": None if pd.isna(window_end) else window_end.strftime("%Y-%m-%d"),
        },
        "summary": {
            "total_sales_rows": int(len(sales)),
            "total_qty": float(pd.to_numeric(sales["qty"], errors="coerce").fillna(0.0).sum()),
            "total_sales_amount": float(pd.to_numeric(sales["net_sales"], errors="coerce").fillna(0.0).sum()),
            "date_min": None if pd.isna(date_min) else date_min.strftime("%Y-%m-%d"),
            "date_max": None if pd.isna(date_max) else date_max.strftime("%Y-%m-%d"),
            "mapped_milk_rows": mapped_rows,
            "mapping_coverage_pct": round(mapped_rows / max(1, len(merged)) * 100.0, 2),
            "selected_category": selected_category,
        },
        "categories": categories,
        "category_dual_axis": category_dual.assign(
            total_qty=lambda x: x["total_qty"].round(2),
            total_sales_amount=lambda x: x["total_sales_amount"].round(2),
        ).to_dict(orient="records"),
        "top_items": top_items.assign(
            total_qty=lambda x: x["total_qty"].round(2),
            total_sales_amount=lambda x: x["total_sales_amount"].round(2),
        ).to_dict(orient="records"),
        "category_composition_over_time": comp_series.assign(
            month=lambda x: pd.to_datetime(x["month"], errors="coerce").dt.strftime("%Y-%m-01"),
            total_qty=lambda x: x["total_qty"].round(2),
            month_total_qty=lambda x: x["month_total_qty"].round(2),
            share_pct=lambda x: x["share_pct"].round(4),
        ).to_dict(orient="records"),
        "item_drilldown_over_time": item_series_payload,
        "category_milk": category_milk.assign(
            total_milk_oz=lambda x: x["total_milk_oz"].round(2),
            total_gallons=lambda x: x["total_gallons"].round(4),
        ).to_dict(orient="records"),
    }


def _ingredient_path() -> Path:
    return _resolve_data_path(_dashboard_settings()["ingredients_path"])


def _save_ingredient_rows(rows: list[IngredientRow]) -> int:
    if not rows:
        raise ValueError("No ingredient rows provided.")

    payload = pd.DataFrame([row.model_dump() for row in rows])
    payload["category"] = payload["category"].fillna("").astype(str).str.strip()
    payload["item"] = payload["item"].fillna("").astype(str).str.strip()
    payload["price_point_name"] = payload["price_point_name"].fillna("").astype(str).str.strip()
    payload["milk_oz"] = pd.to_numeric(payload["milk_oz"], errors="coerce").fillna(0.0)

    payload = payload[
        (payload["category"] != "") & (payload["item"] != "") & (payload["price_point_name"] != "")
    ].copy()
    if payload.empty:
        raise ValueError("At least one valid ingredient row is required.")

    payload = payload.drop_duplicates(subset=["category", "item", "price_point_name"], keep="last")
    to_save = payload.rename(columns={"milk_oz": "Milk (oz)"})
    to_save.to_excel(_ingredient_path(), index=False)
    _load_dashboard_data.cache_clear()
    _startup_retrained_model.cache_clear()
    return int(len(to_save))


@lru_cache(maxsize=1)
def _service():
    settings = _settings()
    return build_forecast_service(
        tracking_uri=settings["tracking_uri"],
        model_name=str(settings["model_name"]),
        model_alias=str(settings["model_alias"]),
    )


def _forecast_response_from_startup_model(payload: InferenceForecastRequest) -> InferenceForecastResponse:
    data = _load_dashboard_data()
    latest_date = pd.to_datetime(data.daily["date"], errors="coerce").max()
    if pd.isna(latest_date):
        raise ValueError("Cannot compute forecast: latest date is missing from daily data.")

    forecast_df, startup_model = _startup_retrained_forecast(horizon_days=payload.horizon_days)
    forecast_df, _ = _normalize_forecast_dates(
        forecast_df,
        latest_date=pd.Timestamp(latest_date),
        horizon_days=payload.horizon_days,
    )
    forecast_df["forecast_gallons"] = (
        pd.to_numeric(forecast_df["forecast_gallons"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    expected_total = float(forecast_df["forecast_gallons"].sum())
    order_gallons = int(math.ceil(expected_total * (1.0 + payload.safety_buffer_pct / 100.0)))

    daily_forecast = [
        {"date": row["date"].date(), "forecast_gallons": float(row["forecast_gallons"])}
        for _, row in forecast_df.iterrows()
    ]
    return InferenceForecastResponse(
        daily_forecast=daily_forecast,
        expected_total_gallons=expected_total,
        order_gallons=order_gallons,
        safety_buffer_pct=payload.safety_buffer_pct,
        model_name=f"{startup_model.model_key}_startup_retrained",
        model_version=None,
        run_id=startup_model.selection.run_id,
    )


def _model_config_payload() -> dict[str, Any]:
    settings = _settings()
    runtime_path = _runtime_model_config_path()
    runtime_cfg = _load_runtime_model_config()
    return {
        "tracking_uri": settings["tracking_uri"],
        "model_name": settings["model_name"],
        "model_alias": settings["model_alias"],
        "runtime_config_path": str(runtime_path),
        "runtime_override_active": bool(runtime_cfg.get("model_name") or runtime_cfg.get("model_alias")),
        "admin_key_required": bool(_admin_api_key()),
    }


def _square_headers(access_token: str | None = None) -> dict[str, str]:
    headers = {
        "Square-Version": "2026-01-22",
        "Content-Type": "application/json",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def _build_eod_sync_config(
    *,
    token_record: dict[str, Any] | None = None,
    token_refresh_callback: Any | None = None,
) -> EODSyncConfig:
    dashboard = _dashboard_settings()
    ops = _ops_settings()
    try:
        lookback_days = max(1, int(str(ops["default_lookback_days"])))
    except ValueError:
        lookback_days = 7
    return EODSyncConfig(
        sales_path=_resolve_data_path(dashboard["sales_path"]),
        ingredients_path=_resolve_data_path(dashboard["ingredients_path"]),
        token_backup_path=_resolve_data_path(ops["token_backup_path"]),
        sqlite_path=_resolve_data_path(ops["sync_db_path"]),
        square_env=ops["square_env"],
        local_timezone=ops["local_timezone"],
        default_lookback_days=lookback_days,
        client_id=str(ops.get("square_client_id") or "") or None,
        client_secret=str(ops.get("square_client_secret") or "") or None,
        token_record=token_record,
        token_refresh_callback=token_refresh_callback,
    )


def _refresh_connection_tokens(connection: dict[str, Any]) -> dict[str, Any]:
    oauth = _square_oauth_settings()
    payload = {
        "client_id": oauth["client_id"],
        "client_secret": oauth["client_secret"],
        "grant_type": "refresh_token",
        "refresh_token": connection["refresh_token"],
    }
    resp = requests.post(
        f"{oauth['base_url']}/oauth2/token",
        json=payload,
        headers=_square_headers(),
        timeout=30,
    )
    data = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail={"error": "Square token refresh failed", "square": data})
    merged = {
        "merchant_id": data.get("merchant_id") or connection["merchant_id"],
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token") or connection["refresh_token"],
        "token_type": data.get("token_type"),
        "scope": data.get("scope"),
        "expires_at": data.get("expires_at"),
    }
    _auth_store().upsert_square_connection(
        user_id=int(connection["user_id"]),
        token_payload=merged,
        square_env=oauth["square_env"],
    )
    refreshed = _auth_store().get_square_connection_by_user(int(connection["user_id"]), include_secrets=True)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Failed to persist refreshed Square token.")
    return refreshed


@app.on_event("startup")
def warm_startup_model() -> None:
    _startup_retrained_model.cache_clear()
    try:
        trained = _startup_retrained_model()
        LOGGER.info(
            "Startup retrain complete: model=%s source=%s metric=%s value=%s rows=%s",
            trained.model_key,
            trained.selection.source,
            trained.selection.metric_name,
            trained.selection.metric_value,
            trained.training_rows,
        )
    except Exception:  # pragma: no cover - startup resilience
        LOGGER.exception("Startup retrain failed; API will attempt fallback behavior on request.")


@app.get("/", include_in_schema=False)
def dashboard_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(index_path)


@app.post("/api/auth/register")
def auth_register(payload: AuthCredentialsRequest, response: Response) -> dict[str, Any]:
    try:
        user = _auth_store().create_user(email=payload.email, password=payload.password)
        session_token = _auth_store().create_session(user_id=user.id)
        _set_session_cookie(response, session_token)
        return {
            "status": "registered",
            "user": {
                "id": user.id,
                "email": user.email,
                "created_at_utc": user.created_at_utc,
            },
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/auth/login")
def auth_login(payload: AuthCredentialsRequest, response: Response) -> dict[str, Any]:
    user = _auth_store().authenticate_user(email=payload.email, password=payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    session_token = _auth_store().create_session(user_id=user.id)
    _set_session_cookie(response, session_token)
    return {
        "status": "authenticated",
        "user": {
            "id": user.id,
            "email": user.email,
            "created_at_utc": user.created_at_utc,
            "last_login_at_utc": user.last_login_at_utc,
        },
    }


@app.post("/api/auth/logout")
def auth_logout(
    response: Response,
    session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, str]:
    if session_token:
        _auth_store().revoke_session(session_token)
    _clear_session_cookie(response)
    return {"status": "logged_out"}


@app.get("/api/auth/me")
def auth_me(session_token: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, Any]:
    user = _auth_store().get_user_by_session(session_token)
    if user is None:
        return {"authenticated": False}
    connection = _auth_store().get_square_connection_by_user(user.id, include_secrets=False)
    return {
        "authenticated": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "created_at_utc": user.created_at_utc,
            "last_login_at_utc": user.last_login_at_utc,
        },
        "square_connection": connection,
    }


@app.get("/api/auth/square/connect")
def auth_square_connect(
    user: UserRecord = Depends(_current_user),
) -> RedirectResponse:
    try:
        oauth = _square_oauth_settings()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    state = _auth_store().create_oauth_state(user_id=user.id)
    params = {
        "client_id": oauth["client_id"],
        "scope": oauth["scopes"],
        "session": "false",
        "state": state,
        "redirect_uri": oauth["redirect_uri"],
    }
    url = f"{oauth['base_url']}/oauth2/authorize?{urlencode(params)}"
    return RedirectResponse(url=url, status_code=302)


@app.get("/api/auth/square/callback")
def auth_square_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> RedirectResponse:
    if error:
        raise HTTPException(status_code=400, detail=f"Square authorization failed: {error}")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Square callback requires code and state.")
    try:
        user_id = _auth_store().consume_oauth_state(state)
        oauth = _square_oauth_settings()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = {
        "client_id": oauth["client_id"],
        "client_secret": oauth["client_secret"],
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": oauth["redirect_uri"],
    }
    resp = requests.post(
        f"{oauth['base_url']}/oauth2/token",
        json=payload,
        headers=_square_headers(),
        timeout=30,
    )
    data = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail={"error": "Square token exchange failed", "square": data})
    _auth_store().upsert_square_connection(user_id=user_id, token_payload=data, square_env=oauth["square_env"])
    return RedirectResponse(url="/?square_connected=1", status_code=302)


@app.get("/api/auth/square/status")
def auth_square_status(user: UserRecord = Depends(_current_user)) -> dict[str, Any]:
    connection = _auth_store().get_square_connection_by_user(user.id, include_secrets=False)
    return {
        "connected": connection is not None,
        "connection": connection,
    }


@app.post("/api/auth/square/disconnect")
def auth_square_disconnect(
    payload: SquareDisconnectRequest,
    user: UserRecord = Depends(_current_user),
) -> dict[str, str]:
    connection = _auth_store().get_square_connection_by_user(user.id, include_secrets=True)
    if connection is None:
        raise HTTPException(status_code=404, detail="No Square connection found for this user.")
    if payload.revoke_in_square:
        try:
            oauth = _square_oauth_settings()
            resp = requests.post(
                f"{oauth['base_url']}/oauth2/revoke",
                json={
                    "client_id": oauth["client_id"],
                    "access_token": connection["access_token"],
                },
                headers=_square_headers(),
                timeout=30,
            )
            if resp.status_code >= 400:
                data = resp.json() if resp.content else {}
                raise HTTPException(status_code=502, detail={"error": "Square revoke failed", "square": data})
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    _auth_store().revoke_square_connection(user_id=user.id)
    return {"status": "disconnected"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        _startup_retrained_model()
        return {"status": "ready"}
    except Exception as exc:  # pragma: no cover - endpoint safety fallback
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc


@app.post("/v1/forecast", response_model=InferenceForecastResponse)
def forecast(payload: InferenceForecastRequest) -> InferenceForecastResponse:
    try:
        return _forecast_response_from_startup_model(payload)
    except HTTPException:
        raise
    except Exception:
        try:
            svc = _service()
            return svc.forecast(payload)
        except Exception as fallback_exc:  # pragma: no cover - endpoint safety fallback
            raise HTTPException(status_code=500, detail=str(fallback_exc)) from fallback_exc


@app.get("/api/dashboard/forecast")
def dashboard_forecast(
    horizon_days: int = Query(default=7, ge=1, le=30),
    lookback_days: int = Query(default=30, ge=7, le=365),
    safety_buffer_pct: float = Query(default=10.0, ge=0.0, le=100.0),
    model: Literal[
        "mlflow_pretrained",
        "auto",
        "weekday_baseline",
        "prophet",
        "moving_average",
        "seasonal_naive",
        "mlflow_production",
        "mlflow_active",
    ] = Query(default="mlflow_pretrained"),
) -> dict[str, Any]:
    try:
        return _build_forecast_payload(
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            safety_buffer_pct=safety_buffer_pct,
            model_name=model,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/dashboard/sales-breakdown")
def dashboard_sales_breakdown(
    top_n: int = Query(default=12, ge=5, le=50),
    window: Literal["all_time", "past_month", "past_quarter", "custom"] = Query(default="all_time"),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    drill_category: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return _sales_breakdown_payload(
            top_n=top_n,
            window=window,
            start_date=start_date,
            end_date=end_date,
            drill_category=drill_category,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/dashboard/ingredients")
def dashboard_ingredients() -> dict[str, Any]:
    try:
        data = _load_dashboard_data()
        rows = data.ingredient_edit.sort_values(["category", "item", "price_point_name"]).reset_index(drop=True)
        rows["milk_oz"] = pd.to_numeric(rows["milk_oz"], errors="coerce").fillna(0.0).round(4)
        return {
            "file_name": _ingredient_path().name,
            "rows": rows.to_dict(orient="records"),
            "row_count": int(len(rows)),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/dashboard/ingredients")
def dashboard_ingredients_update(payload: IngredientUpdateRequest) -> dict[str, Any]:
    try:
        written_rows = _save_ingredient_rows(payload.rows)
        return {
            "status": "saved",
            "file_name": _ingredient_path().name,
            "rows_written": written_rows,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/integrations/square/eod/latest")
def square_latest_eod_sync(user: UserRecord = Depends(_current_user)) -> dict[str, Any]:
    connection = _auth_store().get_square_connection_by_user(user.id, include_secrets=False)
    if connection is None:
        raise HTTPException(status_code=404, detail="No Square connection found for this user.")
    try:
        cfg = _build_eod_sync_config()
        latest = load_latest_sync_run(cfg.sqlite_path, merchant_id=connection["merchant_id"])
        return {
            "sync_db_path": str(cfg.sqlite_path),
            "merchant_id": connection["merchant_id"],
            "latest": latest,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/integrations/square/eod-sync")
def square_eod_sync(
    payload: EODSyncRequest,
    user: UserRecord = Depends(_current_user),
) -> dict[str, Any]:
    connection = _auth_store().get_square_connection_by_user(user.id, include_secrets=True)
    if connection is None:
        raise HTTPException(status_code=404, detail="No Square connection found for this user.")
    try:
        start_at = _parse_optional_utc(payload.start_at_utc)
        end_at = _parse_optional_utc(payload.end_at_utc)
        cfg = _build_eod_sync_config(
            token_record=connection,
            token_refresh_callback=lambda token_record: _refresh_connection_tokens(token_record),
        )
        sync_result = run_eod_sync(
            config=cfg,
            merchant_id=connection["merchant_id"],
            dry_run=payload.dry_run,
            start_at=start_at,
            end_at=end_at,
        )
        _load_dashboard_data.cache_clear()
        _startup_retrained_model.cache_clear()
        forecast = _build_forecast_payload(
            lookback_days=payload.lookback_days,
            horizon_days=payload.horizon_days,
            safety_buffer_pct=payload.safety_buffer_pct,
            model_name="mlflow_pretrained",
        )
        return {
            "status": "ok",
            "merchant_id": connection["merchant_id"],
            "sync": sync_result,
            "forecast_summary": forecast["summary"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/admin/model-config")
def admin_model_config(x_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    _require_admin_key(x_admin_key)
    return _model_config_payload()


@app.put("/api/admin/model-config")
def admin_model_config_update(
    payload: AdminModelConfigUpdateRequest,
    x_admin_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_admin_key(x_admin_key)
    try:
        path = _save_runtime_model_config(model_name=payload.model_name, model_alias=payload.model_alias)
        _service.cache_clear()
        return {
            "status": "saved",
            "runtime_config_path": str(path),
            "model_config": _model_config_payload(),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/admin/eod/latest")
def admin_latest_eod_sync(x_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    _require_admin_key(x_admin_key)
    try:
        cfg = _build_eod_sync_config()
        latest = load_latest_sync_run(cfg.sqlite_path)
        return {
            "sync_db_path": str(cfg.sqlite_path),
            "latest": latest,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/eod-sync")
def admin_eod_sync(
    payload: EODSyncRequest,
    x_admin_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_admin_key(x_admin_key)
    try:
        start_at = _parse_optional_utc(payload.start_at_utc)
        end_at = _parse_optional_utc(payload.end_at_utc)
        cfg = _build_eod_sync_config()
        sync_result = run_eod_sync(
            config=cfg,
            merchant_id=payload.merchant_id,
            dry_run=payload.dry_run,
            start_at=start_at,
            end_at=end_at,
        )
        _load_dashboard_data.cache_clear()
        _startup_retrained_model.cache_clear()

        forecast = _build_forecast_payload(
            lookback_days=payload.lookback_days,
            horizon_days=payload.horizon_days,
            safety_buffer_pct=payload.safety_buffer_pct,
            model_name="mlflow_pretrained",
        )
        sales = _sales_breakdown_payload(
            top_n=12,
            window="past_month",
            start_date=None,
            end_date=None,
            drill_category=None,
        )
        return {
            "status": "ok",
            "sync": sync_result,
            "forecast_summary": forecast["summary"],
            "sales_summary": sales["summary"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/dashboard/refresh")
def dashboard_refresh() -> dict[str, str]:
    _load_dashboard_data.cache_clear()
    _startup_retrained_model.cache_clear()
    _startup_retrained_model()
    return {"status": "refreshed"}
