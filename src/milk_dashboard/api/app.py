"""FastAPI API + UI app."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from prophet import Prophet
from pydantic import BaseModel, Field

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

app = FastAPI(title="Milk Forecast API", version="0.2.0")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@dataclass
class DashboardData:
    sales_clean: pd.DataFrame
    merged: pd.DataFrame
    daily: pd.DataFrame
    ingredient_edit: pd.DataFrame


class IngredientRow(BaseModel):
    category: str = Field(default="")
    item: str = Field(default="")
    price_point_name: str = Field(default="")
    milk_oz: float = Field(default=0.0)


class IngredientUpdateRequest(BaseModel):
    rows: list[IngredientRow]


def _settings() -> dict[str, str | None]:
    return {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "model_name": os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME),
        "model_alias": os.getenv("MLFLOW_MODEL_ALIAS", DEFAULT_MODEL_ALIAS),
    }


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
) -> dict[str, Any]:
    data = _load_dashboard_data()
    daily = data.daily.copy()
    if daily.empty:
        raise ValueError("No daily milk data available")

    latest_date = daily["date"].max()
    lookback_start = latest_date - pd.Timedelta(days=lookback_days - 1)
    past_window = daily[daily["date"] >= lookback_start].copy()
    if past_window.empty:
        raise ValueError("No rows available for the selected lookback window")

    model_used = "prophet"
    note = None
    if lookback_days < 14:
        forecast_df = _simple_weekday_forecast(past_window, horizon_days)
        model_used = "weekday_baseline"
        note = "Short lookback selected, using weekday baseline forecast."
    else:
        try:
            forecast_df = _prophet_forecast(past_window, horizon_days)
        except Exception:
            forecast_df = _simple_weekday_forecast(past_window, horizon_days)
            model_used = "weekday_baseline"
            note = "Prophet fallback triggered, using weekday baseline forecast."

    forecast_df["forecast_gallons"] = pd.to_numeric(
        forecast_df["forecast_gallons"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    expected_total = float(forecast_df["forecast_gallons"].sum())
    order_gallons = int(math.ceil(expected_total * (1.0 + safety_buffer_pct / 100.0)))

    weekday = _weekday_stats(daily)
    monthly = _monthly_stats(daily)

    return {
        "summary": {
            "horizon_days": horizon_days,
            "lookback_days": lookback_days,
            "safety_buffer_pct": safety_buffer_pct,
            "expected_total_gallons": round(expected_total, 4),
            "order_gallons": order_gallons,
            "model_used": model_used,
            "note": note,
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
    return int(len(to_save))


@lru_cache(maxsize=1)
def _service():
    settings = _settings()
    return build_forecast_service(
        tracking_uri=settings["tracking_uri"],
        model_name=str(settings["model_name"]),
        model_alias=str(settings["model_alias"]),
    )


@app.get("/", include_in_schema=False)
def dashboard_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(index_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        _service()
        return {"status": "ready"}
    except Exception as exc:  # pragma: no cover - endpoint safety fallback
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc


@app.post("/v1/forecast", response_model=InferenceForecastResponse)
def forecast(payload: InferenceForecastRequest) -> InferenceForecastResponse:
    try:
        svc = _service()
        return svc.forecast(payload)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - endpoint safety fallback
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/dashboard/forecast")
def dashboard_forecast(
    horizon_days: int = Query(default=7, ge=1, le=30),
    lookback_days: int = Query(default=30, ge=7, le=365),
    safety_buffer_pct: float = Query(default=10.0, ge=0.0, le=100.0),
) -> dict[str, Any]:
    try:
        return _build_forecast_payload(
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            safety_buffer_pct=safety_buffer_pct,
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


@app.post("/api/dashboard/refresh")
def dashboard_refresh() -> dict[str, str]:
    _load_dashboard_data.cache_clear()
    return {"status": "refreshed"}
