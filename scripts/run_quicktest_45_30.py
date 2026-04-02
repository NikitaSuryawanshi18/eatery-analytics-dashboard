from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import cloudpickle
import mlflow
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor

ROOT = Path(r"c:\Users\Bhavesh\Documents\Python Scripts\Jeff\Cafe\milk-dashboard")
INPUT_PATH = ROOT / "artifacts" / "data" / "cleaned_merged_with_ingredients.csv"
OUT_DIR = ROOT / "artifacts" / "expanding_backtest" / "quicktest_45_30"
PRED_DIR = OUT_DIR / "predictions"
MET_DIR = OUT_DIR / "metrics"
SUMMARY_DIR = OUT_DIR / "summary"

EXPERIMENT_NAME = "milk_models_expanding_quicktest"
TRACKING_URI = "sqlite:///mlruns_quicktest.db"

LOOKBACK_DAYS = 45
HORIZON_DAYS = 30
STEP_DAYS = 30
FORCE_FINAL = True

LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]

MODELS = ["ARIMA_212", "Prophet", "XGBoost"]


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name)


def metrics_dict(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mask = actual != 0
    mape = float(np.mean(np.abs(err[mask] / actual[mask])) * 100.0) if np.any(mask) else float("nan")
    accuracy = float(100.0 - mape) if not np.isnan(mape) else float("nan")
    bias = float(np.mean(pred - actual))
    mean_actual = float(np.mean(actual)) if len(actual) else float("nan")
    error_pct = float((rmse / mean_actual) * 100.0) if mean_actual and mean_actual != 0 else float("nan")
    error_std = float(np.std(err))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Accuracy": accuracy,
        "Bias": bias,
        "ErrorPct": error_pct,
        "ErrorStd": error_std,
    }


def generate_splits(n_rows: int, initial_train_days: int, horizon_days: int, step_days: int, force_final: bool):
    out = []
    if n_rows <= initial_train_days + horizon_days:
        return out

    train_end = initial_train_days - 1
    while train_end + horizon_days < n_rows:
        out.append({
            "train_start": 0,
            "train_end": train_end,
            "test_start": train_end + 1,
            "test_end": train_end + horizon_days,
        })
        train_end += step_days

    final_train_end = n_rows - horizon_days - 1
    if force_final and final_train_end >= initial_train_days - 1:
        if not out or out[-1]["train_end"] != final_train_end:
            out.append({
                "train_start": 0,
                "train_end": final_train_end,
                "test_start": final_train_end + 1,
                "test_end": final_train_end + horizon_days,
            })

    return out


def _roll_mean(vals, w):
    if len(vals) < w:
        return float(np.mean(vals)) if len(vals) else 0.0
    return float(np.mean(vals[-w:]))


def _roll_std(vals, w):
    if len(vals) < w:
        return float(np.std(vals)) if len(vals) else 0.0
    return float(np.std(vals[-w:]))


def feature_row(history, current_date):
    current_date = pd.Timestamp(current_date)
    eps = 1e-9
    row = {}
    for lag in LAG_DAYS:
        row[f"lag_{lag}"] = float(history[-lag])

    mean7 = _roll_mean(history, 7)
    mean14 = _roll_mean(history, 14)

    row["mean_7"] = mean7
    row["mean_14"] = mean14
    row["std_7"] = _roll_std(history, 7)
    row["std_14"] = _roll_std(history, 14)
    row["mean_7_vs_14_diff"] = mean7 - mean14
    row["mean_7_vs_14_ratio"] = float(mean7 / (mean14 + eps))
    row["mean_7_vs_14_pct"] = float(((mean7 - mean14) / (abs(mean14) + eps)) * 100.0)

    row["day_of_week"] = int(current_date.dayofweek)
    row["is_weekend"] = int(current_date.dayofweek >= 5)
    row["month"] = int(current_date.month)
    row["day_of_month"] = int(current_date.day)
    row["week_of_year"] = int(current_date.isocalendar().week)

    return row


def build_xy(train_dates, train_values):
    max_lag = max(LAG_DAYS)
    rows = []
    y = []
    for i in range(max_lag, len(train_values)):
        rows.append(feature_row(train_values[:i], train_dates[i]))
        y.append(float(train_values[i]))
    if not rows:
        return pd.DataFrame(), np.array([])
    return pd.DataFrame(rows), np.asarray(y, dtype=float)


def recursive_predict_xgb(model, train_dates, train_values, horizon):
    hist = list(np.asarray(train_values, dtype=float))
    preds = []
    last_date = pd.Timestamp(train_dates[-1])
    for _ in range(horizon):
        next_date = last_date + pd.Timedelta(days=1)
        x_next = pd.DataFrame([feature_row(hist, next_date)])
        p = float(model.predict(x_next)[0])
        p = max(p, 0.0)
        preds.append(p)
        hist.append(p)
        last_date = next_date
    return np.asarray(preds, dtype=float)


def clean_metrics(d):
    out = {}
    for k, v in d.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isnan(fv) or np.isinf(fv):
            continue
        out[k] = fv
    return out


for d in [OUT_DIR, PRED_DIR, MET_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(INPUT_PATH)
raw.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in raw.columns]

if "datetime" in raw.columns:
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
    if "milk_oz_total" in raw.columns:
        val_col = "milk_oz_total"
    elif "milk_oz" in raw.columns:
        val_col = "milk_oz"
    else:
        raise ValueError("Need milk_oz_total or milk_oz in merged file")

    raw[val_col] = pd.to_numeric(raw[val_col], errors="coerce").fillna(0.0)
    daily = (
        raw.dropna(subset=["datetime"])
        .groupby(raw["datetime"].dt.date)[val_col]
        .sum()
        .reset_index()
        .rename(columns={"datetime": "date", val_col: "total_milk_oz"})
    )
elif "date" in raw.columns and "total_milk_oz" in raw.columns:
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["total_milk_oz"] = pd.to_numeric(raw["total_milk_oz"], errors="coerce").fillna(0.0)
    daily = (
        raw.dropna(subset=["date"])
        .groupby(raw["date"].dt.date)["total_milk_oz"]
        .sum()
        .reset_index()
        .rename(columns={"date": "date"})
    )
else:
    raise ValueError("Unable to derive daily series from merged input")

daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)
daily["gallons"] = pd.to_numeric(daily["total_milk_oz"], errors="coerce").fillna(0.0) / 128.0

daily = daily.set_index("date").asfreq("D")
daily["gallons"] = daily["gallons"].fillna(0.0)
daily = daily.reset_index()[["date", "gallons"]]

print("Daily rows:", len(daily), "range:", daily["date"].min(), "->", daily["date"].max())

splits = generate_splits(len(daily), LOOKBACK_DAYS, HORIZON_DAYS, STEP_DAYS, FORCE_FINAL)
print("Split count:", len(splits))
if not splits:
    raise RuntimeError("No splits generated")

mlflow.set_tracking_uri(TRACKING_URI)
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = exp.experiment_id if exp is not None else mlflow.create_experiment(EXPERIMENT_NAME)

all_model_rows = []
parent_runs = []

for model_name in MODELS:
    metrics_rows = []
    pred_rows = []
    latest_model_obj = None
    latest_split_idx = None

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_name}_45_30_parent") as pr:
        parent_run_id = pr.info.run_id

        mlflow.set_tags({
            "run_type": "parent",
            "model_name": model_name,
            "cv_scheme": "expanding_window",
            "window_scheme": "train45_pred30_step30",
        })
        mlflow.log_params({
            "model": model_name,
            "lookback_days": LOOKBACK_DAYS,
            "horizon_days": HORIZON_DAYS,
            "step_days": STEP_DAYS,
            "source_file": str(INPUT_PATH),
            "source_mode": "cleaned_merged_aggregated_daily",
            "lag_days": ",".join(str(x) for x in LAG_DAYS),
        })

        for split_idx, sp in enumerate(splits, start=1):
            tr_s, tr_e = sp["train_start"], sp["train_end"]
            te_s, te_e = sp["test_start"], sp["test_end"]

            train_df = daily.iloc[tr_s:tr_e + 1].copy()
            test_df = daily.iloc[te_s:te_e + 1].copy()

            train_dates = train_df["date"].to_numpy()
            test_dates = test_df["date"].to_numpy()
            train_vals = train_df["gallons"].to_numpy(dtype=float)
            test_vals = test_df["gallons"].to_numpy(dtype=float)

            status = "ok"
            err_msg = ""
            preds = np.full(len(test_vals), np.nan)
            train_metrics = {k: np.nan for k in ["RMSE", "MAE", "MAPE", "Accuracy", "Bias", "ErrorPct", "ErrorStd"]}
            fitted_obj = None

            try:
                if model_name == "ARIMA_212":
                    fit = ARIMA(train_vals, order=(2, 1, 2)).fit()
                    preds = np.maximum(np.asarray(fit.forecast(steps=len(test_vals)), dtype=float), 0.0)

                    fit_train = np.maximum(np.asarray(fit.fittedvalues, dtype=float), 0.0)
                    y_train_tail = train_vals[-len(fit_train):]
                    mask = np.isfinite(fit_train) & np.isfinite(y_train_tail)
                    if int(mask.sum()) >= 5:
                        train_metrics = metrics_dict(y_train_tail[mask], fit_train[mask])
                    fitted_obj = fit

                elif model_name == "Prophet":
                    tdf = pd.DataFrame({"ds": pd.to_datetime(train_dates), "y": train_vals})
                    pm = Prophet(
                        seasonality_mode="multiplicative",
                        weekly_seasonality=True,
                        yearly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=1.0,
                    )
                    pm.fit(tdf)

                    in_sample = np.maximum(pm.predict(tdf[["ds"]])["yhat"].to_numpy(dtype=float), 0.0)
                    train_metrics = metrics_dict(train_vals, in_sample)

                    fut = pm.make_future_dataframe(periods=len(test_vals), freq="D")
                    fc = pm.predict(fut).tail(len(test_vals))
                    preds = np.maximum(fc["yhat"].to_numpy(dtype=float), 0.0)
                    fitted_obj = pm

                elif model_name == "XGBoost":
                    X_train, y_train = build_xy(train_dates, train_vals)
                    if len(X_train) < 10:
                        raise RuntimeError("not enough rows after lag-feature generation")
                    xgb = XGBRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        objective="reg:squarederror",
                    )
                    xgb.fit(X_train, y_train)
                    train_pred = np.maximum(xgb.predict(X_train), 0.0)
                    train_metrics = metrics_dict(y_train, train_pred)

                    preds = recursive_predict_xgb(xgb, train_dates, train_vals, len(test_vals))
                    fitted_obj = xgb
                else:
                    raise ValueError(model_name)

            except Exception as e:
                status = "error"
                err_msg = f"{type(e).__name__}: {e}"

            test_metrics = metrics_dict(test_vals, preds) if not np.isnan(preds).any() else {k: np.nan for k in ["RMSE", "MAE", "MAPE", "Accuracy", "Bias", "ErrorPct", "ErrorStd"]}
            gap_rmse = float(test_metrics["RMSE"] - train_metrics["RMSE"]) if pd.notna(test_metrics["RMSE"]) and pd.notna(train_metrics["RMSE"]) else np.nan

            row = {
                "model": model_name,
                "split_idx": split_idx,
                "status": status,
                "error_message": err_msg,
                "train_start": str(pd.Timestamp(train_dates[0]).date()),
                "train_end": str(pd.Timestamp(train_dates[-1]).date()),
                "test_start": str(pd.Timestamp(test_dates[0]).date()),
                "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                "train_size": int(len(train_vals)),
                "test_size": int(len(test_vals)),
                "generalization_gap_rmse": gap_rmse,
            }
            row.update({f"train_{k.lower()}": v for k, v in train_metrics.items()})
            row.update({f"test_{k.lower()}": v for k, v in test_metrics.items()})
            metrics_rows.append(row)

            for dt, a, p in zip(test_dates, test_vals, preds):
                e = a - p if pd.notna(p) else np.nan
                pred_rows.append({
                    "model": model_name,
                    "split_idx": split_idx,
                    "train_start": row["train_start"],
                    "train_end": row["train_end"],
                    "test_start": row["test_start"],
                    "test_end": row["test_end"],
                    "date": str(pd.Timestamp(dt).date()),
                    "actual": float(a),
                    "prediction": float(p) if pd.notna(p) else np.nan,
                    "error": float(e) if pd.notna(e) else np.nan,
                    "abs_error": float(abs(e)) if pd.notna(e) else np.nan,
                })

            if status == "ok":
                latest_model_obj = fitted_obj
                latest_split_idx = split_idx

            # parent stepped trend metrics
            if pd.notna(test_metrics["RMSE"]):
                mlflow.log_metric("test_rmse", float(test_metrics["RMSE"]), step=split_idx)
            if pd.notna(test_metrics["MAE"]):
                mlflow.log_metric("test_mae", float(test_metrics["MAE"]), step=split_idx)
            if pd.notna(test_metrics["MAPE"]):
                mlflow.log_metric("test_mape", float(test_metrics["MAPE"]), step=split_idx)
            if pd.notna(train_metrics["RMSE"]):
                mlflow.log_metric("train_rmse", float(train_metrics["RMSE"]), step=split_idx)
            if pd.notna(gap_rmse):
                mlflow.log_metric("generalization_gap_rmse", float(gap_rmse), step=split_idx)
            mlflow.log_metric("train_window_size", float(len(train_vals)), step=split_idx)
            mlflow.log_metric("test_window_size", float(len(test_vals)), step=split_idx)

            # child run per split
            with mlflow.start_run(run_name=f"split_{split_idx:03d}", nested=True):
                mlflow.set_tags({"run_type": "child", "model_name": model_name, "split_idx": split_idx})
                mlflow.log_params({
                    "split_idx": split_idx,
                    "train_start": row["train_start"],
                    "train_end": row["train_end"],
                    "test_start": row["test_start"],
                    "test_end": row["test_end"],
                    "train_size": int(len(train_vals)),
                    "test_size": int(len(test_vals)),
                })
                mlflow.log_metrics(clean_metrics({
                    "train_rmse": train_metrics["RMSE"],
                    "train_mae": train_metrics["MAE"],
                    "test_rmse": test_metrics["RMSE"],
                    "test_mae": test_metrics["MAE"],
                    "test_mape": test_metrics["MAPE"],
                    "generalization_gap_rmse": gap_rmse,
                    "test_error_pct": test_metrics["ErrorPct"],
                    "test_error_std": test_metrics["ErrorStd"],
                }))

                split_preds = pd.DataFrame([r for r in pred_rows if r["split_idx"] == split_idx])
                with tempfile.TemporaryDirectory(prefix="quick_split_") as td:
                    td = Path(td)
                    p1 = td / "predictions.csv"
                    p2 = td / "residuals.csv"
                    split_preds.to_csv(p1, index=False)
                    split_preds[["date", "actual", "prediction", "error", "abs_error"]].to_csv(p2, index=False)
                    mlflow.log_artifact(str(p1), artifact_path=f"split_{split_idx:03d}")
                    mlflow.log_artifact(str(p2), artifact_path=f"split_{split_idx:03d}")

        model_metrics_df = pd.DataFrame(metrics_rows)
        model_pred_df = pd.DataFrame(pred_rows)

        safe = sanitize(model_name)
        met_path = MET_DIR / f"{safe}_metrics.csv"
        pred_path = PRED_DIR / f"{safe}_predictions.csv"
        model_path = SUMMARY_DIR / f"{safe}_final_model.pkl"

        model_metrics_df.to_csv(met_path, index=False)
        model_pred_df.to_csv(pred_path, index=False)
        if latest_model_obj is not None:
            with open(model_path, "wb") as f:
                cloudpickle.dump(latest_model_obj, f)
            mlflow.log_artifact(str(model_path), artifact_path="final_model")

        mlflow.log_artifact(str(met_path), artifact_path="summary")
        mlflow.log_artifact(str(pred_path), artifact_path="summary")

        avg_test_rmse = float(model_metrics_df["test_rmse"].mean()) if not model_metrics_df.empty else float("nan")
        last_test_rmse = float(model_metrics_df["test_rmse"].iloc[-1]) if not model_metrics_df.empty else float("nan")
        if len(model_metrics_df) > 1:
            rmse_trend = float(model_metrics_df["test_rmse"].iloc[-1] - model_metrics_df["test_rmse"].iloc[0])
        else:
            rmse_trend = float("nan")

        mlflow.log_metrics(clean_metrics({
            "avg_test_rmse": avg_test_rmse,
            "last_test_rmse": last_test_rmse,
            "rmse_trend_delta": rmse_trend,
            "num_splits": float(len(model_metrics_df)),
        }))

        parent_runs.append({"model": model_name, "parent_run_id": parent_run_id, "splits": len(model_metrics_df), "latest_split_idx": latest_split_idx, "avg_test_rmse": avg_test_rmse, "last_test_rmse": last_test_rmse})
        all_model_rows.extend(metrics_rows)

summary_df = pd.DataFrame(all_model_rows)
summary_path = SUMMARY_DIR / "all_models_metrics.csv"
summary_df.to_csv(summary_path, index=False)

run_summary_path = SUMMARY_DIR / "mlflow_parent_runs.json"
run_summary_path.write_text(json.dumps(parent_runs, indent=2), encoding="utf-8")

print("\nCompleted quicktest.")
print("Summary file:", summary_path)
print("Run manifest:", run_summary_path)
print(pd.DataFrame(parent_runs).sort_values("last_test_rmse"))


