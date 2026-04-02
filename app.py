import math
import pandas as pd
import streamlit as st
import altair as alt
from prophet import Prophet

OZ_PER_GALLON = 128.0

WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTH_ORDER = ["January","February","March","April","May","June","July","August","September","October","November","December"]

# -----------------------------
# Data loading + prep
# -----------------------------
@st.cache_data
def load_and_prepare_data(csv_path: str, ingredient_xlsx_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path, encoding="ISO-8859-1")
    df = df_raw.copy()

    # Standardize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Money columns (safe if present)
    money_cols = ["gross_sales", "discounts", "net_sales", "tax"]
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].replace(r"[$,]", "", regex=True).astype(float)

    # qty numeric
    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    # datetime
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce"
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop missing essentials
    df = df.dropna(subset=["item", "qty", "datetime"])

    # Remove voided items
    df = df[~df["item"].astype(str).str.contains(r"\(voided\)", case=False, na=False)]

    # Ingredient sheet
    ing = pd.read_excel(ingredient_xlsx_path)
    ing = ing.drop(columns=["Unnamed: 8"], errors="ignore")
    ing = ing[["category", "item", "price_point_name", "Milk (oz)"]]
    ing["Milk (oz)"] = pd.to_numeric(ing["Milk (oz)"], errors="coerce").fillna(0)

    # Merge milk oz into sales rows
    df = df.merge(ing, on=["category", "item", "price_point_name"], how="left")
    df["Milk (oz)"] = pd.to_numeric(df["Milk (oz)"], errors="coerce").fillna(0)

    # Assumption: Milk (oz) is per one drink → multiply by qty
    df["milk_oz_total"] = df["Milk (oz)"] * df["qty"]

    # Daily totals (oz)
    daily = (
        df.groupby(df["datetime"].dt.date)["milk_oz_total"]
          .sum()
          .reset_index()
          .rename(columns={"milk_oz_total": "total_milk_oz"})
    )
    daily["date"] = pd.to_datetime(daily["datetime"])
    daily = daily.drop(columns=["datetime"]).sort_values("date")

    # Gallons
    daily["gallons"] = daily["total_milk_oz"] / OZ_PER_GALLON
    return daily


# -----------------------------
# Prophet forecasting
# -----------------------------
@st.cache_resource
def fit_prophet_model(daily_df: pd.DataFrame) -> Prophet:
    prophet_df = daily_df[["date", "total_milk_oz"]].rename(columns={"date": "ds", "total_milk_oz": "y"}).copy()

    # Fill missing dates
    prophet_df = prophet_df.set_index("ds").asfreq("D")
    prophet_df["y"] = prophet_df["y"].interpolate("time")
    prophet_df = prophet_df.reset_index()

    m = Prophet(
        seasonality_mode="multiplicative",
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=1.0
    )
    m.fit(prophet_df)
    return m


def forecast_next_days_prophet(model: Prophet, periods: int) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq="D")
    fc = model.predict(future)
    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()
    out = out.rename(columns={"ds": "date", "yhat": "forecast_oz"})
    out["forecast_gallons"] = out["forecast_oz"] / OZ_PER_GALLON
    out["lower_gallons"] = out["yhat_lower"] / OZ_PER_GALLON
    out["upper_gallons"] = out["yhat_upper"] / OZ_PER_GALLON
    return out


def simple_weekday_forecast(daily_actual: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Stable fallback for very short history:
    - Use last 7 days of actuals
    - Compute average gallons per weekday
    - Forecast future days using that weekday average
    """
    d = daily_actual.sort_values("date").copy()
    last7 = d.tail(7).copy()

    if last7.empty:
        # Shouldn't happen, but safety
        future_dates = pd.date_range(d["date"].max() + pd.Timedelta(days=1), periods=periods, freq="D")
        return pd.DataFrame({"date": future_dates, "forecast_gallons": [0.0]*periods})

    last7["weekday"] = last7["date"].dt.day_name()
    weekday_avg = last7.groupby("weekday")["gallons"].mean().to_dict()
    overall_avg = float(last7["gallons"].mean())

    future_dates = pd.date_range(d["date"].max() + pd.Timedelta(days=1), periods=periods, freq="D")
    preds = []
    for dt in future_dates:
        wd = dt.day_name()
        preds.append(float(weekday_avg.get(wd, overall_avg)))

    fc = pd.DataFrame({"date": future_dates, "forecast_gallons": preds})
    fc["lower_gallons"] = None
    fc["upper_gallons"] = None
    return fc


# -----------------------------
# Seasonality insights (independent)
# -----------------------------
def weekly_pattern_last_7_days(daily: pd.DataFrame) -> pd.DataFrame:
    latest = daily["date"].max()
    start = latest - pd.Timedelta(days=6)
    d7 = daily[(daily["date"] >= start) & (daily["date"] <= latest)].copy()
    if d7.empty:
        return pd.DataFrame()

    d7["weekday"] = d7["date"].dt.day_name()
    out = d7.groupby("weekday", as_index=False)["gallons"].mean()
    out["weekday"] = pd.Categorical(out["weekday"], categories=WEEKDAY_ORDER, ordered=True)
    out = out.sort_values("weekday").reset_index(drop=True)
    return out


def seasonal_pattern_by_month(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Uses ALL available history.
    Shows:
      - total_gallons per month (across all years in the dataset)
      - avg_gallons_per_day (daily average within that month)
    """
    d = daily.copy()
    d["month_num"] = d["date"].dt.month
    d["month"] = d["date"].dt.month_name()

    out = d.groupby(["month_num", "month"], as_index=False).agg(
        total_gallons=("gallons", "sum"),
        avg_gallons_per_day=("gallons", "mean"),
        days=("gallons", "count")
    )
    out = out.sort_values("month_num").reset_index(drop=True)
    return out


# -----------------------------
# Chart helpers (clean dates + horizontal labels)
# -----------------------------
def alt_line_by_date(df: pd.DataFrame, date_col: str, value_col: str, y_title: str = "", height: int = 320):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.date
    d["date_label"] = pd.to_datetime(d[date_col].astype(str)).dt.strftime("%b %d")

    chart = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "date_label:O",
                title="Date",
                sort=list(d["date_label"]),
                axis=alt.Axis(labelAngle=0, labelOverlap="greedy"),
            ),
            y=alt.Y(f"{value_col}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("date_label:O", title="Date"),
                alt.Tooltip(f"{value_col}:Q", title="Gallons", format=".2f"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def alt_bar_weekdays(df: pd.DataFrame, height: int = 320):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("weekday:O", sort=WEEKDAY_ORDER, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("gallons:Q", title="Avg gallons/day"),
            tooltip=[alt.Tooltip("weekday:O"), alt.Tooltip("gallons:Q", format=".2f")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def alt_line_months(df: pd.DataFrame, y_col: str, y_title: str, height: int = 320):
    # month already sorted by month_num, so just provide explicit sort order
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:O", sort=MONTH_ORDER, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            tooltip=[
                alt.Tooltip("month:O"),
                alt.Tooltip("total_gallons:Q", title="Total", format=".1f"),
                alt.Tooltip("avg_gallons_per_day:Q", title="Avg/day", format=".1f"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Milk Ordering Dashboard", layout="wide")
st.title("🥛 Milk Ordering Dashboard")

with st.sidebar:
    st.header("Files")
    csv_path = st.text_input("Sales CSV", value="Jitters data.csv")
    xlsx_path = st.text_input("Ingredient sheet (XLSX)", value="Ingredient Measure.xlsx")

    st.header("Order recommendation")
    order_horizon_days = st.slider("Order for how many days?", 1, 14, 7, 1)

    lookback_days = st.slider("Use past ___ days to learn from", 7, 60, 15, 1)

    st.header("Safety")
    safety_buffer_pct = st.slider("Safety buffer (%)", 0, 30, 10, 1)

    st.header("Extra insights")
    show_weekly_pattern = st.checkbox("Show weekday vs weekend pattern", value=True)
    show_seasonal_pattern = st.checkbox("Show seasonal pattern (summer vs winter)", value=True)
    show_confidence = st.checkbox("Show forecast range (Prophet only)", value=False)

# Load data
daily = load_and_prepare_data(csv_path, xlsx_path)
if daily.empty:
    st.error("No milk data found. Check file paths and ingredient mapping.")
    st.stop()

latest_date = daily["date"].max()

# Lookback window
lookback_start = latest_date - pd.Timedelta(days=lookback_days - 1)
past_window = daily[daily["date"] >= lookback_start].copy()

# Forecast logic:
# - if lookback is very small, use stable fallback
# - else use Prophet
if lookback_days < 14:
    st.info("Short history selected — using a stable last-week pattern forecast.")
    fc = simple_weekday_forecast(past_window, periods=order_horizon_days)
    using_prophet = False
else:
    model = fit_prophet_model(past_window)
    fc = forecast_next_days_prophet(model, periods=order_horizon_days)
    using_prophet = True

# Never allow negative forecasts
fc["forecast_gallons"] = pd.to_numeric(fc["forecast_gallons"], errors="coerce").fillna(0).clip(lower=0)

# Totals
used_past_x = float(past_window["gallons"].sum())
expected_total = float(fc["forecast_gallons"].sum())
expected_with_buffer = expected_total * (1 + safety_buffer_pct / 100.0)
order_gallons = math.ceil(expected_with_buffer)

# Summary metrics
c1, c2, c3 = st.columns(3)
c1.metric(f"✅ Used in last {lookback_days} days", f"{used_past_x:.1f}")
c2.metric(f"🛒 Milk to order for next {order_horizon_days} days", f"{order_gallons}")
c3.metric(f"📈 Expected use next {order_horizon_days} days", f"{expected_total:.1f}")

st.success(f"✅ Order **{order_gallons}** to cover the next **{order_horizon_days} days**.")
st.caption(f"Based on the past **{lookback_days} days** of sales, plus a **{safety_buffer_pct}%** safety buffer.")

st.divider()

# Main charts
left, right = st.columns(2)

with left:
    st.subheader(f"Past {lookback_days} days (what happened)")
    alt_line_by_date(past_window[["date", "gallons"]], "date", "gallons")

    st.write("Daily breakdown:")
    past_table = past_window[["date", "gallons"]].copy()
    past_table["date"] = pd.to_datetime(past_table["date"]).dt.strftime("%b %d, %Y")
    past_table["gallons"] = past_table["gallons"].round(2)
    st.dataframe(past_table, use_container_width=True)

with right:
    st.subheader(f"Next {order_horizon_days} days (what we expect)")
    alt_line_by_date(fc[["date", "forecast_gallons"]], "date", "forecast_gallons")

    st.write("Expected per day:")
    fc_table = fc[["date", "forecast_gallons"]].copy()
    fc_table["date"] = pd.to_datetime(fc_table["date"]).dt.strftime("%b %d, %Y")
    fc_table = fc_table.rename(columns={"forecast_gallons": "expected"})
    fc_table["expected"] = fc_table["expected"].round(2)
    st.dataframe(fc_table, use_container_width=True)

    if show_confidence and using_prophet:
        st.write("Forecast range (Prophet only):")
        ci = fc[["date", "lower_gallons", "forecast_gallons", "upper_gallons"]].copy()
        ci["date"] = pd.to_datetime(ci["date"]).dt.strftime("%b %d, %Y")
        ci = ci.rename(columns={"lower_gallons": "low", "forecast_gallons": "expected", "upper_gallons": "high"})
        for col in ["low", "expected", "high"]:
            ci[col] = pd.to_numeric(ci[col], errors="coerce").round(2)
        st.dataframe(ci, use_container_width=True)
    elif show_confidence and not using_prophet:
        st.info("Forecast range is only available when Prophet is used (lookback ≥ 14).")

st.divider()

# Extra insights (independent)
if show_weekly_pattern or show_seasonal_pattern:
    st.subheader("Extra insights (independent of the forecast)")

extra_left, extra_right = st.columns(2)

with extra_left:
    if show_weekly_pattern:
        st.markdown("### 📅 Weekday vs weekend pattern (last 7 days)")
        weekly_df = weekly_pattern_last_7_days(daily)
        if weekly_df.empty:
            st.info("Not enough recent data to show a weekly pattern yet.")
        else:
            alt_bar_weekdays(weekly_df)
            st.caption("This chart uses only the last 7 days of actual usage.")

with extra_right:
    if show_seasonal_pattern:
        st.markdown("### 🌤️ Seasonal pattern (summer vs winter) (all available data)")
        seasonal_df = seasonal_pattern_by_month(daily)
        if seasonal_df.empty:
            st.info("Not enough data to show a seasonal pattern yet.")
        else:
            alt_line_months(seasonal_df, y_col="total_gallons", y_title="Total gallons")
            st.caption("This chart aggregates all history by month (total + avg/day in tooltip).")
