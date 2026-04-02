"""Reusable cleaning and feature preparation pipeline."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from milk_dashboard.constants import OZ_PER_GALLON


MONEY_COLUMNS = ("gross_sales", "discounts", "net_sales", "tax")
MERGE_KEYS = ("category", "item", "price_point_name")
NON_ESSENTIAL_COLUMNS = (
    "time_zone",
    "sku",
    "transaction_id",
    "payment_id",
    "device_name",
    "details",
    "location",
    "dining_option",
    "customer_id",
    "customer_name",
    "customer_reference_id",
    "unit",
    "commission",
    "employee",
    "fulfillment_note",
    "channel",
    "token",
    "card_brand",
    "pan_suffix",
)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return out


def parse_money_columns(df: pd.DataFrame, money_columns: Iterable[str] = MONEY_COLUMNS) -> pd.DataFrame:
    out = df.copy()
    for col in money_columns:
        if col in out.columns:
            out[col] = out[col].replace(r"[$,]", "", regex=True)
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def attach_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "date" in out.columns and "time" in out.columns:
        out["datetime"] = pd.to_datetime(
            out["date"].astype(str) + " " + out["time"].astype(str),
            errors="coerce",
        )
    else:
        out["datetime"] = pd.to_datetime(out.get("date"), errors="coerce")
    return out


def remove_voided_items(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "item" not in out.columns:
        return out
    return out[~out["item"].astype(str).str.contains(r"\(voided\)", case=False, na=False)]


def drop_non_essential_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    return out.drop(columns=[c for c in NON_ESSENTIAL_COLUMNS if c in out.columns], errors="ignore")


def prepare_ingredient_table(ingredients_df: pd.DataFrame) -> pd.DataFrame:
    ing = standardize_column_names(ingredients_df)
    ing = ing.drop(columns=["unnamed:_8", "unnamed:_7", "unnamed:_6"], errors="ignore")

    required = set(MERGE_KEYS)
    missing = required.difference(set(ing.columns))
    if missing:
        raise ValueError(f"Ingredient table missing required columns: {sorted(missing)}")

    if "milk_(oz)" not in ing.columns and "milk_oz" not in ing.columns and "milk_(oz)." not in ing.columns:
        # Keep compatibility with the original spreadsheet heading.
        possible = [c for c in ing.columns if "milk" in c]
        if possible:
            ing = ing.rename(columns={possible[0]: "milk_(oz)"})
        else:
            ing["milk_(oz)"] = 0.0

    milk_col = "milk_(oz)" if "milk_(oz)" in ing.columns else "milk_oz"
    ing[milk_col] = pd.to_numeric(ing[milk_col], errors="coerce").fillna(0.0)

    return ing[list(MERGE_KEYS) + [milk_col]].rename(columns={milk_col: "milk_oz"})


def merge_sales_and_ingredients(
    sales_df: pd.DataFrame,
    ingredients_df: pd.DataFrame,
    apply_qty_multiplier: bool = False,
) -> pd.DataFrame:
    sales = sales_df.copy()
    ingredients = prepare_ingredient_table(ingredients_df)

    required_sales = {"item", "qty", "datetime", *MERGE_KEYS}
    missing_sales = required_sales.difference(set(sales.columns))
    if missing_sales:
        raise ValueError(f"Sales table missing required columns: {sorted(missing_sales)}")

    merged = sales.merge(ingredients, on=list(MERGE_KEYS), how="left")
    merged["milk_oz"] = pd.to_numeric(merged["milk_oz"], errors="coerce").fillna(0.0)
    merged["qty"] = pd.to_numeric(merged["qty"], errors="coerce")
    merged = merged.dropna(subset=["item", "qty"])

    if apply_qty_multiplier:
        merged["milk_oz_total"] = merged["milk_oz"] * merged["qty"]
    else:
        # DataImport_EDA parity: aggregate milk mapping value per transaction row.
        merged["milk_oz_total"] = merged["milk_oz"]
    return merged


def aggregate_daily_milk_usage(merged_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        merged_df.groupby(merged_df["datetime"].dt.date)["milk_oz_total"]
        .sum()
        .reset_index()
        .rename(columns={"milk_oz_total": "total_milk_oz", "datetime": "date"})
    )
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["gallons"] = out["total_milk_oz"] / OZ_PER_GALLON
    return out


def ensure_daily_frequency(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return daily_df.copy()

    out = daily_df.copy().sort_values("date")
    out = out.set_index("date").asfreq("D")
    out["total_milk_oz"] = out["total_milk_oz"].fillna(0.0)
    out["gallons"] = out["gallons"].fillna(0.0)
    return out.reset_index()


def build_schema_metadata(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    schema: dict[str, dict[str, str]] = {}
    for col, dtype in df.dtypes.items():
        schema[str(col)] = {"dtype": str(dtype)}
    return schema


def prepare_daily_data(
    sales_df: pd.DataFrame,
    ingredients_df: pd.DataFrame,
    fill_missing_days: bool = False,
    apply_qty_multiplier: bool = False,
    drop_non_essential: bool = True,
) -> pd.DataFrame:
    sales = standardize_column_names(sales_df)
    sales = parse_money_columns(sales)
    sales = attach_datetime_columns(sales)
    sales = remove_voided_items(sales)
    if drop_non_essential:
        sales = drop_non_essential_columns(sales)
    sales = sales.dropna(subset=["item", "qty"])

    merged = merge_sales_and_ingredients(
        sales,
        ingredients_df,
        apply_qty_multiplier=apply_qty_multiplier,
    )
    daily = aggregate_daily_milk_usage(merged)

    if fill_missing_days:
        daily = ensure_daily_frequency(daily)
    return daily
