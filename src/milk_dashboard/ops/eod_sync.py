"""Square EOD sync helpers.

This module supports:
- loading the latest OAuth token from JSONL backup,
- pulling completed Square orders for a time window,
- converting line-items to the local Jitters sales CSV schema,
- appending only new rows,
- detecting ingredient mapping gaps (especially coffee items),
- storing sync run metadata in a lightweight sqlite database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests


SQUARE_API_VERSION = "2026-01-22"

CSV_COLUMNS = [
    "Date",
    "Time",
    "Time Zone",
    "Category",
    "Item",
    "Qty",
    "Price Point Name",
    "SKU",
    "Modifiers Applied",
    "Gross Sales",
    "Discounts",
    "Net Sales",
    "Tax",
    "Transaction ID",
    "Payment ID",
    "Device Name",
    "Notes",
    "Details",
    "Event Type",
    "Location",
    "Dining Option",
    "Customer ID",
    "Customer Name",
    "Customer Reference ID",
    "Unit",
    "Count",
    "Itemization Type",
    "Commission",
    "Employee",
    "Fulfillment Note",
    "Channel",
    "Token",
    "Card Brand",
    "PAN Suffix",
]

COFFEE_KEYWORDS = {
    "coffee",
    "espresso",
    "latte",
    "cappuccino",
    "americano",
    "mocha",
    "macchiato",
    "cortado",
    "flat white",
    "cold brew",
    "drip",
    "pourover",
    "pour over",
    "shot",
}
FOOD_KEYWORDS = {
    "sandwich",
    "bagel",
    "muffin",
    "cookie",
    "croissant",
    "scone",
    "salad",
    "toast",
    "wrap",
    "cake",
    "brownie",
}


@dataclass
class EODSyncConfig:
    sales_path: Path
    ingredients_path: Path
    token_backup_path: Path
    sqlite_path: Path
    square_env: str = "production"
    local_timezone: str = "America/New_York"
    default_lookback_days: int = 7
    client_id: str | None = None
    client_secret: str | None = None
    token_record: dict[str, Any] | None = None
    token_refresh_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _square_base_url(square_env: str) -> str:
    env = (square_env or "production").strip().lower()
    if env == "sandbox":
        return "https://connect.squareupsandbox.com"
    return "https://connect.squareup.com"


def _format_money(amount: float) -> str:
    return f"${amount:.2f}"


def _money_amount(money_obj: Any) -> float:
    if not isinstance(money_obj, dict):
        return 0.0
    amount = money_obj.get("amount")
    if amount is None:
        return 0.0
    try:
        return float(amount) / 100.0
    except (TypeError, ValueError):
        return 0.0


def _parse_iso_timestamp(ts: str) -> datetime:
    out = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if out.tzinfo is None:
        return out.replace(tzinfo=timezone.utc)
    return out


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_latest_token_record(token_backup_path: Path, merchant_id: str | None = None) -> dict[str, Any]:
    records = _load_json_lines(token_backup_path)
    if not records:
        raise ValueError(f"No token backup records found: {token_backup_path}")

    filtered = [r for r in records if r.get("access_token")]
    if merchant_id:
        filtered = [r for r in filtered if str(r.get("merchant_id", "")).strip() == merchant_id.strip()]
    if not filtered:
        scope = f"merchant_id={merchant_id}" if merchant_id else "any merchant"
        raise ValueError(f"No access token record found for {scope}.")
    return filtered[-1]


def refresh_access_token(
    *,
    token_record: dict[str, Any],
    square_base_url: str,
    client_id: str | None,
    client_secret: str | None,
    token_backup_path: Path,
) -> dict[str, Any]:
    refresh_token = str(token_record.get("refresh_token") or "").strip()
    if not refresh_token:
        raise ValueError("Cannot refresh token: refresh_token missing from backup record.")
    if not client_id or not client_secret:
        raise ValueError("Cannot refresh token: set SQUARE_CLIENT_ID and SQUARE_CLIENT_SECRET.")

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    headers = {
        "Square-Version": SQUARE_API_VERSION,
        "Content-Type": "application/json",
    }
    url = f"{square_base_url}/oauth2/token"
    resp = requests.post(url, json=payload, headers=headers, timeout=45)
    data = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        raise RuntimeError(f"Token refresh failed: {data}")

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": "refresh_token",
        "merchant_id": data.get("merchant_id") or token_record.get("merchant_id"),
        "square_env": token_record.get("square_env", "production"),
        "redirect_uri": token_record.get("redirect_uri"),
        "token_type": data.get("token_type"),
        "scope": data.get("scope"),
        "expires_at": data.get("expires_at"),
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
    }
    token_backup_path.parent.mkdir(parents=True, exist_ok=True)
    with token_backup_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=True) + "\n")
    return out


def refresh_access_token_inline(
    *,
    token_record: dict[str, Any],
    square_base_url: str,
    client_id: str | None,
    client_secret: str | None,
) -> dict[str, Any]:
    refresh_token = str(token_record.get("refresh_token") or "").strip()
    if not refresh_token:
        raise ValueError("Cannot refresh token: refresh_token missing from token record.")
    if not client_id or not client_secret:
        raise ValueError("Cannot refresh token: set SQUARE_CLIENT_ID and SQUARE_CLIENT_SECRET.")

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    headers = {
        "Square-Version": SQUARE_API_VERSION,
        "Content-Type": "application/json",
    }
    url = f"{square_base_url}/oauth2/token"
    resp = requests.post(url, json=payload, headers=headers, timeout=45)
    data = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        raise RuntimeError(f"Token refresh failed: {data}")
    return {
        "merchant_id": data.get("merchant_id") or token_record.get("merchant_id"),
        "square_env": token_record.get("square_env", "production"),
        "redirect_uri": token_record.get("redirect_uri"),
        "token_type": data.get("token_type"),
        "scope": data.get("scope"),
        "expires_at": data.get("expires_at"),
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token") or token_record.get("refresh_token"),
    }


def _square_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Square-Version": SQUARE_API_VERSION,
        "Content-Type": "application/json",
    }


def _fetch_locations(access_token: str, square_base_url: str) -> dict[str, str]:
    url = f"{square_base_url}/v2/locations"
    resp = requests.get(url, headers=_square_headers(access_token), timeout=45)
    if resp.status_code >= 400:
        return {}
    data = resp.json() if resp.content else {}
    locations = data.get("locations") or []
    out: dict[str, str] = {}
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        loc_id = str(loc.get("id") or "").strip()
        if not loc_id:
            continue
        out[loc_id] = str(loc.get("name") or loc.get("business_name") or "").strip()
    return out


def fetch_completed_orders(
    *,
    access_token: str,
    square_base_url: str,
    start_at: datetime,
    end_at: datetime,
    location_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    url = f"{square_base_url}/v2/orders/search"
    headers = _square_headers(access_token)
    start_iso = start_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    end_iso = end_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    all_orders: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        payload: dict[str, Any] = {
            "limit": 500,
            "query": {
                "filter": {
                    "date_time_filter": {
                        "created_at": {
                            "start_at": start_iso,
                            "end_at": end_iso,
                        }
                    },
                    "state_filter": {"states": ["COMPLETED"]},
                },
                "sort": {"sort_field": "CREATED_AT", "sort_order": "ASC"},
            },
            "return_entries": False,
        }
        if location_ids:
            payload["location_ids"] = location_ids
        if cursor:
            payload["cursor"] = cursor

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            raise RuntimeError(f"Square orders/search failed: {data}")
        orders = data.get("orders") or []
        if orders:
            all_orders.extend([o for o in orders if isinstance(o, dict)])
        cursor = data.get("cursor")
        if not cursor:
            break

    return all_orders


def _category_maps(existing_sales: pd.DataFrame) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    if existing_sales.empty:
        return {}, {}

    sales = existing_sales.copy()
    sales.columns = sales.columns.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    needed = {"category", "item", "price_point_name"}
    if not needed.issubset(set(sales.columns)):
        return {}, {}

    sales["category"] = sales["category"].fillna("").astype(str)
    sales["item"] = sales["item"].fillna("").astype(str)
    sales["price_point_name"] = sales["price_point_name"].fillna("").astype(str)

    pair = (
        sales[sales["item"] != ""]
        .groupby(["item", "price_point_name", "category"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    pair_map: dict[tuple[str, str], str] = {}
    for _, row in pair.iterrows():
        key = (_normalize_text(row["item"]), _normalize_text(row["price_point_name"]))
        if key in pair_map:
            continue
        pair_map[key] = str(row["category"]).strip()

    item = (
        sales[sales["item"] != ""]
        .groupby(["item", "category"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    item_map: dict[str, str] = {}
    for _, row in item.iterrows():
        key = _normalize_text(row["item"])
        if key in item_map:
            continue
        item_map[key] = str(row["category"]).strip()

    return pair_map, item_map


def _infer_category(
    *,
    item: str,
    variation: str,
    pair_map: dict[tuple[str, str], str],
    item_map: dict[str, str],
) -> str:
    item_key = _normalize_text(item)
    variation_key = _normalize_text(variation)
    if (item_key, variation_key) in pair_map:
        return pair_map[(item_key, variation_key)]
    if item_key in item_map:
        return item_map[item_key]

    joined = f"{item_key} {variation_key}"
    if any(k in joined for k in COFFEE_KEYWORDS):
        return "Coffee"
    if any(k in joined for k in FOOD_KEYWORDS):
        return "Food"
    return "Uncategorized"


def _to_sync_rows(
    *,
    orders: list[dict[str, Any]],
    existing_sales: pd.DataFrame,
    local_timezone: str,
    location_names: dict[str, str],
) -> pd.DataFrame:
    pair_map, item_map = _category_maps(existing_sales)
    tz = ZoneInfo(local_timezone)

    rows: list[dict[str, Any]] = []
    for order in orders:
        line_items = order.get("line_items") or []
        if not isinstance(line_items, list) or not line_items:
            continue

        order_id = str(order.get("id") or "").strip()
        created_raw = str(order.get("created_at") or "").strip()
        if not created_raw:
            continue
        try:
            created_utc = _parse_iso_timestamp(created_raw).astimezone(timezone.utc)
            local_dt = created_utc.astimezone(tz)
        except Exception:
            continue

        location_id = str(order.get("location_id") or "").strip()
        location_name = location_names.get(location_id, location_id)
        customer_id = str(order.get("customer_id") or "").strip()

        tender_payment_id = ""
        tenders = order.get("tenders") or []
        if isinstance(tenders, list) and tenders:
            first_tender = tenders[0] if isinstance(tenders[0], dict) else {}
            tender_payment_id = str(first_tender.get("payment_id") or "").strip()

        for line in line_items:
            if not isinstance(line, dict):
                continue

            item = str(line.get("name") or "").strip()
            variation = str(line.get("variation_name") or "").strip()
            if not item:
                continue

            qty_raw = line.get("quantity")
            try:
                qty = float(qty_raw) if qty_raw is not None else 0.0
            except (TypeError, ValueError):
                qty = 0.0
            if qty <= 0:
                continue

            category = _infer_category(item=item, variation=variation, pair_map=pair_map, item_map=item_map)
            gross = _money_amount(line.get("gross_sales_money"))
            discount = _money_amount(line.get("total_discount_money"))
            net = _money_amount(line.get("total_money"))
            tax = _money_amount(line.get("total_tax_money"))

            details_url = ""
            if order_id:
                details_url = f"https://app.squareup.com/dashboard/sales/transactions/{order_id}"

            row = {col: "" for col in CSV_COLUMNS}
            row.update(
                {
                    "Date": local_dt.strftime("%Y-%m-%d"),
                    "Time": local_dt.strftime("%H:%M:%S"),
                    "Time Zone": str(local_dt.tzinfo),
                    "Category": category,
                    "Item": item,
                    "Qty": qty,
                    "Price Point Name": variation,
                    "Gross Sales": _format_money(gross),
                    "Discounts": _format_money(discount),
                    "Net Sales": _format_money(net),
                    "Tax": _format_money(tax),
                    "Transaction ID": order_id,
                    "Payment ID": tender_payment_id,
                    "Details": details_url,
                    "Event Type": "Payment",
                    "Location": location_name,
                    "Dining Option": str(order.get("fulfillment_type") or ""),
                    "Customer ID": customer_id,
                    "Unit": "ea",
                    "Count": int(round(qty)) if abs(qty - round(qty)) < 1e-9 else qty,
                    "Itemization Type": str(line.get("item_type") or "Physical Good"),
                    "Channel": "Square API Sync",
                }
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def _ensure_sales_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CSV_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[CSV_COLUMNS]


def _dedupe_subset(columns: list[str]) -> list[str]:
    preferred = ["Transaction ID", "Item", "Price Point Name", "Qty", "Date", "Time"]
    return [c for c in preferred if c in columns]


def _latest_existing_datetime(existing_sales: pd.DataFrame, local_timezone: str) -> datetime | None:
    if existing_sales.empty:
        return None
    sales = existing_sales.copy()
    cols = {c.lower(): c for c in sales.columns}
    if "date" not in cols:
        return None
    date_col = cols["date"]
    if "time" in cols:
        dt_series = pd.to_datetime(
            sales[date_col].astype(str).str.strip() + " " + sales[cols["time"]].astype(str).str.strip(),
            errors="coerce",
        )
    else:
        dt_series = pd.to_datetime(sales[date_col], errors="coerce")
    dt_series = dt_series.dropna()
    if dt_series.empty:
        return None

    tz = ZoneInfo(local_timezone)
    latest_naive = dt_series.max().to_pydatetime()
    if latest_naive.tzinfo is None:
        return latest_naive.replace(tzinfo=tz)
    return latest_naive.astimezone(tz)


def append_rows_to_sales_csv(
    *,
    sales_path: Path,
    new_rows: pd.DataFrame,
    dry_run: bool = False,
) -> tuple[int, int, pd.DataFrame]:
    if not sales_path.exists():
        raise FileNotFoundError(f"Sales CSV not found: {sales_path}")

    existing = pd.read_csv(sales_path, encoding="ISO-8859-1", low_memory=False)
    existing = _ensure_sales_columns(existing)
    incoming = _ensure_sales_columns(new_rows)

    if incoming.empty:
        return 0, int(len(existing)), existing

    combined = pd.concat([existing, incoming], ignore_index=True)
    dedupe_cols = _dedupe_subset(combined.columns.tolist())
    if dedupe_cols:
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="first")
    else:
        combined = combined.drop_duplicates(keep="first")

    appended_count = max(0, len(combined) - len(existing))
    if not dry_run and appended_count > 0:
        combined.to_csv(sales_path, index=False, encoding="ISO-8859-1")
    return int(appended_count), int(len(combined)), combined


def detect_new_item_gaps(new_rows: pd.DataFrame, ingredients_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if new_rows.empty:
        return {"priority_alerts": [], "non_priority_new_items": []}

    ing = ingredients_df.copy()
    ing.columns = ing.columns.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    for col in ("item", "price_point_name", "category"):
        if col not in ing.columns:
            ing[col] = ""

    ing["item_key"] = ing["item"].map(_normalize_text)
    ing["price_key"] = ing["price_point_name"].map(_normalize_text)
    ing["category_key"] = ing["category"].map(_normalize_text)
    ingredient_categories = set(ing["category_key"].dropna().tolist())

    ing_pairs = set(zip(ing["item_key"], ing["price_key"]))
    ing_items = set(ing["item_key"])

    rows = new_rows.copy()
    rows["item_key"] = rows["Item"].map(_normalize_text)
    rows["price_key"] = rows["Price Point Name"].map(_normalize_text)
    rows["category_key"] = rows["Category"].map(_normalize_text)
    rows["qty_numeric"] = pd.to_numeric(rows["Qty"], errors="coerce").fillna(0.0)

    missing = rows[
        (~rows[["item_key", "price_key"]].apply(tuple, axis=1).isin(ing_pairs)) & (~rows["item_key"].isin(ing_items))
    ].copy()
    if missing.empty:
        return {"priority_alerts": [], "non_priority_new_items": []}

    grouped = (
        missing.groupby(["Category", "Item", "Price Point Name"], as_index=False)
        .agg(total_qty=("qty_numeric", "sum"), rows=("item_key", "count"))
        .sort_values(["total_qty", "rows"], ascending=False)
    )

    priority_alerts: list[dict[str, Any]] = []
    non_priority_new_items: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        item = str(row["Item"]).strip()
        price_point = str(row["Price Point Name"]).strip()
        category = str(row["Category"]).strip()
        is_priority = _normalize_text(category) in ingredient_categories
        record = {
            "category": category,
            "item": item,
            "price_point_name": price_point,
            "total_qty": float(row["total_qty"]),
            "rows": int(row["rows"]),
            "suggested_action": (
                "Priority: add this item to Ingredient Measure before next EOD run."
                if is_priority
                else "Optional mapping update."
            ),
        }
        if is_priority:
            priority_alerts.append(record)
        else:
            non_priority_new_items.append(record)

    return {
        "priority_alerts": priority_alerts,
        "non_priority_new_items": non_priority_new_items,
    }


def _ensure_sync_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts_utc TEXT NOT NULL,
            merchant_id TEXT,
            square_env TEXT,
            start_at_utc TEXT,
            end_at_utc TEXT,
            fetched_orders INTEGER NOT NULL,
            fetched_rows INTEGER NOT NULL,
            appended_rows INTEGER NOT NULL,
            total_rows_after INTEGER NOT NULL,
            dry_run INTEGER NOT NULL,
            status TEXT NOT NULL,
            details_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            alert_type TEXT NOT NULL,
            category TEXT,
            item TEXT,
            price_point_name TEXT,
            total_qty REAL,
            rows_count INTEGER,
            suggested_action TEXT,
            FOREIGN KEY(run_id) REFERENCES sync_runs(id)
        )
        """
    )
    conn.commit()


def persist_sync_run(
    *,
    sqlite_path: Path,
    run_payload: dict[str, Any],
    alerts: dict[str, list[dict[str, Any]]],
) -> int:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        _ensure_sync_tables(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sync_runs (
                run_ts_utc, merchant_id, square_env, start_at_utc, end_at_utc,
                fetched_orders, fetched_rows, appended_rows, total_rows_after,
                dry_run, status, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_payload.get("run_ts_utc"),
                run_payload.get("merchant_id"),
                run_payload.get("square_env"),
                run_payload.get("start_at_utc"),
                run_payload.get("end_at_utc"),
                int(run_payload.get("fetched_orders", 0)),
                int(run_payload.get("fetched_rows", 0)),
                int(run_payload.get("appended_rows", 0)),
                int(run_payload.get("total_rows_after", 0)),
                int(bool(run_payload.get("dry_run"))),
                str(run_payload.get("status", "ok")),
                json.dumps(run_payload.get("details", {}), ensure_ascii=True),
            ),
        )
        run_id = int(cur.lastrowid)

        for alert_type, rows in alerts.items():
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO sync_alerts (
                        run_id, alert_type, category, item, price_point_name,
                        total_qty, rows_count, suggested_action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        str(alert_type),
                        str(row.get("category") or ""),
                        str(row.get("item") or ""),
                        str(row.get("price_point_name") or ""),
                        float(row.get("total_qty") or 0.0),
                        int(row.get("rows") or 0),
                        str(row.get("suggested_action") or ""),
                    ),
                )
        conn.commit()
        return run_id


def load_latest_sync_run(sqlite_path: Path, merchant_id: str | None = None) -> dict[str, Any] | None:
    if not sqlite_path.exists():
        return None
    with sqlite3.connect(sqlite_path) as conn:
        _ensure_sync_tables(conn)
        conn.row_factory = sqlite3.Row
        if merchant_id:
            row = conn.execute(
                "SELECT * FROM sync_runs WHERE merchant_id = ? ORDER BY id DESC LIMIT 1",
                (merchant_id,),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM sync_runs ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None

        run_id = int(row["id"])
        alerts_rows = conn.execute(
            "SELECT * FROM sync_alerts WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        alerts: dict[str, list[dict[str, Any]]] = {
            "priority_alerts": [],
            "non_priority_new_items": [],
            # Backward compatibility with older run records:
            "coffee_alerts": [],
            "non_coffee_new_items": [],
        }
        for ar in alerts_rows:
            rec = {
                "category": ar["category"],
                "item": ar["item"],
                "price_point_name": ar["price_point_name"],
                "total_qty": ar["total_qty"],
                "rows": ar["rows_count"],
                "suggested_action": ar["suggested_action"],
            }
            key = str(ar["alert_type"])
            alerts.setdefault(key, []).append(rec)

        details_json = row["details_json"]
        details: dict[str, Any] = {}
        if details_json:
            try:
                details = json.loads(details_json)
            except json.JSONDecodeError:
                details = {}
        return {
            "run_id": run_id,
            "run_ts_utc": row["run_ts_utc"],
            "merchant_id": row["merchant_id"],
            "square_env": row["square_env"],
            "start_at_utc": row["start_at_utc"],
            "end_at_utc": row["end_at_utc"],
            "fetched_orders": row["fetched_orders"],
            "fetched_rows": row["fetched_rows"],
            "appended_rows": row["appended_rows"],
            "total_rows_after": row["total_rows_after"],
            "dry_run": bool(row["dry_run"]),
            "status": row["status"],
            "details": details,
            "alerts": alerts,
        }


def run_eod_sync(
    *,
    config: EODSyncConfig,
    merchant_id: str | None = None,
    dry_run: bool = False,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> dict[str, Any]:
    square_base = _square_base_url(config.square_env)
    token_record = config.token_record or load_latest_token_record(config.token_backup_path, merchant_id=merchant_id)
    access_token = str(token_record.get("access_token") or "").strip()
    if not access_token:
        raise ValueError("No access token available.")

    existing_sales = pd.read_csv(config.sales_path, encoding="ISO-8859-1", low_memory=False)
    existing_sales = _ensure_sales_columns(existing_sales)
    ingredients = pd.read_excel(config.ingredients_path)

    now_utc = datetime.now(timezone.utc)
    if end_at is None:
        end_at = now_utc

    if start_at is None:
        latest_local = _latest_existing_datetime(existing_sales, config.local_timezone)
        if latest_local is not None:
            start_at = latest_local.astimezone(timezone.utc) + timedelta(seconds=1)
        else:
            start_at = now_utc - timedelta(days=config.default_lookback_days)

    if start_at >= end_at:
        raise ValueError("start_at must be earlier than end_at.")

    location_names = _fetch_locations(access_token, square_base)
    location_ids = [loc_id for loc_id in location_names.keys() if str(loc_id).strip()]

    def _pull_with_token(token: str) -> list[dict[str, Any]]:
        return fetch_completed_orders(
            access_token=token,
            square_base_url=square_base,
            start_at=start_at,
            end_at=end_at,
            location_ids=location_ids,
        )

    try:
        orders = _pull_with_token(access_token)
    except RuntimeError as exc:
        msg = str(exc)
        if "UNAUTHORIZED" not in msg and "AUTHENTICATION_ERROR" not in msg:
            raise
        if config.token_refresh_callback is not None:
            refreshed = config.token_refresh_callback(token_record)
        elif config.token_record is not None:
            refreshed = refresh_access_token_inline(
                token_record=token_record,
                square_base_url=square_base,
                client_id=config.client_id,
                client_secret=config.client_secret,
            )
        else:
            refreshed = refresh_access_token(
                token_record=token_record,
                square_base_url=square_base,
                client_id=config.client_id,
                client_secret=config.client_secret,
                token_backup_path=config.token_backup_path,
            )
        refreshed_token = str(refreshed.get("access_token") or "").strip()
        if not refreshed_token:
            raise ValueError("Token refresh succeeded but no access token returned.")
        token_record = refreshed
        access_token = refreshed_token
        location_names = _fetch_locations(access_token, square_base)
        location_ids = [loc_id for loc_id in location_names.keys() if str(loc_id).strip()]
        orders = _pull_with_token(access_token)
    sync_rows = _to_sync_rows(
        orders=orders,
        existing_sales=existing_sales,
        local_timezone=config.local_timezone,
        location_names=location_names,
    )
    appended_rows, total_rows_after, combined = append_rows_to_sales_csv(
        sales_path=config.sales_path,
        new_rows=sync_rows,
        dry_run=dry_run,
    )

    if appended_rows > 0:
        added_slice = combined.tail(appended_rows).copy()
    else:
        added_slice = pd.DataFrame(columns=CSV_COLUMNS)
    alerts = detect_new_item_gaps(added_slice, ingredients)

    payload = {
        "run_ts_utc": datetime.now(timezone.utc).isoformat(),
        "merchant_id": merchant_id or token_record.get("merchant_id"),
        "square_env": config.square_env,
        "start_at_utc": start_at.astimezone(timezone.utc).isoformat(),
        "end_at_utc": end_at.astimezone(timezone.utc).isoformat(),
        "fetched_orders": int(len(orders)),
        "fetched_rows": int(len(sync_rows)),
        "appended_rows": int(appended_rows),
        "total_rows_after": int(total_rows_after),
        "dry_run": bool(dry_run),
        "status": "ok",
        "details": {
            "sales_path": str(config.sales_path),
            "ingredients_path": str(config.ingredients_path),
            "token_backup_path": str(config.token_backup_path),
            "token_source": "inline" if config.token_record is not None else "backup_file",
            "timezone": config.local_timezone,
            "priority_alert_count": len(alerts.get("priority_alerts") or []),
            "new_non_priority_item_count": len(alerts.get("non_priority_new_items") or []),
        },
    }
    run_id = persist_sync_run(sqlite_path=config.sqlite_path, run_payload=payload, alerts=alerts)
    return {
        "run_id": run_id,
        **payload,
        "alerts": alerts,
    }
