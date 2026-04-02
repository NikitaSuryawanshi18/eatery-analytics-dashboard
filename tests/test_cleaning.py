import pandas as pd

from milk_dashboard.data.cleaning import prepare_daily_data


def test_prepare_daily_data_computes_expected_gallons():
    sales = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "Time": ["08:00:00", "09:00:00", "08:30:00"],
            "Category": ["Espresso", "Espresso", "Espresso"],
            "Item": ["Latte", "Latte", "Latte (Voided)"],
            "Qty": [1, 2, 1],
            "Price Point Name": ["Medium", "Medium", "Medium"],
            "Gross Sales": ["$5.00", "$10.00", "$5.00"],
            "Discounts": ["$0.00", "$0.00", "$0.00"],
            "Net Sales": ["$5.00", "$10.00", "$5.00"],
            "Tax": ["$0.35", "$0.70", "$0.35"],
        }
    )
    ingredients = pd.DataFrame(
        {
            "category": ["Espresso"],
            "item": ["Latte"],
            "price_point_name": ["Medium"],
            "Milk (oz)": [10],
        }
    )

    daily = prepare_daily_data(sales, ingredients, fill_missing_days=False)

    assert list(daily.columns) == ["date", "total_milk_oz", "gallons"]
    assert len(daily) == 1
    # DataImport_EDA parity default: sum mapped milk by row, without qty scaling.
    assert float(daily.loc[0, "total_milk_oz"]) == 20.0
    assert float(daily.loc[0, "gallons"]) == 20.0 / 128.0


def test_prepare_daily_data_with_qty_multiplier():
    sales = pd.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-01"],
            "Time": ["08:00:00", "09:00:00"],
            "Category": ["Espresso", "Espresso"],
            "Item": ["Latte", "Latte"],
            "Qty": [1, 2],
            "Price Point Name": ["Medium", "Medium"],
        }
    )
    ingredients = pd.DataFrame(
        {
            "category": ["Espresso"],
            "item": ["Latte"],
            "price_point_name": ["Medium"],
            "Milk (oz)": [10],
        }
    )

    daily = prepare_daily_data(
        sales,
        ingredients,
        fill_missing_days=False,
        apply_qty_multiplier=True,
    )
    assert float(daily.loc[0, "total_milk_oz"]) == 30.0
