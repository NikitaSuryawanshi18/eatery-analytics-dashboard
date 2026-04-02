from pathlib import Path

import pandas as pd

from milk_dashboard.pipeline import build_daily_from_cleaned_dataframe, prepare_from_cleaned_file


def test_build_daily_from_cleaned_dataframe_row_level():
    df = pd.DataFrame(
        {
            "datetime": ["2026-01-01 09:00:00", "2026-01-01 10:00:00", "2026-01-02 11:00:00"],
            "milk_oz_total": [10, 20, 15],
        }
    )
    out = build_daily_from_cleaned_dataframe(df)
    assert list(out.columns) == ["date", "total_milk_oz", "gallons"]
    assert len(out) == 2
    assert float(out.loc[0, "total_milk_oz"]) == 30.0


def test_prepare_from_cleaned_file_daily_csv(tmp_path: Path):
    src = tmp_path / "cleaned_daily.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "total_milk_oz": [128.0, 256.0],
            "gallons": [1.0, 2.0],
        }
    ).to_csv(src, index=False)

    prepared = prepare_from_cleaned_file(str(src), source_id="cleaned_test")
    assert prepared.source_id == "cleaned_test"
    assert len(prepared.daily_df) == 2
    assert float(prepared.daily_df.loc[1, "gallons"]) == 2.0

