import pandas as pd

from milk_dashboard.data.splits import generate_time_splits


def test_generate_time_splits_returns_cv_and_holdout():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=70, freq="D"),
            "gallons": [float(i % 10 + 1) for i in range(70)],
        }
    )

    cv_splits, holdout = generate_time_splits(
        df,
        holdout_days=14,
        min_train_days=30,
        horizon_days=7,
        step_days=7,
        max_splits=4,
    )

    assert len(holdout.train) == 56
    assert len(holdout.test) == 14
    assert len(cv_splits) > 0
    assert len(cv_splits) <= 4
    assert all(len(s.test) == 7 for s in cv_splits)

