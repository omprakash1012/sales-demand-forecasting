import pandas as pd

from generate_data import generate_sales


def test_generate_sales_returns_requested_row_count():
    df = generate_sales(periods=200)
    assert len(df) == 200


def test_generate_sales_has_expected_columns():
    df = generate_sales(periods=50)
    assert set(df.columns) == {"date", "sales", "is_promo"}


def test_sales_values_are_never_below_floor():
    df = generate_sales(periods=500)
    assert df["sales"].min() >= 10


def test_is_promo_is_binary():
    df = generate_sales(periods=500)
    assert set(df["is_promo"].unique()).issubset({0, 1})


def test_dates_are_daily_and_sequential():
    df = generate_sales(start="2023-01-01", periods=30)
    diffs = df["date"].diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all()
