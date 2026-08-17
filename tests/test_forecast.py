import numpy as np
import pytest

from forecast import evaluate, run_arima, train_test_split_ts
from generate_data import generate_sales


def test_train_test_split_ts_sizes():
    df = generate_sales(periods=120)
    train, test = train_test_split_ts(df, horizon=20)
    assert len(train) == 100
    assert len(test) == 20


def test_train_test_split_ts_is_chronological():
    df = generate_sales(periods=120)
    train, test = train_test_split_ts(df, horizon=20)
    assert train["date"].max() < test["date"].min()


def test_evaluate_computes_correct_mae_and_mape():
    results = {}
    actual = np.array([100, 200, 300, 400])
    predicted = np.array([110, 190, 300, 380])

    evaluate("Test Model", actual, predicted, results)

    expected_mae = np.mean(np.abs(actual - predicted))
    expected_mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    assert results["Test Model"]["MAE"] == round(expected_mae, 2)
    assert results["Test Model"]["MAPE"] == round(expected_mape, 2)


def test_evaluate_zero_error_gives_zero_mae_and_mape():
    results = {}
    actual = np.array([50, 60, 70])
    evaluate("Perfect", actual, actual.copy(), results)
    assert results["Perfect"]["MAE"] == 0.0
    assert results["Perfect"]["MAPE"] == 0.0


def test_run_arima_returns_forecast_of_requested_length():
    df = generate_sales(periods=150)
    train, test = train_test_split_ts(df, horizon=10)

    preds = run_arima(train, test, horizon=10, order=(1, 1, 1))

    assert len(preds) == 10
    assert np.all(np.isfinite(preds))
