"""
Sales / Demand Forecasting
---------------------------
Builds short-term demand forecasts using Prophet (with an ARIMA baseline
for comparison) and evaluates against a held-out test window.

Usage:
    python generate_data.py
    python forecast.py --horizon 30
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

DATA_PATH = "data/daily_sales.csv"
REPORT_DIR = "reports"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def train_test_split_ts(df, horizon):
    train = df.iloc[:-horizon].copy()
    test = df.iloc[-horizon:].copy()
    return train, test


def run_prophet(train, test, horizon):
    from prophet import Prophet

    prophet_df = train.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    preds = forecast.tail(horizon)["yhat"].values
    return preds, model, forecast


def run_arima(train, test, horizon, order=(5, 1, 2)):
    model = ARIMA(train["sales"], order=order)
    fitted = model.fit()
    preds = fitted.forecast(steps=horizon)
    return np.asarray(preds)


def evaluate(name, actual, predicted, results):
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    results[name] = {"MAE": round(mae, 2), "MAPE": round(mape * 100, 2)}
    print(f"{name}: MAE={mae:.2f}, MAPE={mape * 100:.2f}%")


def plot_forecasts(train, test, prophet_preds, arima_preds, out_path):
    plt.figure(figsize=(12, 5))
    plt.plot(train["date"].tail(90), train["sales"].tail(90), label="Train (last 90d)", color="gray")
    plt.plot(test["date"], test["sales"], label="Actual", color="black", linewidth=2)
    plt.plot(test["date"], prophet_preds, label="Prophet Forecast", linestyle="--")
    plt.plot(test["date"], arima_preds, label="ARIMA Forecast", linestyle="--")
    plt.legend()
    plt.title("Demand Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(horizon=30):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run `python generate_data.py` first.")

    df = load_data()
    train, test = train_test_split_ts(df, horizon)

    results = {}

    prophet_preds, prophet_model, _ = run_prophet(train, test, horizon)
    evaluate("Prophet", test["sales"].values, prophet_preds, results)

    arima_preds = run_arima(train, test, horizon)
    evaluate("ARIMA", test["sales"].values, arima_preds, results)

    # naive baseline: repeat last observed value
    naive_preds = np.full(horizon, train["sales"].iloc[-1])
    evaluate("Naive Baseline", test["sales"].values, naive_preds, results)

    os.makedirs(REPORT_DIR, exist_ok=True)
    plot_forecasts(train, test, prophet_preds, arima_preds, f"{REPORT_DIR}/forecast_comparison.png")

    out = test[["date", "sales"]].copy()
    out["prophet_forecast"] = prophet_preds
    out["arima_forecast"] = arima_preds
    out.to_csv(f"{REPORT_DIR}/forecast_results.csv", index=False)

    import json
    with open(f"{REPORT_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    best = min(results, key=lambda k: results[k]["MAPE"])
    baseline_mape = results["Naive Baseline"]["MAPE"]
    best_mape = results[best]["MAPE"]
    improvement = (baseline_mape - best_mape) / baseline_mape * 100
    print(f"\nBest model: {best} — {improvement:.1f}% MAPE improvement over naive baseline")
    print(f"Reports saved to {REPORT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=30, help="Days to forecast/hold out")
    args = parser.parse_args()
    main(args.horizon)
