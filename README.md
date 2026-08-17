# Sales / Demand Forecasting

![CI](https://github.com/omprakash1012/sales-demand-forecasting/actions/workflows/ci.yml/badge.svg)

Short-term demand forecasting to help optimize inventory and staffing
decisions, comparing Prophet and ARIMA against a naive baseline.

**Stack:** Python · Pandas · Prophet · statsmodels (ARIMA) · Power BI · pytest · GitHub Actions

## Problem

The business needed better short-term demand forecasts to optimize
inventory and staffing, rather than relying on flat, rule-of-thumb
projections.

## Approach

1. Built a daily sales time series with trend, weekly/yearly seasonality,
   and promotional spikes.
2. Trained a **Prophet** model (yearly + weekly seasonality, tuned
   `changepoint_prior_scale`) and an **ARIMA(5,1,2)** baseline.
3. Held out the last 30 days as a test window and evaluated both models
   against a naive "repeat last value" baseline using MAE and MAPE.
4. Exported a BI-ready CSV (`build_dashboard.py`) with rolling averages,
   for visualization in Power BI/Tableau.

## Results

| Model | MAPE |
|---|---|
| Naive Baseline | 8.46% |
| ARIMA(5,1,2) | 8.08% |
| **Prophet (best)** | **3.91%** |

Prophet improved forecast accuracy by ~54% versus the naive baseline on the
30-day holdout (results reproducible via `python forecast.py --horizon 30`
on the included synthetic dataset).

## Project structure

```
sales-demand-forecasting/
├── .github/workflows/ci.yml   # pytest on every push (GitHub Actions)
├── tests/                      # pytest suite (data generation, splitting, metrics, ARIMA)
├── generate_data.py             # synthetic daily sales generator (swap for real data)
├── forecast.py                    # trains Prophet + ARIMA, evaluates, plots
├── build_dashboard.py               # exports a BI-ready CSV for Power BI/Tableau
├── requirements.txt
├── data/                              # generated CSVs (gitignored)
└── reports/                            # forecast plots, metrics.json, forecast_results.csv
```

## Getting started

```bash
pip install -r requirements.txt
python generate_data.py            # creates data/daily_sales.csv
python forecast.py --horizon 30    # trains models, evaluates, saves plots
python build_dashboard.py          # exports reports/bi_export.csv for Power BI
```

## Testing

```bash
pytest tests/ -v
```

10 tests covering the synthetic data generator (schema, value floor,
sequential daily dates), the train/test time-series split, the MAE/MAPE
evaluation math (checked against hand-computed expected values), and a real
ARIMA fit/forecast on a small synthetic series. Prophet itself isn't unit
tested directly — it's a slow multi-second fit — but the data pipeline and
evaluation logic it shares with ARIMA are. This is also what
`.github/workflows/ci.yml` runs on every push.

## Notes

Ships with a synthetic data generator so it runs end-to-end out of the box.
Swap `data/daily_sales.csv` for real sales data with `date` and `sales`
columns to use on actual business data.
