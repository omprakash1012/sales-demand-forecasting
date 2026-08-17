# Sales / Demand Forecasting

Short-term demand forecasting to help optimize inventory and staffing decisions, comparing Prophet and ARIMA against a naive baseline.

**Stack:** Python, Pandas, Prophet, statsmodels (ARIMA), Power BI

## Problem

The business needed better short-term demand forecasts to optimize inventory and staffing, rather than relying on flat, rule-of-thumb projections.

## Approach

Built a daily sales time series with trend, weekly/yearly seasonality, and promotional spikes. Trained a Prophet model (yearly and weekly seasonality, tuned changepoint_prior_scale) and an ARIMA(5,1,2) baseline. Held out the last 30 days as a test window and evaluated both models against a naive "repeat last value" baseline using MAE and MAPE. Exported a BI-ready CSV (build_dashboard.py) with rolling averages, for visualization in Power BI/Tableau.

## Results

| Model | MAPE |
|-------|------|
| Naive Baseline | 8.46% |
| ARIMA(5,1,2) | 8.08% |
| Prophet (best) | 3.91% |

Prophet improved forecast accuracy by roughly 54% versus the naive baseline on the 30-day holdout. Results are reproducible via `python forecast.py --horizon 30` on the included synthetic dataset.

## Project structure

```
sales-demand-forecasting/
  generate_data.py     (synthetic daily sales generator, swap for real data)
  forecast.py            (trains Prophet + ARIMA, evaluates, plots)
  build_dashboard.py      (exports a BI-ready CSV for Power BI/Tableau)
  requirements.txt
  data/                    (generated CSVs, gitignored)
  reports/                  (forecast plots, metrics.json, forecast_results.csv)
```

## Getting started

```bash
pip install -r requirements.txt
python generate_data.py
python forecast.py --horizon 30
python build_dashboard.py
```

## Notes

Ships with a synthetic data generator so it runs end-to-end out of the box. Swap `data/daily_sales.csv` for real sales data with `date` and `sales` columns to use on actual business data.
