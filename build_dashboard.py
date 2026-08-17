"""
Builds a simple Power BI / Tableau-friendly export: a tidy CSV with actuals,
forecasts, and rolling averages, ready to drop straight into a BI tool.

Usage:
    python build_dashboard.py
"""
import os

import pandas as pd

DATA_PATH = "data/daily_sales.csv"
REPORT_DIR = "reports"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["rolling_7d_avg"] = df["sales"].rolling(7).mean().round(1)
    df["rolling_30d_avg"] = df["sales"].rolling(30).mean().round(1)
    df["day_of_week"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.to_period("M").astype(str)

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = f"{REPORT_DIR}/bi_export.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote BI-ready export -> {out_path}")
    print("Import this file into Power BI / Tableau to build the demand dashboard.")


if __name__ == "__main__":
    main()
