"""
Generates synthetic daily sales/demand data with trend, weekly seasonality,
yearly seasonality, and promotional spikes — mimicking real retail demand.
"""
import numpy as np
import pandas as pd

np.random.seed(7)


def generate_sales(start="2022-01-01", periods=1000):
    dates = pd.date_range(start=start, periods=periods, freq="D")
    t = np.arange(periods)

    trend = 150 + 0.08 * t
    weekly_seasonality = 25 * np.sin(2 * np.pi * t / 7 + 1.5)
    yearly_seasonality = 40 * np.sin(2 * np.pi * t / 365.25 + 4)
    noise = np.random.normal(0, 12, periods)

    promo_days = np.random.choice(periods, size=periods // 25, replace=False)
    promo_effect = np.zeros(periods)
    promo_effect[promo_days] = np.random.uniform(40, 90, len(promo_days))

    sales = trend + weekly_seasonality + yearly_seasonality + noise + promo_effect
    sales = np.clip(sales, 10, None).round().astype(int)

    df = pd.DataFrame({
        "date": dates,
        "sales": sales,
        "is_promo": np.isin(np.arange(periods), promo_days).astype(int),
    })
    return df


if __name__ == "__main__":
    df = generate_sales()
    df.to_csv("data/daily_sales.csv", index=False)
    print(f"Generated {len(df)} rows -> data/daily_sales.csv")
    print(df.tail())
