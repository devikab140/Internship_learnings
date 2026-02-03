import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ----------------------------
# 1. LOAD DATA
# ----------------------------

df = pd.read_csv(
    "monthly_demand_data.csv",
    parse_dates=["Month"]
)

df = df.sort_values("Month")

# ----------------------------
# 2. COMMON DIAGNOSTIC FUNCTIONS
# ----------------------------

def check_missing_periods(series, name):
    print(f"\n[{name}] Missing periods:", series.isna().sum())


def variance_check(series, name):
    plt.figure(figsize=(10, 4))
    plt.plot(series)
    plt.title(f"Level & Variance Check - {name}")
    # plt.show()


def adf_test(series, name):
    result = adfuller(series.dropna())
    print(f"\nADF Test - {name}")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.4f}")
    print("Stationary" if result[1] < 0.05 else "Non-stationary")


def decompose(series, name):
    decomposition = seasonal_decompose(
        series,
        model="additive",
        period=12
    )
    decomposition.plot()
    plt.suptitle(f"STI Decomposition - {name}", fontsize=14)
    # plt.show()


def acf_pacf(series, name):
    series = series.dropna()
    n = len(series)

    if n < 10:
        print(f"Skipping ACF/PACF for {name} (too few points)")
        return

    max_lags = min(24, n // 2 - 1)

    plt.figure(figsize=(12, 4))
    plot_acf(series, lags=max_lags)
    plt.title(f"ACF - {name}")
    # plt.show()

    plt.figure(figsize=(12, 4))
    plot_pacf(series, lags=max_lags, method="ywm")
    plt.title(f"PACF - {name}")
    # plt.show()


# ======================================================
# 3. OVERALL DEMAND DIAGNOSTICS
# ======================================================

print("\n================ OVERALL DEMAND =================")

overall = (
    df.groupby("Month")["Order_Demand"]
    .sum()
    .asfreq("MS")
)

check_missing_periods(overall, "Overall")
variance_check(overall, "Overall")
decompose(overall, "Overall")
acf_pacf(overall, "Overall")
adf_test(overall, "Overall")

# ======================================================
# 4. CATEGORY LEVEL DIAGNOSTICS (ALL)
# ======================================================

print("\n================ CATEGORY LEVEL =================")

for category, grp in df.groupby("Category_Name"):

    series = (
        grp.groupby("Month")["Order_Demand"]
        .sum()
        .asfreq("MS")
    )

    if len(series) < 24:
        continue

    print(f"\n--- Category: {category} ---")

    check_missing_periods(series, category)
    variance_check(series, category)
    decompose(series, category)
    acf_pacf(series, category)
    adf_test(series, category)

# ======================================================
# 5. PRODUCT LEVEL DIAGNOSTICS (ALL)
# ======================================================

print("\n================ PRODUCT LEVEL =================")

for product, grp in df.groupby("Product_Name"):

    series = (
        grp.set_index("Month")["Order_Demand"]
        .asfreq("MS")
    )

    if len(series) < 24:
        continue

    print(f"\n--- Product: {product} ---")

    check_missing_periods(series, product)
    variance_check(series, product)
    decompose(series, product)
    acf_pacf(series, product)
    adf_test(series, product)

print("\n All time series diagnostics completed")
