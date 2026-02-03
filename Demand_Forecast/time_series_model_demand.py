import pandas as pd
import numpy as np
import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SEASONAL_PERIOD = 12
MIN_HISTORY = 24
TEST_MONTHS = 6
FUTURE_MONTHS = 12
MAX_FORECAST_MULTIPLIER = 3

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(
    "monthly_demand_data.csv",
    parse_dates=["Month"]
).sort_values("Month")

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def last_value_baseline(train, steps):
    """Forecast by repeating last observed value"""
    return np.repeat(train.iloc[-1], steps)

def clip_forecast(pred, max_hist):
    return np.clip(pred, 0, MAX_FORECAST_MULTIPLIER * max_hist)

# ------------------------------------------------------------
# OUTPUT STORAGE
# ------------------------------------------------------------
product_eval_rows = []
product_comparison_rows = []
product_forecast_rows = []

# ============================================================
# PRODUCT LEVEL — HOLT-WINTERS + BASELINE COMPARISON
# ============================================================
for product, grp in df.groupby("Product_Name"):

    series = (
        grp.set_index("Month")["Order_Demand"]
        .asfreq("MS")
        .fillna(0)
    )

    if len(series) < MIN_HISTORY:
        continue

    # ------------------ EVALUATION ------------------
    train_eval = series[:-TEST_MONTHS]
    test_eval = series[-TEST_MONTHS:]

    avg_demand = train_eval.mean()
    max_hist = train_eval.max()

    try:
        # Holt-Winters Model
        hw_model = ExponentialSmoothing(
            train_eval,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD
        ).fit()

        hw_pred = hw_model.forecast(TEST_MONTHS)
        hw_pred = clip_forecast(hw_pred, max_hist)

        # Last Value Baseline
        baseline_pred = last_value_baseline(train_eval, TEST_MONTHS)

        # RMSE Calculation
        hw_rmse_val = rmse(test_eval.values, hw_pred)
        baseline_rmse_val = rmse(test_eval.values, baseline_pred)

        hw_rmse_pct = hw_rmse_val / avg_demand if avg_demand > 0 else np.inf
        baseline_rmse_pct = baseline_rmse_val / avg_demand if avg_demand > 0 else np.inf

        better_than_baseline = hw_rmse_val < baseline_rmse_val

        # ------------------ MONTH-LEVEL EVALUATION ------------------
        for dt, actual, pred in zip(test_eval.index, test_eval.values, hw_pred):
            product_eval_rows.append({
                "Product_Name": product,
                "Category_Name": grp["Category_Name"].iloc[0],
                "Month": dt,
                "Actual_Demand": actual,
                "HW_Predicted": round(pred, 0),
                "HW_RMSE": round(hw_rmse_val, 2),
                "HW_RMSE_%": round(hw_rmse_pct * 100, 2),
                "Last_Value_RMSE": round(baseline_rmse_val, 2),
                "Last_Value_RMSE_%": round(baseline_rmse_pct * 100, 2),
                "Better_Than_Baseline": better_than_baseline
            })

        # ------------------ PRODUCT SUMMARY ------------------
        product_comparison_rows.append({
            "Product_Name": product,
            "Category_Name": grp["Category_Name"].iloc[0],
            "HW_RMSE": round(hw_rmse_val, 2),
            "Last_Value_RMSE": round(baseline_rmse_val, 2),
            "HW_RMSE_%": round(hw_rmse_pct * 100, 2),
            "Last_Value_RMSE_%": round(baseline_rmse_pct * 100, 2),
            "Better_Than_Baseline": better_than_baseline,
            "Model_Status": "OK"
            if (better_than_baseline and hw_rmse_pct < 40)
            else "UNSTABLE"
        })

    except Exception:
        continue

    # ------------------ FUTURE FORECAST ------------------
    try:
        final_model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD
        ).fit()

        future_pred = final_model.forecast(FUTURE_MONTHS)
        future_pred = clip_forecast(future_pred, series.max())

        for dt, val in zip(future_pred.index, future_pred):
            product_forecast_rows.append({
                "Level": "Product",
                "Category_Name": grp["Category_Name"].iloc[0],
                "Product_Name": product,
                "Forecast_Month": dt,
                "Forecast_Demand": round(val, 0)
            })

    except Exception:
        continue

# ============================================================
# EXPORT CSVs
# ============================================================
pd.DataFrame(product_eval_rows).to_csv(
    "product_evaluation_with_baseline.csv", index=False
)

pd.DataFrame(product_comparison_rows).to_csv(
    "model_comparison_product_level.csv", index=False
)

pd.DataFrame(product_forecast_rows).to_csv(
    "product_forecast.csv", index=False
)

print(" Product-Level Forecasting Completed Successfully")
print("Files generated:")
print(" - product_evaluation_with_baseline.csv")
print(" - model_comparison_product_level.csv")
print(" - product_forecast.csv")























































































# import pandas as pd
# import numpy as np
# import warnings

# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.stattools import adfuller
# from sklearn.metrics import mean_squared_error

# warnings.filterwarnings("ignore")

# # ------------------------------------------------------------
# # CONFIG
# # ------------------------------------------------------------
# SEASONAL_PERIOD = 12
# MIN_HISTORY = 24
# TEST_MONTHS = 6
# FUTURE_MONTHS = 12
# MAX_FORECAST_MULTIPLIER = 3

# # ------------------------------------------------------------
# # LOAD DATA
# # ------------------------------------------------------------
# df = pd.read_csv(
#     "monthly_demand_data.csv",
#     parse_dates=["Month"]
# ).sort_values("Month")

# # ------------------------------------------------------------
# # HELPER FUNCTIONS
# # ------------------------------------------------------------
# def detect_d(series):
#     """Return d=0 if series stationary, else d=1"""
#     return 0 if adfuller(series.dropna())[1] < 0.05 else 1

# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))

# def naive_forecast(train, steps):
#     return np.repeat(train.iloc[-1], steps)

# def clip_forecast(pred, max_hist):
#     return np.clip(pred, 0, MAX_FORECAST_MULTIPLIER * max_hist)

# def fit_sarima(series):
#     d = detect_d(series)
#     model = SARIMAX(
#         series,
#         order=(1, d, 1),
#         seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
#         enforce_stationarity=False,
#         enforce_invertibility=False
#     )
#     return model.fit(disp=False)

# # ------------------------------------------------------------
# # OUTPUT STORAGE
# # ------------------------------------------------------------
# overall_forecast_rows = []
# category_forecast_rows = []
# product_forecast_rows = []

# product_eval_rows = []
# product_comparison_rows = []

# # ============================================================
# # PRODUCT LEVEL — HOLT-WINTERS + EVALUATION
# # ============================================================
# for product, grp in df.groupby("Product_Name"):

#     series = grp.set_index("Month")["Order_Demand"].asfreq("MS").fillna(0)
#     if len(series) < MIN_HISTORY:
#         continue

#     # --- EVALUATION AGAINST NAIVE ---
#     train_eval = series[:-TEST_MONTHS]
#     test_eval = series[-TEST_MONTHS:]
#     avg_demand = train_eval.mean()
#     max_hist_eval = train_eval.max()

#     try:
#         # Holt-Winters
#         hw_model = ExponentialSmoothing(
#             train_eval,
#             trend="add",
#             seasonal="add",
#             seasonal_periods=SEASONAL_PERIOD
#         ).fit()

#         hw_pred = hw_model.forecast(TEST_MONTHS)
#         hw_pred = clip_forecast(hw_pred, max_hist_eval)

#         # Naive
#         naive_pred = naive_forecast(train_eval, TEST_MONTHS)

#         # RMSE
#         hw_rmse_val = rmse(test_eval.values, hw_pred)
#         naive_rmse_val = rmse(test_eval.values, naive_pred)

#         hw_rmse_pct = hw_rmse_val / avg_demand if avg_demand > 0 else np.inf
#         naive_rmse_pct = naive_rmse_val / avg_demand if avg_demand > 0 else np.inf

#         better_than_naive = hw_rmse_val < naive_rmse_val

#         # Product Evaluation CSV
#         for dt, actual, pred in zip(test_eval.index, test_eval.values, hw_pred):
#             product_eval_rows.append({
#                 "Product_Name": product,
#                 "Category_Name": grp["Category_Name"].iloc[0],
#                 "Month": dt,
#                 "Actual_Demand": actual,
#                 "HW_Predicted": round(pred, 0),
#                 "HW_RMSE": round(hw_rmse_val, 2),
#                 "HW_RMSE_%": round(hw_rmse_pct * 100, 2),
#                 "Naive_RMSE": round(naive_rmse_val, 2),
#                 "Naive_RMSE_%": round(naive_rmse_pct * 100, 2),
#                 "Better_Than_Naive": better_than_naive
#             })

#         # Product Comparison CSV
#         product_comparison_rows.append({
#             "Product_Name": product,
#             "Category_Name": grp["Category_Name"].iloc[0],
#             "HW_RMSE": round(hw_rmse_val, 2),
#             "Naive_RMSE": round(naive_rmse_val, 2),
#             "HW_RMSE_%": round(hw_rmse_pct * 100, 2),
#             "Naive_RMSE_%": round(naive_rmse_pct * 100, 2),
#             "Better_Than_Naive": better_than_naive,
#             "Model_Status": "OK" if (better_than_naive and hw_rmse_pct < 40) else "UNSTABLE"
#         })

#     except Exception:
#         continue

#     # --- FUTURE FORECAST ---
#     try:
#         final_model = ExponentialSmoothing(
#             series,
#             trend="add",
#             seasonal="add",
#             seasonal_periods=SEASONAL_PERIOD
#         ).fit()

#         future_pred = final_model.forecast(FUTURE_MONTHS)
#         future_pred = clip_forecast(future_pred, series.max())

#         for dt, val in zip(future_pred.index, future_pred):
#             product_forecast_rows.append({
#                 "Level": "Product",
#                 "Category_Name": grp["Category_Name"].iloc[0],
#                 "Product_Name": product,
#                 "Forecast_Month": dt,
#                 "Forecast_Demand": round(val, 0)
#             })

#     except Exception:
#         continue

# # ============================================================
# #  CATEGORY LEVEL — SARIMA
# # ============================================================
# for category, grp in df.groupby("Category_Name"):

#     series = grp.groupby("Month")["Order_Demand"].sum().asfreq("MS").interpolate()
#     if len(series) < MIN_HISTORY:
#         continue

#     try:
#         model = fit_sarima(series)
#         forecast = model.get_forecast(FUTURE_MONTHS).predicted_mean

#         for dt, val in forecast.items():
#             category_forecast_rows.append({
#                 "Level": "Category",
#                 "Category_Name": category,
#                 "Product_Name": "ALL",
#                 "Forecast_Month": dt,
#                 "Forecast_Demand": round(max(val, 0), 0)
#             })
#     except Exception:
#         continue

# # ============================================================
# #  OVERALL LEVEL — SARIMA
# # ============================================================
# overall_series = df.groupby("Month")["Order_Demand"].sum().asfreq("MS").interpolate()
# overall_model = fit_sarima(overall_series)
# overall_forecast = overall_model.get_forecast(FUTURE_MONTHS).predicted_mean

# for dt, val in overall_forecast.items():
#     overall_forecast_rows.append({
#         "Level": "Overall",
#         "Category_Name": "ALL",
#         "Product_Name": "ALL",
#         "Forecast_Month": dt,
#         "Forecast_Demand": round(max(val, 0), 0)
#     })

# # ============================================================
# # EXPORT CSVs
# # ============================================================
# pd.DataFrame(product_eval_rows).to_csv("product_evaluation_with_naive.csv", index=False)
# pd.DataFrame(product_comparison_rows).to_csv("model_comparison_product_level.csv", index=False)
# pd.DataFrame(product_forecast_rows).to_csv("product_forecast.csv", index=False)
# pd.DataFrame(category_forecast_rows).to_csv("category_forecast.csv", index=False)
# pd.DataFrame(overall_forecast_rows).to_csv("overall_forecast.csv", index=False)

# print("Hierarchical Forecasting Completed Successfully ")
# print("Files generated:")
# print(" - product_evaluation_with_naive.csv")
# print(" - model_comparison_product_level.csv")
# print(" - product_forecast.csv")
# print(" - category_forecast.csv")
# print(" - overall_forecast.csv")

