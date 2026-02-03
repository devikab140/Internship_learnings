import pandas as pd
import numpy as np
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------

SEASONAL_PERIOD = 12
MIN_HISTORY = 24
TEST_MONTHS = 6
FUTURE_MONTHS = 12      # Forecast full next year
MAX_FORECAST_MULTIPLIER = 3   # Safety cap

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

df = pd.read_csv(
    "monthly_demand_data.csv",
    parse_dates=["Month"]
).sort_values("Month")

# ------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------

def detect_d(series):
    """Detect differencing using ADF test"""
    p = adfuller(series.dropna())[1]
    return 0 if p < 0.05 else 1


def train_sarima(series, d):
    """Safe SARIMA configuration"""
    model = SARIMAX(
        series,
        order=(1, d, 1),
        seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_forecast(values, max_hist):
    """Clip extreme forecasts"""
    return np.clip(values, 0, MAX_FORECAST_MULTIPLIER * max_hist)


# ------------------------------------------------------
# STORAGE
# ------------------------------------------------------

evaluation_rows = []
future_rows = []

# ------------------------------------------------------
# PRODUCT-LEVEL FORECAST LOOP
# ------------------------------------------------------

for product, grp in df.groupby("Product_Name"):

    series = (
        grp.set_index("Month")["Order_Demand"]
        .asfreq("MS")
        .fillna(0)
    )

    if len(series) < MIN_HISTORY:
        continue

    train = series[:-TEST_MONTHS]
    test = series[-TEST_MONTHS:]

    avg_demand = train.mean()
    max_hist = train.max()

    try:
        d = detect_d(train)
        model = train_sarima(train, d)

        # ----------------------------
        # TEST FORECAST (EVALUATION)
        # ----------------------------
        test_pred = model.get_forecast(TEST_MONTHS).predicted_mean
        test_pred.index = test.index
        test_pred = safe_forecast(test_pred, max_hist)

        error = rmse(test, test_pred)
        rmse_pct = error / avg_demand if avg_demand > 0 else np.inf

        stable = (
            np.isfinite(error)
            and error < MAX_FORECAST_MULTIPLIER * max_hist
            and rmse_pct < 0.40
        )

        # Store evaluation
        for dt in test.index:
            evaluation_rows.append({
                "Level": "Product",
                "Product_Name": product,
                "Category_Name": grp["Category_Name"].iloc[0],
                "Month": dt,
                "Actual_Demand": test.loc[dt],
                "Predicted_Demand": round(test_pred.loc[dt], 0),
                "RMSE": round(error, 2),
                "RMSE_%": round(rmse_pct * 100, 2),
                "Model_Status": "OK" if stable else "UNSTABLE"
            })

        # ----------------------------
        # FUTURE FORECAST
        # ----------------------------
        if stable:
            final_model = train_sarima(series, d)
            future_pred = final_model.get_forecast(FUTURE_MONTHS).predicted_mean
            future_pred = safe_forecast(future_pred, max_hist)

            future_index = pd.date_range(
                start=series.index[-1] + pd.offsets.MonthBegin(1),
                periods=FUTURE_MONTHS,
                freq="MS"
            )

            for dt, val in zip(future_index, future_pred):
                future_rows.append({
                    "Forecast_Level": "Product",
                    "Product_Name": product,
                    "Category_Name": grp["Category_Name"].iloc[0],
                    "Forecast_Month": dt,
                    "Forecast_Demand": round(val, 0),
                    "RMSE_%": round(rmse_pct * 100, 2)
                })

    except Exception:
        continue

# ------------------------------------------------------
# CATEGORY FALLBACK (FOR UNSTABLE PRODUCTS)
# ------------------------------------------------------

for category, grp in df.groupby("Category_Name"):

    series = (
        grp.groupby("Month")["Order_Demand"]
        .sum()
        .asfreq("MS")
        .fillna(0)
    )

    if len(series) < MIN_HISTORY:
        continue

    try:
        d = detect_d(series)
        model = train_sarima(series, d)
        future_pred = model.get_forecast(FUTURE_MONTHS).predicted_mean

        future_index = pd.date_range(
            start=series.index[-1] + pd.offsets.MonthBegin(1),
            periods=FUTURE_MONTHS,
            freq="MS"
        )

        for dt, val in zip(future_index, future_pred):
            future_rows.append({
                "Forecast_Level": "Category",
                "Product_Name": "ALL",
                "Category_Name": category,
                "Forecast_Month": dt,
                "Forecast_Demand": round(max(val, 0), 0),
                "RMSE_%": None
            })

    except Exception:
        continue

# ------------------------------------------------------
# EXPORT FOR POWER BI
# ------------------------------------------------------

eval_df = pd.DataFrame(evaluation_rows)
future_df = pd.DataFrame(future_rows)

eval_df.to_csv("evaluation_results_powerbi_2.csv", index=False)
future_df.to_csv("future_forecast_powerbi_2.csv", index=False)

print(" PRODUCTION FORECASTING COMPLETED")
print(" evaluation_results_powerbi_2.csv")
print(" future_forecast_powerbi_2.csv")



# # ======================================================
# # DEMAND FORECASTING PIPELINE
# # Evaluation (Test RMSE) + Future Forecast
# # Product-level | SARIMA | Power BI ready
# # ======================================================

# import pandas as pd
# import numpy as np
# import warnings

# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.stattools import adfuller
# from sklearn.metrics import mean_squared_error

# warnings.filterwarnings("ignore")

# # ------------------------------------------------------
# # CONFIG
# # ------------------------------------------------------

# SEASONAL_PERIOD = 12
# MIN_HISTORY = 24
# TEST_MONTHS = 6
# FUTURE_MONTHS = 12   # e.g. full 2018

# # ------------------------------------------------------
# # LOAD DATA
# # ------------------------------------------------------

# df = pd.read_csv(
#     "monthly_demand_data.csv",
#     parse_dates=["Month"]
# ).sort_values("Month")

# # ------------------------------------------------------
# # HELPER FUNCTIONS
# # ------------------------------------------------------

# def detect_d(series):
#     """ADF-based differencing detection"""
#     p = adfuller(series.dropna())[1]
#     return 0 if p < 0.05 else 1


# def train_sarima(series, d):
#     model = SARIMAX(
#         series,
#         order=(1, d, 1),
#         seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
#         enforce_stationarity=False,
#         enforce_invertibility=False
#     )
#     return model.fit(disp=False)


# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))


# # ------------------------------------------------------
# # RESULTS STORAGE
# # ------------------------------------------------------

# evaluation_results = []
# future_results = []

# # ------------------------------------------------------
# # FORECAST LOOP (PRODUCT LEVEL)
# # ------------------------------------------------------

# for product, grp in df.groupby("Product_Name"):

#     series = (
#         grp.set_index("Month")["Order_Demand"]
#         .asfreq("MS")
#         .fillna(0)
#     )

#     if len(series) < MIN_HISTORY:
#         continue

#     # ----------------------------
#     # TRAIN / TEST SPLIT
#     # ----------------------------
#     train = series[:-TEST_MONTHS]
#     test = series[-TEST_MONTHS:]

#     try:
#         # ----------------------------
#         # MODEL TRAINING
#         # ----------------------------
#         d = detect_d(train)
#         model = train_sarima(train, d)

#         # ----------------------------
#         # TEST FORECAST (EVALUATION)
#         # ----------------------------
#         test_forecast = model.get_forecast(
#             steps=TEST_MONTHS
#         ).predicted_mean

#         test_forecast.index = test.index

#         error = rmse(test, test_forecast)

#         # Store evaluation results
#         for date in test.index:
#             evaluation_results.append({
#                 "Product_Name": product,
#                 "Category_Name": grp["Category_Name"].iloc[0],
#                 "Month": date,
#                 "Actual_Demand": round(test.loc[date], 0),
#                 "Predicted_Demand": round(max(test_forecast.loc[date], 0), 0),
#                 "RMSE": round(error, 2)
#             })

#         # ----------------------------
#         # RETRAIN ON FULL DATA
#         # ----------------------------
#         final_model = train_sarima(series, d)

#         last_month = series.index[-1]
#         future_index = pd.date_range(
#             start=last_month + pd.offsets.MonthBegin(1),
#             periods=FUTURE_MONTHS,
#             freq="MS"
#         )

#         future_forecast = final_model.get_forecast(
#             steps=FUTURE_MONTHS
#         ).predicted_mean

#         future_forecast.index = future_index

#         # Store future forecasts
#         for date, value in future_forecast.items():
#             future_results.append({
#                 "Product_Name": product,
#                 "Category_Name": grp["Category_Name"].iloc[0],
#                 "Forecast_Month": date,
#                 "Forecast_Demand": round(max(value, 0), 0),
#                 "Model_RMSE": round(error, 2),
#                 "Differencing_d": d
#             })

#     except Exception:
#         continue

# # ------------------------------------------------------
# # EXPORT FILES
# # ------------------------------------------------------

# eval_df = pd.DataFrame(evaluation_results)
# future_df = pd.DataFrame(future_results)

# eval_df.to_csv("model_evaluation_test_forecast.csv", index=False)
# future_df.to_csv("future_forecast_2018.csv", index=False)

# print("Model evaluation & future forecasting completed")
# print("Evaluation file : model_evaluation_test_forecast.csv")
# print(" Future forecast : future_forecast_2018.csv")

