# # ============================================================
# # FINAL CORRECTED FORECASTING PIPELINE (WITH ALLOCATED SARIMA)
# # ============================================================

# import pandas as pd
# import numpy as np
# import warnings

# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.stattools import adfuller
# from sklearn.metrics import mean_squared_error

# warnings.filterwarnings("ignore")

# # ============================================================
# # CONFIG
# # ============================================================

# MIN_HISTORY = 36
# TEST_MONTHS = 6
# FUTURE_MONTHS = 12
# SEASONAL_PERIOD = 12
# MAX_RMSE_PCT = 50
# MAX_ZERO_MONTHS = 24

# # ============================================================
# # LOAD DATA
# # ============================================================

# df = pd.read_csv(
#     "monthly_demand_data.csv",
#     parse_dates=["Month"]
# ).sort_values(["Category_Name", "Product_Name", "Month"])

# # ============================================================
# # HELPERS
# # ============================================================

# def rmse(y, yhat):
#     return np.sqrt(mean_squared_error(y, yhat))

# def cap_outliers(series):
#     q1, q3 = series.quantile([0.25, 0.75])
#     iqr = q3 - q1
#     return np.clip(series, 0, q3 + 1.5 * iqr)

# def log_transform(series):
#     return np.log1p(series)

# def inverse_log(series):
#     return np.expm1(series)

# def is_stationary(series):
#     return adfuller(series.dropna())[1] < 0.05

# # ============================================================
# # OUTPUT STORAGE
# # ============================================================

# summary_rows = []
# test_eval_rows = []
# forecast_rows = []

# # ============================================================
# # PRODUCT LOOP
# # ============================================================

# for (cat, prod), grp in df.groupby(["Category_Name", "Product_Name"]):

#     series = grp.set_index("Month")["Order_Demand"].asfreq("MS").fillna(0)

#     if len(series) < MIN_HISTORY:
#         continue

#     zero_months = int((series == 0).sum())

#     # ----------------------------
#     # PREPROCESS
#     # ----------------------------
#     capped = cap_outliers(series)
#     transformed = log_transform(capped)

#     train = transformed[:-TEST_MONTHS]
#     test_actual = series[-TEST_MONTHS:]

#     use_product = False
#     rmse_pct = np.inf

#     # ========================================================
#     # PRODUCT LEVEL — HOLT-WINTERS
#     # ========================================================

#     try:
#         hw_model = ExponentialSmoothing(
#             train,
#             trend="add",
#             seasonal="add",
#             seasonal_periods=SEASONAL_PERIOD
#         ).fit()

#         hw_pred_log = hw_model.forecast(TEST_MONTHS)
#         hw_pred = inverse_log(hw_pred_log)

#         rmse_val = rmse(test_actual.values, hw_pred.values)
#         rmse_pct = (rmse_val / series.mean()) * 100 if series.mean() > 0 else np.inf

#         if zero_months <= MAX_ZERO_MONTHS and rmse_pct <= MAX_RMSE_PCT:
#             use_product = True

#             for dt, act, pred in zip(test_actual.index, test_actual.values, hw_pred.values):
#                 test_eval_rows.append({
#                     "Category_Name": cat,
#                     "Product_Name": prod,
#                     "Month": dt,
#                     "Actual_Demand": round(act, 0),
#                     "Predicted_Demand": round(max(pred, 0), 0),
#                     "Error": round(act - pred, 0),
#                     "Model": "Holt-Winters",
#                     "Level": "Product"
#                 })

#     except Exception:
#         pass

#     # ========================================================
#     # CATEGORY LEVEL — SARIMA WITH PRODUCT ALLOCATION
#     # ========================================================

#     if not use_product:

#         cat_series = (
#             df[df["Category_Name"] == cat]
#             .groupby("Month")["Order_Demand"]
#             .sum()
#             .asfreq("MS")
#             .fillna(0)
#         )

#         train_c = cat_series[:-TEST_MONTHS]
#         test_c = cat_series[-TEST_MONTHS:]

#         d = 0 if is_stationary(train_c) else 1

#         model = SARIMAX(
#             train_c,
#             order=(1, d, 1),
#             seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
#             enforce_stationarity=False,
#             enforce_invertibility=False
#         ).fit(disp=False)

#         sarima_pred = model.forecast(TEST_MONTHS)

#         # ----------------------------
#         # PRODUCT SHARE ALLOCATION
#         # ----------------------------
#         product_avg = series[:-TEST_MONTHS].mean()
#         category_avg = train_c.mean()

#         product_share = product_avg / category_avg if category_avg > 0 else 0

#         product_level_pred = sarima_pred * product_share

#         rmse_val = rmse(test_actual.values, product_level_pred.values)
#         rmse_pct = (rmse_val / series.mean()) * 100 if series.mean() > 0 else np.inf

#         for dt, act, pred in zip(
#             test_actual.index,
#             test_actual.values,
#             product_level_pred.values
#         ):
#             test_eval_rows.append({
#                 "Category_Name": cat,
#                 "Product_Name": prod,
#                 "Month": dt,
#                 "Actual_Demand": round(act, 0),
#                 "Predicted_Demand": round(max(pred, 0), 0),
#                 "Error": round(act - pred, 0),
#                 "Model": "SARIMA (Allocated)",
#                 "Level": "Product"
#             })

#         future_cat = model.forecast(FUTURE_MONTHS)
#         future_forecast = future_cat * product_share

#         model_used = "SARIMA (Allocated)"
#         level = "Product"

#     else:
#         final_model = ExponentialSmoothing(
#             transformed,
#             trend="add",
#             seasonal="add",
#             seasonal_periods=SEASONAL_PERIOD
#         ).fit()

#         future_log = final_model.forecast(FUTURE_MONTHS)
#         future_forecast = inverse_log(future_log)

#         model_used = "Holt-Winters"
#         level = "Product"

#     # ========================================================
#     # STORE SUMMARY
#     # ========================================================

#     summary_rows.append({
#         "Category_Name": cat,
#         "Product_Name": prod,
#         "Model": model_used,
#         "Level": level,
#         "Zero_Months": zero_months,
#         "RMSE_%": round(rmse_pct, 2)
#     })

#     # ========================================================
#     # STORE FUTURE FORECAST
#     # ========================================================

#     future_dates = pd.date_range(
#         series.index.max() + pd.offsets.MonthBegin(1),
#         periods=FUTURE_MONTHS,
#         freq="MS"
#     )

#     for dt, val in zip(future_dates, future_forecast):
#         forecast_rows.append({
#             "Category_Name": cat,
#             "Product_Name": prod,
#             "Forecast_Month": dt,
#             "Forecast_Demand": round(max(val, 0), 0),
#             "Model": model_used,
#             "Level": level
#         })

# # ============================================================
# # EXPORT FILES
# # ============================================================

# pd.DataFrame(summary_rows).to_csv("model_summary_final.csv", index=False)
# pd.DataFrame(test_eval_rows).to_csv("test_period_actual_vs_predicted_final.csv", index=False)
# pd.DataFrame(forecast_rows).to_csv("future_forecast_final.csv", index=False)

# print(" Pipeline completed successfully")


















# ============================================================
# FINAL END-TO-END DEMAND FORECASTING PIPELINE
# Model Comparison → Best Model → Future Forecast
# ============================================================

import pandas as pd
import numpy as np
import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MIN_HISTORY = 36
TEST_MONTHS = 6
FUTURE_MONTHS = 12
SEASONAL_PERIOD = 12

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "monthly_demand_data.csv",
    parse_dates=["Month"]
).sort_values(["Category_Name", "Product_Name", "Month"])

# ============================================================
# HELPERS
# ============================================================

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

# ============================================================
# STORAGE
# ============================================================

comparison_rows = []
performance_rows = []
best_model_rows = []
future_rows = []

# ============================================================
# CATEGORY SARIMA MODELS
# ============================================================

category_models = {}

for cat, grp in df.groupby("Category_Name"):

    cat_series = (
        grp.groupby("Month")["Order_Demand"]
        .sum()
        .asfreq("MS")
        .fillna(0)
    )

    if len(cat_series) < MIN_HISTORY:
        continue

    model = SARIMAX(
        cat_series[:-TEST_MONTHS],
        order=(1,1,1),
        seasonal_order=(1,1,1,SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    category_models[cat] = model

# ============================================================
# PRODUCT LOOP
# ============================================================

for (cat, prod), grp in df.groupby(["Category_Name", "Product_Name"]):

    series = (
        grp.set_index("Month")["Order_Demand"]
        .asfreq("MS")
        .fillna(0)
    )

    if len(series) < MIN_HISTORY or cat not in category_models:
        continue

    train = series[:-TEST_MONTHS]
    test = series[-TEST_MONTHS:]

    model_scores = {}

    # ========================================================
    # 1️⃣ HOLT-WINTERS
    # ========================================================

    try:
        hw = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD
        ).fit()

        pred = hw.forecast(TEST_MONTHS)

        rmse_val = rmse(test.values, pred.values)
        rmse_pct = (rmse_val / series.mean()) * 100

        model_scores["Holt-Winters"] = rmse_val

        for dt, a, p in zip(test.index, test.values, pred.values):
            comparison_rows.append({
                "Category_Name": cat,
                "Product_Name": prod,
                "Month": dt,
                "Model": "Holt-Winters",
                "Actual": round(a,0),
                "Predicted": round(p,0),
                "Error": round(a-p,0)
            })

        performance_rows.append({
            "Category_Name": cat,
            "Product_Name": prod,
            "Model": "Holt-Winters",
            "RMSE": round(rmse_val,2),
            "RMSE_%": round(rmse_pct,2)
        })

    except:
        pass

    # ========================================================
    # 2️⃣ PROPHET
    # ========================================================

    try:
        p_df = train.reset_index()
        p_df.columns = ["ds", "y"]

        m = Prophet(yearly_seasonality=True)
        m.fit(p_df)

        future = m.make_future_dataframe(periods=TEST_MONTHS, freq="MS")
        fcst = m.predict(future).tail(TEST_MONTHS)

        pred = fcst["yhat"].values

        rmse_val = rmse(test.values, pred)
        rmse_pct = (rmse_val / series.mean()) * 100

        model_scores["Prophet"] = rmse_val

        for dt, a, p in zip(test.index, test.values, pred):
            comparison_rows.append({
                "Category_Name": cat,
                "Product_Name": prod,
                "Month": dt,
                "Model": "Prophet",
                "Actual": round(a,0),
                "Predicted": round(p,0),
                "Error": round(a-p,0)
            })

        performance_rows.append({
            "Category_Name": cat,
            "Product_Name": prod,
            "Model": "Prophet",
            "RMSE": round(rmse_val,2),
            "RMSE_%": round(rmse_pct,2)
        })

    except:
        pass

    # ========================================================
    # 3️⃣ SARIMA (CATEGORY → PRODUCT)
    # ========================================================

    try:
        cat_model = category_models[cat]
        cat_pred = cat_model.forecast(TEST_MONTHS)

        product_share = series.mean() / df[df["Category_Name"] == cat]["Order_Demand"].mean()
        prod_pred = cat_pred * product_share

        rmse_val = rmse(test.values, prod_pred.values)
        rmse_pct = (rmse_val / series.mean()) * 100

        model_scores["SARIMA"] = rmse_val

        for dt, a, p in zip(test.index, test.values, prod_pred.values):
            comparison_rows.append({
                "Category_Name": cat,
                "Product_Name": prod,
                "Month": dt,
                "Model": "SARIMA",
                "Actual": round(a,0),
                "Predicted": round(p,0),
                "Error": round(a-p,0)
            })

        performance_rows.append({
            "Category_Name": cat,
            "Product_Name": prod,
            "Model": "SARIMA",
            "RMSE": round(rmse_val,2),
            "RMSE_%": round(rmse_pct,2)
        })

    except:
        pass

    if not model_scores:
        continue

    # ========================================================
    # PICK BEST MODEL
    # ========================================================

    best_model = min(model_scores, key=model_scores.get)

    best_model_rows.append({
        "Category_Name": cat,
        "Product_Name": prod,
        "Best_Model": best_model,
        "Best_RMSE": round(model_scores[best_model],2)
    })

    # ========================================================
    # FUTURE FORECAST USING BEST MODEL
    # ========================================================

    if best_model == "Holt-Winters":
        final = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD
        ).fit()
        future = final.forecast(FUTURE_MONTHS)

    elif best_model == "Prophet":
        p_df = series.reset_index()
        p_df.columns = ["ds", "y"]

        m = Prophet(yearly_seasonality=True)
        m.fit(p_df)

        f_df = m.make_future_dataframe(periods=FUTURE_MONTHS, freq="MS")
        future = m.predict(f_df)["yhat"].tail(FUTURE_MONTHS).values

    else:
        cat_future = category_models[cat].forecast(FUTURE_MONTHS)
        future = cat_future * product_share

    future_dates = pd.date_range(
        series.index.max() + pd.offsets.MonthBegin(1),
        periods=FUTURE_MONTHS,
        freq="MS"
    )

    for d, v in zip(future_dates, future):
        future_rows.append({
            "Category_Name": cat,
            "Product_Name": prod,
            "Forecast_Month": d,
            "Forecast_Demand": round(max(v,0),0),
            "Model": best_model
        })

# ============================================================
# EXPORT FILES
# ============================================================

pd.DataFrame(comparison_rows).to_csv("model_comparison_test_period.csv", index=False)
pd.DataFrame(performance_rows).to_csv("model_performance_summary.csv", index=False)
pd.DataFrame(best_model_rows).to_csv("best_model_per_product.csv", index=False)
pd.DataFrame(future_rows).to_csv("best_model_future_forecast.csv", index=False)

print(" PIPELINE COMPLETED SUCCESSFULLY")







# # ============================================================
# # FINAL CORRECTED FORECASTING PIPELINE
# # ============================================================

# import pandas as pd
# import numpy as np
# import warnings

# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.stattools import adfuller
# from sklearn.metrics import mean_squared_error

# warnings.filterwarnings("ignore")

# # ============================================================
# # CONFIG
# # ============================================================

# MIN_HISTORY = 36
# TEST_MONTHS = 6          # change this freely (6, 9, 12...)
# FUTURE_MONTHS = 12
# SEASONAL_PERIOD = 12
# MAX_RMSE_PCT = 50
# MAX_ZERO_MONTHS = 24

# # ============================================================
# # LOAD DATA
# # ============================================================

# df = pd.read_csv(
#     "monthly_demand_data.csv",
#     parse_dates=["Month"]
# ).sort_values(["Category_Name", "Product_Name", "Month"])

# # ============================================================
# # HELPERS
# # ============================================================

# def rmse(y, yhat):
#     return np.sqrt(mean_squared_error(y, yhat))

# def cap_outliers(series):
#     q1, q3 = series.quantile([0.25, 0.75])
#     iqr = q3 - q1
#     upper = q3 + 1.5 * iqr
#     return np.clip(series, 0, upper)

# def log_transform(series):
#     return np.log1p(series)

# def inverse_log(series):
#     return np.expm1(series)

# def is_stationary(series):
#     return adfuller(series.dropna())[1] < 0.05

# # ============================================================
# # OUTPUT STORAGE
# # ============================================================

# summary_rows = []
# test_eval_rows = []
# forecast_rows = []

# # ============================================================
# # PRODUCT LOOP
# # ============================================================

# for (cat, prod), grp in df.groupby(["Category_Name", "Product_Name"]):

#     series = (
#         grp.set_index("Month")["Order_Demand"]
#         .asfreq("MS")
#         .fillna(0)
#     )

#     if len(series) < MIN_HISTORY:
#         continue

#     zero_months = int((series == 0).sum())

#     # ----------------------------
#     # PREPROCESS
#     # ----------------------------
#     capped = cap_outliers(series)
#     transformed = log_transform(capped)

#     train = transformed[:-TEST_MONTHS]
#     test_actual = series[-TEST_MONTHS:]

#     use_product = False
#     rmse_pct = np.inf

#     # ========================================================
#     # PRODUCT LEVEL — HOLT-WINTERS
#     # ========================================================

#     try:
#         hw_model = ExponentialSmoothing(
#             train,
#             trend="add",
#             seasonal="add",
#             seasonal_periods=SEASONAL_PERIOD
#         ).fit()

#         hw_pred_log = hw_model.forecast(TEST_MONTHS)
#         hw_pred = inverse_log(hw_pred_log)

#         rmse_val = rmse(test_actual.values, hw_pred.values)
#         rmse_pct = (rmse_val / series.mean()) * 100 if series.mean() > 0 else np.inf

#         if zero_months <= MAX_ZERO_MONTHS and rmse_pct <= MAX_RMSE_PCT:
#             use_product = True

#             # ----- Store test period rows (ONLY when HW is used) -----
#             for dt, act, pred in zip(test_actual.index, test_actual.values, hw_pred.values):
#                 test_eval_rows.append({
#                     "Category_Name": cat,
#                     "Product_Name": prod,
#                     "Month": dt,
#                     "Actual_Demand": round(act, 0),
#                     "Predicted_Demand": round(max(pred, 0), 0),
#                     "Error": round(act - pred, 0),
#                     "Model": "Holt-Winters",
#                     "Level": "Product"
#                 })

#     except Exception:
#         pass

#     # ========================================================
#     # FINAL MODEL SELECTION
#     # ========================================================

#     if use_product:
#         level = "Product"
#         model_used = "Holt-Winters (Log)"

#         final_model = ExponentialSmoothing(
#             transformed,
#             trend="add",
#             seasonal="add",
#             seasonal_periods=SEASONAL_PERIOD
#         ).fit()

#         future_log = final_model.forecast(FUTURE_MONTHS)
#         future_forecast = inverse_log(future_log)

#     else:
#         # ----------------------------
#         # CATEGORY LEVEL — SARIMA
#         # (NO test-period comparison)
#         # ----------------------------
#         cat_series = (
#             df[df["Category_Name"] == cat]
#             .groupby("Month")["Order_Demand"]
#             .sum()
#             .asfreq("MS")
#             .fillna(0)
#         )

#         d = 0 if is_stationary(cat_series) else 1

#         model = SARIMAX(
#             cat_series,
#             order=(1, d, 1),
#             seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
#             enforce_stationarity=False,
#             enforce_invertibility=False
#         ).fit(disp=False)

#         future_forecast = model.forecast(FUTURE_MONTHS)

#         level = "Category"
#         model_used = "SARIMA"

#     # ========================================================
#     # STORE SUMMARY
#     # ========================================================

#     summary_rows.append({
#         "Category_Name": cat,
#         "Product_Name": prod,
#         "Model": model_used,
#         "Level": level,
#         "Zero_Months": zero_months,
#         "RMSE_%": round(rmse_pct, 2) if use_product else None
#     })

#     # ========================================================
#     # STORE FUTURE FORECAST
#     # ========================================================

#     future_dates = pd.date_range(
#         series.index.max() + pd.offsets.MonthBegin(1),
#         periods=FUTURE_MONTHS,
#         freq="MS"
#     )

#     for dt, val in zip(future_dates, future_forecast):
#         forecast_rows.append({
#             "Category_Name": cat,
#             "Product_Name": prod,
#             "Forecast_Month": dt,
#             "Forecast_Demand": round(max(val, 0), 0),
#             "Model": model_used,
#             "Level": level
#         })

# # ============================================================
# # EXPORT FILES
# # ============================================================

# pd.DataFrame(summary_rows).to_csv("model_summary_1.csv", index=False)
# pd.DataFrame(test_eval_rows).to_csv("test_period_actual_vs_predicted_1.csv", index=False)
# pd.DataFrame(forecast_rows).to_csv("future_forecast_1.csv", index=False)

# print(" Pipeline completed successfully")
# print(" Files generated:")
# print(" - model_summary_1.csv")
# print(" - test_period_actual_vs_predicted_1.csv")
# print(" - future_forecast_1.csv")











# ============================================================
# PRODUCT LEVEL DEMAND FORECASTING (BEST PRACTICE)
# Holt-Winters + Log + Outlier Capping + Rolling Evaluation
# ============================================================

import pandas as pd
import numpy as np
import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MIN_HISTORY = 36          # minimum months required
TEST_MONTHS = 6           # evaluation window
FUTURE_MONTHS = 12        # forecast horizon
SEASONAL_PERIOD = 12
MAX_ZERO_MONTHS = 24      # remove sparse SKUs

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "monthly_demand_data.csv",
    parse_dates=["Month"]
).sort_values(["Category_Name", "Product_Name", "Month"])

# ============================================================
# HELPERS
# ============================================================

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

def cap_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    return np.clip(series, 0, upper)

def log_transform(series):
    return np.log1p(series)

def inverse_log(series):
    return np.expm1(series)

# ============================================================
# OUTPUT STORAGE
# ============================================================

summary_rows = []
test_eval_rows = []
forecast_rows = []

# ============================================================
# PRODUCT LOOP
# ============================================================

for (cat, prod), grp in df.groupby(["Category_Name", "Product_Name"]):

    series = (
        grp.set_index("Month")["Order_Demand"]
        .asfreq("MS")
        .fillna(0)
    )

    # ----------------------------
    # REMOVE SPARSE / SHORT SERIES
    # ----------------------------
    if len(series) < MIN_HISTORY:
        continue

    zero_months = int((series == 0).sum())
    if zero_months > MAX_ZERO_MONTHS:
        continue

    # ----------------------------
    # PREPROCESSING
    # ----------------------------
    capped = cap_outliers(series)
    transformed = log_transform(capped)

    train = transformed[:-TEST_MONTHS]
    test_actual = series[-TEST_MONTHS:]

    # ========================================================
    # HOLT-WINTERS MODEL
    # ========================================================
    try:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD
        ).fit()

        pred_log = model.forecast(TEST_MONTHS)
        pred = inverse_log(pred_log)

        rmse_val = rmse(test_actual.values, pred.values)
        rmse_pct = (rmse_val / series.mean()) * 100 if series.mean() > 0 else np.nan

        # ----------------------------
        # STORE TEST PERIOD RESULTS
        # ----------------------------
        for dt, act, pr in zip(test_actual.index, test_actual.values, pred.values):
            test_eval_rows.append({
                "Category_Name": cat,
                "Product_Name": prod,
                "Month": dt,
                "Actual_Demand": round(act, 0),
                "Predicted_Demand": round(max(pr, 0), 0),
                "Error": round(act - pr, 0),
                "Model": "Holt-Winters",
                "Level": "Product"
            })

        # ----------------------------
        # FINAL MODEL (FULL DATA)
        # ----------------------------
        final_model = ExponentialSmoothing(
            transformed,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIOD
        ).fit()

        future_log = final_model.forecast(FUTURE_MONTHS)
        future_forecast = inverse_log(future_log)

        future_dates = pd.date_range(
            series.index.max() + pd.offsets.MonthBegin(1),
            periods=FUTURE_MONTHS,
            freq="MS"
        )

        for dt, val in zip(future_dates, future_forecast):
            forecast_rows.append({
                "Category_Name": cat,
                "Product_Name": prod,
                "Forecast_Month": dt,
                "Forecast_Demand": round(max(val, 0), 0),
                "Model": "Holt-Winters",
                "Level": "Product"
            })

        # ----------------------------
        # SUMMARY
        # ----------------------------
        summary_rows.append({
            "Category_Name": cat,
            "Product_Name": prod,
            "Model": "Holt-Winters",
            "Level": "Product",
            "Zero_Months": zero_months,
            "RMSE_%": round(rmse_pct, 2)
        })

    except Exception:
        continue

# ============================================================
# EXPORT FILES
# ============================================================

pd.DataFrame(summary_rows).to_csv("hw_model_summary.csv", index=False)
pd.DataFrame(test_eval_rows).to_csv("hw_test_actual_vs_predicted.csv", index=False)
pd.DataFrame(forecast_rows).to_csv("hw_future_forecast.csv", index=False)

print(" Product-level Holt-Winters forecasting completed")
print("Files generated:")
print(" - hw_model_summary.csv")
print(" - hw_test_actual_vs_predicted.csv")
print(" - hw_future_forecast.csv")
