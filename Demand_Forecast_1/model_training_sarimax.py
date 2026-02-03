# import pandas as pd
# import numpy as np
# import holidays
# import matplotlib.pyplot as plt

# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.graphics.tsaplots import plot_acf
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # =========================
# # LOAD DAILY DATA
# # =========================
# daily = pd.read_csv(
#     "daily_demand_prepared.csv",
#     parse_dates=['date']
# )

# daily.sort_values('date', inplace=True)
# daily.set_index('date', inplace=True)

# # =========================
# # WEEKLY AGGREGATION BY CATEGORY
# # =========================
# weekly_list = []

# for cat in daily['product_category'].unique():
#     temp = daily[daily['product_category'] == cat]
#     weekly_cat = temp.resample('W').agg({
#         'units_sold': 'sum',
#         'is_festival': 'max',
#         'is_holiday': 'max'
#     })
#     weekly_cat['product_category'] = cat
#     weekly_list.append(weekly_cat)

# weekly = pd.concat(weekly_list).reset_index()
# weekly.to_csv("weekly_demand_prepared.csv", index=False)
# print("Weekly category-wise data ready")
# print(weekly.head())

# # =========================
# # ACF SEASONALITY CHECK
# # =========================
# sample_cat = weekly['product_category'].iloc[0]
# sample_series = (
#     weekly[weekly['product_category'] == sample_cat]
#     .set_index('date')['units_sold']
# )

# plt.figure(figsize=(12,4))
# plot_acf(sample_series.dropna(), lags=60)
# plt.title(f"ACF – Weekly Demand ({sample_cat})")
# plt.show()

# # =========================
# # SARIMAX TRAINING FUNCTION
# # =========================
# DATA_FREQUENCY = "W"      
# FORECAST_HORIZON = 8       # forecast next 8 weeks
# SEASONAL_PERIOD = 52       # yearly seasonality

# def train_sarimax(series, exog):
#     train_size = int(len(series) * 0.8)
#     y_train = series.iloc[:train_size]
#     y_test = series.iloc[train_size:]

#     X_train = exog.iloc[:train_size]
#     X_test = exog.iloc[train_size:]

#     model = SARIMAX(
#         y_train,
#         exog=X_train,
#         order=(1, 1, 1),
#         seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
#         enforce_stationarity=False,
#         enforce_invertibility=False
#     )
    
#     # fit model safely
#     fitted = model.fit(disp=False)

#     forecast = fitted.get_forecast(
#         steps=len(y_test),
#         exog=X_test
#     )
#     y_pred = forecast.predicted_mean

#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     mae = mean_absolute_error(y_test, y_pred)
#     return fitted, rmse, mae

# # =========================
# # TRAIN MODELS FOR ALL CATEGORIES
# # =========================
# # =========================
# # TRAIN MODELS FOR ALL CATEGORIES
# # =========================
# results = []
# models = {}

# for cat in weekly['product_category'].unique():
#     cat_df = weekly[weekly['product_category'] == cat].copy()
#     cat_df.set_index('date', inplace=True)

#     y = cat_df['units_sold']
#     X = cat_df[['is_festival', 'is_holiday']]

#     # allow categories with at least 1 year of weekly data
#     if len(y) < SEASONAL_PERIOD:
#         print(f"Skipping {cat} – not enough data ({len(y)} weeks)")
#         continue

#     try:
#         fitted, rmse, mae = train_sarimax(y, X)
#         models[cat] = fitted
#         results.append({
#             'product_category': cat,
#             'rmse': rmse,
#             'mae': mae
#         })
#         print(f"Trained SARIMAX for {cat} – RMSE: {rmse:.2f}, MAE: {mae:.2f}")
#     except Exception as e:
#         print(f"Failed to train {cat}: {e}")

# # save evaluation if we have results
# if results:
#     eval_df = pd.DataFrame(results).sort_values('rmse')
#     eval_df.to_csv("weekly_model_evaluation.csv", index=False)
#     print(eval_df.head())
# else:
#     print("No categories have enough data for training!")


# # =========================
# # FUTURE FORECAST FUNCTIONS
# # =========================
# def create_future_exog(dates):
#     br_holidays = holidays.Brazil()
#     return pd.DataFrame({
#         'is_festival': 0,
#         'is_holiday': dates.isin(br_holidays).astype(int)
#     }, index=dates)

# def forecast_future(fitted_model, last_date, horizon=FORECAST_HORIZON):
#     future_dates = pd.date_range(
#         start=last_date + pd.tseries.frequencies.to_offset(DATA_FREQUENCY),
#         periods=horizon,
#         freq=DATA_FREQUENCY
#     )

#     future_exog = create_future_exog(future_dates)

#     forecast = fitted_model.get_forecast(
#         steps=horizon,
#         exog=future_exog
#     )

#     return pd.DataFrame({
#         'date': future_dates,
#         'forecast_units_sold': forecast.predicted_mean.values,
#         'lower_ci': forecast.conf_int().iloc[:, 0].values,
#         'upper_ci': forecast.conf_int().iloc[:, 1].values
#     })

# # =========================
# # FORECAST ALL CATEGORIES
# # =========================
# forecast_list = []

# for cat, model in models.items():
#     last_date = weekly[weekly['product_category'] == cat]['date'].max()
#     forecast_df = forecast_future(model, last_date)
#     forecast_df['product_category'] = cat
#     forecast_list.append(forecast_df)

# final_forecast = pd.concat(forecast_list)
# final_forecast.to_csv("weekly_category_forecast.csv", index=False)
# print("Future weekly forecasts saved")
# print(final_forecast.head())



# ============================================================
# PER-CATEGORY ARIMAX — TRAIN / TEST / PLOTS / INSIGHTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# USER CONTROLS
# ============================================================

FORECAST_HORIZON = 7     # 7 = next week | 30 = next month
MAX_PLOTS = 3           # plot only few categories

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "daily_demand_prepared.csv",
    parse_dates=["date"]
)

df.sort_values(["product_category", "date"], inplace=True)

# ============================================================
# 2. STORAGE
# ============================================================

results = []
plot_count = 0

# ============================================================
# 3. LOOP PER CATEGORY
# ============================================================

for category in df["product_category"].unique():

    df_cat = df[df["product_category"] == category].copy()
    df_cat = df_cat.sort_values("date").reset_index(drop=True)

    # skip very small categories
    if len(df_cat) < 80:
        continue

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    df_cat["lag1"] = df_cat["units_sold"].shift(1)
    df_cat["rolling7"] = df_cat["units_sold"].rolling(7).mean()
    df_cat.dropna(inplace=True)

    # --------------------------------------------------------
    # Train / Test Split (80 / 20)
    # --------------------------------------------------------

    split = int(len(df_cat) * 0.8)

    train = df_cat.iloc[:split].copy()
    test  = df_cat.iloc[split:].copy()

    y_train = train.set_index("date")["units_sold"]
    y_test  = test.set_index("date")["units_sold"]

    exog_cols = ["is_festival", "is_holiday", "lag1", "rolling7"]

    X_train = train.set_index("date")[exog_cols]
    X_test  = test.set_index("date")[exog_cols]

    # --------------------------------------------------------
    # Train ARIMAX (KEEP DATETIME INDEX)
    # --------------------------------------------------------

    try:
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=(1, 0, 1),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        model_fit = model.fit(disp=False)

        # ----------------------------------------------------
        # TEST FORECAST
        # ----------------------------------------------------

        forecast_test = model_fit.get_forecast(
            steps=len(X_test),
            exog=X_test
        )

        y_test_pred = pd.Series(
            forecast_test.predicted_mean.values,
            index=y_test.index
        )

        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae  = mean_absolute_error(y_test, y_test_pred)

        # ----------------------------------------------------
        # PLOT (LIMITED)
        # ----------------------------------------------------

        if plot_count < MAX_PLOTS and y_test.sum() > 0:

            plt.figure(figsize=(10, 4))

            plt.plot(
                y_test.index,
                y_test.values,
                label="Actual",
                linewidth=2
            )

            plt.plot(
                y_test_pred.index,
                y_test_pred.values,
                label="Forecast",
                linestyle="--"
            )

            plt.title(f"Actual vs Forecast — {category}")
            plt.xlabel("Date")
            plt.ylabel("Units Sold")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

            plot_count += 1

        # ----------------------------------------------------
        # FUTURE FORECAST (NEXT WEEK / MONTH)
        # ----------------------------------------------------

        last_lag = df_cat["units_sold"].iloc[-1]
        last_roll = df_cat["units_sold"].tail(7).mean()

        future_exog = pd.DataFrame({
            "is_festival": [0] * FORECAST_HORIZON,
            "is_holiday":  [0] * FORECAST_HORIZON,
            "lag1":        [last_lag] * FORECAST_HORIZON,
            "rolling7":    [last_roll] * FORECAST_HORIZON
        })

        future_forecast = model_fit.get_forecast(
            steps=FORECAST_HORIZON,
            exog=future_exog
        )

        future_avg = future_forecast.predicted_mean.mean()
        last_avg   = df_cat["units_sold"].tail(FORECAST_HORIZON).mean()

        trend = "INCREASING" if future_avg > last_avg else "DECREASING"

        results.append({
            "category": category,
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "last_avg_demand": round(last_avg, 2),
            "future_avg_demand": round(future_avg, 2),
            "trend": trend
        })

    except Exception as e:
        print(f"Failed for {category}: {e}")

# ============================================================
# 4. FINAL INSIGHTS
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("future_avg_demand", ascending=False)

print("\n HIGHEST DEMAND CATEGORIES (FUTURE)")
print(results_df.head(5))

print("\n CATEGORIES WITH INCREASING DEMAND")
print(results_df[results_df["trend"] == "INCREASING"].head(10))

# ============================================================
# 5. SAVE RESULTS
# ============================================================

results_df.to_csv("category_demand_forecast_insights.csv", index=False)
print("\n Saved: category_demand_forecast_insights.csv")
