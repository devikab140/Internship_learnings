import pandas as pd
import numpy as np
import warnings

from prophet import Prophet
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MIN_HISTORY = 24          # minimum months required
TEST_MONTHS = 6           # evaluation window
FUTURE_MONTHS = 12        # forecast horizon

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    "monthly_demand_data.csv",
    parse_dates=["Month"]
).sort_values(["Product_Name", "Month"])

# --------------------------------------------------
# OUTPUT STORAGE
# --------------------------------------------------

evaluation_output = []
forecast_output = []

# --------------------------------------------------
# PRODUCT-LEVEL PROPHET LOOP
# --------------------------------------------------

for product, grp in df.groupby("Product_Name"):

    # Prepare Prophet format
    ts = (
        grp[["Month", "Order_Demand"]]
        .rename(columns={"Month": "ds", "Order_Demand": "y"})
        .sort_values("ds")
    )

    if len(ts) < MIN_HISTORY:
        continue

    # ---------------------------
    # Train / Test Split
    # ---------------------------
    train_ts = ts.iloc[:-TEST_MONTHS]
    test_ts = ts.iloc[-TEST_MONTHS:]

    try:
        # ---------------------------
        # Train Prophet (Evaluation)
        # ---------------------------
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.1
        )

        model.fit(train_ts)

        # ---------------------------
        # Evaluate on Last 6 Months
        # ---------------------------
        future_test = model.make_future_dataframe(
            periods=TEST_MONTHS,
            freq="MS"
        )

        forecast_test = model.predict(future_test).tail(TEST_MONTHS)

        preds = np.clip(forecast_test["yhat"].values, 0, None)

        rmse = np.sqrt(mean_squared_error(test_ts["y"], preds))
        rmse_pct = rmse / test_ts["y"].mean() if test_ts["y"].mean() > 0 else np.inf

        for dt, actual, pred in zip(test_ts["ds"], test_ts["y"], preds):
            evaluation_output.append({
                "Product_Name": product,
                "Category_Name": grp["Category_Name"].iloc[0],
                "Month": dt,
                "Actual_Demand": actual,
                "Predicted_Demand": round(pred, 0),
                "RMSE": round(rmse, 2),
                "RMSE_%": round(rmse_pct * 100, 2),
                "Model": "Prophet_Product"
            })

        # ---------------------------
        # Final Model for Future
        # ---------------------------
        final_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.1
        )

        final_model.fit(ts)

        future = final_model.make_future_dataframe(
            periods=FUTURE_MONTHS,
            freq="MS"
        )

        forecast = final_model.predict(future).tail(FUTURE_MONTHS)

        for dt, val in zip(forecast["ds"], forecast["yhat"]):
            forecast_output.append({
                "Product_Name": product,
                "Category_Name": grp["Category_Name"].iloc[0],
                "Forecast_Month": dt,
                "Forecast_Demand": round(max(val, 0), 0),
                "Model": "Prophet_Product"
            })

    except Exception:
        continue

# --------------------------------------------------
# EXPORT
# --------------------------------------------------

pd.DataFrame(evaluation_output).to_csv(
    "prophet_product_evaluation.csv",
    index=False
)

pd.DataFrame(forecast_output).to_csv(
    "prophet_product_forecast.csv",
    index=False
)




