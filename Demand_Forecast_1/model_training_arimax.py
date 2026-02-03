# # =========================
# # ARIMAX MODEL WITH FESTIVALS & HOLIDAYS
# # =========================

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# # -------------------------
# # 1. Load prepared data
# # -------------------------
# daily_demand = pd.read_csv("daily_demand_prepared.csv")
# # -------------------------
# # 2. Define target & exogenous variables
# # -------------------------
# target = daily_demand['units_sold']

# # Exogenous variables: festivals + holidays
# exog = daily_demand[['is_festival', 'is_holiday']]

# # -------------------------
# # 3. Train-test split (80-20)
# # -------------------------
# train_size = int(len(target) * 0.8)
# y_train, y_test = target[:train_size], target[train_size:]
# exog_train, exog_test = exog[:train_size], exog[train_size:]

# print(f"Training points: {len(y_train)}, Testing points: {len(y_test)}")

# # -------------------------
# # 4. Fit ARIMAX model
# # -------------------------
# # Starting with ARIMA(1,0,1) + exogenous regressors
# model = SARIMAX(y_train, exog=exog_train, order=(1, 0, 1), enforce_stationarity=True, enforce_invertibility=True)
# arimax_result = model.fit(disp=False)

# print(arimax_result.summary())

# # -------------------------
# # 5. Forecast on test set
# # -------------------------
# forecast = arimax_result.get_forecast(steps=len(y_test), exog=exog_test)
# y_pred = forecast.predicted_mean

# conf_int = forecast.conf_int()

# # -------------------------
# # 6. Evaluation
# # -------------------------
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# mae = mean_absolute_error(y_test, y_pred)

# print(f"ARIMAX(1,0,1) + festival/holiday RMSE: {rmse:.2f}")
# print(f"ARIMAX(1,0,1) + festival/holiday MAE: {mae:.2f}")

# # -------------------------
# # 7. Plot actual vs predicted
# # -------------------------
# plt.figure(figsize=(15,6))
# plt.plot(y_test.index, y_test, label='Actual', color='blue')
# plt.plot(y_test.index, y_pred, label='ARIMAX Forecast', color='red')
# plt.fill_between(y_test.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink', alpha=0.3)
# plt.title("ARIMAX Forecast vs Actual Units Sold")
# plt.xlabel("Date")
# plt.ylabel("Units Sold")
# plt.legend()
# plt.show()

# # -------------------------
# # 8. Plot residuals
# # -------------------------
# residuals = y_test - y_pred
# plt.figure(figsize=(12,5))
# plt.plot(residuals)
# plt.title("Residuals of ARIMAX Model")
# plt.show()

# # Histogram of residuals
# plt.figure(figsize=(10,5))
# sns.histplot(residuals, bins=50, kde=True)
# plt.title("Residual Distribution")
# plt.show()

# # -------------------------
# # 9. Optional: Zoom-in on spikes (festival impact)
# # -------------------------


# festival_days = daily_demand[daily_demand['is_festival']==1].index
# plt.figure(figsize=(15,5))
# plt.plot(y_test.index, y_test, label='Actual', color='blue')
# plt.scatter(festival_days, y_test.loc[festival_days], color='orange', label='Festival Days', s=50)
# plt.title("Festival Days vs Actual Sales")
# plt.legend()
# plt.show()






# =========================
# ARIMAX MODEL WITH FESTIVALS & HOLIDAYS
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# 1. Load prepared data
# -------------------------
daily_demand = pd.read_csv("daily_demand_prepared.csv", parse_dates=['date'])
daily_demand.sort_values('date', inplace=True)
daily_demand.reset_index(drop=True, inplace=True)

# -------------------------
# 2. Define target & exogenous variables
# -------------------------
target = daily_demand['units_sold']
exog = daily_demand[['is_festival', 'is_holiday']]

# -------------------------
# 3. Train-test split (80-20)
# -------------------------
train_size = int(len(target) * 0.8)
y_train, y_test = target[:train_size], target[train_size:]
exog_train, exog_test = exog[:train_size], exog[train_size:]

print(f"Training points: {len(y_train)}, Testing points: {len(y_test)}")

# -------------------------
# 4. Fit ARIMAX model
# -------------------------
model = SARIMAX(y_train, exog=exog_train, order=(1, 0, 1),
                enforce_stationarity=True, enforce_invertibility=True)
arimax_result = model.fit(disp=False)

print(arimax_result.summary())

# -------------------------
# 5. Forecast on test set
# -------------------------
forecast = arimax_result.get_forecast(steps=len(y_test), exog=exog_test)
y_pred = forecast.predicted_mean
conf_int = forecast.conf_int()

# -------------------------
# 6. Evaluation
# -------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"ARIMAX(1,0,1) + festival/holiday RMSE: {rmse:.2f}")
print(f"ARIMAX(1,0,1) + festival/holiday MAE: {mae:.2f}")

# -------------------------
# 7. Plot actual vs predicted
# -------------------------
plt.figure(figsize=(15,6))
plt.plot(daily_demand['date'][train_size:], y_test, label='Actual', color='blue')
plt.plot(daily_demand['date'][train_size:], y_pred, label='ARIMAX Forecast', color='red')
plt.fill_between(daily_demand['date'][train_size:],
                 conf_int.iloc[:,0], conf_int.iloc[:,1],
                 color='pink', alpha=0.3)
plt.title("ARIMAX Forecast vs Actual Units Sold")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.show()

# -------------------------
# 8. Plot residuals
# -------------------------
residuals = y_test - y_pred
plt.figure(figsize=(12,5))
plt.plot(daily_demand['date'][train_size:], residuals)
plt.title("Residuals of ARIMAX Model")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residual Distribution")
plt.show()

# -------------------------
# 9. Optional: Zoom-in on festival spikes
# -------------------------
festival_days = daily_demand[daily_demand['is_festival']==1].index
plt.figure(figsize=(15,5))
plt.plot(daily_demand['date'][train_size:], y_test, label='Actual', color='blue')
# Only mark festival days in test set
festival_test_days = [i for i in festival_days if i >= train_size]
plt.scatter(daily_demand['date'][festival_test_days], y_test.loc[festival_test_days],
            color='orange', label='Festival Days', s=50)
plt.title("Festival Days vs Actual Sales")
plt.legend()
plt.show()

# -------------------------
# 10. Save predicted values to CSV
# -------------------------
# Create a copy of the dataset
output_df = daily_demand.copy()
# Add forecast column (NaN for train set)
output_df['forecast_units_sold'] = np.nan
output_df.loc[train_size:, 'forecast_units_sold'] = y_pred.values

# Save to CSV
output_df.to_csv("daily_demand_with_forecast.csv", index=False)
print("CSV saved with forecasted values: daily_demand_with_forecast.csv")
