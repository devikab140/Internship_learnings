import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ================================
# 1. LOAD DATA
# ================================
daily_demand = pd.read_csv(
    "daily_demand_prepared.csv",
    parse_dates=['date']
)

daily_demand.sort_values('date', inplace=True)
daily_demand.set_index('date', inplace=True)

y = daily_demand['units_sold']
X = daily_demand[['is_festival', 'is_holiday']]

# ================================
# 2. TRAIN FINAL MODEL ON FULL DATA
# ================================

model = SARIMAX(
    y,
    exog=X,
    order=(1, 0, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)

print("Final ARIMAX model trained on full data")

# ================================
# 3. DEFINE FUTURE PERIOD
# ================================

FORECAST_DAYS = 365  # forecast 1 year beyond 2018

last_date = daily_demand.index.max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=FORECAST_DAYS,
    freq='D'
)

# ================================
# 4. CREATE FUTURE EXOG VARIABLES
# ================================

# Brazilian holidays
br_holidays = holidays.Brazil()

future_is_holiday = future_dates.isin(br_holidays).astype(int)

# Festival dates (update yearly if needed)
festival_dates = {
    'Carnival': ['2019-03-04'],
    'Black_Friday': ['2019-11-29'],
    'Christmas': ['2019-12-25'],
    'New_Year': ['2019-01-01']
}

future_is_festival = []

for d in future_dates:
    flag = 0
    for dates in festival_dates.values():
        if d in pd.to_datetime(dates):
            flag = 1
    future_is_festival.append(flag)

future_exog = pd.DataFrame({
    'is_festival': future_is_festival,
    'is_holiday': future_is_holiday
}, index=future_dates)

# ================================
# 5. FORECAST FUTURE
# ================================

forecast = model_fit.get_forecast(
    steps=FORECAST_DAYS,
    exog=future_exog
)

forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# ================================
# 6. BUILD FORECAST DATAFRAME
# ================================

forecast_df = pd.DataFrame({
    'date': future_dates,
    'forecast_units_sold': forecast_mean.values,
    'lower_ci': forecast_ci.iloc[:, 0].values,
    'upper_ci': forecast_ci.iloc[:, 1].values,
    'is_festival': future_exog['is_festival'].values,
    'is_holiday': future_exog['is_holiday'].values
})

# ================================
# 7. VISUALIZE
# ================================

plt.figure(figsize=(14, 6))

# Historical
plt.plot(
    daily_demand.index,
    daily_demand['units_sold'],
    label='Historical',
    alpha=0.6
)

# Forecast
plt.plot(
    forecast_df['date'],
    forecast_df['forecast_units_sold'],
    label='Forecast (Post-2018)',
    color='red'
)

# Confidence Interval
plt.fill_between(
    forecast_df['date'],
    forecast_df['lower_ci'],
    forecast_df['upper_ci'],
    color='red',
    alpha=0.2,
    label='Confidence Interval'
)

# Festival markers
festival_mask = forecast_df['is_festival'] == 1
plt.scatter(
    forecast_df.loc[festival_mask, 'date'],
    forecast_df.loc[festival_mask, 'forecast_units_sold'],
    color='orange',
    s=60,
    label='Festival Days'
)

plt.title('Future Demand Forecast Beyond 2018 (ARIMAX)')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# 8. SAVE FOR DASHBOARD
# ================================

forecast_df.to_csv(
    "future_demand_forecast_2019_onwards.csv",
    index=False
)

print("Future forecast saved successfully!")
