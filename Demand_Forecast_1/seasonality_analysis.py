# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import adfuller, acf, pacf
# import numpy as np

# # Load final prepared dataset
# daily_demand = pd.read_csv("daily_demand_prepared.csv")

# print(daily_demand.head())
# print(daily_demand.info())

# # Ensure it's sorted
# daily_demand = daily_demand.sort_values('date')
# # ----------------------------
# # SET DATE AS INDEX
# # ----------------------------
# daily_demand.set_index('date', inplace=True)

# # ----------------------------
# # PLOT DAILY TREND
# # ----------------------------
# plt.figure(figsize=(15,5))
# plt.plot(daily_demand['units_sold'], color='blue')
# plt.title('Daily Units Sold')
# plt.xlabel('Date')
# plt.ylabel('Units Sold')
# plt.grid(True)
# plt.show()

# # ----------------------------
# # DECOMPOSITION (Additive)
# # ----------------------------
# decomposition = seasonal_decompose(daily_demand['units_sold'], model='additive', period=7)  # weekly seasonality
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid

# plt.figure(figsize=(15,10))
# plt.subplot(411)
# plt.plot(daily_demand['units_sold'], label='Original')
# plt.legend(loc='upper left')
# plt.subplot(412)
# plt.plot(trend, label='Trend', color='orange')
# plt.legend(loc='upper left')
# plt.subplot(413)
# plt.plot(seasonal, label='Seasonality', color='green')
# plt.legend(loc='upper left')
# plt.subplot(414)
# plt.plot(residual, label='Residual', color='red')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# # ----------------------------
# # PLOT WEEKLY SEASONALITY
# # ----------------------------
# plt.figure(figsize=(10,5))
# sns.boxplot(x='day_of_week', y='units_sold', data=daily_demand.reset_index())
# plt.title('Units Sold by Day of Week')
# plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
# plt.ylabel('Units Sold')
# plt.show()

# # ----------------------------
# # PLOT MONTHLY SEASONALITY
# # ----------------------------
# plt.figure(figsize=(10,5))
# sns.boxplot(x='month', y='units_sold', data=daily_demand.reset_index())
# plt.title('Units Sold by Month')
# plt.xlabel('Month')
# plt.ylabel('Units Sold')
# plt.show()

# # ----------------------------
# # HOLIDAY AND FESTIVAL IMPACT
# # ----------------------------
# plt.figure(figsize=(10,5))
# sns.boxplot(x='is_holiday', y='units_sold', data=daily_demand.reset_index())
# plt.title('Units Sold: Holiday vs Non-Holiday')
# plt.show()

# plt.figure(figsize=(10,5))
# sns.boxplot(x='is_festival', y='units_sold', data=daily_demand.reset_index())
# plt.title('Units Sold: Festival vs Non-Festival')
# plt.show()

# # ----------------------------
# # YEARLY TREND
# # ----------------------------
# plt.figure(figsize=(12,5))
# sns.lineplot(x='date', y='units_sold', hue='year', data=daily_demand.reset_index(), marker='o')
# plt.title('Yearly Units Sold Trend')
# plt.show()

# # ----------------------------
# # AUTOCORRELATION AND PARTIAL AUTOCORRELATION
# # ----------------------------
# lag_acf = acf(daily_demand['units_sold'].dropna(), nlags=30)
# lag_pacf = pacf(daily_demand['units_sold'].dropna(), nlags=30)

# plt.figure(figsize=(12,5))
# plt.subplot(121)
# plt.stem(lag_acf)
# plt.title('Autocorrelation (ACF)')
# plt.subplot(122)
# plt.stem(lag_pacf)
# plt.title('Partial Autocorrelation (PACF)')
# plt.tight_layout()
# plt.show()

# # ----------------------------
# # CHECK STATIONARITY
# # ----------------------------
# result = adfuller(daily_demand['units_sold'].dropna())
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# if result[1] <= 0.05:
#     print("Series is likely stationary")
# else:
#     print("Series is likely non-stationary")

# # ----------------------------
# # CORRELATION WITH CALENDAR FEATURES
# # ----------------------------
# calendar_features = ['day_of_week', 'week_of_year', 'month', 'is_weekend', 'is_holiday', 'is_festival']
# corr = daily_demand[calendar_features + ['units_sold']].corr()
# plt.figure(figsize=(10,6))
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title('Correlation: Units Sold vs Calendar Features')
# plt.show()

























# ============================================
# Robust ARIMAX Forecasting with Auto ARIMA
# ============================================

import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -------------------------------
# 1. Load and Prepare Data
# -------------------------------

daily = pd.read_csv("daily_demand_prepared.csv", parse_dates=['date'])
daily.dropna(inplace=True)

# Weekly aggregation
weekly = (
    daily.groupby(['product_category', pd.Grouper(key='date', freq='W-MON')])['units_sold']
    .sum()
    .reset_index()
)
weekly.rename(columns={'units_sold':'weekly_units'}, inplace=True)

# -------------------------------
# 2. Add Holidays/Festivals as Exogenous
# -------------------------------

ind_holidays = holidays.India(years=range(2016, 2027))

weekly['is_holiday'] = weekly['date'].apply(lambda x: 1 if x in ind_holidays else 0)
exog_cols = ['is_holiday']

# -------------------------------
# 3. Forecast Function with Auto ARIMA
# -------------------------------

def forecast_arimax_auto(df, category, exog_cols, forecast_periods=104, log_transform=True):
    """
    df: weekly data
    category: category to forecast
    exog_cols: list of exogenous variables
    forecast_periods: number of weeks to forecast
    log_transform: whether to log-transform series to reduce spikes
    """
    data = df[df['product_category']==category].copy()
    data.set_index('date', inplace=True)
    
    y = data['weekly_units']
    X = data[exog_cols]

    # Log-transform for irregular/volatile series
    if log_transform:
        y = np.log1p(y)
    
    # Auto ARIMA to select best (p,d,q)
    stepwise_model = pm.auto_arima(
        y, exogenous=X,
        seasonal=False,  # change to True if strong seasonality detected
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        max_p=5, max_q=5, max_d=2
    )
    
    order = stepwise_model.order
    print(f"{category}: Selected ARIMA order {order}")
    
    # Fit final ARIMAX using SARIMAX
    model = SARIMAX(
        y,
        exog=X,
        order=order,
        seasonal_order=(0,0,0,0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    
    # Forecast future periods
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=forecast_periods, freq='W-MON')
    
    # Future exogenous
    future_exog = pd.DataFrame({'date': future_dates})
    future_exog['is_holiday'] = future_exog['date'].apply(lambda x: 1 if x in ind_holidays else 0)
    
    forecast = model_fit.get_forecast(steps=forecast_periods, exog=future_exog[exog_cols])
    
    forecast_mean = forecast.predicted_mean
    if log_transform:
        forecast_mean = np.expm1(forecast_mean)  # inverse log-transform

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast_units': forecast_mean,
        'lower_ci': np.expm1(forecast.conf_int().iloc[:,0]) if log_transform else forecast.conf_int().iloc[:,0],
        'upper_ci': np.expm1(forecast.conf_int().iloc[:,1]) if log_transform else forecast.conf_int().iloc[:,1],
        'product_category': category
    })
    
    return forecast_df

# -------------------------------
# 4. Forecast All Categories
# -------------------------------

categories = weekly['product_category'].unique()
all_forecasts = []

for cat in categories:
    print(f"Forecasting category: {cat}")
    fc = forecast_arimax_auto(weekly, cat, exog_cols, forecast_periods=104)
    all_forecasts.append(fc)

forecast_all = pd.concat(all_forecasts)
forecast_all.reset_index(drop=True, inplace=True)

# -------------------------------
# 5. Plot Example
# -------------------------------

example_cat = 'health_beauty'
plt.figure(figsize=(12,5))
orig = weekly[weekly['product_category']==example_cat].set_index('date')['weekly_units']
fc = forecast_all[forecast_all['product_category']==example_cat].set_index('date')['forecast_units']
plt.plot(orig, label='Actual')
plt.plot(fc, label='Forecast', color='orange')
plt.fill_between(fc.index,
                 forecast_all[forecast_all['product_category']==example_cat]['lower_ci'].values,
                 forecast_all[forecast_all['product_category']==example_cat]['upper_ci'].values,
                 color='orange', alpha=0.2)
plt.title(f'ARIMAX Forecast - {example_cat}')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.legend()
plt.show()

# -------------------------------
# 6. Save Forecast
# -------------------------------
forecast_all.to_csv("weekly_forecast_arimax_auto.csv", index=False)
print("Forecast saved to weekly_forecast_arimax_auto.csv")




# daily_sales['date'] = pd.to_datetime(daily_sales['date'])

# # Overall daily demand (all categories combined)
# overall_daily = (
#     daily_sales
#     .groupby('date')['units_sold']
#     .sum()
#     .reset_index()
# )

# overall_daily.set_index('date', inplace=True)

# #plot overall demand
# plt.figure(figsize=(12,4))
# plt.plot(overall_daily.index, overall_daily['units_sold'])
# plt.title("Overall Daily Demand (All Categories)")
# plt.xlabel("Date")
# plt.ylabel("Units Sold")
# plt.show()

# #decompose overall demand
# decomp = seasonal_decompose(
#     overall_daily['units_sold'],
#     model='additive',
#     period=7   # check weekly first
# )

# decomp.plot()
# plt.show()

# #category level analysis
# category_name = "electronics"

# cat_data = daily_sales[daily_sales['category'] == category_name]
# cat_data = cat_data.sort_values('date')
# cat_data.set_index('date', inplace=True)

# #acf plot for category
# plt.figure(figsize=(10,4))
# plot_acf(cat_data['units_sold'], lags=60)
# plt.title("ACF - Daily Units Sold")
# plt.show()

# #monthly aggregation and acf plot
# monthly_data = (
#     cat_data['units_sold']
#     .resample('M')
#     .sum()
# )

# max_lags = min(12, len(monthly_data)//2)

# plt.figure(figsize=(10,4))
# plot_acf(monthly_data, lags=max_lags)
# plt.title("ACF - Monthly Units Sold")
# plt.show()
