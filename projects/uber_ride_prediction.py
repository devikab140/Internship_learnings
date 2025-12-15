import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# ============================================================
# 0. Load Datasets
# ============================================================
data1 = pd.read_csv("C:/Users/devik/Downloads/booking_data.csv")
data2 = pd.read_csv("C:/Users/devik/OneDrive/Desktop/Coderzon/modified_datetime_data.csv", low_memory=False)

# ============================================================
# 1. Select Useful Features
# ============================================================
booking_features = [
    'Date', 'Time', 'Booking Status',
    'Pickup Location', 'Drop Location',
    'Vehicle Type', 'Avg VTAT', 'Avg CTAT',
    'Booking Value', 'Ride Distance',
    'Driver Ratings', 'Customer Rating',
    'Payment Method',
    'Cancelled Rides by Customer',
    'Cancelled Rides by Driver',
    'Incomplete Rides'
]

weather_features = [
    'datetime',
    'source', 'destination',
    'price', 'distance', 'surge_multiplier',
    'temperature', 'apparentTemperature',
    'short_summary', 'long_summary',
    'precipIntensity', 'precipProbability',
    'humidity', 'windSpeed', 'windGust',
    'visibility'
]

booking = data1[booking_features].copy()
weather = data2[weather_features].copy()

# ============================================================
# 2. Clean Booking Data
# ============================================================
booking["datetime"] = pd.to_datetime(
    booking["Date"] + " " + booking["Time"],
    format="%d-%m-%Y %H:%M:%S"
)
booking.drop(["Date", "Time"], axis=1, inplace=True)

num_cols = ['Avg VTAT','Avg CTAT','Booking Value','Ride Distance',
            'Driver Ratings','Customer Rating']
for col in num_cols:
    booking[col] = booking[col].fillna(booking[col].median())

booking['Payment Method'] = booking['Payment Method'].fillna(
    booking['Payment Method'].mode()[0]
)

zero_cols = [
    'Cancelled Rides by Customer',
    'Cancelled Rides by Driver',
    'Incomplete Rides'
]
for col in zero_cols:
    booking[col] = booking[col].fillna(0)

booking = booking.rename(columns={
    "Pickup Location": "source",
    "Drop Location": "destination"
})

booking["date"] = booking["datetime"].dt.date
booking["hour"] = booking["datetime"].dt.hour

# ============================================================
# 3. Clean Weather Data
# ============================================================
weather["datetime"] = pd.to_datetime(weather["datetime"], format="%d-%m-%Y %H:%M")
weather["price"] = weather["price"].fillna(weather["price"].median())
weather["source"] = weather["source"].fillna(weather["source"].mode()[0])
weather["destination"] = weather["destination"].fillna(weather["destination"].mode()[0])
weather["date"] = weather["datetime"].dt.date
weather["hour"] = weather["datetime"].dt.hour

# ============================================================
# 4. Merge Datasets
# ============================================================
merged_df = pd.merge(
    booking, weather,
    on=["source", "destination", "date", "hour"],
    how="left"
)
merged_df = merged_df.drop_duplicates().reset_index(drop=True)
merged_df = merged_df.rename(columns={"datetime_x": "datetime"})
if "datetime_y" in merged_df.columns:
    merged_df.drop(columns=["datetime_y"], inplace=True)

# ============================================================
# 5. Ride Count (Target)
# ============================================================
ride_count_df = merged_df.groupby(
    ["source", "date", "hour"]
).size().reset_index(name="ride_count")

merged_df = pd.merge(
    merged_df, ride_count_df,
    on=["source", "date", "hour"],
    how="left"
)

# ============================================================
# 6. Time Features
# ============================================================
merged_df["day_of_week"] = merged_df["datetime"].dt.dayofweek
merged_df["is_weekend"] = merged_df["day_of_week"].isin([5, 6]).astype(int)

# ============================================================
# 7. Label Encoding
# ============================================================
cat_cols = merged_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    merged_df[col] = le.fit_transform(merged_df[col].astype(str))

# ============================================================
# 8. Prepare X & y
# ============================================================
drop_cols = [
    "datetime", "timestamp",
    "Booking Status",
    "Driver Ratings", "Customer Rating",
    "Avg CTAT", "Avg VTAT",
    "Booking Value", "Payment Method"
]
merged_df = merged_df.drop(columns=drop_cols, errors='ignore')

X = merged_df.drop(columns=["ride_count"], errors='ignore')
y = merged_df["ride_count"]

# ============================================================
# 9. Handle Remaining NaN Values
# ============================================================
for col in X.columns:
    if X[col].dtype in [np.float64, np.int64]:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(X[col].mode()[0])

# ============================================================
# 10. Train-Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 11A. Random Forest
# ============================================================
rf_model = RandomForestRegressor(
    n_estimators=80,
    max_depth=12,
    min_samples_split=40,
    min_samples_leaf=20,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n===== RANDOM FOREST RESULTS =====")
print("MSE :", mean_squared_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2 Score:", r2_score(y_test, rf_pred))


# ============================================================
# 11C. XGBoost Regressor
# ============================================================
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("\n===== XGBOOST RESULTS =====")
print("MSE :", mean_squared_error(y_test, xgb_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))
print("R2 Score:", r2_score(y_test, xgb_pred))

# ============================================================
# 11B. SVR (Scaled)
# ============================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

svr_model = SVR(kernel="rbf", C=10, epsilon=0.1)
svr_model.fit(X_train_s, y_train)
svr_pred = svr_model.predict(X_test_s)

print("\n===== SVR RESULTS =====")
print("MSE :", mean_squared_error(y_test, svr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, svr_pred)))
print("R2 Score:", r2_score(y_test, svr_pred))
