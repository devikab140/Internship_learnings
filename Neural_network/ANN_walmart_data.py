import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
# ---------------------------
# 1. Load Walmart dataset
# ---------------------------
df = pd.read_csv("C:/Users/devik/Downloads/Walmart.csv")

# Basic checks
print(df.head())
print(df.columns)
print(df.isnull().sum())

# ---------------------------
# 2. Data cleaning
# ---------------------------
df = df.dropna()

# ---------------------------
# 3. Feature selection
# ---------------------------
X = df[['Store', 'Unemployment', 'Holiday_Flag',
        'Temperature', 'Fuel_Price', 'CPI']]
y = df['Weekly_Sales']

# ---------------------------
# 4. Train-test split FIRST
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 5. Scale features (NO leakage)
# ---------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_ = y_train.values.reshape(-1, 1)
y_train_Scaled = scaler.fit_transform(y_train_)
y_test_ = y_test.values.reshape(-1, 1)
y_test_Scaled = scaler.transform(y_test_)

# ---------------------------
# 6. Build ANN model
# ---------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='tanh'),
    Dense(16, activation='relu'),
    Dense(1)   
])

# ---------------------------
# 7. Compile model
# ---------------------------
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mse']
)

# ---------------------------
# 8. Train model
# ---------------------------
model.fit(
    X_train_scaled,
    y_train_Scaled,
    epochs=1000,
    verbose=0
)

# ---------------------------
# 9. Evaluate model
# ---------------------------
train_loss, train_mse = model.evaluate(
    X_train_scaled, y_train_Scaled, verbose=0
)
test_loss, test_mse = model.evaluate(
    X_test_scaled, y_test_Scaled, verbose=0
)

print(f"Train MSE: {train_mse}")
print(f"Train Loss: {train_loss}")
print(f"Test MSE: {test_mse}")
print(f"Test Loss: {test_loss}")
print("RMSE on Test Set:", test_mse ** 0.5)
print("RMSE on Train Set:", train_mse ** 0.5)

#printing the r2 score

y_test_pred_scaled = model.predict(X_test_scaled)
y_test_pred = scaler.inverse_transform(y_test_pred_scaled)
r2 = r2_score(y_test, y_test_pred)
print(f"R2 Score on Test Set: {r2}")


########################################################################################

#comparing with ml models
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

Rf_model=RandomForestRegressor(n_estimators=100,random_state=42)
Rf_model.fit(X_train,y_train)

y_pred=Rf_model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Random Forest Regressor MSE: {mse}")
print(f"Random Forest Regressor R2 Score: {r2}")
