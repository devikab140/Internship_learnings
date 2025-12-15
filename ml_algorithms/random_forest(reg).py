import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("C:/Users/devik/Downloads/house_price_data.csv")
print(data.columns)

# Data preprocessing
X=data[['bedrooms','sqft_living', 'sqft_lot','floors','sqft_above','sqft_basement']]
scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)
y=data['price']

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Using RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print("Predicted Y", y_pred[0:5])
print("Actual Y", y_test.head(5).values)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

