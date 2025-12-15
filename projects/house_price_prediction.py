import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.model_selection import train_test_split

data=pd.read_csv("C:/Users/devik/Downloads/price_data.csv")
data_frame=pd.DataFrame(data)
print(data_frame.columns)
df=data_frame[['price', 'bedrooms','sqft_living', 'sqft_lot','floors','sqft_above','sqft_basement']]
# print(df.head())

X=df[['bedrooms','sqft_living', 'sqft_lot','floors','sqft_above','sqft_basement']]
y=df['price']

# print(X.head())
# print(y.head())

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
y_scaled= (y - y.mean()) / y.std()
# print(X_scaled)
# print(y_scaled)

model=LinearRegression()
model.fit(X_scaled,y_scaled)
pred_y=model.predict(X_scaled)
print(pred_y)

coef=pd.Series(model.coef_,index=X.columns)
print('Coeficients are ')
print(coef)
print('Intercept is ',model.intercept_)
print("R2 score",r2_score(y_scaled,pred_y))
print("RMSE",root_mean_squared_error(y_scaled,pred_y))

print(model.predict([[-.75,-1.07,-1.02,0.5,-.48,-1.01]]))



#train test
print("#----------------------------------------------------------------------#")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
model1=LinearRegression()
model1.fit(X_train, y_train)

# --- Predict on test data ---
y_pred = model1.predict(X_test)

# --- Evaluate performance ---
print("Coefficients:", model1.coef_)
print("Intercept:", model1.intercept_)
print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


#without scaling
print("#-----------------------------------------------------------------------------#")

X=df[['bedrooms','sqft_living', 'sqft_lot','floors','sqft_above','sqft_basement']]
y=df['price']

model2=LinearRegression()
model2.fit(X,y)
pred=model2.predict(X)

print("Coefficients:", model2.coef_)
print("Intercept:", model2.intercept_)
print("RMSE:", root_mean_squared_error(y,pred))
print("R2 Score:", r2_score(y,pred))

print(model2.predict([[3,1340,7912,15,1340,0]]))

#train test

print("#------------------------------------------------------------------#")

x_train, x_test, Y_train, Y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model3=LinearRegression()
model3.fit(x_train, Y_train)

# --- Predict on test data ---
Y_pred = model3.predict(x_test)

# --- Evaluate performance ---
print("Coefficients:", model3.coef_)
print("Intercept:", model3.intercept_)
print("RMSE:", root_mean_squared_error(Y_test, Y_pred))
print("R2 Score:", r2_score(Y_test, Y_pred))



