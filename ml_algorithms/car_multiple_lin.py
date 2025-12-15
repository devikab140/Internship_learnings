# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression

# data=pd.read_csv("C:/Users/devik/Downloads/car_data.csv")
# df=pd.DataFrame(data)

# X=df[['Engine_Size (L)','Weight (kg)','Horsepower']]
# y=df['MPG (Miles_per_Gallon)']

# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)
# y_scaled=(y - y.mean()) / y.std()

# model=LinearRegression()
# model.fit(X_scaled,y_scaled)
# coef=pd.Series(model.coef_,index=X.columns)
# print(coef)





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
import numpy as np

def MBA(n,y_cap,y_or):
    return ((1/n)*np.sum(y_cap-y_or))

df=pd.read_csv("C:/Users/devik/Downloads/car_data.csv")
# print(df.head())

X_vals=df[['Engine_Size (L)','Weight (kg)','Horsepower']]  #sklearn expect X as 2D
MPG=df['MPG (Miles_per_Gallon)']
n1=len(MPG)
# print(X_vals)
# print(MPG)

model=LinearRegression()
model.fit(X_vals,MPG)
pred=model.predict(X_vals)
print(model.predict([[3.2,1500,130]]))
# print(pred)

# print(f"Coeficient is {model.coef_}")
# print(f"Intercept is {model.intercept_}")

# print("MAE: ",mean_absolute_error(MPG,pred))
# print("MSE: ",mean_squared_error(MPG,pred))
# print("RMSE: ",root_mean_squared_error(MPG,pred))
# print("R2 Score: ",r2_score(MPG,pred))
# print("MBA: ",MBA(n1,pred,MPG))

# features = ['Engine_Size (L)', 'Weight (kg)', 'Horsepower']

# plt.figure(figsize=(15, 5))

# for i, feature in enumerate(features):
#     plt.subplot(1, 3, i + 1)
#     plt.scatter(df[feature], MPG, color='blue', label='Actual MPG', alpha=0.7)
#     plt.scatter(df[feature], pred, color='red', label='Predicted MPG', alpha=0.7)
    
#     plt.xlabel(feature)
#     plt.ylabel("MPG")
#     plt.title(f"{feature} vs MPG")
#     plt.legend()
#     plt.grid(True)

# plt.tight_layout()
# plt.show()

# # print("#-------------------------------------------#")

# # --- Import libraries ---
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# # --- Load dataset ---
# df = pd.read_csv("C:/Users/devik/Downloads/car_data.csv")

# # --- Define features (X) and target (y) ---
# X = df[['Engine_Size (L)', 'Weight (kg)', 'Horsepower']]
# y = df['MPG (Miles_per_Gallon)']

# # --- Split into training and testing sets ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # --- Create and train model ---
# model = LinearRegression()
# model.fit(X_train, y_train)

# # --- Predict on test data ---
# y_pred = model.predict(X_test)

# # --- Evaluate performance ---
# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)
# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("RMSE:", root_mean_squared_error(y_test, y_pred))
# print("R² Score:", r2_score(y_test, y_pred))
# print("MBA: ",MBA(n1,y_pred,y_test))
# # --- Plot Actual vs Predicted MPG ---
# plt.figure(figsize=(7,6))
# plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
#          color='red', linewidth=2, label='Perfect Prediction Line')

# plt.xlabel("Actual MPG")
# plt.ylabel("Predicted MPG")
# plt.title("Actual vs Predicted MPG (Test Data)")
# plt.legend()
# plt.grid(True)
# plt.show()





# # plt.scatter(X,y,color="blue",marker="o")
# # plt.xlabel("Engine")
# # plt.ylabel("MPG")
# # plt.plot(X,Model.predict(X),color="red")
# # plt.show()