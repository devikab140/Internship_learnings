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

engine=df[['Engine_Size (L)']]  #sklearn expect X as 2D
MPG=df['MPG (Miles_per_Gallon)']
n1=len(engine)
# print(engine)
# print(MPG)

model=LinearRegression()
model.fit(engine,MPG)
pred=model.predict(engine)
# print(pred)

print(f"Coeficient is {model.coef_}")
print(f"Intercept is {model.intercept_}")

print("MAE: ",mean_absolute_error(MPG,pred))
print("MSE: ",mean_squared_error(MPG,pred))
print("RMSE: ",root_mean_squared_error(MPG,pred))
print("R2 Score: ",r2_score(MPG,pred))
print("MBA: ",MBA(n1,pred,MPG))

plt.scatter(engine,MPG,color="blue",marker="o")
plt.xlabel("Engine")
plt.ylabel("MPG")
plt.plot(engine,pred,color="red")
plt.show()


print("#-------------------------------------------#")

X=df[['Engine_Size (L)']]
y=df['MPG (Miles_per_Gallon)']
n=len(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

Model=LinearRegression()
Model.fit(X_train,y_train)
print(f"Coeficient is {Model.coef_}")
print(f"Intercept is {Model.intercept_}")

pred_y=Model.predict(X_test)
# print(pred_y)

print("MAE: ",mean_absolute_error(y_test,pred_y))
print("MSE: ",mean_squared_error(y_test,pred_y))
print("RMSE: ",root_mean_squared_error(y_test,pred_y))
print("R2 Score: ",r2_score(y_test,pred_y))
print("MBA: ",MBA(n,pred_y,y_test))



plt.scatter(X,y,color="blue",marker="o")
plt.xlabel("Engine")
plt.ylabel("MPG")
plt.plot(X,Model.predict(X),color="red")
plt.show()