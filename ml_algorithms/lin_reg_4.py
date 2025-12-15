import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
from sklearn.model_selection import train_test_split



def MBA(n,y_cap,y_or):
    return ((1/n)*np.sum(y_cap-y_or))

df=pd.DataFrame({
    'area':[1,2,3,4,5],
    'price':[75,150,290,310,350]
})

X1=df.drop("price",axis='columns')
y1=df.price
n1=len(X1)

model1=LinearRegression()
model1.fit(X1,y1)
print(f"Coeficient is {model1.coef_}")
print(f"Intercept is {model1.intercept_}")
y_pred=model1.predict(X1)

print("prediction ",model1.predict([[6]]))
#
print("MAE: ",mean_absolute_error(y1,y_pred))
print("MSE: ",mean_squared_error(y1,y_pred))
print("RMSE: ",root_mean_squared_error(y1,y_pred))
print("R2 Score: ",r2_score(y1,y_pred))
print("MBA: ",MBA(n1,y_pred,y1))


plt.scatter(X1,y1,color="blue",marker="o")
plt.xlabel("Sleep")
plt.ylabel("Production")
plt.plot(X1,y_pred,color="red")
plt.show()


print("#-----------------------------------------------------#")

df=pd.DataFrame({
    'area':[1,2,3,4,5],
    'price':[75,150,290,310,350]
})
X=df[['area']]
y=df[['price']]

# print(X)
# print(y)
n=len(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
Model=LinearRegression()

Model.fit(X_train,y_train)
print(f"Coeficient is {Model.coef_}")
print(f"Intercept is {Model.intercept_}")

pred_y=Model.predict(X_test)
print(pred_y)

print("MAE: ",mean_absolute_error(y_test,pred_y))
print("MSE: ",mean_squared_error(y_test,pred_y))
print("RMSE: ",root_mean_squared_error(y_test,pred_y))
print("R2 Score: ",r2_score(y_test,pred_y))
print("MBA: ",MBA(n,pred_y,y_test))
print("prediction ",Model.predict([[6]]))

plt.scatter(X,y,color="blue",marker="o")
plt.xlabel("Area")
plt.ylabel("Price")
plt.plot(X,Model.predict(X),color="red")
plt.show()