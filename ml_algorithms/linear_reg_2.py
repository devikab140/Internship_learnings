import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df=pd.DataFrame({
    'area':[2600,3000,3200,3600,4000],
    'price':[550000,565000,610000,680000,725000]
})

X=df[['area']]
Y=df[['price']]

# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

model=LinearRegression()
model.fit(X_train,Y_train)
print(f"coeficeint is {model.coef_}")
print(f"intercept is {model.intercept_}")

Y_pred=model.predict(X_test)
print(Y_pred)

print(f"MSE: ",mean_squared_error(Y_test,Y_pred))
print(f"R2 score: ",r2_score(Y_test,Y_pred))

plt.scatter(X,Y,color="blue")
plt.plot(X,model.predict(X),color="red")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression")
plt.show()