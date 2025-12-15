import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score

df=pd.DataFrame({
    'area':[2600,3000,3200,3600,4000],
    'price':[550000,565000,610000,680000,725000]
})
# print(df)

plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.area,df.price,color='red',marker='+')
# plt.show()

new_area=df.drop('price',axis='columns')
price=df.price

reg=linear_model.LinearRegression()
reg.fit(new_area,price)

price_pred=reg.predict(new_area)
print("MSE: ",mean_squared_error(price,price_pred))  # y_test --> y_value(price)  ,y_pred --> pred of x_test (here pred od area)
print("R2 score: ",r2_score(price,price_pred))


# print(f"price for {3300} is {reg.predict([[3300]])}")
# print(f"price for {1000} is {reg.predict([[1000]])}")
# print(f"price for {10000} is {reg.predict([[10000]])}")
print(reg.intercept_)
print(reg.coef_)


plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(new_area,reg.predict(new_area))

# plt.show()