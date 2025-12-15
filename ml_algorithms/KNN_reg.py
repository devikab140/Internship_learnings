#import libraries
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#1.house price
# # Load dataset
# data = pd.read_csv("C:/Users/devik/Downloads/house_price_data.csv")
# print(data.columns)

# # Data preprocessing
# X=data[['bedrooms','sqft_living', 'sqft_lot','floors','sqft_above','sqft_basement']]
# y=data['price']

# #scaling
# scaler_x = StandardScaler()
# X_scaled = scaler_x.fit_transform(X)

# # scaler_y = StandardScaler()
# # y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
# #splitting
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Using KNeighborsRegressor
# model = KNeighborsRegressor(n_neighbors=7)
# model.fit(X_train, y_train)

# # Prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# # Evaluation
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("R2 Score:", r2)


#2.student-mat

# df=pd.read_csv("C:/Users/devik/Downloads/student-mat.csv")
# #data cleaning
# print("Sum of null values",df.isna().sum())
# print("Sum of Duplicates",df.duplicated().sum())

# df['pass'] = df['G3'].apply(lambda x: 1 if x > 10 else 0)
# X=df[['activities','higher','traveltime','studytime','failures','romantic','internet','freetime','goout','Dalc','Walc','absences','G1','G2']]
# y=df['pass']
# X_onehot = pd.get_dummies(X[['activities','higher','romantic','internet']], drop_first=True)
# X = pd.concat([X.drop(['activities','higher','romantic','internet'], axis=1), X_onehot], axis=1)

# # 4. Scale numeric columns

# num_cols = ['traveltime','studytime','failures','freetime','goout',
#             'Dalc','Walc','absences','G1','G2']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X[num_cols])

# # Convert back to DataFrame for readability
# X_scaled_df = pd.DataFrame(X_scaled, columns=num_cols, index=X.index)

# # 5. Combine scaled numeric + encoded categorical
# cat_cols = ['activities_yes', 'higher_yes', 'romantic_yes', 'internet_yes']
# X_final = pd.concat([X_scaled_df, X[cat_cols]], axis=1)
# print("Final feature set X",X_final.head(3))

# X_train,X_test,y_train,y_test=train_test_split(X_final,y,test_size=0.2,random_state=42)

# # Using KNeighborsRegressor
# model = KNeighborsRegressor(n_neighbors=5)
# model.fit(X_train, y_train)

# # Prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# # Evaluation
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("R2 Score:", r2)


#3.high dim

# df=pd.read_csv("C:/Users/devik/Downloads/high_dim.csv")
# print(df.columns)

# X = df.drop('target', axis=1)
# y = df['target']

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# # Using KNeighborsRegressor
# model = KNeighborsRegressor(n_neighbors=11)
# model.fit(X_train, y_train)

# # Prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# # Evaluation
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("R2 Score:", r2)


#4.car data

# --- Load dataset ---
df = pd.read_csv("C:/Users/devik/Downloads/car_data.csv")

# --- Define features (X) and target (y) ---
X = df[['Engine_Size (L)', 'Weight (kg)', 'Horsepower']]
y = df['MPG (Miles_per_Gallon)']

# --- Split into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print("Predicted Y", y_pred[0:5])

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
