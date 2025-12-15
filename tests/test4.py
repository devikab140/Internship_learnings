import pandas as pd
import matplotlib.pyplot as plt

#PART A

df=pd.read_csv("C:/Users/devik/Downloads/car_data (1).csv")
#1
# print(df.shape)
# #2
# print(df.info())
# #3
# print(df.isnull().sum())
# #4
# print(df.describe())
# #5
# X=df[['Engine_Size(L)','Horsepower','Weight(kg)','MPG','Price($1000)']]

# plt.hist(X,bins=10)
# # plt.legend()
# plt.xlabel(X)
# # plt.show()

# #6
# correlation=X.corr()
# print(correlation)
#7
"""price - horsepower,engine size and weight are highly correlated
    mpg-none , slitly  price more related
"""


#PART B
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error,accuracy_score

# #1
# x=df[['Engine_Size(L)','Horsepower','Weight(kg)','MPG']]
# y=df['Price($1000)']

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
# model=LinearRegression()
# model.fit(x_train,y_train)
# #2
# y_pred=model.predict(x_test)
# print(y_pred)
# #3
# print("R2 score",r2_score(y_test,y_pred))
# print("MAE",mean_absolute_error(y_test,y_pred))
# print("RMSE",root_mean_squared_error(y_test,y_pred))


# #4
# plt.scatter(y_test,y_pred)
# plt.xlabel("Actual price")
# plt.ylabel("Predicted price")
# 
# # plt.show()

# #5
# print(df[['Engine_Size(L)','Horsepower','Weight(kg)','MPG','Price($1000)']].corr())
# #6
# print("Accuracy",accuracy_score(y_train,model.predict(x_train)))


# PART C
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve,RocCurveDisplay
from sklearn import metrics

#1
# X=df[['Engine_Size(L)','Horsepower','Weight(kg)','MPG']]
# y=df['High_Price']

# x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
# model=LogisticRegression()

# model.fit(x_train,y_train)
# #2

# print(model.predict_proba(x_test))

# #3
# print("Predicted probabilities ",model.predict(x_test)) 

# #4
# cm=confusion_matrix(y_train,model.predict(x_train))
# print(cm)

# #5
# print("Accuracy",accuracy_score(y_train,model.predict(x_train)))
# print("Precision",precision_score(y_train,model.predict(x_train)))
# print("Recall",recall_score(y_train,model.predict(x_train)))
# print("F1 score",f1_score(y_train,model.predict(x_train)))

# #6
# fpr,tpr,thresholds=roc_curve(y_train,model.predict(x_train))
# print(thresholds)
# roc_auc = metrics.auc(fpr, tpr)
# display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
# display.plot()
# # plt.show()

# #7
# # Set a new threshold
# threshold = 0.7  # example: make the model more sensitive (detect more positives)
# y_prob=model.predict(x_test)[:,1]
# y_pred_custom = (y_prob >= threshold).astype(int)

# print(confusion_matrix(y_test, y_pred_custom))








# #PART E

# from sklearn.linear_model import LinearRegression,LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
# import numpy as np
# #1
# x=df[['Engine_Size(L)','Weight(kg)']]
# y=df['MPG']

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
# model=LinearRegression()
# model.fit(x_train,y_train)
# #2
# print(df[['Engine_Size(L)','Weight(kg)','MPG']].corr())
# #3
# mean_value=df['MPG'].mean()
# df['High_MPG']=df['MPG'].apply(lambda x: 1 if x > mean_value else 0)
# print(df.head())
# #4
# X=df[['Engine_Size(L)','Weight(kg)']]
# Y=df['High_MPG']

# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
# Model=LogisticRegression()
# Model.fit(x_train,Y_train)

# print(model.predict(X_test))
# CM=confusion_matrix(Y_train,Model.predict(X_train))
# print(CM)

# print("Logistic - Accuracy",accuracy_score(Y_train,Model.predict(X_train)))
# print("Logistic - F1 score",f1_score(Y_train,Model.predict(X_train)))



# # cm=confusion_matrix(y_train,model.predict(x_train))
# # print(cm)

# # print("linear-Accuracy",accuracy_score(y_train,model.predict(x_train)))
# # print("Linear-F1 score",f1_score(y_train,model.predict(x_train)))



















#part c threshold setting

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

# Data
X = df[['Engine_Size(L)', 'Horsepower', 'Weight(kg)', 'MPG']]
y = df['High_Price']

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(x_train, y_train)

# --- Default threshold (0.5) ---
print("Default predicted probabilities (first 5):")
print(model.predict_proba(x_test)[:5])

# --- Custom threshold ---
threshold = 0.7  # change this as you wish (e.g., 0.3, 0.5, 0.7)
y_prob = model.predict_proba(x_test)[:, 1]   # probability of class 1
y_pred_custom = (y_prob >= threshold).astype(int)

# --- Evaluation ---
cm = confusion_matrix(y_test, y_pred_custom)
print("\nConfusion Matrix (threshold =", threshold, "):\n", cm)

print("\nAccuracy:", accuracy_score(y_test, y_pred_custom))
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall:", recall_score(y_test, y_pred_custom))
print("F1 Score:", f1_score(y_test, y_pred_custom))

# --- ROC curve (optional) ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
