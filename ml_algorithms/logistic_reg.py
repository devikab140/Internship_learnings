# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression()

# X=[[22],[45],[26],[39],[50],[48],[50],[23]]
# y=[0,1,0,1,1,1,1,0]
# X_test=[[30],[40],[49]]

# model.fit(X,y)
# y_pred=model.predict(X_test)
# print(y_pred)

import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,f1_score,accuracy_score,precision_score,recall_score,ConfusionMatrixDisplay

data=pd.read_csv("C:/Users/devik/Downloads/insurance_data.csv")
# print(data.head())

plt.scatter(data.age,data.bought_insurance,marker='+',color='blue')
# plt.show()

X_train,X_test,y_train,y_test=train_test_split(data[['age']],data.bought_insurance,train_size=0.8)
# print(X_test)

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)  

# print(model.predict_proba(X_test))  # for x test values it will print prob for classs1(0) ans prob for class2(1)
# print(model.score(X_test,y_test))   # log model performance (83%)

# print(y_pred)
# print(model.coef_)  # coef and inter have no value here
# print(model.intercept_)

# print(model.predict([[25]]))
cm=confusion_matrix(y_train,model.predict(X_train))
print(cm)
print("Accuracy",accuracy_score(y_train,model.predict(X_train)))
print("Precision",precision_score(y_train,model.predict(X_train)))
print("Recall",recall_score(y_train,model.predict(X_train)))
print("F1-score",f1_score(y_train,model.predict(X_train)))


disp=ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn")
plt.show()