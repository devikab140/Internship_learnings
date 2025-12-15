from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
#Q1
# model=LogisticRegression()
# X=[[2],[4],[6],[8],[9]]
# y=[0,0,1,1,1]


# model.fit(X,y)
# print(model.predict_proba(X))
# y_pred=model.predict(X)
# print(y_pred)

# print(model.coef_)
# print(model.intercept_)



#multiple logistic regression

print("Multile normal method Q1")
Model=LogisticRegression()
df=pd.DataFrame({
    'hrs':[2,4,6,8,9],
    'attendance':[60,80,75,90,88],
    'pass':[0,0,1,1,1]
})
X=df[['hrs','attendance']]
y=df['pass']
Model.fit(X,y)

CM=confusion_matrix(y,Model.predict(X))
print(CM)
print("Accuracy",accuracy_score(y,Model.predict(X)))
print("Precision",precision_score(y,Model.predict(X)))
print("Recall",recall_score(y,Model.predict(X)))
print("F1-score",f1_score(y,Model.predict(X)))
DISP=ConfusionMatrixDisplay(CM)
DISP.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn")
plt.show()

print("#---------------------------------------------------------------------------------#")

#train test
print('multiple train test method Q1')
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)
print(X_test)

model1=LogisticRegression()
model1.fit(X_train,y_train)
# pred_y=model1.predict(X_test) 
# print("train_test model predicted",pred_y) 
# print(model1.predict_proba(X_test))
# print("train_test model score is ",model1.score(X_test,y_test))

cm=confusion_matrix(y_train,model1.predict(X_train))
print(cm)
print("Accuracy",accuracy_score(y_train,model1.predict(X_train)))
print("Precision",precision_score(y_train,model1.predict(X_train)))
print("Recall",recall_score(y_train,model1.predict(X_train)))
print("F1-score",f1_score(y_train,model1.predict(X_train)))
disp=ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn")
plt.show()

