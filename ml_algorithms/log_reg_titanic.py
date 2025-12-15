from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay,precision_score,recall_score,f1_score,log_loss
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/devik/Downloads/titanic_dataset.csv")
print(data.columns)
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

X=data[['Pclass','Sex', 'Age','Fare']]
y=data['Survived']
model=LogisticRegression()
model.fit(X,y)
print("normal prediction",model.predict([[2,0,39,89.325]]))
print("jack normal prediction",model.predict([[3,1,20,5.25]]))
print("Rose normal prediction",model.predict([[1,0,34,120.25]]))

cm=confusion_matrix(y,model.predict(X))
print(cm)
print("Accuracy",accuracy_score(y,model.predict(X)))
print("Precision",precision_score(y,model.predict(X)))
print("Recall",recall_score(y,model.predict(X)))
print("F1-score",f1_score(y,model.predict(X)))
disp=ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn")
# plt.show()
Loss=log_loss(y,model.predict(X))
print("Logistic Loss:",Loss)

#---------------------------------------------------------------------------#
print("Train test method")
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)
Model=LogisticRegression()
Model.fit(X_train,y_train)
# y_pred=Model.predict(X_test)  
print("train test prediction",Model.predict([[2,0,39,89.325]]))
print("jack train test prediction",Model.predict([[3,1,20,5.25]]))
print("Rose train test prediction",Model.predict([[1,0,34,120.25]]))


CM=confusion_matrix(y_train,Model.predict(X_train))
print(CM)
print("Accuracy",accuracy_score(y_train,Model.predict(X_train)))
print("Precision",precision_score(y_train,Model.predict(X_train)))
print("Recall",recall_score(y_train,Model.predict(X_train)))
print("F1-score",f1_score(y_train,Model.predict(X_train)))


Disp=ConfusionMatrixDisplay(CM)
Disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn")
# plt.show()

#logg loss
# from sklearn.metrics import log_loss
loss=log_loss(y_test,Model.predict(X_test))
print("Logistic Loss:",loss)





#logistic optimization using gradient descent
# opt=LogisticRegression(solver='liblinear',max_iter=200)
# opt.fit(X_train,y_train)
# print("Optimized train test prediction",opt.predict([[2,0,39,89.325]]))
# print("jack Optimized train test prediction",opt.predict([[3,1,20,5.25]]))
# print("Rose Optimized train test prediction",opt.predict([[1,0,34,120.25]]))    