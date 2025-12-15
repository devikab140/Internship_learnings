from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
#Q2

print("Multile normal method Q2")


data=pd.DataFrame({
    'age':[22,25,28,35,40],
    'income':[30,50,45,80,95],
    'cs':[650,720,680,750,800],
    'approval':[0,1,0,1,1]
})
X=data[['age','income','cs']]
y=data['approval']

Model=LogisticRegression()
Model.fit(X,y)
cm=confusion_matrix(y,Model.predict(X))
print(cm)
print("Accuracy",accuracy_score(y,Model.predict(X)))
print("Precision",precision_score(y,Model.predict(X)))
print("Recall",recall_score(y,Model.predict(X)))
print("F1-score",f1_score(y,Model.predict(X)))

disp=ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn")
plt.show()

print("#--------------------------------------------------------------------------------#")

# train test 

print("Train test method Q2")
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)  

CM=confusion_matrix(y_train,model.predict(X_train))
print(CM)
print("Accuracy",accuracy_score(y_train,model.predict(X_train)))
print("Precision",precision_score(y_train,model.predict(X_train)))
print("Recall",recall_score(y_train,model.predict(X_train)))
print("F1-score",f1_score(y_train,model.predict(X_train)))


Disp=ConfusionMatrixDisplay(CM)
Disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn")
plt.show()