#Bank Loan Prediction using Naive Bayes Classifier

#Importing Libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#load dataset
df=pd.read_csv("C:/Users/devik/Downloads/bankloan.csv")
print(df.columns)
print(df.info())

#data cleaning
print("Sum of null values",df.isna().sum())
print("Sum of Duplicates",df.duplicated().sum())
X=df[['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities.Account','CD.Account','Online','CreditCard']]
y=df['Personal.Loan']

#preprocessing
scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)
print("Scaled X",X_scaled[0])
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

#Using GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)

#prediction
# print(X_test[0:5])
y_pred=model.predict(X_test)
print("Predicted Y",y_pred[0:5])

#evaluation
cm=confusion_matrix(y_test,y_pred)
print(cm)
disp=ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn (train test) - GaussianNB")
plt.show()  
print("Accuracy(GaussianNB) train test",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

print('prediction on new data:',model.predict(scalar.transform([[45,20,60000,4,2.5,3,0,0,0,1,0]])))
