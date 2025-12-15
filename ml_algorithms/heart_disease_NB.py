#heart disease prediction using Naive Bayes Classifier
#Importing Libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#load dataset
df=pd.read_csv("C:/Users/devik/Downloads/framingham.csv")
print(df.columns)
print(df.head(3))

#data cleaning
print("Sum of null values",df.isna().sum())
print("Sum of Duplicates",df.duplicated().sum())
df=df.dropna()
print("Sum of null values",df.isna().sum())


X=df[['male','age','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']]
y=df['TenYearCHD']

#preprocessing
scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)
print("Scaled X",X_scaled[0])
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2)

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
print('prediction on new data:',model.predict(scalar.transform([[1,55,0,20,0,0,1,0,200,130,90,28.5,80,100]])))