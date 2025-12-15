import pandas as pd
from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
#load dataset
data=pd.read_csv("C:/Users/devik/Downloads/spam.csv")
print(data.columns)

#data preprocessing
X=data['Message']
y=data['Category'] 
cv=CountVectorizer()
X_vectorized=cv.fit_transform(X)
print("Vectorized X",X_vectorized.toarray()[0])

#Using GaussianNB
model=GaussianNB()
model.fit(X_vectorized.toarray(),y)
y_pred=model.predict(X_vectorized.toarray())
print("Predicted Y",y_pred[0:5])        
cm=confusion_matrix(y,y_pred)
print(cm)
disp=ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn (normal method)- GaussianNB")
plt.show()
print("Accuracy(GaussianNB) normal method",accuracy_score(y,y_pred))
print(classification_report(y,y_pred))

#train test method
X_train,X_test,y_train,y_test=train_test_split(X_vectorized,y,test_size=0.2,random_state=42)
model.fit(X_train.toarray(),y_train)
Y_pred=model.predict(X_test.toarray())
CM=confusion_matrix(y_test,Y_pred)
print(CM)
Disp=ConfusionMatrixDisplay(cm)
Disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn(train test) - GaussianNB")
plt.show()
print("Accuracy(GaussianNB) train test",accuracy_score(y_test,Y_pred))
print(classification_report(y_test,Y_pred))


#Using MultinomialNB
model2=MultinomialNB()

model2.fit(X_vectorized.toarray(),y)
Y_pred2=model2.predict(X_vectorized.toarray())
print("Predicted Y",Y_pred2[0:5])        
CM2=confusion_matrix(y,Y_pred2)
print(CM2)
Disp2=ConfusionMatrixDisplay(cm)
Disp2.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn (normal method)- MultinomialNB")
plt.show()
print("Accuracy(MultinomialNB) normal method",accuracy_score(y,Y_pred2))
print(classification_report(y,Y_pred2))

#train test method

model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)
disp2=ConfusionMatrixDisplay(cm2)
disp2.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn (train test- MultinomialNB")
plt.show()
print("Accuracy(MultinomialNB) trai test",accuracy_score(y_test,y_pred2))
print(classification_report(y_test,y_pred2))


#Using BernoulliNB

model3=BernoulliNB()
model3.fit(X_vectorized.toarray(),y)
Y_pred3=model3.predict(X_vectorized.toarray())
print("Predicted Y",Y_pred3[0:5])        
CM3=confusion_matrix(y,Y_pred3)
print(CM3)
Disp3=ConfusionMatrixDisplay(CM3)
Disp3.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn (normal method) - BernoulliNB")
plt.show()
print("Accuracy(BernoulliNB) normal method",accuracy_score(y,Y_pred3))
print(classification_report(y,Y_pred3))


#train test method
model3.fit(X_train,y_train)
y_pred3=model3.predict(X_test)  
cm3=confusion_matrix(y_test,y_pred3)
print(cm3)
disp3=ConfusionMatrixDisplay(cm3)
disp3.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn(traintest) - BernoulliNB")
plt.show()
print("Accuracy (BernoulliNB) train test",accuracy_score(y_test,y_pred3))
print(classification_report(y_test,y_pred3))



# #predicting new messages
# new_messages=["Congratulations! You've won a free ticket to Bahamas. Call now!","Hey, are we still meeting for lunch today?"]
# new_messages_vectorized=cv.transform(new_messages)  
# print("Predictions for new messages - GaussianNB:",model.predict(new_messages_vectorized.toarray()))
# print("Predictions for new messages - MultinomialNB:",model2.predict(new_messages_vectorized))
# print("Predictions for new messages - BernoulliNB:",model3.predict(new_messages_vectorized))