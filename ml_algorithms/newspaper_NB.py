#newspaper classification using Naive Bayes Classifier

#Importing Libraries
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

#load dataset
data=pd.read_csv("C:/Users/devik/Downloads/Uci-newssport.xlsx - Train.csv")
print(data.columns)

#data preprocessing
X=data['NEWS']
y=data['CLASS']

#tfidf Gives high weight to rare but important words, and low weight to common ones like “the”, “and” so we used it here.
tfidf=TfidfVectorizer()
X_vectorized=tfidf.fit_transform(X)
print("Vectorized X",X_vectorized.toarray()[0])
#train test method
X_train,X_test,y_train,y_test=train_test_split(X_vectorized,y,test_size=0.2,random_state=42)

#Using MultinomialNB
model=MultinomialNB()
model.fit(X_train.toarray(),y_train)
Y_pred=model.predict(X_test.toarray())

#evaluation
CM=confusion_matrix(y_test,Y_pred)  
print(CM)
Disp=ConfusionMatrixDisplay(CM)
Disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn (train test) - MultinomialNB")
plt.show()
print("Accuracy(MultinomialNB) train test",accuracy_score(y_test,Y_pred))
print(classification_report(y_test,Y_pred))
print('prediction on new data:',model.predict(tfidf.transform(["The team secured a fantastic deal with a company"]).toarray()))
print('prediction on new data:',model.predict(tfidf.transform(["2012 13 Jeep Grand Cherokee and Dodge Durango, 2014 Fiat 500L All Set to "]).toarray()))
print('prediction on new data:',model.predict(tfidf.transform(["Catherine Giudici Disses Juan PaBlo On 'After The Final Rose' For No Proposal"]).toarray()))
print('prediction on new data:',model.predict(tfidf.transform(["Blood test can predict Alzheimer's disease three years out at 90% accuracy."]).toarray()))