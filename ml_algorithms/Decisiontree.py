# regression decision tree and classification decision tree

import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score,classification_report

# #load dataset
# df=pd.read_csv("C:/Users/devik/Downloads/spam.csv") 
# print(df.head())

# #data preprocessing
# X_data=df[['Message']]
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(X_data['Message']).toarray()

# Y=df['Category']
# y = Y.map({'ham': 0, 'spam': 1})
# print(y.head())

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #Using DecisionTreeClassifier
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])
# #evaluation

# print("Accuracy(DecisionTreeClassifier) train test",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# #plotting tree
# plt.figure(figsize=(20,10))
# plot_tree(model,max_depth=4, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=['ham', 'spam'])
# plt.title("Decision Tree Classifier for Spam Detection")
# plt.show()


# #another set for decision tree classification

# data=pd.read_csv("C:/Users/devik/Downloads/Uci-newssport.xlsx - Train.csv")
# print(data.columns)

# #data preprocessing
# X=data['NEWS']
# y=data['CLASS']
# tfidf=TfidfVectorizer()
# X_vectorized=tfidf.fit_transform(X).toarray()
# X_train,X_test,y_train,y_test=train_test_split(X_vectorized,y,test_size=0.2,random_state=42)

# #Using DecisionTreeClassifier
# Model=DecisionTreeClassifier()
# Model.fit(X_train,y_train)

# #prediction
# y_pred=Model.predict(X_test)
# print("Predicted Y",y_pred[0:5])
# #plotting tree
# plt.figure(figsize=(20,10))
# plot_tree(Model,max_depth=4, filled=True, feature_names=tfidf.get_feature_names_out(), class_names=model.classes_)
# plt.title("Decision Tree Classifier newspaper classification")
# plt.show()



#regression decision tree example
#load dataset
data = pd.read_csv("C:/Users/devik/Downloads/house_price_data.csv")
print(data.head())
X=data[['bedrooms','sqft_living', 'sqft_lot','floors','sqft_above','sqft_basement']]
scalar=StandardScaler()
x_scaled=scalar.fit_transform(X)
y=data['price']
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

#Using DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
#prediction
y_pred = model.predict(X_test)
print("Predicted Y", y_pred[0:5])
print(y_test.head(5))
#evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


#plotting tree
plt.figure(figsize=(20,10))
plot_tree(model,max_depth=4, filled=True, feature_names=X.columns)
plt.title("Decision Tree Regressor for House Price Prediction")
# plt.show()



