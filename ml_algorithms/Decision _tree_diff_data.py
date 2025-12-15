import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report

#1
# breast cancer dataset
# #load dataset
# df=pd.read_csv("C:/Users/devik/Downloads/data.csv")
# print(df.columns)
# X=df[['radius_mean', 'texture_mean', 'perimeter_mean',
#        'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
#        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
#        'fractal_dimension_se', 'radius_worst', 'texture_worst',
#        'perimeter_worst', 'area_worst', 'smoothness_worst',
#        'compactness_worst', 'concavity_worst', 'concave points_worst',
#        'symmetry_worst', 'fractal_dimension_worst']]

# y=df['diagnosis'].map({'M':1,'B':0})

# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# #Using DecisionTreeClassifier
# model = DecisionTreeClassifier(criterion="entropy")
# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy(DecisionTreeClassifier) entropy",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# #plotting tree
# plt.figure(figsize=(20,10))
# plot_tree(model,max_depth=4, filled=True, feature_names=X.columns, class_names=['Benign', 'Malignant'])
# plt.title("Decision Tree Classifier for Breast Cancer Detection")
# plt.show()


#2
# #diabetes dataset
# data=pd.read_csv("C:/Users/devik/Downloads/diabetes2.csv")
# print(data.columns)
# x=data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
# y=data['Outcome']

# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(x)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# #Using DecisionTreeClassifier
# model = DecisionTreeClassifier(criterion="gini")
# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy(DecisionTreeClassifier) gini",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# #plotting tree
# plt.figure(figsize=(20,10))
# plot_tree(model,max_depth=4, filled=True, feature_names=x.columns, class_names=['No Diabetes', 'Diabetes'])
# plt.title("Decision Tree Classifier for Diabetes Detection")
# plt.show()


#3
# #heart disease dataset
# df=pd.read_csv("C:/Users/devik/Downloads/framingham.csv")
# print(df.columns)
# X=df[['age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
#        'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
#        'diaBP', 'BMI', 'heartRate', 'glucose']]
# y=df['TenYearCHD']
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# #Using DecisionTreeClassifier
# model = DecisionTreeClassifier(criterion="entropy")
# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy(DecisionTreeClassifier) gini",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# #plotting tree
# plt.figure(figsize=(20,10))
# plot_tree(model,max_depth=4, filled=True, feature_names=X.columns, class_names=['No CHD', 'CHD'])
# plt.title("Decision Tree Classifier for Heart Disease Detection")
# plt.show()



#4
# #iris dataset
# data=pd.read_csv("C:/Users/devik/Downloads/iris.csv")
# print(data.columns)
# X=data[['x0', 'x1', 'x2', 'x3', 'x4']]
# y=data['type']
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# #Using DecisionTreeClassifier
# model = DecisionTreeClassifier(criterion="gini")
# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy(DecisionTreeClassifier) gini",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# #plotting tree
# plt.figure(figsize=(20,10))
# plot_tree(model,max_depth=4, filled=True, feature_names=X.columns, class_names=model.classes_)
# plt.title("Decision Tree Classifier for Iris Species Classification")
# plt.show()



#5
# #insurance dataset
# data=pd.read_csv("C:/Users/devik/Downloads/insurance_data.csv")
# print(data.columns)

# #data preprocessing
# X=data[['age']]
# y=data['bought_insurance']

# model=DecisionTreeClassifier(criterion="gini")
# model.fit(X,y)

# #prediction
# y_pred=model.predict(X)
# print("Predicted Y",y_pred[0:5])

# #evaluation
# print("Accuracy(DecisionTreeClassifier) entropy",accuracy_score(y,y_pred))
# print(classification_report(y,y_pred))

# #plotting tree
# plt.figure(figsize=(10,6))
# plot_tree(model,max_depth=4, filled=True, feature_names=X.columns, class_names=['Not Bought', 'Bought'])
# plt.title("Decision Tree Classifier for Insurance Purchase Prediction")
# plt.show()


#6
# #bank loan dataset for decision tree classification
# df=pd.read_csv("C:/Users/devik/Downloads/bankloan.csv")
# print(df.columns)

# X=df[['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities.Account','CD.Account','Online','CreditCard']]
# y=df['Personal.Loan']

# #preprocessing
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)

# X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# #Using DecisionTreeClassifier
# model=DecisionTreeClassifier(criterion="gini")
# model.fit(X_train,y_train)

# #prediction
# y_pred=model.predict(X_test)
# print("Predicted Y",y_pred[0:5])

# #evaluation
# print("Accuracy(DecisionTreeClassifier) gini",accuracy_score(y_test,y_pred))
# print(classification_report(y_test,y_pred))

# #plotting tree
# plt.figure(figsize=(20,10))
# plot_tree(model,max_depth=4, filled=True, feature_names=X.columns, class_names=['No Loan', 'Loan'])
# plt.title("Decision Tree Classifier for Bank Loan Prediction")  
# plt.show()


#7
#marketing campaign dataset for decision tree classification
# #load dataset
# df=pd.read_csv("C:/Users/devik/Downloads/marketing_campaign_corrected.csv")
# print(df.columns)

# X=df[['Income','Kidhome','Teenhome','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain']]
# print(X[0:5])
# y=df['Response']

# #preprocessing
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)

# X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3)

# #Using DecisionTreeClassifier
# model=DecisionTreeClassifier(criterion="entropy")
# model.fit(X_train,y_train)

# #prediction
# y_pred=model.predict(X_test)
# print("Predicted Y",y_pred[0:5])

# #evaluation
# print("Accuracy(DecisionTreeClassifier) entropy",accuracy_score(y_test,y_pred))
# print(classification_report(y_test,y_pred))

# #plotting tree
# plt.figure(figsize=(20,10))
# plot_tree(model,max_depth=4, filled=True, feature_names=X.columns, class_names=['No Response', 'Response'])
# plt.title("Decision Tree Classifier for Marketing Campaign Response Prediction")
# plt.show()
