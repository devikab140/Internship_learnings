#import libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score,classification_report

# # 1.bankloan
# #load dataset
# df=pd.read_csv("C:/Users/devik/Downloads/bankloan.csv")
# print(df.columns)

# #data cleaning 
# print(df.isna().sum())
# df.drop_duplicates()

# #data preprocessing
# X=df[['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities.Account','CD.Account','Online','CreditCard']]
# y=df['Personal.Loan']
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# # model selection and training
# model=KNeighborsClassifier(n_neighbors=5)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())

# model.fit(X_train,y_train)

# #prediction
# y_pred=model.predict(X_test)
# print("Predicted Y",y_pred[0:5])

# #evaluation
# print("Accuracy",accuracy_score(y_test,y_pred))
# print(classification_report(y_test,y_pred))


#2. car data

#load dataset
df=pd.read_csv("C:/Users/devik/Downloads/car_data (1).csv")
print(df.columns)

#data cleaning 
print(df.isna().sum())
df.drop_duplicates()

#data preprocessing
X=df[[ 'Engine_Size(L)', 'Horsepower', 'Weight(kg)', 'MPG']]
y=df['High_Price']
scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# model selection and training
model=KNeighborsClassifier(n_neighbors=7)
skf=StratifiedKFold(n_splits=7)
cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
print("cross_val_score results:", cv_scores)
print("cross_val_score mean:", cv_scores.mean())

model.fit(X_train,y_train)

#prediction
y_pred=model.predict(X_test)
print("Predicted Y",y_pred[0:5])

#evaluation
print("Accuracy",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


#3. cancer
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

# # Model selection
# model = KNeighborsClassifier(n_neighbors=5)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())


# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# #4.date fruit
# df=pd.read_csv("C:/Users/devik/Downloads/Date_Fruit_Datasets.xlsx - Date_Fruit_Datasets.csv")
# print(df.columns)

# # Data preprocessing
# X=df[['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY',
#        'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO',
#        'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2',
#        'SHAPEFACTOR_3', 'SHAPEFACTOR_4', 'MeanRR', 'MeanRG', 'MeanRB',
#        'StdDevRR', 'StdDevRG', 'StdDevRB', 'SkewRR', 'SkewRG', 'SkewRB',
#        'KurtosisRR', 'KurtosisRG', 'KurtosisRB', 'EntropyRR', 'EntropyRG',
#        'EntropyRB', 'ALLdaub4RR', 'ALLdaub4RG', 'ALLdaub4RB']]
# y=df['Class']

# # Encoding class labels
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# #logistic regression
# model = KNeighborsClassifier(n_neighbors=5)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())


# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred, target_names=le.classes_))


# #5.diabetics

# data=pd.read_csv("C:/Users/devik/Downloads/diabetes2.csv")
# print(data.columns)
# x=data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
# y=data['Outcome']

# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(x)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # model selection
# model = KNeighborsClassifier(n_neighbors=19)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())


# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


#6.heart disease
# df=pd.read_csv("C:/Users/devik/Downloads/framingham.csv")
# print(df.columns)
# print(df.isna().sum())
# num_cols = ['cigsPerDay','BPMeds','totChol','BMI','heartRate']
# df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# df['glucose'] = df['glucose'].fillna(df['glucose'].median())
# print(df.isna().sum())
# X=df[['age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
#        'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
#        'diaBP', 'BMI', 'heartRate', 'glucose']]
# y=df['TenYearCHD']
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # model selection and training
# model = KNeighborsClassifier(n_neighbors=5)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())


# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


#7.insurance
# data=pd.read_csv("C:/Users/devik/Downloads/insurance_data.csv")
# print(data.columns)

# #data preprocessing
# X=data[['age']]
# y=data['bought_insurance']
## model selction and training
# model=KNeighborsClassifier(n_neighbors=5)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())


# model.fit(X,y)

# #prediction
# y_pred=model.predict(X)
# print("Predicted Y",y_pred[0:5])

# #evaluation
# print("Accuracy",accuracy_score(y,y_pred))
# print(classification_report(y,y_pred))

#8.iris

# data=pd.read_csv("C:/Users/devik/Downloads/iris.csv")
# print(data.columns)
# X=data[['x0', 'x1', 'x2', 'x3', 'x4']]
# y=data['type']
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # model selection and training
# model = KNeighborsClassifier(n_neighbors=7)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())


# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


#9. marketing

# df=pd.read_csv("C:/Users/devik/Downloads/marketing_campaign_corrected.csv")
# print(df.columns)
# print(df.info())

# #data cleaning
# print("Sum of null values",df.isna().sum())
# print("Sum of Duplicates",df.duplicated().sum())
# df=df.dropna()
# print("Sum of null values",df.isna().sum())

# X=df[['Income','Kidhome','Teenhome','Recency','MntWines','MntFruits',
#       'MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds',
#       'NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases',
#       'NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1',
#       'AcceptedCmp2','Complain']]
# y=df['Response']

# #preprocessing
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# #Using knn
# model=KNeighborsClassifier(n_neighbors=5)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())


# model.fit(X_train,y_train)

# #prediction
# y_pred=model.predict(X_test)
# print("Predicted Y",y_pred[0:5])


# #evaluation
# print("Accuracy",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# 10.pumpkin seed

# data = pd.read_csv("C:/Users/devik/Downloads/Pumpkin_Seeds_Dataset.xlsx - Pumpkin_Seeds_Dataset.csv")
# print(data.columns)

# # Data preprocessing
# X=data[['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
#        'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity', 'Extent',
#        'Roundness', 'Aspect_Ration', 'Compactness']]
# y=data['Class']
# print(y.value_counts())
# # Encoding class labels
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# # model selection and training
# model = KNeighborsClassifier(n_neighbors=11)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())


# model.fit(X_train, y_train)


# # Prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])
# # Evaluation
# print("Accuracy",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


#11.spam

# data=pd.read_csv("C:/Users/devik/Downloads/spam.csv")
# print(data.columns)

# #data preprocessing
# X=data['Message']
# y=data['Category'] 
# cv=CountVectorizer()
# X_vectorized=cv.fit_transform(X)
# #train test method
# X_train,X_test,y_train,y_test=train_test_split(X_vectorized,y,test_size=0.2,random_state=42)

# #model selection and training
# model=KNeighborsClassifier(n_neighbors=7)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())


# model.fit(X_train,y_train)

# #prediction
# Y_pred=model.predict(X_test)
# print(Y_pred[0:50])

# #Evaluation
# print("Accuracy",accuracy_score(y_test,Y_pred))
# print(classification_report(y_test,Y_pred))


#12.newspaper
# #load dataset
# data=pd.read_csv("C:/Users/devik/Downloads/Uci-newssport.xlsx - Train.csv")
# print(data.columns)

# #data preprocessing
# X=data['NEWS']
# y=data['CLASS']

# #tfidf Gives high weight to rare but important words, and low weight to common ones like “the”, “and” so we used it here.
# tfidf=TfidfVectorizer()
# X_vectorized=tfidf.fit_transform(X)
# print("Vectorized X",X_vectorized.toarray()[0])
# #train test method
# X_train,X_test,y_train,y_test=train_test_split(X_vectorized,y,test_size=0.2,random_state=42)

# # model selection and training
# model = KNeighborsClassifier(n_neighbors=11)

# skf=StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model,X_scaled, y, cv=skf)
# print("cross_val_score results:", cv_scores)
# print("cross_val_score mean:", cv_scores.mean())


# model.fit(X_train, y_train)

# # Prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# # Evaluation
# print("Accuracy",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))