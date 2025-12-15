#svc and svr
import pandas as pd
from sklearn.svm import SVC,SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#SVC

# #bank loan dataset
# df=pd.read_csv("C:/Users/devik/Downloads/bankloan.csv")
# print(df.columns)

# #data preprocessing
# X=df[['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities.Account','CD.Account','Online','CreditCard']]
# y=df['Personal.Loan']
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# #Using SVC
# model = SVC(kernel='rbf')
# model.fit(X_train, y_train)
# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy(SVC) rbf", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Visualizing decision boundary for SVC using only 2 features: Age and Income
# # Select 2 features only for visualization
# X2 = df[['Age', 'Income']]
# y2 = df['Personal.Loan']

# # Scaling only these 2 features
# scaler2 = StandardScaler()
# X2_scaled = scaler2.fit_transform(X2)

# # Train a separate SVM for boundary visualization
# svm_vis = SVC(kernel='rbf')
# svm_vis.fit(X2_scaled, y2)

# # Meshgrid for decision boundary
# x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
# y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1

# xx, yy = np.meshgrid(
#     np.linspace(x_min, x_max, 600),
#     np.linspace(y_min, y_max, 600)
# )

# Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# plt.figure(figsize=(10, 6))
# # Decision boundary
# plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.25)
# # Scatter plot of actual points
# plt.scatter(X2_scaled[:, 0], X2_scaled[:, 1], c=y2, cmap='coolwarm', edgecolors='k')
# plt.title("SVM Decision Boundary (Age vs Income)")
# plt.xlabel("Age (scaled)")
# plt.ylabel("Income (scaled)")
# plt.show()


# #car dataset
# data = pd.read_csv("C:/Users/devik/Downloads/car_data (1).csv")
# print(data.columns)

# #data preprocessing
# X = data[['Engine_Size(L)', 'Horsepower', 'Weight(kg)', 'MPG','Price($1000)']]
# y = data['High_Price']

# scalar = StandardScaler()
# X_scaled = scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# #Using SVC
# model = SVC(kernel='linear')
# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy(SVC) linear", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Visualizing decision boundary for SVC using only 2 features: Horsepower and MPG
# # Select 2 features only for visualization
# X2 = data[['Horsepower', 'MPG']]
# y2 = data['High_Price']
# # Scaling only these 2 features
# scaler2 = StandardScaler()
# X2_scaled = scaler2.fit_transform(X2)
# # Train a separate SVM for boundary visualization
# svm_vis = SVC(kernel='linear')
# svm_vis.fit(X2_scaled, y2)
# # Meshgrid for decision boundary
# x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
# y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
# xx, yy = np.meshgrid(
#     np.linspace(x_min, x_max, 600),
#     np.linspace(y_min, y_max, 600)
# )
# Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape) 
# plt.figure(figsize=(10, 6))
# # Decision boundary
# plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.25)
# # Scatter plot of actual points
# plt.scatter(X2_scaled[:, 0], X2_scaled[:, 1], c=y2, cmap='coolwarm', edgecolors='k')
# plt.title("SVM Decision Boundary (Horsepower vs MPG)")
# plt.xlabel("Horsepower (scaled)")
# plt.ylabel("MPG (scaled)")
# plt.show()


# #cancer dataset
# data = pd.read_csv("C:/Users/devik/Downloads/data.csv")
# print(data.columns)

# #data preprocessing
# X = data[['radius_mean', 'texture_mean', 'perimeter_mean',
#        'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
#        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
#        'fractal_dimension_se', 'radius_worst', 'texture_worst',
#        'perimeter_worst', 'area_worst', 'smoothness_worst',
#        'compactness_worst', 'concavity_worst', 'concave points_worst',
#        'symmetry_worst', 'fractal_dimension_worst']]
# y = data['diagnosis'].map({'M':1,'B':0})
# scalar = StandardScaler()
# X_scaled = scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# #Using SVC
# model = SVC(kernel='poly')
# model.fit(X_train, y_train)
# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])
# #evaluation
# print("Accuracy(SVC) poly", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# # Visualizing decision boundary for SVC using only 2 features: radius_mean and texture_mean
# # Select 2 features only for visualization
# X2 = data[['radius_mean', 'texture_mean']]
# y2 = data['diagnosis'].map({'M':1,'B':0})
# # Scaling only these 2 features
# scaler2 = StandardScaler()  
# X2_scaled = scaler2.fit_transform(X2)
# # Train a separate SVM for boundary visualization
# svm_vis = SVC(kernel='poly')
# svm_vis.fit(X2_scaled, y2)
# # Meshgrid for decision boundary
# x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
# y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
# xx, yy = np.meshgrid(
#     np.linspace(x_min, x_max, 600),
#     np.linspace(y_min, y_max, 600)
# )
# Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure(figsize=(10, 6))
# # Decision boundary
# plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.25)
# # Scatter plot of actual points
# plt.scatter(X2_scaled[:, 0], X2_scaled[:, 1], c=y2, cmap='coolwarm', edgecolors='k')
# plt.title("SVM Decision Boundary (Radius Mean vs Texture Mean)")
# plt.xlabel("Radius Mean (scaled)")
# plt.ylabel("Texture Mean (scaled)")
# plt.show()

# #datefruit dataset
# data = pd.read_csv("C:/Users/devik/Downloads/Date_Fruit_Datasets.xlsx - Date_Fruit_Datasets.csv")
# print(data.columns)

# #data preprocessing
# X = data[['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY',
#        'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO',
#        'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2',
#        'SHAPEFACTOR_3', 'SHAPEFACTOR_4', 'MeanRR', 'MeanRG', 'MeanRB',
#        'StdDevRR', 'StdDevRG', 'StdDevRB', 'SkewRR', 'SkewRG', 'SkewRB',
#        'KurtosisRR', 'KurtosisRG', 'KurtosisRB', 'EntropyRR', 'EntropyRG',
#        'EntropyRB', 'ALLdaub4RR', 'ALLdaub4RG', 'ALLdaub4RB']]
# y = data['Class']
# scalar = StandardScaler()
# X_scaled = scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# #Using SVC
# model = SVC(kernel='rbf')
# model.fit(X_train, y_train)
# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])
# #evaluation
# print("Accuracy(SVC) rbf", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # print(data['Class'].unique())
# # Visualizing decision boundary for SVC using only 2 features: AREA and PERIMETER
# # Select 2 features only for visualization
# X2 = data[['AREA', 'PERIMETER']]    

# le = LabelEncoder()
# y2 = le.fit_transform(data['Class'])
# print(le.classes_)


# # Scaling only these 2 features
# scaler2 = StandardScaler()
# X2_scaled = scaler2.fit_transform(X2)
# # Train a separate SVM for boundary visualization
# svm_vis = SVC(kernel='rbf')
# svm_vis.fit(X2_scaled, y2)
# # Meshgrid for decision boundary
# x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
# y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
# xx, yy = np.meshgrid(
#     np.linspace(x_min, x_max, 600),
#     np.linspace(y_min, y_max, 600)
# )
# Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure(figsize=(10, 6))
# # Decision boundary
# plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.25)
# # Scatter plot of actual points 
# plt.scatter(X2_scaled[:, 0], X2_scaled[:, 1], c=y2, cmap='coolwarm', edgecolors='k')
# plt.title("SVM Decision Boundary (AREA vs PERIMETER)")
# plt.xlabel("AREA (scaled)")
# plt.ylabel("PERIMETER (scaled)")
# plt.show()

# # diabetes dataset
# data = pd.read_csv("C:/Users/devik/Downloads/diabetes2.csv")
# print(data.columns)

# x = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
# y = data['Outcome']
# scalar = StandardScaler()
# X_scaled = scalar.fit_transform(x)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# # Using SVC
# model = SVC(kernel='rbf')    
# model.fit(X_train, y_train)
# # prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])
# # evaluation
# print("Accuracy(SVC) rbf", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Visualizing decision boundary for SVC using only 2 features: Glucose and BMI
# # Select 2 features only for visualization
# X2 = data[['Glucose', 'BMI']]
# y2 = data['Outcome']
# # Scaling only these 2 features
# scaler2 = StandardScaler()
# X2_scaled = scaler2.fit_transform(X2)
# # Train a separate SVM for boundary visualization
# svm_vis = SVC(kernel='rbf')
# svm_vis.fit(X2_scaled, y2)
# # Meshgrid for decision boundary
# x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
# y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
# xx, yy = np.meshgrid(
#     np.linspace(x_min, x_max, 600),
#     np.linspace(y_min, y_max, 600)
# )
# Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure(figsize=(10, 6))
# # Decision boundary
# plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.25)
# # Scatter plot of actual points
# plt.scatter(X2_scaled[:, 0], X2_scaled[:, 1], c=y2, cmap='coolwarm', edgecolors='k')
# plt.title("SVM Decision Boundary (Glucose vs BMI)")
# plt.xlabel("Glucose (scaled)")
# plt.ylabel("BMI (scaled)")
# plt.show()

# #heart disease dataset
# df=pd.read_csv("C:/Users/devik/Downloads/framingham.csv")
# print(df.columns)
# features = ['age','currentSmoker', 'cigsPerDay', 'BPMeds',
#             'prevalentStroke', 'prevalentHyp', 'diabetes',
#             'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

# df_clean = df[features + ['TenYearCHD']].dropna()   # DROP NANS FROM ALL COLUMNS TOGETHER

# X = df_clean[features]
# y = df_clean['TenYearCHD']

# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# #Using SVC
# model = SVC(kernel='rbf')
# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy(SVC) rbf",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Visualizing decision boundary for SVC using only 2 features: age and glucose
# # Select 2 features only for visualization
# df_vis = df[['age', 'glucose', 'TenYearCHD']].dropna()  # CLEAN ONLY THESE FEATURES

# X2 = df_vis[['age', 'glucose']]
# y2 = df_vis['TenYearCHD']

# # Scaling only these 2 features
# scaler2 = StandardScaler()
# X2_scaled = scaler2.fit_transform(X2)
# # Train a separate SVM for boundary visualization
# svm_vis = SVC(kernel='rbf')
# svm_vis.fit(X2_scaled, y2)
# # Meshgrid for decision boundary
# x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
# y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
# xx, yy = np.meshgrid(
#     np.linspace(x_min, x_max, 600),
#     np.linspace(y_min, y_max, 600)
# )   
# Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure(figsize=(10, 6))
# # Decision boundary
# plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.25)
# # Scatter plot of actual points
# plt.scatter(X2_scaled[:, 0], X2_scaled[:, 1], c=y2, cmap='coolwarm', edgecolors='k')
# plt.title("SVM Decision Boundary (Age vs Glucose)")
# plt.xlabel("Age (scaled)")
# plt.ylabel("Glucose (scaled)")
# plt.show()

# #insurance dataset
# df=pd.read_csv("C:/Users/devik/Downloads/insurance_data.csv")
# print(df.columns)
# X=df[['age']]
# y=df['bought_insurance']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #Using SVC
# model = SVC(kernel='linear')    
# model.fit(X_train, y_train)

# #prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# #evaluation
# print("Accuracy(SVC) linear",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Visualizing decision boundary for SVC using only 1 feature: age
# # Since we have only one feature, we will plot the decision boundary as a vertical line
# plt.figure(figsize=(10, 6))
# # Scatter plot of actual points
# plt.scatter(X, y, c=y, cmap='coolwarm', edgecolors='k') 
# # Decision boundary
# x_min, x_max = X['age'].min() - 1, X['age'].max() + 1
# xx = np.linspace(x_min, x_max, 600)
# yy = model.decision_function(xx.reshape(-1, 1))
# plt.plot(xx, yy, color='black', linestyle='--')
# plt.title("SVM Decision Boundary (Age vs Bought Insurance)")
# plt.xlabel("Age")
# plt.ylabel("Decision Function")
# plt.show()  


# #iris dataset
# data=pd.read_csv("C:/Users/devik/Downloads/iris.csv")
# print(data.columns)
# X=data[['x0', 'x1', 'x2', 'x3', 'x4']]
# y=data['type']

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

# #svc
# model=SVC(kernel='rbf')
# model.fit(X_train,y_train)

# #prediction
# pred=model.predict(X_test)
# print("Predicted Y", pred[0:5])

# #evaluation
# print("Accuracy(SVC) rbf",accuracy_score(y_test, pred))
# print(classification_report(y_test, pred))

# # # Visualizing decision boundary for SVC using only 2 features: age and glucose
# # # Select 2 features only for visualization

# X2 = data[['x0', 'x1']]
# le = LabelEncoder()
# y2=data['type']
# y2_encoded = le.fit_transform(y2)

# # Scaling only these 2 features
# scaler2 = StandardScaler()
# X2_scaled = scaler2.fit_transform(X2)
# # Train a separate SVM for boundary visualization
# svm_vis = SVC(kernel='rbf')
# svm_vis.fit(X2_scaled, y2_encoded)
# # Meshgrid for decision boundary
# x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
# y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
# xx, yy = np.meshgrid(
#     np.linspace(x_min, x_max, 600),
#     np.linspace(y_min, y_max, 600)
# )   
# Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure(figsize=(10, 6))
# # Decision boundary
# plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.25)
# # Scatter plot of actual points
# plt.scatter(X2_scaled[:, 0], X2_scaled[:, 1], c=y2_encoded, cmap='coolwarm', edgecolors='k')
# plt.title("SVM Decision Boundary (X0 vs X1)")
# plt.xlabel("X0")
# plt.ylabel("X1")
# plt.show()


# #pumpkin seed dataset

# df=pd.read_csv("C:/Users/devik/Downloads/Pumpkin_Seeds_Dataset.xlsx - Pumpkin_Seeds_Dataset.csv")
# print(df.columns)
# X=df[['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
#        'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity', 'Extent',
#        'Roundness', 'Aspect_Ration', 'Compactness']]
# le=LabelEncoder()
# y=df['Class']
# y_lab=le.fit_transform(y)
# scalar=StandardScaler()
# x_scaled=scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_lab, test_size=0.2, random_state=42)

# # Using SVC
# model = SVC(kernel='rbf')    
# model.fit(X_train, y_train)
# # prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])
# # evaluation
# print("Accuracy(SVC) rbf", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Visualizing decision boundary for SVC using only 2 features: Glucose and BMI
# # Select 2 features only for visualization
# X2 = df[['Area', 'Perimeter']]
# y2=df['Class']
# y2_encod = le.fit_transform(y2)
# # Scaling only these 2 features
# scaler2 = StandardScaler()
# X2_scaled = scaler2.fit_transform(X2)
# # Train a separate SVM for boundary visualization
# svm_vis = SVC(kernel='rbf')
# svm_vis.fit(X2_scaled, y2_encod)
# # Meshgrid for decision boundary
# x_min, x_max = X2_scaled[:, 0].min() - 1, X2_scaled[:, 0].max() + 1
# y_min, y_max = X2_scaled[:, 1].min() - 1, X2_scaled[:, 1].max() + 1
# xx, yy = np.meshgrid(
#     np.linspace(x_min, x_max, 600),
#     np.linspace(y_min, y_max, 600)
# )
# Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure(figsize=(10, 6))
# # Decision boundary
# plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.25)
# # Scatter plot of actual points
# plt.scatter(X2_scaled[:, 0], X2_scaled[:, 1], c=y2_encod, cmap='coolwarm', edgecolors='k')
# plt.title("SVM Decision Boundary (Area perimeter)")
# plt.xlabel("Area")
# plt.ylabel("perimeter")
# plt.show()

#SVR

df=pd.read_csv("C:/Users/devik/Downloads/house_price_data.csv")
print(df.columns)

X=df[['bedrooms','sqft_living', 'sqft_lot','floors','sqft_above','sqft_basement']]
y=df['price']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
y_scaled=scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y_scaled, test_size=0.2, random_state=42)
    
svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr.fit(X_train, y_train)

# predict + evaluate
y_pred = svr.predict(X_test)
# evaluation
print("Accuracy(SVC) rbf", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



# # quick scatter: predicted vs actual
# plt.figure(figsize=(7,7))
# plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1)
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("SVR: Predicted vs Actual")
# plt.grid(True)