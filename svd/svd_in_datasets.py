import pandas as pd
import numpy as np
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score, root_mean_squared_error,mean_squared_error,accuracy_score,classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# # -----------------------------------------------------------
# # Load Dataset
# # -----------------------------------------------------------
# data = pd.read_csv("C:/Users/devik/Downloads/house_price_data.csv")

# df = data[['price', 'bedrooms','bathrooms','sqft_living','sqft_lot',
#            'floors','waterfront','view','condition',
#            'sqft_above','sqft_basement','yr_built','yr_renovated']]

# X = df.drop('price', axis=1)
# y = df['price']

# # -----------------------------------------------------------
# # Scaling
# # -----------------------------------------------------------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# y_scaled = (y - y.mean()) / y.std()

# # -----------------------------------------------------------
# # Train-Test Split
# # -----------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y_scaled, test_size=0.2, random_state=42
# )

# # -----------------------------------------------------------
# # SVD Dimensionality Reduction (TRAIN ONLY)
# # -----------------------------------------------------------
# U, S, Vt = np.linalg.svd(X_train, full_matrices=False)

# # CHOOSE NUMBER OF COMPONENTS TO KEEP
# k = 8    
# Vt_k = Vt[:k, :]    

# # Project train and test onto reduced space
# X_train_reduced = X_train @ Vt_k.T
# X_test_reduced = X_test @ Vt_k.T

# print("Original Features:", X_train.shape[1])
# print("Reduced Features:", X_train_reduced.shape[1])

# # -----------------------------------------------------------
# # Linear Regression on Reduced Data
# # -----------------------------------------------------------
# model_svd = LinearRegression()
# model_svd.fit(X_train_reduced, y_train)

# # Predictions
# y_pred_train = model_svd.predict(X_train_reduced)
# y_pred_test = model_svd.predict(X_test_reduced)

# # -----------------------------------------------------------
# # Evaluation
# # -----------------------------------------------------------
# print("\n================= SVD DIMENSION REDUCTION REGRESSION =================")
# print("Training RMSE:", root_mean_squared_error(y_train, y_pred_train))
# print("Training R2:", r2_score(y_train, y_pred_train))

# print("\nTesting RMSE:", root_mean_squared_error(y_test, y_pred_test))
# print("Testing R2:", r2_score(y_test, y_pred_test))

# print("\nModel Coefficients:", model_svd.coef_)
# print("Intercept:", model_svd.intercept_)
# print("=======================================================================\n")


# #3.high dim
# 
# # ================== LOAD DATA ==================
# df = pd.read_csv("C:/Users/devik/Downloads/high_dim.csv")

# X = df.drop('target', axis=1)
# y = df['target']

# # ================== TRAIN TEST SPLIT ==================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # ================== SCALING ==================
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ================== APPLY SVD ==================
# # You can choose components based on dataset size
# svd = TruncatedSVD(n_components=50, random_state=42)

# X_train_svd = svd.fit_transform(X_train_scaled)
# X_test_svd = svd.transform(X_test_scaled)

# print("Original dimensions:", X_train.shape[1])
# print("Reduced dimensions (after SVD):", X_train_svd.shape[1])

# # ================== LINEAR REGRESSION ==================
# model = LinearRegression()
# model.fit(X_train_svd, y_train)

# # ================== PREDICTION ==================
# y_pred = model.predict(X_test_svd)

# # ================== EVALUATION ==================
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Linear Regression with SVD")
# print("MSE:", mse)
# print("R2:", r2)
# print("Coefficients shape:", model.coef_)



# # ==============================================
# # Load Data
# # ==============================================
# data = pd.read_csv("C:/Users/devik/Downloads/data.csv")

# # Selecting features
# X = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
#           'smoothness_mean', 'compactness_mean', 'concavity_mean',
#           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#           'compactness_se', 'concavity_se', 'concave points_se',
#           'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
#           'perimeter_worst', 'area_worst', 'smoothness_worst',
#           'compactness_worst', 'concavity_worst', 'concave points_worst',
#           'symmetry_worst', 'fractal_dimension_worst']]

# # Target encoding
# y = data['diagnosis'].map({'M': 1, 'B': 0})

# # Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42
# )

# # ====================================================
# # WITHOUT SVD (Base RandomForest)
# # ====================================================
# start1 = time.time()
# base_model = RandomForestClassifier(
#     n_estimators=100,
#     min_samples_leaf=2,
#     max_depth=7,
#     max_features=10,
#     random_state=42
# )
# base_model.fit(X_train, y_train)
# pred_base = base_model.predict(X_test)
# time_no_svd = time.time() - start1

# acc_no_svd = accuracy_score(y_test, pred_base)

# # ====================================================
# # APPLY SVD
# # ====================================================
# svd = TruncatedSVD(n_components=20, random_state=42)
# X_train_svd = svd.fit_transform(X_train)
# X_test_svd = svd.transform(X_test)

# print("Original dimensions:", X_train.shape[1])
# print("Reduced dimensions after SVD:", X_train_svd.shape[1])
# print("Variance explained by SVD:", svd.explained_variance_ratio_.sum())

# # ====================================================
# # WITH SVD (RandomForest)
# # ====================================================
# start2 = time.time()
# svd_model = LogisticRegression(
#     max_iter=1000,
#     random_state=42
# )
# svd_model.fit(X_train_svd, y_train)
# pred_svd = svd_model.predict(X_test_svd)
# time_svd = time.time() - start2

# acc_svd = accuracy_score(y_test, pred_svd)

# # ====================================================
# # Results
# # ====================================================
# print("\n========== RESULTS ==========")
# print("Accuracy WITHOUT SVD:", acc_no_svd)
# print("Accuracy WITH SVD:", acc_svd)
# print("Training Time WITHOUT SVD:", time_no_svd)
# print("Training Time WITH SVD:", time_svd)
# print("\nClassification Report (Using SVD):")
# print(classification_report(y_test, pred_svd))





# # ====================================================
# # Load Data
# # ====================================================
# data = pd.read_csv("C:/Users/devik/Downloads/data.csv")

# X = data[['radius_mean','texture_mean','perimeter_mean','area_mean',
#           'smoothness_mean','compactness_mean','concavity_mean',
#           'concave points_mean','symmetry_mean','fractal_dimension_mean',
#           'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
#           'compactness_se','concavity_se','concave points_se','symmetry_se',
#           'fractal_dimension_se','radius_worst','texture_worst',
#           'perimeter_worst','area_worst','smoothness_worst','compactness_worst',
#           'concavity_worst','concave points_worst','symmetry_worst',
#           'fractal_dimension_worst']]

# y = data['diagnosis'].map({'M':1,'B':0})

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42
# )

# # ====================================================
# # Apply SVD
# # ====================================================
# svd = TruncatedSVD(n_components=20, random_state=42)
# X_train_svd = svd.fit_transform(X_train)
# X_test_svd = svd.transform(X_test)

# # ====================================================
# # MODELS
# # ====================================================
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=500),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "SVM": SVC(kernel='rbf', probability=True)
# }

# results_acc_no_svd = {}
# results_acc_svd = {}
# time_no_svd = {}
# time_svd = {}

# # ====================================================
# # Evaluation Function
# # ====================================================
# def train_and_evaluate(model, X_train, X_test, label):
#     start = time.time()
#     model.fit(X_train, y_train)
#     pred = model.predict(X_test)
#     end = time.time()
    
#     accuracy = accuracy_score(y_test, pred)
#     return accuracy, end - start

# # ====================================================
# # Run models (WITHOUT SVD)
# # ====================================================
# for name, model in models.items():
#     acc, t = train_and_evaluate(model, X_train, X_test, name)
#     results_acc_no_svd[name] = acc
#     time_no_svd[name] = t

# # ====================================================
# # Run models (WITH SVD)
# # ====================================================
# for name, model in models.items():
#     acc, t = train_and_evaluate(model, X_train_svd, X_test_svd, name)
#     results_acc_svd[name] = acc
#     time_svd[name] = t

# # ====================================================
# # Plot Accuracy Comparison
# # ====================================================
# plt.figure(figsize=(10, 5))
# labels = list(models.keys())
# acc1 = [results_acc_no_svd[l] for l in labels]
# acc2 = [results_acc_svd[l] for l in labels]

# x = np.arange(len(labels))
# width = 0.35

# plt.bar(x - width/2, acc1, width, label='Without SVD')
# plt.bar(x + width/2, acc2, width, label='With SVD')

# plt.ylabel("Accuracy")
# plt.title("Accuracy Comparison (With vs Without SVD)")
# plt.xticks(x, labels)
# plt.legend()
# plt.show()

# # ====================================================
# # Plot Training Time Comparison
# # ====================================================
# plt.figure(figsize=(10, 5))
# t1 = [time_no_svd[l] for l in labels]
# t2 = [time_svd[l] for l in labels]

# plt.bar(x - width/2, t1, width, label='Without SVD')
# plt.bar(x + width/2, t2, width, label='With SVD')

# plt.ylabel("Training Time (seconds)")
# plt.title("Training Time (With vs Without SVD)")
# plt.xticks(x, labels)
# plt.legend()
# plt.show()

# # ====================================================
# # Print Results
# # ====================================================
# print("\n=== FINAL RESULTS ===")
# for name in models.keys():
#     print(f"\nModel: {name}")
#     print(f"Accuracy WITHOUT SVD: {results_acc_no_svd[name]}")
#     print(f"Accuracy WITH SVD:    {results_acc_svd[name]}")
#     print(f"Time WITHOUT SVD:     {time_no_svd[name]}")
#     print(f"Time WITH SVD:        {time_svd[name]}")

'''
SVD is most useful for linear margin-based models like SVM.

Not helpful for Logistic Regression on this dataset (only tiny speed gain).

Not suitable for tree-based models like Random Forest.

Best to use SVD when:

    Features are correlated

    Noise exists

    You use linear models (SVM, Linear Regression)      '''



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

# #Applying SVD
# svd = TruncatedSVD(n_components=6, random_state=42)
# X_train_svd = svd.fit_transform(X_train)
# X_test_svd = svd.transform(X_test)
# print("Original dimensions:", X_train.shape[1])
# print("Reduced dimensions after SVD:", X_train_svd.shape[1])
# print("Variance explained by SVD:", svd.explained_variance_ratio_.sum())
# #Using SVC after SVD
# model_svd = SVC(kernel='rbf')
# model_svd.fit(X_train_svd, y_train)
# #prediction
# y_pred_svd = model_svd.predict(X_test_svd)
# print("Predicted Y after SVD", y_pred_svd[0:5])
# #evaluation
# print("Accuracy(SVC) rbf after SVD", accuracy_score(y_test, y_pred_svd))
# print(classification_report(y_test, y_pred_svd))

# svd decreases the accuracy here slightly but training is faster.


# df=pd.read_csv("C:/Users/devik/Downloads/car_data.csv")
# print(df.columns)

# engine=df[['Engine_Size (L)','Weight (kg)', 'Horsepower']] 
# MPG=df['MPG (Miles_per_Gallon)']

# model=LinearRegression()
# model.fit(engine,MPG)
# pred=model.predict(engine)
# # print(pred)
# print("MSE: ",mean_squared_error(MPG,pred))
# print("RMSE: ",root_mean_squared_error(MPG,pred))
# print("R2 Score: ",r2_score(MPG,pred))

# svd = TruncatedSVD(n_components=1, random_state=42)
# engine_svd = svd.fit_transform(engine)  
# print("Original dimensions:", engine.shape[1])
# print("Reduced dimensions after SVD:", engine_svd.shape[1])
# print("Variance explained by SVD:", svd.explained_variance_ratio_.sum())
# model_svd=LinearRegression()
# model_svd.fit(engine_svd,MPG)
# pred_svd=model_svd.predict(engine_svd)
# print("MSE after SVD: ",mean_squared_error(MPG,pred_svd))
# print("RMSE after SVD: ",root_mean_squared_error(MPG,pred_svd))
# print("R2 Score after SVD: ",r2_score(MPG,pred_svd))

# # SVD reduces feature dimensions but model performance degrades here significantly.



# marketing campaign dataset for decision tree classification
#load dataset
df=pd.read_csv("C:/Users/devik/Downloads/marketing_campaign.csv")
print(df.columns)

X=df[['Income','Kidhome','Teenhome','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain']]
print(X[0:5])
y=df['Response']
X.dropna(inplace=True)
y=y[X.index]
#preprocessing
scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

#Using DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="entropy")
model.fit(X_train,y_train)

#prediction
y_pred=model.predict(X_test)
print("Predicted Y",y_pred[0:5])

#evaluation
print("Accuracy(DecisionTreeClassifier) entropy",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

#applying SVD
svd = TruncatedSVD(n_components=18, random_state=42)
X_train_svd = svd.fit_transform(X_train)        
X_test_svd = svd.transform(X_test)
print("Original dimensions:", X_train.shape[1])
print("Reduced dimensions after SVD:", X_train_svd.shape[1])
print("Variance explained by SVD:", svd.explained_variance_ratio_.sum())
#Using DecisionTreeClassifier after SVD
model_svd=DecisionTreeClassifier(criterion="entropy")
model_svd.fit(X_train_svd,y_train)
#prediction
y_pred_svd=model_svd.predict(X_test_svd)
print("Predicted Y after SVD",y_pred_svd[0:5])
#evaluation
print("Accuracy(DecisionTreeClassifier) entropy after SVD",accuracy_score(y_test,y_pred_svd))
print(classification_report(y_test,y_pred_svd))

#not much difference in accuracy 

