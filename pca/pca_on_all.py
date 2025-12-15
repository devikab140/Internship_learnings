from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# #cancer dataset
# df=pd.read_csv('C:/Users/devik/Downloads/data.csv')
# # print(df.columns)
# print(df.shape)
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
# # Standardize the features
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)

# # Apply PCA
# pca=PCA(n_components=0.95)
# X_pca=pca.fit_transform(X_scaled)
# print(X_pca.shape)
# print("PCA Components:\n", X_pca[0:5])
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)   
# print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))

# #after pca done you can use X_pca for further model training and evaluation

# #splitting data
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
# #Using DecisionTreeClassifier

# model=DecisionTreeClassifier(criterion="entropy")
# model.fit(X_train, y_train)
# y_pred=model.predict(X_test)
# print("Predicted Y", y_pred[0:5])
# print("Accuracy after PCA:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))



# #bankloan dataset
# # Load the dataset
# df=pd.read_csv('C:/Users/devik/Downloads/bankloan.csv')
# print(df.columns)

# # Separate features and target variable
# X=df[['Age','Experience','Income','Family','CCAvg','Mortgage','Securities.Account','CD.Account','Online','CreditCard']]
# y=df['Personal.Loan']

# # Standardize the features
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)
# print("Scaled Data:\n", X_scaled[0:5])

# # Apply PCA
# pca=PCA(n_components=0.95)  
# X_pca=pca.fit_transform(X_scaled)
# print(X_pca.shape)
# print("PCA Components:\n", X_pca[0:5])
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)
# print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))


#high dim dataset
# df=pd.read_csv('C:/Users/devik/Downloads/high_dim.csv')
# print(df.columns)
# print(df.shape)
# X=df.drop('target', axis=1)
# # y=df['target']
# # Standardize the features
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)
# print("Scaled Data:\n", X_scaled[0:5])
# # Apply PCA
# pca=PCA(n_components=0.95)
# X_pca=pca.fit_transform(X_scaled)
# print(X_pca.shape)
# print("PCA Components:\n", X_pca[0:5])
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)
# print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))



# #house price dataset
# df=pd.read_csv('C:/Users/devik/Downloads/house_price_data.csv')
# print(df.columns)
# X=df[['bedrooms','sqft_living', 'sqft_lot','floors','sqft_above','sqft_basement']]
# y=df['price']

# # Standardize the features
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)
# print("Scaled Data:\n", X_scaled[0:5])

# # Apply PCA
# pca=PCA(n_components=0.90)
# X_pca=pca.fit_transform(X_scaled)
# print(X_pca.shape)
# print("PCA Components:\n", X_pca[0:5])
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)
# print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))



# #iris dataset
# df=pd.read_csv("C:/Users/devik/Downloads/iris.csv")
# print(df.columns)
# X=df[['x0', 'x1', 'x2', 'x3', 'x4']]
# y=df['type'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
# # Standardize the features
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)
# print("Scaled Data:\n", X_scaled[0:5])

# # Apply PCA
# pca=PCA(n_components=0.95)
# X_pca=pca.fit_transform(X_scaled)
# print(X_pca.shape)
# print("PCA Components:\n", X_pca[0:5])
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)
# print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))

# plotting data after pca with 2 components 
# plt.figure(figsize=(8,5))
# plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', edgecolor='k', s=100)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA of Iris Dataset')
# plt.colorbar(label='Species')
# plt.grid(True)
# plt.show()


# #diabetes dataset
# df=pd.read_csv('C:/Users/devik/Downloads/diabetes2.csv')
# print(df.columns)
# X=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
#       'BMI', 'DiabetesPedigreeFunction', 'Age']]
# y=df['Outcome']
# # Standardize the features
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)
# print("Scaled Data:\n", X_scaled[0:5])

# # Apply PCA
# pca=PCA(n_components=0.95)
# X_pca=pca.fit_transform(X_scaled)
# print(X_pca.shape)
# print("PCA Components:\n", X_pca[0:5])
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)
# print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))


# #heart disease dataset
# df=pd.read_csv('C:/Users/devik/Downloads/framingham.csv')
# print(df.columns)
# X=df[['age',  'currentSmoker', 'cigsPerDay', 'BPMeds',
#        'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
#        'diaBP', 'BMI', 'heartRate', 'glucose']]
# X=X.copy()
# X.dropna(inplace=True)
# y=df['TenYearCHD']
# # Standardize the features
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)
# print("Scaled Data:\n", X_scaled[0:5])
# # Apply PCA
# pca=PCA(n_components=0.95)
# X_pca=pca.fit_transform(X_scaled)
# print(X_pca.shape)
# print("PCA Components:\n", X_pca[0:5])
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)
# print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))




# #kidney disease dataset
# # Load data
# df = pd.read_csv('C:/Users/devik/Downloads/kidney_disease.csv')
# print(df.columns)

# # Select feature columns
# X = df[['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
#         'sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad',
#         'appet','pe','ane']].copy()

# y = df['classification'].map({'ckd': 1, 'notckd': 0})

# print("Feature dtypes:\n", X.dtypes)

# #  Drop rows where target is missing
# y = y.dropna()
# X = X.loc[y.index]  # Keep only rows that have target

# #  Fill missing values
# def clean_data(X):
#     # Fill categorical NaN with mode
#     for col in X.select_dtypes(include='object').columns:
#         X[col] = X[col].fillna(X[col].mode()[0]).str.strip()
    
#     # Convert numeric-looking columns with dirty values
#     num_dirty_cols = ["pcv", "wc", "rc"]
#     for col in num_dirty_cols:
#         X[col] = pd.to_numeric(X[col], errors='coerce')  # invalid → NaN
#         X[col] = X[col].fillna(X[col].median())
    
#     # Fill remaining numeric NaN with median
#     for col in X.select_dtypes(include=['float64','int64']).columns:
#         X[col] = X[col].fillna(X[col].median())
    
#     return X

# X = clean_data(X)
# print("Missing after cleaning:\n", X.isnull().sum())

# #  Convert binary categorical columns to 0/1
# binary_map = {
#     'normal': 0, 'abnormal': 1,
#     'notpresent': 0, 'present': 1,
#     'good': 1, 'poor': 0,
#     'yes': 1, 'no': 0
# }

# binary_cols = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
# for col in binary_cols:
#     X[col] = X[col].map(binary_map)

# #  Scale the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print("Scaled Data Sample:\n", X_scaled[:5])

# #  Apply PCA (retain 95% variance)
# pca = PCA(n_components=0.95)
# X_pca = pca.fit_transform(X_scaled)
# print("X_pca shape:", X_pca.shape)
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)
# print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))

# #  Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X_pca, y, test_size=0.2, random_state=42
# )

# #  Train Decision Tree Classifier
# # model = DecisionTreeClassifier(criterion="entropy")
# model =LogisticRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print("Predicted Y Sample:", y_pred[:5])
# print("Accuracy after PCA:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

#############################################################################################

# #pca after svd 
# print("Kidney Disease Dataset with PCA after SVD")
# #  Perform SVD
# X_svd=TruncatedSVD(n_components=min(X_scaled.shape)-1).fit_transform(X_scaled)
# print("SVD Data Sample:\n", X_svd[:5])
# print("X_svd shape:", X_svd.shape)

# #  Apply PCA (retain 95% variance)
# pca = PCA(n_components=0.95)
# X_pca_s = pca.fit_transform(X_svd)
# print("X_pca shape:", X_pca_s.shape)
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)
# print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))

# #  Split data
# X_train1, X_test1, y_train1, y_test1 = train_test_split(
#     X_pca_s, y, test_size=0.2, random_state=42
# )

# #  Train Decision Tree Classifier
# # model1 = DecisionTreeClassifier(criterion="entropy")
# model1 =LogisticRegression()
# model1.fit(X_train1, y_train1)
# y_pred1 = model1.predict(X_test1)

# print("Predicted Y Sample:", y_pred1[:5])
# print("Accuracy after PCA:", accuracy_score(y_test1, y_pred1))
# print(classification_report(y_test1, y_pred1))