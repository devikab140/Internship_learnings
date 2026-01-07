import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# # -----------------------------
# # 1. Load Data
# # -----------------------------
# data = pd.read_csv("C:/Users/devik/Downloads/data.csv")

# # -----------------------------
# # 2. Select Features
# # -----------------------------
# X = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
#           'smoothness_mean', 'compactness_mean', 'concavity_mean',
#           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#           'compactness_se', 'concavity_se', 'concave points_se',
#           'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
#           'perimeter_worst', 'area_worst', 'smoothness_worst',
#           'compactness_worst', 'concavity_worst', 'concave points_worst',
#           'symmetry_worst', 'fractal_dimension_worst']]

# y = data['diagnosis']   # not used for training (unsupervised)

# # -----------------------------
# # 3. Standardize Features
# # -----------------------------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # -----------------------------
# # 4.1Finding optimal k using BIC and AIC
# # -----------------------------
# bic_scores = []
# aic_scores = []
# k_values = range(1, 7)

# for k in k_values:
#     gmm = GaussianMixture(
#         n_components=k,
#         covariance_type='full',
#         random_state=42
#     )
#     gmm.fit(X_scaled)
    
#     bic_scores.append(gmm.bic(X_scaled))
#     aic_scores.append(gmm.aic(X_scaled))
    
#     print(f"k={k} | BIC={bic_scores[-1]:.2f} | AIC={aic_scores[-1]:.2f}")

# best_k_bic = k_values[np.argmin(bic_scores)]
# best_k_aic = k_values[np.argmin(aic_scores)]

# print("Best k according to BIC:", best_k_bic)
# print("Best k according to AIC:", best_k_aic)
# plt.plot(k_values, bic_scores, marker='o', label='BIC')
# plt.plot(k_values, aic_scores, marker='o', label='AIC')
# plt.xlabel("Number of Components (k)")
# plt.ylabel("Score")
# plt.title("BIC & AIC for GMM")
# plt.legend()
# plt.grid()
# plt.show()
# # -----------------------------
# # 4.2 Train GMM on FULL (30-D) DATA
# # -----------------------------
# final_k = best_k_bic

# final_gmm = GaussianMixture(
#     n_components=final_k,
#     covariance_type='full',
#     random_state=42
# )

# final_gmm.fit(X_scaled)
# final_labels = final_gmm.predict(X_scaled)
# print("Probabilities for first sample",final_gmm.predict_proba(X_scaled)[0])  
# # print("Mean of first sample",final_gmm.means_[0])
# # print("Covariance of first  component",final_gmm.covariances_[0])

# # -----------------------------
# # 5. PCA ONLY for Visualization
# # -----------------------------
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # Project GMM centers into PCA space
# centers_2d = pca.transform(final_gmm.means_)

# # -----------------------------
# # 6. Visualization
# # -----------------------------
# plt.figure(figsize=(10, 6))

# plt.scatter(
#     X_pca[:, 0],
#     X_pca[:, 1],
#     c=final_labels,
#     cmap='viridis',
#     s=40,
#     edgecolor='k'
# )

# plt.scatter(
#     centers_2d[:, 0],
#     centers_2d[:, 1],
#     c='red',
#     marker='X',
#     s=200,
#     label='GMM Centers'
# )

# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.title('GMM Trained in 30D (Visualized in 2D using PCA)')
# plt.legend()
# plt.grid()
# plt.show()

# # -----------------------------
# # 7. Optional: Check shapes
# # -----------------------------
# print("Original feature shape:", X.shape)
# print("GMM means shape:", final_gmm.means_.shape)

#--------------------------------------------------------------------------------------

# #2.date fruit
# df=pd.read_excel("C:/Users/devik/Downloads/Date_Fruit_Datasets.xlsx")
# print(df.columns)

# # Data preprocessing
# X=df[['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY',
#        'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO',
#        'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2',
#        'SHAPEFACTOR_3', 'SHAPEFACTOR_4', 'MeanRR', 'MeanRG', 'MeanRB',
#        'StdDevRR', 'StdDevRG', 'StdDevRB', 'SkewRR', 'SkewRG', 'SkewRB',
#        'KurtosisRR', 'KurtosisRG', 'KurtosisRB', 'EntropyRR', 'EntropyRG',
#        'EntropyRB', 'ALLdaub4RR', 'ALLdaub4RG', 'ALLdaub4RB']]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Finding optimal k using BIC and AIC
# bic_scores = []
# aic_scores = []
# k_values = range(1, 7)
# for k in k_values:
#     gmm = GaussianMixture(
#         n_components=k,
#         covariance_type='full',
#         random_state=42
#     )
#     gmm.fit(X_scaled)
    
#     bic_scores.append(gmm.bic(X_scaled))
#     aic_scores.append(gmm.aic(X_scaled))
    
#     print(f"k={k} | BIC={bic_scores[-1]:.2f} | AIC={aic_scores[-1]:.2f}")

# best_k_bic = k_values[np.argmin(bic_scores)]
# print("Best k according to BIC:", best_k_bic)

# plt.plot(k_values, bic_scores, marker='o', label='BIC')
# plt.plot(k_values, aic_scores, marker='o', label='AIC')
# plt.xlabel("Number of Components (k)")
# plt.ylabel("Score")
# plt.title("BIC & AIC for GMM")
# plt.legend()
# plt.grid()
# plt.show()

# # Train GMM on FULL DATA
# final_k = best_k_bic
# final_gmm = GaussianMixture(
#     n_components=final_k,
#     covariance_type='full',
#     random_state=42
# )
# final_gmm.fit(X_scaled)
# final_labels = final_gmm.predict(X_scaled)
# print("Probabilities for first sample",final_gmm.predict_proba(X_scaled)[0])
# # print("Mean of first sample",final_gmm.means_[0])
# # print("Covariance of first  component",final_gmm.covariances_[0])

#--------------------------------------------------------------------------------------

# #3. Kidney disease
# df = pd.read_csv('C:/Users/devik/Downloads/kidney_disease.csv')
# print(df.columns)

# # Select feature columns
# X = df[['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
#         'sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad',
#         'appet','pe','ane']].copy()
# #-----------------------------
# # 2. Identify numeric & categorical columns
# # -----------------------------
# numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
# categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# # -----------------------------
# # 3. Handle missing values
# # -----------------------------
# # Numeric columns: fill with mean
# X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

# # Categorical columns: fill with 'missing'
# X[categorical_cols] = X[categorical_cols].fillna('missing')

# # -----------------------------
# # 4. One-hot encode categorical columns
# # -----------------------------
# X = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Finding optimal k using BIC and AIC
# bic_scores = [] 
# aic_scores = []
# k_values = range(1, 7)
# for k in k_values:
#     gmm = GaussianMixture(
#         n_components=k,
#         covariance_type='full',
#         random_state=42
#     )
#     gmm.fit(X_scaled)
    
#     bic_scores.append(gmm.bic(X_scaled))
#     aic_scores.append(gmm.aic(X_scaled))
    
#     print(f"k={k} | BIC={bic_scores[-1]:.2f} | AIC={aic_scores[-1]:.2f}")
# best_k_bic = k_values[np.argmin(bic_scores)]
# print("Best k according to BIC:", best_k_bic)
# plt.plot(k_values, bic_scores, marker='o', label='BIC')
# plt.plot(k_values, aic_scores, marker='o', label='AIC')
# plt.xlabel("Number of Components (k)")
# plt.ylabel("Score")
# plt.title("BIC & AIC for GMM")
# plt.legend()
# plt.grid()
# plt.show()

# # Train GMM on FULL DATA
# final_k = best_k_bic
# final_gmm = GaussianMixture(
#     n_components=final_k,
#     covariance_type='full',
#     random_state=42
# )
# final_gmm.fit(X_scaled)
# final_labels = final_gmm.predict(X_scaled)
# print("Probabilities for first sample",final_gmm.predict_proba(X_scaled)[0])
# # print("Mean of first sample",final_gmm.means_[0])
# # print("Covariance of first  component",final_gmm.covariances_[0])

#------------------------------------------------------------------------------------

# # 4.house price dataset
# df=pd.read_csv('C:/Users/devik/Downloads/house_price_data.csv')
# print(df.columns)
# X=df[['bedrooms','sqft_living', 'sqft_lot','floors','sqft_above','sqft_basement']]
# y=df['price']

# # Standardize the features
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)

# # Finding optimal k using BIC and AIC
# bic_scores=[]
# aic_sores=[]
# k_vals=range(1,7)
# for k in k_vals:
#     gmm=GaussianMixture(n_components=k,covariance_type='full',random_state=42)
#     gmm.fit(X_scaled)
#     bic_scores.append(gmm.bic(X_scaled))
#     aic_sores.append(gmm.aic(X_scaled))
#     print(f"k={k}  | BIC={bic_scores[-1]:.2f} | AIC={aic_sores[-1]:.2f}") #-1 will give the most recent appended value
# best_k_bic=k_vals[np.argmin(bic_scores)]
# print("Best k according to BIC:",best_k_bic)

# plt.plot(k_vals,bic_scores,marker='o',label='BIC')
# plt.plot(k_vals,aic_sores,marker='o',label='AIC')
# plt.xlabel("Number of Components (k)")
# plt.ylabel("Score")
# plt.title("BIC & AIC for GMM")
# plt.legend()
# plt.grid()
# plt.show()

# # Train GMM on FULL DATA
# final_k=best_k_bic  
# final_gmm=GaussianMixture(n_components=final_k,covariance_type='full',random_state=42)
# final_gmm.fit(X_scaled)
# final_labels=final_gmm.predict(X_scaled)
# print("Probabilities for first sample",final_gmm.predict_proba(X_scaled)[0])
# # print("Mean of first sample",final_gmm.means_[0])
# # print("Covariance of first  component",final_gmm.covariances_[0])

#------------------------------------------------------------------------------------

# #5.iris dataset
# df=pd.read_csv("C:/Users/devik/Downloads/iris.csv")
# print(df.columns)
# X=df[['x0', 'x1', 'x2', 'x3', 'x4']]
# y=df['type'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})

# # Standardize the features
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)

# # Finding optimal k using BIC and AIC
# bic_scores=[]
# aic_sores=[]
# k_vals=range(1,7)
# for k in k_vals:
#     gmm=GaussianMixture(n_components=k,covariance_type='full',random_state=42)
#     gmm.fit(X_scaled)
#     bic_scores.append(gmm.bic(X_scaled))
#     aic_sores.append(gmm.aic(X_scaled))
#     print(f"k={k}  | BIC={bic_scores[-1]:.2f} | AIC={aic_sores[-1]:.2f}") #-1 will give the most recent appended value
# best_k_bic=k_vals[np.argmin(bic_scores)]
# print("Best k according to BIC:",best_k_bic)

# plt.plot(k_vals,bic_scores,marker='o',label='BIC')
# plt.plot(k_vals,aic_sores,marker='o',label='AIC')
# plt.xlabel("Number of Components (k)")
# plt.ylabel("Score")
# plt.title("BIC & AIC for GMM")
# plt.legend()
# plt.grid()
# plt.show()

# # Train GMM on FULL DATA
# final_k=best_k_bic  
# final_gmm=GaussianMixture(n_components=final_k,covariance_type='full',random_state=42)
# final_gmm.fit(X_scaled)
# final_labels=final_gmm.predict(X_scaled)
# print("Probabilities for first sample",final_gmm.predict_proba(X_scaled)[0])
# # print("Mean of first sample",final_gmm.means_[0])
# # print("Covariance of first  component",final_gmm.covariances_[0])

#------------------------------------------------------------------------------------

# #6.diabetes dataset
# df=pd.read_csv('C:/Users/devik/Downloads/diabetes2.csv')
# print(df.columns)
# X=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
#       'BMI', 'DiabetesPedigreeFunction', 'Age']]

# # Standardize the features
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)

# # Finding optimal k using BIC and AIC
# bic_scores=[]
# aic_sores=[]
# k_vals=range(1,7)
# for k in k_vals:
#     gmm=GaussianMixture(n_components=k,covariance_type='full',random_state=42)
#     gmm.fit(X_scaled)
#     bic_scores.append(gmm.bic(X_scaled))
#     aic_sores.append(gmm.aic(X_scaled))
#     print(f"k={k}  | BIC={bic_scores[-1]:.2f} | AIC={aic_sores[-1]:.2f}") #-1 will give the most recent appended value
# best_k_bic=k_vals[np.argmin(bic_scores)]
# print("Best k according to BIC:",best_k_bic)

# plt.plot(k_vals,bic_scores,marker='o',label='BIC')
# plt.plot(k_vals,aic_sores,marker='o',label='AIC')
# plt.xlabel("Number of Components (k)")
# plt.ylabel("Score")
# plt.title("BIC & AIC for GMM")
# plt.legend()
# plt.grid()
# plt.show()

# # Train GMM on FULL DATA
# final_k=best_k_bic  
# final_gmm=GaussianMixture(n_components=final_k,covariance_type='full',random_state=42)
# final_gmm.fit(X_scaled)
# final_labels=final_gmm.predict(X_scaled)
# print("Probabilities for first sample",final_gmm.predict_proba(X_scaled)[0])
# # print("Mean of first sample",final_gmm.means_[0])
# # print("Covariance of first  component",final_gmm.covariances_[0])

#-------------------------------------------------------------------------------------

# #7.insurance
# data=pd.read_csv("C:/Users/devik/Downloads/insurance_data.csv")
# print(data.columns)

# #data preprocessing
# X=data[['age']]
# scaler=StandardScaler()
# X_scaled=scaler.fit_transform(X)

# # Finding optimal k using BIC and AIC
# bic_scores=[]
# aic_sores=[]
# k_vals=range(1,7)
# for k in k_vals:
#     gmm=GaussianMixture(n_components=k,covariance_type='full',random_state=42)
#     gmm.fit(X_scaled)
#     bic_scores.append(gmm.bic(X_scaled))
#     aic_sores.append(gmm.aic(X_scaled))
#     print(f"k={k}  | BIC={bic_scores[-1]:.2f} | AIC={aic_sores[-1]:.2f}") #-1 will give the most recent appended value
# best_k_bic=k_vals[np.argmin(bic_scores)]
# print("Best k according to BIC:",best_k_bic)

# plt.plot(k_vals,bic_scores,marker='o',label='BIC')
# plt.plot(k_vals,aic_sores,marker='o',label='AIC')
# plt.xlabel("Number of Components (k)")
# plt.ylabel("Score")
# plt.title("BIC & AIC for GMM")
# plt.legend()
# plt.grid()
# plt.show()

# # Train GMM on FULL DATA
# final_k=best_k_bic  
# final_gmm=GaussianMixture(n_components=final_k,covariance_type='full',random_state=42)
# final_gmm.fit(X_scaled)
# final_labels=final_gmm.predict(X_scaled)
# print("Probabilities for first sample",final_gmm.predict_proba(X_scaled)[0])
# # print("Mean of first sample",final_gmm.means_[0])
# # print("Covariance of first  component",final_gmm.covariances_[0])

#--------------------------------------------------------------------------------------

#8. marketing

df=pd.read_csv("C:/Users/devik/Downloads/marketing_campaign_corrected.csv")
print(df.columns)
print(df.info())

#data cleaning
print("Sum of null values",df.isna().sum())
print("Sum of Duplicates",df.duplicated().sum())
df=df.dropna()
print("Sum of null values",df.isna().sum())

X=df[['Income','Kidhome','Teenhome','Recency','MntWines','MntFruits',
      'MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds',
      'NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases',
      'NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1',
      'AcceptedCmp2','Complain']]
y=df['Response']

#preprocessing
scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)


# Finding optimal k using BIC and AIC
bic_scores=[]
aic_sores=[]
k_vals=range(1,20)
for k in k_vals:
    gmm=GaussianMixture(n_components=k,covariance_type='full',random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_sores.append(gmm.aic(X_scaled))
    print(f"k={k}  | BIC={bic_scores[-1]:.2f} | AIC={aic_sores[-1]:.2f}") #-1 will give the most recent appended value
best_k_bic=k_vals[np.argmin(bic_scores)]
print("Best k according to BIC:",best_k_bic)

plt.plot(k_vals,bic_scores,marker='o',label='BIC')
plt.plot(k_vals,aic_sores,marker='o',label='AIC')
plt.xlabel("Number of Components (k)")
plt.ylabel("Score")
plt.title("BIC & AIC for GMM")
plt.legend()
plt.grid()
plt.show()

# Train GMM on FULL DATA
final_k=best_k_bic  
final_gmm=GaussianMixture(n_components=final_k,covariance_type='full',random_state=42)
final_gmm.fit(X_scaled)
final_labels=final_gmm.predict(X_scaled)
print("Probabilities for first sample",final_gmm.predict_proba(X_scaled)[0])
# print("Mean of first sample",final_gmm.means_[0])
# print("Covariance of first  component",final_gmm.covariances_[0])
