#Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from k_finding import best_k_combined
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#load dataset
df=pd.read_csv("C:/Users/devik/Downloads/Mall_Customers.csv")
# print(df.head())

#data cleaning
print(df.shape)
print(df.isna().sum())
print(df.duplicated().sum())

df = df.drop_duplicates()

df = df.dropna()
#data preprocessing
X=df[['Annual Income (k$)','Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k=best_k_combined(X_scaled)

#model tarining
  
kmeans_final = KMeans(n_clusters=k, random_state=42)
clusters = kmeans_final.fit_predict(X_scaled)

df['Cluster'] = clusters
print(df.head())

# If X was scaled, convert cluster centers back to original scale
centers = scaler.inverse_transform(kmeans_final.cluster_centers_)

# Create a dataframe for easier inspection
cluster_centers = pd.DataFrame(centers, columns=['Annual Income', 'Spending Score', 'Age'] if 'Age' in X.columns else ['Annual Income', 'Spending Score'])
cluster_centers['Cluster'] = range(k)
# Define mapping based on your interpretation
cluster_labels_mapping = {
    0: 'High Income-Low Expenditure',
    1: 'Old Age',
    2: 'Low Income-Low Expenditure',
    3: 'Low Income-High Expenditure'
}

# Apply mapping
df['Cluster_Label'] = df['Cluster'].map(cluster_labels_mapping)

# -------------------------------------------------------
# 7. Plot Clusters with Labels
# -------------------------------------------------------
plt.figure(figsize=(8,6))
colors = {
    'High Income-Low Expenditure': 'red',
    'Old Age': 'purple',
    'Low Income-Low Expenditure': 'blue',
    'Low Income-High Expenditure': 'green'
}

for label, color in colors.items():
    subset = df[df['Cluster_Label'] == label]
    plt.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
                label=label, color=color, s=80)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clusters with Custom Labels")
plt.legend()
plt.show()

# -------------------------------------------------------
# 8. Optional: Check first few rows
# -------------------------------------------------------
print(df[['Age','Annual Income (k$)','Spending Score (1-100)','Cluster','Cluster_Label']].head())

# # -----------------------------------------
# # 6. Cluster Visualization
# # -----------------------------------------

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'],
#            c=df['Cluster'], s=60)

# ax.set_xlabel("Age")
# ax.set_ylabel("Annual Income")
# ax.set_zlabel("Spending Score")
# plt.title("3D K-Means Clusters")
# plt.show()


# ###################################################################
# #two features
# #data preprocessing
# X=df[['Annual Income (k$)','Spending Score (1-100)']]
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# k=best_k_combined(X_scaled)

# #model tarining
  
# kmeans_final = KMeans(n_clusters=k, random_state=42)
# clusters = kmeans_final.fit_predict(X_scaled)

# df['Cluster'] = clusters
# print(df.head())

# # -----------------------------------------
# # 6. Cluster Visualization
# # -----------------------------------------
# col1 = "Annual Income (k$)"
# col2 = "Spending Score (1-100)"

# plt.scatter(df[col1], df[col2], c=clusters)
# plt.xlabel(col1)
# plt.ylabel(col2)
# plt.title("K-Means Clusters")
# plt.show()




# # -------------------------------------------------------
# # Without k means just segmentation and plotting
# # -------------------------------------------------------

# print(df.head())

# # -------------------------------------------------------
# # 3. Define segmentation rules
# # -------------------------------------------------------
# def assign_segment(row):
#     if row['Age'] < 18:
#         return 'Childrens'
#     elif row['Age'] > 60:
#         return 'Old Age'
#     elif row['Annual Income (k$)'] < 40 and row['Spending Score (1-100)'] < 50:
#         return 'Low Income-Low Expenditure'
#     elif row['Annual Income (k$)'] < 40 and row['Spending Score (1-100)'] >= 50:
#         return 'Low Income-High Expenditure'
#     elif row['Annual Income (k$)'] >= 40 and row['Spending Score (1-100)'] < 50:
#         return 'High Income-Low Expenditure'
#     else:
#         return 'High Income-High Expenditure'  # optional, if needed

# # Apply the function
# df['Segment'] = df.apply(assign_segment, axis=1)

# # -------------------------------------------------------
# # 4. Check results
# # -------------------------------------------------------
# print(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Segment']].head(10))

# # -------------------------------------------------------
# # 5. Visualization
# # -------------------------------------------------------
# plt.figure(figsize=(8,6))

# colors = {
#     'Childrens': 'cyan',
#     'Low Income-Low Expenditure': 'blue',
#     'Low Income-High Expenditure': 'green',
#     'High Income-Low Expenditure': 'red',
#     'Old Age': 'purple'
# }

# for segment, color in colors.items():
#     subset = df[df['Segment'] == segment]
#     plt.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
#                 label=segment, color=color, s=80)

# plt.xlabel("Annual Income (k$)")
# plt.ylabel("Spending Score (1-100)")
# plt.title("Custom Customer Segments")
# plt.legend()
# plt.show()
