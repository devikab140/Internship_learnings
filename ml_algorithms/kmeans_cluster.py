# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler

# -----------------------------------------
# # 1. Load / Create Your Dataset
# # -----------------------------------------
# data = load_iris()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# print(df.shape)

# X = df.select_dtypes(include=['int64', 'float64'])

# # Scaling is important for K-Means
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # -----------------------------------------
# # 2. Elbow Method
# # -----------------------------------------
# inertia_list = []
# K_range = range(2, 11)

# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X_scaled)
#     inertia_list.append(kmeans.inertia_)

# plt.figure(figsize=(6, 4))
# plt.plot(K_range, inertia_list, marker='o')
# plt.title("Elbow Method - Optimal K")
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("Inertia (Within-Cluster SSE)")
# plt.grid(True)
# plt.show()

# # -----------------------------------------
# # 3. Silhouette Score Method
# # -----------------------------------------
# silhouette_scores = []

# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X_scaled)
#     sil_score = silhouette_score(X_scaled, labels)
#     silhouette_scores.append(sil_score)

# plt.figure(figsize=(6, 4))
# plt.plot(K_range, silhouette_scores, marker='o')
# plt.title("Silhouette Score vs K")
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Score")
# plt.grid(True)
# plt.show()

# best_k_silhouette = K_range[np.argmax(silhouette_scores)]
# print(f"Best K by Silhouette Score: {best_k_silhouette}")

# # -----------------------------------------
# # 4. GAP STATISTIC
# # -----------------------------------------

# def gap_statistic(X, K_range, n_refs=10):
#     gaps = []
#     std_devs = []

#     bounds = np.min(X, axis=0), np.max(X, axis=0)

#     for k in K_range:
#         ref_disps = []
        
#         # Generate Reference Datasets
#         for _ in range(n_refs):
#             random_data = np.random.uniform(bounds[0], bounds[1], size=X.shape)
#             km = KMeans(n_clusters=k, random_state=42)
#             km.fit(random_data)
#             ref_disps.append(km.inertia_)

#         # Real K-Means
#         km = KMeans(n_clusters=k, random_state=42)
#         km.fit(X)

#         log_ref = np.log(ref_disps).mean()
#         log_orig = np.log(km.inertia_)

#         gap = log_ref - log_orig
#         gaps.append(gap)

#         std_devs.append(np.std(np.log(ref_disps)))

#     return gaps, std_devs


# gaps, gap_std = gap_statistic(X_scaled, K_range)

# plt.figure(figsize=(6, 4))
# plt.plot(K_range, gaps, marker='o')
# plt.title("Gap Statistic vs K")
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("Gap Value")
# plt.grid(True)
# plt.show()

# best_k_gap = K_range[np.argmax(gaps)]
# print(f"Best K by Gap Statistic: {best_k_gap}")

# # -----------------------------------------
# # 5. Final K-Means with Best K
# # -----------------------------------------
# best_k = best_k_silhouette   
# kmeans_final = KMeans(n_clusters=best_k, random_state=42)
# clusters = kmeans_final.fit_predict(X_scaled)

# df['Cluster'] = clusters
# print(df.head())

# # -----------------------------------------
# # 6. Cluster Visualization
# # -----------------------------------------
# col1 = "sepal length (cm)"
# col2 = "sepal width (cm)"

# plt.scatter(df[col1], df[col2], c=clusters)
# plt.xlabel(col1)
# plt.ylabel(col2)
# plt.title("K-Means Clusters")
# plt.show()



# # # for bankloan data

# # # -----------------------------------------
# # # 1. Load / Create Your Dataset
# # # -----------------------------------------
# # df=pd.read_csv("C:/Users/devik/Downloads/bankloan.csv")
# # print(df.shape)

# # X = df.select_dtypes(include=['int64', 'float64'])

# # # Scaling is important for K-Means
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # # -----------------------------------------
# # # 2. Elbow Method
# # # -----------------------------------------
# # inertia_list = []
# # K_range = range(2, 15)

# # for k in K_range:
# #     kmeans = KMeans(n_clusters=k, random_state=42)
# #     kmeans.fit(X_scaled)
# #     inertia_list.append(kmeans.inertia_)

# # plt.figure(figsize=(6, 4))
# # plt.plot(K_range, inertia_list, marker='o')
# # plt.title("Elbow Method - Optimal K")
# # plt.xlabel("Number of Clusters (K)")
# # plt.ylabel("Inertia (Within-Cluster SSE)")
# # plt.grid(True)
# # plt.show()

# # # -----------------------------------------
# # # 3. Silhouette Score Method
# # # -----------------------------------------
# # silhouette_scores = []

# # for k in K_range:
# #     kmeans = KMeans(n_clusters=k, random_state=42)
# #     labels = kmeans.fit_predict(X_scaled)
# #     sil_score = silhouette_score(X_scaled, labels)
# #     silhouette_scores.append(sil_score)

# # plt.figure(figsize=(6, 4))
# # plt.plot(K_range, silhouette_scores, marker='o')
# # plt.title("Silhouette Score vs K")
# # plt.xlabel("Number of Clusters")
# # plt.ylabel("Silhouette Score")
# # plt.grid(True)
# # plt.show()

# # best_k_silhouette = K_range[np.argmax(silhouette_scores)]
# # print(f"Best K by Silhouette Score: {best_k_silhouette}")

# # # -----------------------------------------
# # # 4. GAP STATISTIC
# # # -----------------------------------------

# # def gap_statistic(X, K_range, n_refs=10):
# #     gaps = []
# #     std_devs = []

# #     bounds = np.min(X, axis=0), np.max(X, axis=0)

# #     for k in K_range:
# #         ref_disps = []

# #         # Generate Reference Datasets
# #         for _ in range(n_refs):
# #             random_data = np.random.uniform(bounds[0], bounds[1], size=X.shape)
# #             km = KMeans(n_clusters=k, random_state=42)
# #             km.fit(random_data)
# #             ref_disps.append(km.inertia_)

# #         # Real K-Means
# #         km = KMeans(n_clusters=k, random_state=42)
# #         km.fit(X)

# #         log_ref = np.log(ref_disps).mean()
# #         log_orig = np.log(km.inertia_)

# #         gap = log_ref - log_orig
# #         gaps.append(gap)

# #         std_devs.append(np.std(np.log(ref_disps)))

# #     return gaps, std_devs


# # gaps, gap_std = gap_statistic(X_scaled, K_range)

# # plt.figure(figsize=(6, 4))
# # plt.plot(K_range, gaps, marker='o')
# # plt.title("Gap Statistic vs K")
# # plt.xlabel("Number of Clusters (K)")
# # plt.ylabel("Gap Value")
# # plt.grid(True)
# # plt.show()

# # best_k_gap = K_range[np.argmax(gaps)]
# # print(f"Best K by Gap Statistic: {best_k_gap}")

# # # -----------------------------------------
# # # 5. Final K-Means with Best K
# # # -----------------------------------------
# # best_k = best_k_gap   
# # kmeans_final = KMeans(n_clusters=best_k, random_state=42)
# # clusters = kmeans_final.fit_predict(X_scaled)

# # df['Cluster'] = clusters
# # print(df.head())

# # # -----------------------------------------
# # # 6. Cluster Visualization
# # # -----------------------------------------
# # col1 = "Income"
# # col2 = "Age"

# # plt.scatter(df[col1], df[col2], c=clusters)
# # plt.xlabel(col1)
# # plt.ylabel(col2)
# # plt.title("K-Means Clusters")
# # plt.show()


# # #house price

# # # -----------------------------------------
# # # 1. Load / Create Your Dataset
# # # -----------------------------------------
# # df=pd.read_csv("C:/Users/devik/Downloads/house_price_data.csv")
# # print(df.shape)

# # X = df.select_dtypes(include=['int64', 'float64'])

# # # Scaling is important for K-Means
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # # -----------------------------------------
# # # 2. Elbow Method
# # # -----------------------------------------
# # inertia_list = []
# # K_range = range(2, 20)

# # for k in K_range:
# #     kmeans = KMeans(n_clusters=k, random_state=42)
# #     kmeans.fit(X_scaled)
# #     inertia_list.append(kmeans.inertia_)

# # plt.figure(figsize=(6, 4))
# # plt.plot(K_range, inertia_list, marker='o')
# # plt.title("Elbow Method - Optimal K")
# # plt.xlabel("Number of Clusters (K)")
# # plt.ylabel("Inertia (Within-Cluster SSE)")
# # plt.grid(True)
# # plt.show()

# # # -----------------------------------------
# # # 3. Silhouette Score Method
# # # -----------------------------------------
# # silhouette_scores = []

# # for k in K_range:
# #     kmeans = KMeans(n_clusters=k, random_state=42)
# #     labels = kmeans.fit_predict(X_scaled)
# #     sil_score = silhouette_score(X_scaled, labels)
# #     silhouette_scores.append(sil_score)

# # plt.figure(figsize=(6, 4))
# # plt.plot(K_range, silhouette_scores, marker='o')
# # plt.title("Silhouette Score vs K")
# # plt.xlabel("Number of Clusters")
# # plt.ylabel("Silhouette Score")
# # plt.grid(True)
# # plt.show()

# # best_k_silhouette = K_range[np.argmax(silhouette_scores)]
# # print(f"Best K by Silhouette Score: {best_k_silhouette}")

# # # -----------------------------------------
# # # 4. GAP STATISTIC
# # # -----------------------------------------

# # def gap_statistic(X, K_range, n_refs=10):
# #     gaps = []
# #     std_devs = []

# #     bounds = np.min(X, axis=0), np.max(X, axis=0)

# #     for k in K_range:
# #         ref_disps = []

# #         # Generate Reference Datasets
# #         for _ in range(n_refs):
# #             random_data = np.random.uniform(bounds[0], bounds[1], size=X.shape)
# #             km = KMeans(n_clusters=k, random_state=42)
# #             km.fit(random_data)
# #             ref_disps.append(km.inertia_)

# #         # Real K-Means
# #         km = KMeans(n_clusters=k, random_state=42)
# #         km.fit(X)

# #         log_ref = np.log(ref_disps).mean()
# #         log_orig = np.log(km.inertia_)

# #         gap = log_ref - log_orig
# #         gaps.append(gap)

# #         std_devs.append(np.std(np.log(ref_disps)))

# #     return gaps, std_devs


# # gaps, gap_std = gap_statistic(X_scaled, K_range)

# # plt.figure(figsize=(6, 4))
# # plt.plot(K_range, gaps, marker='o')
# # plt.title("Gap Statistic vs K")
# # plt.xlabel("Number of Clusters (K)")
# # plt.ylabel("Gap Value")
# # plt.grid(True)
# # plt.show()

# # best_k_gap = K_range[np.argmax(gaps)]
# # print(f"Best K by Gap Statistic: {best_k_gap}")

# # # -----------------------------------------
# # # 5. Final K-Means with Best K
# # # -----------------------------------------
# # best_k = best_k_gap   
# # kmeans_final = KMeans(n_clusters=best_k, random_state=42)
# # clusters = kmeans_final.fit_predict(X_scaled)

# # df['Cluster'] = clusters
# # print(df.head())

# # # -----------------------------------------
# # # 6. Cluster Visualization
# # # -----------------------------------------
# # col1 = "price"
# # col2 = "floors"

# # plt.scatter(df[col1], df[col2], c=clusters)
# # plt.xlabel(col1)
# # plt.ylabel(col2)
# # plt.title("K-Means Clusters")
# # plt.show()

