import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np


# -----------------------------------------
# # 1. Load / Create Your Dataset
# # -----------------------------------------
# data = load_iris()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# print(df.shape)

# X = df.select_dtypes(include=['int64', 'float64'])

# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

# def best_k_by_silhouette(X, k_min=2, k_max=10):

#     best_k = k_min
#     best_score = -1

#     for k in range(k_min, k_max + 1):

#         kmeans = KMeans(n_clusters=k).fit(X)
#         labels = kmeans.labels_

#         score = silhouette_score(X, labels)

#         print(f"K={k}, Silhouette={score:.4f}")

#         if score > best_score:
#             best_score = score
#             best_k = k

#     print("\n Best K =", best_k)
#     return best_k
# print(best_k_by_silhouette(X))

##############################################################
# combined one
def best_k_combined(X, k_min=2, k_max=10):

    wcss = []
    silhouettes = []

    # Step 1: compute WCSS + Silhouette for each k
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        labels = kmeans.labels_
        
        wcss_k = kmeans.inertia_
        sil_k = silhouette_score(X, labels)
        
        wcss.append(wcss_k)
        silhouettes.append(sil_k)

        print(f"K={k}  WCSS={wcss_k:.2f}  Silhouette={sil_k:.4f}")

    # Step 2: compute WCSS percentage drop
    wcss_drop = []
    for i in range(1, len(wcss)):
        drop = (wcss[i-1] - wcss[i]) / wcss[i-1]
        wcss_drop.append(drop)

    # Add a dummy 0 drop for k_min
    wcss_drop = [0] + wcss_drop  

    print("\nWCSS Drop % for each K:")
    for k in range(k_min, k_max + 1):
        print(f"K={k}, Drop={wcss_drop[k-k_min]:.4f}")

    # Step 3: normalize silhouette and drop to combine fairly
    sil_norm = (silhouettes - np.min(silhouettes)) / (np.max(silhouettes) - np.min(silhouettes))
    drop_norm = (wcss_drop - np.min(wcss_drop)) / (np.max(wcss_drop) - np.min(wcss_drop))

    # Step 4: final score = silhouette + drop
    final_score = sil_norm + drop_norm

    best_k = (np.argmax(final_score) + k_min)

    print("\nFinal Combined Scores:")
    for k in range(k_min, k_max + 1):
        print(f"K={k}, Combined Score={final_score[k-k_min]:.4f}")

    print("\nBest K (Combined Method) =", best_k)

    return best_k

# print(best_k_combined(X))