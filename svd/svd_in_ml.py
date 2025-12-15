#SVD for Dimensionality Reduction (like PCA)


# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import load_iris

# # Load dataset
# iris = load_iris()
# X = iris.data  # features

# # Standardize (important for SVD/PCA)
# X = StandardScaler().fit_transform(X)

# # Apply SVD (reduce 4 → 2)
# svd = TruncatedSVD(n_components=2)
# X_svd = svd.fit_transform(X)

# print("Reduced Shape:", X_svd.shape)
# print("Reduced Data (first 5 rows):\n", X_svd[:5])



#####################################################################
#SVD for Recommendation Systems (Matrix Factorization)

# import numpy as np
# from numpy.linalg import svd

# # User-Item rating matrix (0 = missing)
# R = np.array([[5, 3, 0, 1],
#               [4, 0, 0, 1],
#               [1, 1, 0, 5],
#               [0, 0, 5, 4],
#               [0, 1, 5, 4]])

# # Perform SVD
# U, S, Vt = svd(R, full_matrices=False)

# # Keep top 2 singular values
# k = 2
# U_k = U[:, :k]
# S_k = np.diag(S[:k])
# Vt_k = Vt[:k, :]

# # Reconstruct matrix (predict missing ratings)
# R_pred = U_k @ S_k @ Vt_k

# print("Predicted Ratings:\n", np.round(R_pred, 2))


######################################################################
#SVD for Image Compression

# import numpy as np
# import matplotlib.pyplot as plt
# from numpy.linalg import svd
# from skimage import data

# # Load sample image (grayscale)
# img = data.camera()
# img = img / 255.0
# plt.imshow(img, cmap="gray")
# plt.title("Original Image ")
# plt.axis('off')
# plt.show()

# # SVD
# U, S, Vt = svd(img, full_matrices=False)

# # Keep only top k singular values
# k = 50
# compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# plt.imshow(compressed, cmap="gray")
# plt.title("Compressed Image using SVD (k=50)")
# plt.axis('off')
# plt.show()



###################################################
#Latent Semantic Analysis (LSA) in NLP

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

docs = [
    "Machine learning is amazing",
    "Deep learning and AI",
    "SVD is used in NLP",
    "Topic extraction using LSA"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

svd = TruncatedSVD(n_components=2)
X_topics = svd.fit_transform(X)

print("Topic representation:\n", X_topics)
