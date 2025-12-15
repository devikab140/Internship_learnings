import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Example dataset: replace this with your actual dataset
from sklearn.datasets import load_iris
data = load_iris()
X = data.data  # features


# Step 1: scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # X = your features

# Step 2: apply PCA (no n_components — take all)
pca = PCA()
pca.fit(X_scaled)

# Step 3: cumulative explained variance
cum_var = np.cumsum(pca.explained_variance_ratio_)

# Step 4: plot
plt.figure(figsize=(8,5))
plt.plot(cum_var, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative EVR to Choose Number of PCs")
plt.grid(True)
plt.show()

# Step 5 (optional): print index where improvement becomes small
diff = np.diff(cum_var)
print("Difference between consecutive EVR values:\n", diff)
threshold = 0.01  
num_components = np.where(diff < threshold)[0]
if len(num_components) > 0:
    print(f"Number of components where improvement < {threshold}: {num_components[0] }")
else:
    print("No components found where improvement is below the threshold.")