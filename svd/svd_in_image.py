#loading image using PIL
from PIL import Image
import numpy as np
from numpy.linalg import svd

# Load sample image (grayscale)
img = Image.open("C:/Users/devik/Downloads/rose_flower.jpg").convert("L")
img.show(title="Original Image")

# Convert image to numpy array and normalize
print(np.array(img))
img_array = np.array(img) / 255.0
print("Original Image Shape:", img_array.shape)

# SVD
U, S, Vt = svd(img_array, full_matrices=False)

for k in [5,20,50,100]:
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    # Convert back to image
    compressed_img = Image.fromarray(np.uint8(compressed * 255))
    compressed_img.show(title="Compressed Image using SVD (k={})".format(k))
    
    # #finding frobenius norm
    # frobenius_norm = np.linalg.norm(img_array - compressed, 'fro')
    # print("Frobenius Norm for k={}: {:.4f}".format(k, frobenius_norm))

    # error using only singular values
    error_fast = np.sqrt(np.sum(S[k:]**2))
    print("Error (using only singular values) for k={}: {:.4f}".format(k,error_fast) )
