# simple perceptron implementation in Python
import numpy as np
from sklearn.linear_model import Perceptron

# Define the training data (AND logic gate)
X = np.array([[0, 0],
              [0, 1],   
              [1, 0],
              [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate output
# Create the Perceptron model
model = Perceptron(max_iter=1000, tol=1e-3)
# Train the model
model.fit(X, y)

# Test the model
test_data = np.array([[0, 0],  
                      [0, 1],
                      [1, 0],
                      [1, 1]])
predictions = model.predict(test_data)
print("Predictions for AND gate:")
for i, test in enumerate(test_data):
    print(f"Input: {test}, Predicted Output: {predictions[i]}")

print("Weights:", model.coef_)
print("Bias:", model.intercept_)

#or gate
y_or = np.array([0, 1, 1, 1])  # OR gate output

# Create the Perceptron model for OR gate
model_or = Perceptron(max_iter=1000, tol=1e-3)
model_or.fit(X, y_or)

# Test the model for OR gate
predictions_or = model_or.predict(test_data)
print("\nPredictions for OR gate:")
for i, test in enumerate(test_data):
    print(f"Input: {test}, Predicted Output: {predictions_or[i]}")

print("Weights (OR):", model_or.coef_)
print("Bias (OR):", model_or.intercept_)

#xor gate
y_xor = np.array([0, 1, 1, 0])  # XOR gate output

# Create the Perceptron model for XOR gate
model_xor = Perceptron(max_iter=1000, tol=1e-3)
model_xor.fit(X, y_xor)

# Test the model for XOR gate
predictions_xor = model_xor.predict(test_data)
print("\nPredictions for XOR gate:")
for i, test in enumerate(test_data):
    print(f"Input: {test}, Predicted Output: {predictions_xor[i]}")
print("Weights (XOR):", model_xor.coef_)
print("Bias (XOR):", model_xor.intercept_)
print(model_xor.score(X, y_xor))