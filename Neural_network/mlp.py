import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# AND gate dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1])   # AND gate output

# Build the model
model = Sequential([
    Dense(2, activation='tanh', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X, y, epochs=500, verbose=0)

# Predict
predictions = model.predict(X)

print("Predictions for AND gate:")
for i, x in enumerate(X):
    print(f"Input: {x}, Predicted Output: {round(predictions[i][0])}")

#xor gate dataset
y_xor = np.array([0, 1, 1, 0])

# Build the model for XOR gate
model_xor = Sequential([
    Dense(4, activation='tanh', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

# Compile
model_xor.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model_xor.fit(X, y_xor, epochs=500, verbose=0)

# Predict
predictions_xor = model_xor.predict(X)

print("Predictions for XOR gate:")
for i, x in enumerate(X):
    print(f"Input: {x}, Predicted Output: {round(predictions_xor[i][0])}")
