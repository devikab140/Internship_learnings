#solving XOR using ANN
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the XOR input and output
X=np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]])
y=np.array([[0],[1],[1],[0]])

# Build the ANN model
model=Sequential(
    [
        Dense(4,activation='tanh',input_shape=(2,)),
        Dense(1,activation='sigmoid')
    ]
)

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X,y,epochs=1000,verbose=0)

# Predict the outputs
predictions=model.predict(X)
for i,x in enumerate(X):
    print(f"Input: {x} --> predicted output: {round(predictions[i][0])}")