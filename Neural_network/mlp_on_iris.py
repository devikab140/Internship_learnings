import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

#one hot encoding
encoder=OneHotEncoder(sparse_output=False)
y=encoder.fit_transform(y.reshape(-1,1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build the MLP model
model=Sequential(
    [
        Dense(10,activation='relu',input_shape=(4,)),
        Dense(3,activation='softmax')
    ]
)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X_train,y_train,epochs=200,verbose=1)

# Evaluate the model
Loss,accuracy=model.evaluate(X_test,y_test,verbose=1)
print(f'Test Loss: {Loss}, Test Accuracy: {accuracy}')

# Make predictions
predictions=model.predict(X_test)
print("Predictions on test set:")
for i,pred in enumerate(predictions):
    print(f"Input: {X_test[i]}, Predicted class: {pred.argmax()}, Probabilities: {pred}")
