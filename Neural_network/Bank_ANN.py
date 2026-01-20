import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


train_data=pd.read_csv("C:/Users/devik/Downloads/bank-full_corctd.csv")
test_data=pd.read_csv("C:/Users/devik/Downloads/bank_corctd.csv")

# Convert categorical columns to numeric using 'category' dtype
categorical_cols = train_data.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('y')  # exclude target for now

for col in categorical_cols:
    # Convert train column to category codes
    train_data[col] = train_data[col].astype('category').cat.codes
    # Ensure same categories mapping in test
    test_data[col] = test_data[col].astype('category').cat.codes

# Step 4: Convert target 'y' to 0/1
train_data['y'] = train_data['y'].map({'no': 0, 'yes': 1})
test_data['y'] = test_data['y'].map({'no': 0, 'yes': 1})

# Step 5: Separate features and target
X_train = train_data.drop('y', axis=1)
y_train = train_data['y']
X_test = test_data.drop('y', axis=1)
y_test = test_data['y']

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Build ANN Model
model = Sequential([
    (Dense(32, input_dim=X_train.shape[1], activation='relu')) ,
    (Dense(16, activation='tanh')),
    (Dense(8, activation='relu')),
    (Dense(1, activation='sigmoid'))
])

# Step 8: Compile Model
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy'])

# Step 9: Train Model
model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=0)

# Step 10: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

##########################################################################
#comparing with ml models

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

Rf_model=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
Rf_model.fit(X_train,y_train)

y_pred=Rf_model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_test,y_pred))
