import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder


# # Load cancer dataset
# data = pd.read_csv("C:/Users/devik/Downloads/data.csv")
# print(data.columns)

# # Data preprocessing
# X=data[['radius_mean', 'texture_mean', 'perimeter_mean',
#        'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
#        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
#        'fractal_dimension_se', 'radius_worst', 'texture_worst',
#        'perimeter_worst', 'area_worst', 'smoothness_worst',
#        'compactness_worst', 'concavity_worst', 'concave points_worst',
#        'symmetry_worst', 'fractal_dimension_worst']]

# y=data['diagnosis'].map({'M':1,'B':0})
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Using RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100,min_samples_leaf=2,max_depth=7,max_features=10, random_state=42)
# model.fit(X_train, y_train)

# # Prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])

# # Evaluation
# print("Accuracy(RandomForestClassifier)",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# #pumkin dataset
# data = pd.read_csv("C:/Users/devik/Downloads/Pumpkin_Seeds_Dataset.xlsx - Pumpkin_Seeds_Dataset.csv")
# print(data.columns)

# # Data preprocessing
# X=data[['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
#        'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity', 'Extent',
#        'Roundness', 'Aspect_Ration', 'Compactness']]
# y=data['Class']
# print(y.value_counts())
# # Encoding class labels
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# scalar=StandardScaler()
# X_scaled=scalar.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# # Using RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_leaf=2,max_features=10, random_state=42)
# model.fit(X_train, y_train)
# # Prediction
# y_pred = model.predict(X_test)
# print("Predicted Y", y_pred[0:5])
# # Evaluation
# print("Accuracy(RandomForestClassifier) pumpkin",accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred, target_names=le.classes_))


# dates dataset
df=pd.read_csv("C:/Users/devik/Downloads/Date_Fruit_Datasets.xlsx - Date_Fruit_Datasets.csv")
print(df.columns)

# Data preprocessing
X=df[['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY',
       'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO',
       'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2',
       'SHAPEFACTOR_3', 'SHAPEFACTOR_4', 'MeanRR', 'MeanRG', 'MeanRB',
       'StdDevRR', 'StdDevRG', 'StdDevRB', 'SkewRR', 'SkewRG', 'SkewRB',
       'KurtosisRR', 'KurtosisRG', 'KurtosisRB', 'EntropyRR', 'EntropyRG',
       'EntropyRB', 'ALLdaub4RR', 'ALLdaub4RG', 'ALLdaub4RB']]
y=df['Class']
print(y.value_counts())
# Encoding class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Using RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10,min_samples_leaf=2,max_features=10,random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
print("Predicted Y", y_pred[0:5])

# Evaluation
print("Accuracy(RandomForestClassifier) dates",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))