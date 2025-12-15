#comparing all method for date fruit classification ,choose best one

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder


# Load dataset
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

#logistic regression
model_lr = LogisticRegression(max_iter=200)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, target_names=le.classes_))

# Decision Tree Classifier
model_dt = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=2)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
print("Decision Tree Classifier Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

# Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, max_features=10, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

#comparison
if accuracy_score(y_test, y_pred_lr) >= accuracy_score(y_test, y_pred_nb) and accuracy_score(y_test, y_pred_lr) >= accuracy_score(y_test, y_pred_dt) and accuracy_score(y_test, y_pred_lr) >= accuracy_score(y_test, y_pred_rf):
    print(f"Best Model: Logistic Regression ,since compared to others it has highest accuracy {accuracy_score(y_test, y_pred_lr)} ")
elif accuracy_score(y_test, y_pred_nb) >= accuracy_score(y_test, y_pred_lr) and accuracy_score(y_test, y_pred_nb) >= accuracy_score(y_test, y_pred_dt) and accuracy_score(y_test, y_pred_nb) >= accuracy_score(y_test, y_pred_rf):
    print(f"Best Model: Naive Bayes ,since compared to others it has highest accuracy {accuracy_score(y_test, y_pred_nb)} ")
elif accuracy_score(y_test, y_pred_dt) >= accuracy_score(y_test, y_pred_lr) and accuracy_score(y_test, y_pred_dt) >= accuracy_score(y_test, y_pred_nb) and accuracy_score(y_test, y_pred_dt) >= accuracy_score(y_test, y_pred_rf):
    print(f"Best Model: Decision Tree Classifier ,since compared to others it has highest accuracy {accuracy_score(y_test, y_pred_dt)}")
else:
    print(f"Best Model: Random Forest Classifier ,since compared to others it has highest accuracy {accuracy_score(y_test, y_pred_rf)}")