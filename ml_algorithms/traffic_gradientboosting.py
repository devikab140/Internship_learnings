# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("C:/Users/devik/Downloads/all_features_traffic_dataset.csv")

# =========================
# 2. DATETIME FEATURES
# =========================
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month
    df["Year"] = df["Timestamp"].dt.year
    df["Weekday"] = df["Timestamp"].dt.weekday
    # DROP original Timestamp to avoid dtype errors
    df = df.drop(columns=["Timestamp"])

# =========================
# 3. DROP CONSTANT COLUMNS
# =========================
df = df.loc[:, df.nunique() > 1]

# =========================
# 4. REGRESSION: Travel_Time
# =========================
reg_target = "Travel_Time"
X_reg = df.drop(columns=[reg_target], errors="ignore")
y_reg = df[reg_target]

cat_cols_reg = X_reg.select_dtypes(include=['object']).columns.tolist()
num_cols_reg = X_reg.select_dtypes(include=['float','int']).columns.tolist()

# Fill missing
X_reg[num_cols_reg] = X_reg[num_cols_reg].fillna(X_reg[num_cols_reg].mean())
X_reg[cat_cols_reg] = X_reg[cat_cols_reg].fillna("missing")

# One-hot encode categorical
X_reg = pd.get_dummies(X_reg, columns=cat_cols_reg, drop_first=True)

# Train-test split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)
gb_reg.fit(X_reg_train, y_reg_train)
reg_preds = gb_reg.predict(X_reg_test)

print("\n=== Gradient Boosting Regression (Travel_Time) ===")
print("RMSE:", np.sqrt(mean_squared_error(y_reg_test, reg_preds)))
print("R2 Score:", r2_score(y_reg_test, reg_preds))

# =========================
# 5. CLASSIFICATION: Congestion_Level
# =========================
clf_target = "Congestion_Level"
X_clf = df.drop(columns=[clf_target], errors="ignore")
y_clf = df[clf_target]

# Encode target
le_clf = LabelEncoder()
y_clf = le_clf.fit_transform(y_clf)

cat_cols_clf = X_clf.select_dtypes(include=['object']).columns.tolist()
num_cols_clf = X_clf.select_dtypes(include=['float','int']).columns.tolist()

X_clf[num_cols_clf] = X_clf[num_cols_clf].fillna(X_clf[num_cols_clf].mean())
X_clf[cat_cols_clf] = X_clf[cat_cols_clf].fillna("missing")

# One-hot encode categorical
X_clf = pd.get_dummies(X_clf, columns=cat_cols_clf, drop_first=True)

# Train-test split
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)
gb_clf.fit(X_clf_train, y_clf_train)
clf_preds = gb_clf.predict(X_clf_test)

print("\n=== Gradient Boosting Classification (Congestion_Level) ===")
print("Accuracy:", accuracy_score(y_clf_test, clf_preds))
print("\nClassification Report:\n", classification_report(y_clf_test, clf_preds))
