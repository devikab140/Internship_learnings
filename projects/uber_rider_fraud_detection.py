# ===============================================================
# Import Libraries
# ===============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# ===============================================================
# 1. LOAD  DATA 
# ===============================================================
df = pd.read_csv(
    "C:/Users/devik/Downloads/fraud_labeled.csv",
    low_memory=False
)

print("Loaded Shape:", df.shape)
df = df.head(15500)
print("After limiting rows:", df.shape)

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]
print("After removing dup columns:", df.shape)


# ===============================================================
# 3. CLEAN & LABEL ENCODE
# ===============================================================

# Drop timestamps (optional)
drop_cols = [
    "datetime", "datetime_x", "datetime_y",
    "date", "date_x", "date_y",
    "Time", "hour", "hour_x", "hour_y"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Convert numeric columns (NO FutureWarning)
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass  # ignore string columns

# Identify categorical and numeric
categorical_cols = df.select_dtypes(include="object").columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "fraud"]

# Encode categoricals
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# ===============================================================
# 4. BUILD X, y
# ===============================================================
X = df.drop(columns=["fraud"])
y = df["fraud"]

X = X.fillna(0)

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ===============================================================
# 5. TRAIN-TEST SPLIT
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================================================
# 6. MODELS
# ===============================================================

# XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)
acc_xgb = accuracy_score(y_test, xgb.predict(X_test))

# SVC
svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)
acc_svc = accuracy_score(y_test, svc.predict(X_test))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test))

# ===============================================================
# 7. RESULTS
# ===============================================================
print("=====================================")
print("XGBoost Accuracy:", acc_xgb)
print("SVC Accuracy:", acc_svc)
print("Random Forest Accuracy:", acc_rf)
print("=====================================")








































































# # # # # https://www.kaggle.com/datasets/advaitasen/ride-safety-dataset-of-mumbai-and-delhi/data?utm_source=chatgpt.com 
# # # # # https://www.kaggle.com/datasets/datasetengineer/metropolitan-regions-traffic-monitoring-systems/data?utm_source=chatgpt.com 
# # # # # https://www.kaggle.com/datasets/rishabhrajsharma/cityride-dataset-rides-data-drivers-data?utm_source=chatgpt.com 
# # # # # https://www.kaggle.com/datasets/mickhirsh/taxi-data-set?utm_source=chatgpt.com
# # # # # https://www.kaggle.com/datasets/zoya77/vanet-real-time-route-optimization-dataset
