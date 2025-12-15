# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import lightgbm as lgb

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
    # Drop original Timestamp to avoid dtype issues
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

# Identify categorical and numeric columns
cat_cols_reg = X_reg.select_dtypes(include=['object']).columns.tolist()
num_cols_reg = X_reg.select_dtypes(include=['float','int']).columns.tolist()

# Fill missing values
X_reg[num_cols_reg] = X_reg[num_cols_reg].fillna(X_reg[num_cols_reg].mean())
X_reg[cat_cols_reg] = X_reg[cat_cols_reg].fillna("missing")

# Convert categorical columns to 'category' type
for col in cat_cols_reg:
    X_reg[col] = X_reg[col].astype('category')

# Train-test split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# LightGBM Regressor
lgb_train_reg = lgb.Dataset(X_reg_train, label=y_reg_train, categorical_feature=cat_cols_reg)
lgb_model_reg = lgb.train(
    params={'objective':'regression', 'metric':'rmse', 'learning_rate':0.05},
    train_set=lgb_train_reg,
    num_boost_round=400
)

reg_preds = lgb_model_reg.predict(X_reg_test)
print("\n=== LightGBM Regression (Travel_Time) ===")
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

# Identify categorical and numeric columns
cat_cols_clf = X_clf.select_dtypes(include=['object']).columns.tolist()
num_cols_clf = X_clf.select_dtypes(include=['float','int']).columns.tolist()

# Fill missing values
X_clf[num_cols_clf] = X_clf[num_cols_clf].fillna(X_clf[num_cols_clf].mean())
X_clf[cat_cols_clf] = X_clf[cat_cols_clf].fillna("missing")

# Convert categorical columns to 'category' type
for col in cat_cols_clf:
    X_clf[col] = X_clf[col].astype('category')

# Train-test split
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Determine LightGBM objective
if len(np.unique(y_clf)) > 2:
    objective = 'multiclass'
    num_class = len(np.unique(y_clf))
    metric = 'multi_logloss'
else:
    objective = 'binary'
    num_class = None
    metric = 'binary_logloss'

lgb_train_clf = lgb.Dataset(X_clf_train, label=y_clf_train, categorical_feature=cat_cols_clf)
params = {'objective': objective, 'learning_rate':0.05, 'metric': metric}
if num_class:
    params['num_class'] = num_class

lgb_model_clf = lgb.train(params, lgb_train_clf, num_boost_round=400)

clf_preds_probs = lgb_model_clf.predict(X_clf_test)
if objective == 'multiclass':
    clf_preds = np.argmax(clf_preds_probs, axis=1)
else:
    clf_preds = (clf_preds_probs > 0.5).astype(int)

print("\n=== LightGBM Classification (Congestion_Level) ===")
print("Accuracy:", accuracy_score(y_clf_test, clf_preds))
print("\nClassification Report:\n", classification_report(y_clf_test, clf_preds))
