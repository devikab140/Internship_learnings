#Importing necessary libraries
import pandas as pd
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score,classification_report,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
data1=pd.read_csv("C:/Users/devik/Downloads/music_features_with_mood.csv")
data2=pd.read_csv("C:/Users/devik/Downloads/skip_list_metrics.csv")

# -------------------------------------------------------
# 1. MERGING
# -------------------------------------------------------
features = data1[['acousticness', 'danceability','energy', 'instrumentalness',
                  'loudness','tempo','valence','speechiness','popularity','mood_auto']].copy()

skip_feat = data2[['skip_1', 'skip_2', 'skip_3', 'not_skipped',
                   'context_switch','no_pause_before_play', 'short_pause_before_play',
                   'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
                   'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
                   'hist_user_behavior_reason_end']].copy()

features['fake_id'] = range(len(features))
skip_feat['fake_id'] = range(len(skip_feat))

merged = pd.merge(features, skip_feat, on='fake_id', how='inner')
merged.drop('fake_id', axis=1, inplace=True)

merged = merged.drop_duplicates()

# -------------------------------------------------------
# 2. PREPROCESSING
# -------------------------------------------------------
numeric_cols = ['acousticness','danceability','energy','instrumentalness',
                'loudness','tempo','valence','speechiness','popularity']

boolean_cols = ['skip_1','skip_2','skip_3','not_skipped','context_switch',
                'no_pause_before_play','short_pause_before_play','long_pause_before_play']

binary_numeric_cols = ['hist_user_behavior_n_seekfwd',
                       'hist_user_behavior_n_seekback',
                       'hist_user_behavior_is_shuffle']

cat_col = 'hist_user_behavior_reason_end'
target = 'mood_auto'

df = merged.copy()
df[boolean_cols] = df[boolean_cols].astype(int)

# Label encoder (SAVE the encoder)
cat_encoder = LabelEncoder()
df[cat_col] = cat_encoder.fit_transform(df[cat_col])

# Scale numeric (SAVE the scaler)
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

df = df.sample(n=10000, random_state=42)

# Feature matrix
X = df[numeric_cols + boolean_cols + binary_numeric_cols + [cat_col]]
y = df[target]

# -------------------------------------------------------
# 3. CLASSIFICATION MODEL
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='rbf')

# 5-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold)

print("Cross Validation Scores kfold:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

skf=StratifiedKFold(n_splits=5)
cv_scores=cross_val_score(model,X, y, cv=skf)
print("cross_val_score results _skf:", cv_scores)
print("cross_val_score mean:", cv_scores.mean())


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("SVC Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
