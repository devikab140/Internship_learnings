# #Importing necessary libraries
# import pandas as pd
# from sklearn.svm import SVC,SVR
# from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
# from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
# from sklearn.metrics import accuracy_score,classification_report,mean_squared_error,r2_score
# from sklearn.preprocessing import StandardScaler,LabelEncoder
# import matplotlib.pyplot as plt
# import seaborn as sns

# #load dataset
# data1=pd.read_csv("C:/Users/devik/Downloads/music_features_with_mood.csv")
# data2=pd.read_csv("C:/Users/devik/Downloads/skip_list_metrics.csv")

# # -------------------------------------------------------
# # 1. MERGING
# # -------------------------------------------------------
# features = data1[['acousticness', 'danceability','energy', 'instrumentalness',
#                   'loudness','tempo','valence','speechiness','popularity','mood_auto']].copy()

# skip_feat = data2[['skip_1', 'skip_2', 'skip_3', 'not_skipped',
#                    'context_switch','no_pause_before_play', 'short_pause_before_play',
#                    'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
#                    'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
#                    'hist_user_behavior_reason_end']].copy()

# features['fake_id'] = range(len(features))
# skip_feat['fake_id'] = range(len(skip_feat))

# merged = pd.merge(features, skip_feat, on='fake_id', how='inner')
# merged.drop('fake_id', axis=1, inplace=True)

# merged = merged.drop_duplicates()

# # -------------------------------------------------------
# # 2. PREPROCESSING
# # -------------------------------------------------------
# numeric_cols = ['acousticness','danceability','energy','instrumentalness',
#                 'loudness','tempo','valence','speechiness','popularity']

# boolean_cols = ['skip_1','skip_2','skip_3','not_skipped','context_switch',
#                 'no_pause_before_play','short_pause_before_play','long_pause_before_play']

# binary_numeric_cols = ['hist_user_behavior_n_seekfwd',
#                        'hist_user_behavior_n_seekback',
#                        'hist_user_behavior_is_shuffle']


# target = 'mood_auto'

# df = merged.copy()
# df[boolean_cols] = df[boolean_cols].astype(int)


# # Scale numeric (SAVE the scaler)
# scaler = StandardScaler()
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# # df.to_csv("merged_music.csv", index=False)


# df = df.sample(n=10000, random_state=42)

# # Feature matrix
# X = df[numeric_cols + boolean_cols + binary_numeric_cols]
# y = df[target]

# # -------------------------------------------------------
# # 3. CLASSIFICATION MODEL
# # -------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = SVC(kernel='rbf')

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("SVC Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# model1=RandomForestClassifier()
# model1.fit(X_train, y_train)

# y1_pred = model.predict(X_test)

# print("Random forest Accuracy:", accuracy_score(y_test, y1_pred))
# print("\nClassification Report:\n", classification_report(y_test, y1_pred))


# # -------------------------------------------------------
# # 4. NEW SONG PREDICTION (CLASSIFICATION)
# # -------------------------------------------------------
# new_song_raw = [
#     0.95, 0.40, 0.60, 0.1, -4.2, 80.0, 0.55, 0.60, 48,
#     1,1,0,1,0,1,0,0,
#     2,1,1,
#     "fwdbtn"
# ]

# feature_order = numeric_cols + boolean_cols + binary_numeric_cols + [cat_col]

# # Convert to DF
# new_df = pd.DataFrame([new_song_raw], columns=feature_order)

# # # Apply SAME transformations (NO re-fitting)
# new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])
# new_df[cat_col] = cat_encoder.transform(new_df[cat_col])

# mood_prediction = model.predict(new_df)[0]
# print("\nPredicted Mood:", mood_prediction)

# # -------------------------------------------------------
# # 5. LISTEN PERCENTAGE TARGET (REGRESSION)
# # -------------------------------------------------------
# df['listen_percentage'] = (
#     df['skip_1']*20 + 
#     df['skip_2']*40 + 
#     df['skip_3']*60 + 
#     df['not_skipped']*100
# )

# df['listen_scaled'] = df['listen_percentage'] / 100
# y1 = df['listen_scaled']

# X_train_r, X_test_r, y1_train, y1_test = train_test_split(
#     X, y1, test_size=0.2, random_state=42
# )

# svr_model = SVR(kernel='rbf')
# svr_model.fit(X_train_r, y1_train)

# y1_pred = svr_model.predict(X_test_r)

# print("\nSVR Regression Metrics:")
# print("MSE:", mean_squared_error(y1_test, y1_pred))
# print("R2 Score:", r2_score(y1_test, y1_pred))

# rfr_model=RandomForestRegressor()
# rfr_model.fit(X_train_r, y1_train)

# y2_pred = rfr_model.predict(X_test_r)

# print("\n Random forest Regression Metrics:")
# print("MSE:", mean_squared_error(y1_test, y2_pred))
# print("R2 Score:", r2_score(y1_test, y2_pred))

# # -------------------------------------------------------
# # 6. NEW SONG — LISTEN PERCENTAGE PREDICTION
# # -------------------------------------------------------
# new_df = pd.DataFrame([new_song_raw], columns=feature_order)

# # Apply transformations again
# new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])
# new_df[cat_col] = cat_encoder.transform(new_df[cat_col])

# listen_pred_scaled = svr_model.predict(new_df)[0]
# listen_pred_scaled = max(0, min(1, listen_pred_scaled))   # clip between 0–1
# listen_pred = listen_pred_scaled * 100

# print("\nPredicted Listen Percentage:", listen_pred)



# #------------------------------------------------------------
# # Identifying the most influential features affecting listener skip behavior
# #-------------------------------------------------------------

# # Select only numeric + boolean + binary features + skip behavior
# corr_cols = numeric_cols + boolean_cols + binary_numeric_cols

# # Compute correlation matrix
# corr_matrix = df[corr_cols].corr()

# # Plot
# plt.figure(figsize=(14,10))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
# plt.title("Correlation Heatmap - Audio & Engagement Features")
# plt.show()


# #---------------------------------------------------------
# # Visualizations & insights
# #---------------------------------------------------------

# # 1 : skip vs mood
# df["skip"] = df[["skip_1", "skip_2", "skip_3"]].max(axis=1)
# skip_counts = df.groupby("mood_auto")["skip"].sum()
# print(skip_counts)

# skip_counts.plot(kind="bar", figsize=(7,5))

# plt.xlabel("Mood")
# plt.ylabel("Number of Skipped Songs")
# plt.title("Skip Count for Each Mood")
# plt.show()


# # 2:

# # audio features vs skip
# df["skip"] = df[["skip_1", "skip_2", "skip_3"]].max(axis=1)

# audio_cols = ['acousticness','danceability','energy','instrumentalness',
#               'loudness','tempo','valence','speechiness','popularity']

# plt.figure(figsize=(10,6))
# sns.heatmap(df[audio_cols + ["skip"]].corr(), annot=True, cmap="coolwarm")
# plt.title("Audio Features vs Skip (Correlation Heatmap)")
# plt.show()


# #user engagement vs skip
# engagement_cols = ['context_switch','no_pause_before_play','short_pause_before_play',
#                    'long_pause_before_play','hist_user_behavior_n_seekfwd',
#                    'hist_user_behavior_n_seekback','hist_user_behavior_is_shuffle']

# plt.figure(figsize=(10,6))
# sns.heatmap(df[engagement_cols + ["skip"]].corr(), annot=True, cmap="coolwarm")
# plt.title("Engagement Features vs Skip (Correlation Heatmap)")
# plt.show()

 