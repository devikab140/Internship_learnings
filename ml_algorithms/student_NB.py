#student mark classification using Naive Bayes (G3 >10 passed else failed)

#Importing Libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#load dataset
df=pd.read_csv("C:/Users/devik/Downloads/student-mat.csv")
print(df.columns)
print(df.info())

#data cleaning
print("Sum of null values",df.isna().sum())
print("Sum of Duplicates",df.duplicated().sum())

df['pass'] = df['G3'].apply(lambda x: 1 if x > 10 else 0)
X=df[['activities','higher','traveltime','studytime','failures','romantic','internet','freetime','goout','Dalc','Walc','absences','G1','G2']]
y=df['pass']
X_onehot = pd.get_dummies(X[['activities','higher','romantic','internet']], drop_first=True)
X = pd.concat([X.drop(['activities','higher','romantic','internet'], axis=1), X_onehot], axis=1)

# 4. Scale numeric columns

num_cols = ['traveltime','studytime','failures','freetime','goout',
            'Dalc','Walc','absences','G1','G2']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[num_cols])

# Convert back to DataFrame for readability
X_scaled_df = pd.DataFrame(X_scaled, columns=num_cols, index=X.index)

# 5. Combine scaled numeric + encoded categorical
cat_cols = ['activities_yes', 'higher_yes', 'romantic_yes', 'internet_yes']
X_final = pd.concat([X_scaled_df, X[cat_cols]], axis=1)
print("Final feature set X",X_final.head(3))

X_train,X_test,y_train,y_test=train_test_split(X_final,y,test_size=0.2,random_state=42)

#Using GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)

#prediction
# print(X_test[0:5])
y_pred=model.predict(X_test)
print("Predicted Y",y_pred[0:5])

#evaluation
cm=confusion_matrix(y_test,y_pred)
print(cm)
disp=ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix using sklearn (train test) - GaussianNB")
plt.show()
print("Accuracy(GaussianNB) train test",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Example new data for prediction
numeric_values = [[2, 2, 0, 1, 0, 1, 1, 3, 12, 14]]
numeric_scaled = scaler.transform(numeric_values)


categorical_values = np.array([[1, 1, 0, 1]]) # activities_yes, higher_yes, romantic_yes, internet_yes
final_input = np.concatenate([numeric_scaled, categorical_values], axis=1)

prediction = model.predict(final_input)
print("Prediction on new data:", prediction)
















# # Select relevant columns
# cols = ['G3','sex','age','Medu','Fedu','traveltime','studytime','failures',
#         'schoolsup','famsup','internet','romantic','higher',
#         'famrel','freetime','goout','Dalc','Walc','health',
#         'absences','G1','G2']

# # Convert categorical to numeric if not already
# df_encoded = df.copy()
# df_encoded = pd.get_dummies(df_encoded[cols], drop_first=True)

# # Plot correlation heatmap
# plt.figure(figsize=(12,8))
# sns.heatmap(df_encoded.corr()[['G3']].sort_values(by='G3', ascending=False), 
#             annot=True, cmap='coolwarm', center=0)
# plt.title("Correlation of Features with Final Grade (G3)")
# plt.show()
