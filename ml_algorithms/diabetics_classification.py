# Importig libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay,classification_report,log_loss



#load dataset

df=pd.read_csv("C:/Users/devik/Downloads/diabetes2.csv")

#data cleaning
print(df.info())
print("Sum of null values",df.isna().sum())
print("Sum of Duplicates",df.duplicated().sum())

#outliers
sns.boxplot(df['Pregnancies'])
plt.show()

#preprocessing
X=df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y=df['Outcome']
scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)
print("Scaled X",X_scaled[5])

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# model selection anf fitting
model=LogisticRegression()
model.fit(X_train,y_train)

#prediction
print("Predicted probabilities \n ",model.predict_proba(X_test)[0:5,])
y_pred=model.predict(X_test)
print("Predicted Y",y_pred[0:5])

#evaluation
cm=confusion_matrix(y_test,y_pred)
print(cm)
disp=ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.title("confusion matrix for Diebetics dataset")
plt.show()

new_df=df[['Glucose','Outcome']]
print("correlation",new_df.corr())


print("Accuracy",accuracy_score(y_test,y_pred))
print("coeficients ",model.coef_)
print(classification_report(y_test,y_pred))
print(model.predict([[-1.14185152,  0.5040552,  -1.50468724  ,0.90726993 , 0.76583594,  1.4097456 ,5.4849091 , -0.0204964 ]]))
loss=log_loss(y_test,model.predict(X_test))
print("Logistic Loss:",loss)
#visualization 

feature_index = 1  # choose feature  Glucose
# Create smooth range for that feature
x_min, x_max = X_scaled[:, feature_index].min(), X_scaled[:, feature_index].max()
x_range = np.linspace(x_min, x_max, 200).reshape(-1, 1)

# Keep other features fixed at mean
X_mean = np.mean(X_scaled, axis=0)
X_plot = np.tile(X_mean, (200, 1))
X_plot[:, feature_index] = x_range.flatten()

# Predicted probabilities (sigmoid curve)
y_prob = model.predict_proba(X_plot)[:, 1]

# ---- Add actual data points ----
X_test_feature = X_test[:, feature_index]   # scaled feature from test data
y_test_prob = model.predict_proba(X_test)[:, 1]  # model's predicted probability for test set
y_test_actual = y_test                       # 0 or 1 actual labels

plt.figure(figsize=(8,5))

# Sigmoid curve
plt.plot(x_range, y_prob, color='blue', linewidth=2, label='Sigmoid Curve')

# Data points (actual)
plt.scatter(X_test_feature[y_test_actual==0], y_test_prob[y_test_actual==0],
            color='red', alpha=0.6, label='Class 0 (Non-Diabetic)')
plt.scatter(X_test_feature[y_test_actual==1], y_test_prob[y_test_actual==1],
            color='green', alpha=0.6, label='Class 1 (Diabetic)')

# Decision boundary
plt.axhline(0.5, color='gray', linestyle='--', label='Decision Boundary (0.5)')

plt.xlabel('Glucose')
plt.ylabel('Predicted Probability')
plt.title('Sigmoid Curve with Actual Data Points')
plt.legend()
plt.grid(True)
plt.show()

