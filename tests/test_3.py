import numpy as np
np.random.seed(42)

#1
# M=np.random.randint(1,30,(3,3))
# print(M)
# det=np.linalg.det(M)
# print(f"Determinant is {det}")
# inverse=np.linalg.inv(M)
# print(f"Inverse is: \n {inverse}")
# I=np.eye(3)
# print(I)
# prod=np.round(np.matmul(M,inverse))
# print(prod)
# if np.array_equal(I,prod):
#     print(f"Product  of Matrix 'M' and its inverse is I")
#2
# Arr=np.random.rand(24)
# print(Arr)
# Arr_3d=Arr.reshape((3,4,2))
# print(Arr_3d)
# Arr_3d_t=Arr_3d.T
# print(Arr_3d_t)
# print(Arr_3d_t.shape)
# print(Arr_3d.shape)

# #3

# dice=np.random.randint(1,7,1000)
# print(dice)

# print(np.unique(dice,return_counts=True))

#4
# X=np.random.rand(10)
# Y=np.random.rand(10)
# print(X)
# print(Y)
# eucl_dist=np.sqrt(np.sum((X-Y)**2))
# print(f"Euclidean distance is {eucl_dist}")


#5
import pandas as pd
df=pd.DataFrame({
    'Name':['Alice','Bob','Charlie','David','Sam'],
    'Age':[25,30,35,28,23],
    'Salary':[50000,60000,70000,55000,None],
    'City':['NY','LA','NY','LA','NY']
})
# print(df.head(2))
# print(df['Name'][df["City"]=="NY"])
# avg_bycity=df.groupby('City')['Salary'].mean()
# print(avg_bycity)
# avg_sal=df['Salary'].mean()
# print(avg_sal)
# print(df['Salary'].isna())
# # df['Salary'].fillna(avg_sal,inplace=True)
# df.fillna({'Salary':avg_sal},inplace=True)
# print(df)


#6
dataset=pd.read_csv("C:/Users/devik/Downloads/titanic_dataset.csv")
##1
print(dataset.head(6))
##2
print(dataset.columns)
# perctg=(dataset['Survived'].sum()/dataset['Survived'].count())*100
# print(perctg)

# ##3
# print(dataset['Pclass'].value_counts().idxmax())
# ##4
# print(dataset['Name'][dataset['Fare']>100])
# ##5
# print(dataset['Cabin'].isna())
# dataset.fillna({'Cabin':'Unknown'},inplace=True)
# print(dataset['Cabin'].head(6))

# #6
# print(dataset.groupby('Survived')['Fare'].mean())
'''
since the  mean value for class survived is less than non survived ,higher ticket fare does not
indicate a better survival chance
'''

# #7
# print(dataset.groupby('Survived')['Sex'].value_counts())

# #8
# print(dataset.groupby('Survived')['Pclass'].value_counts())

# #9
# print(dataset.groupby('Survived')['Age'].mean())

# #10
print(dataset['Embarked'].value_counts())