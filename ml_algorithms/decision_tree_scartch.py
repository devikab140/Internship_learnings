import pandas as pd
import numpy as np
from math import log2
#insurance dataset
data=pd.read_csv("C:/Users/devik/Downloads/insurance_data.csv")
print(data.columns)

#data preprocessing
X=data[['age']]
y=data['bought_insurance']

print(X.head())
print(y.head())

#entropy 
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

print("Entropy of bought_insurance:", entropy(y))


#function to calculate information gain
def information_gain(X, y, feature):
    original_entropy = entropy(y)
    
    # Get unique values of the feature
    values, counts = np.unique(X[feature], return_counts=True)
    
    # Calculate the weighted entropy after the split
    weighted_entropy = 0
    for v, c in zip(values, counts):
        y_subset = y[X[feature] == v]
        weighted_entropy += (c / len(y)) * entropy(y_subset)
    
    # Information Gain is the difference in entropy
    return original_entropy - weighted_entropy

print("Information Gain for 'age':", information_gain(X, y, 'age'))

#gini index equation = 1- sum(p_i^2)

def gini_index(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)
print("Gini Index of bought_insurance:", gini_index(y))

#gain ratio= information gain / split info
def split_info(X, feature):
    values, counts = np.unique(X[feature], return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def gain_ratio(X, y, feature):
    ig = information_gain(X, y, feature)
    si = split_info(X, feature)
    if si == 0:
        return 0
    return ig / si
print("Gain Ratio for 'age':", gain_ratio(X, y, 'age'))


