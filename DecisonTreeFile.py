import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv('car.data')
print(df.shape)

X = df[[
    "buying",
    "maint",
    "safety"
]].values
Y = df[['class']]
le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = le.fit_transform(X[:, i])
print(X)
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
Y['class'] = Y['class'].map(label_mapping)
Y = np.array(Y)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
drugTree.fit(X_train, Y_train)
predTree = drugTree.predict(X_test)
print(predTree[0:5])
print(Y_test[0:5])
print("Decision Trees's Accuracy", metrics.accuracy_score(Y_test, predTree))

