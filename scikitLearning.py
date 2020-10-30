from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

X = iris.data
Y = iris.target

print(X, Y)
print(X.shape)
print(Y.shape)

# houes of students vs good?bad grades
# 10 students, train with 8 and test with 2
# allows to determine the accuracy

X_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.2)
print(X_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)