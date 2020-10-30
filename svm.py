from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
Y = iris.target
classes = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
print(X, Y)

X_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.2)
model = svm.SVC()
model.fit(X_train, y_train)
prediction = model.predict(x_test)
print(prediction)
acc = accuracy_score(y_test, prediction)
print(acc)