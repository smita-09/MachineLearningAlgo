from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()
#features and Labels
X = boston.data
y = boston.target
print(X, y)
print(X.shape)
print(y.shape)

#algorithm
l_reg = linear_model.LinearRegression()
plt.scatter(X.T[5] ,y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = l_reg.fit(X_train,y_train)
predict = model.predict(X_test)
print(predict)
print('R^2 value:', l_reg.score(X, y))
print('coeff:', l_reg.coef_)
print('intercept:', l_reg.intercept_)