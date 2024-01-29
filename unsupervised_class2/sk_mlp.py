# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import division, print_function

from sklearn.neural_network import MLPRegressor

from util import getKaggleMNIST

# Note: you may need to update your version of future
# sudo pip install -U future

# get data
X, _, Xt, _ = getKaggleMNIST()

# create the model and train it
model = MLPRegressor()
model.fit(X, X)

# test the model
print("Train R^2:", model.score(X, X))
print("Test R^2:", model.score(Xt, Xt))

Xhat = model.predict(X)
mse = ((Xhat - X) ** 2).mean()
print("Train MSE:", mse)

Xhat = model.predict(Xt)
mse = ((Xhat - Xt) ** 2).mean()
print("Test MSE:", mse)
