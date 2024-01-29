# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import division, print_function

from datetime import datetime

from sklearn.linear_model import LogisticRegression
from umap import UMAP

from util import getKaggleMNIST

# Note: you may need to update your version of future
# sudo pip install -U future

# get the data
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

print("Score without transformation:")
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print(model.score(Xtrain, Ytrain))
print(model.score(Xtest, Ytest))

umapper = UMAP(n_neighbors=5, n_components=10)
t0 = datetime.now()
Ztrain = umapper.fit_transform(Xtrain)
print("umap fit_transform took:", datetime.now() - t0)
t0 = datetime.now()
Ztest = umapper.transform(Xtest)
print("umap transform took:", datetime.now() - t0)

print("Score with transformation")
model = LogisticRegression()
t0 = datetime.now()
model.fit(Ztrain, Ytrain)
print("logistic regression fit took:", datetime.now() - t0)
print(model.score(Ztrain, Ytrain))
print(model.score(Ztest, Ytest))
