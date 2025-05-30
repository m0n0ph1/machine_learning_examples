# https://deeplearningcourses.com/c/support-vector-machines-in-python
# https://www.udemy.com/support-vector-machines-in-python
from __future__ import division, print_function

from datetime import datetime

from sklearn.svm import SVC

from util import getKaggleMNIST

# Note: you may need to update your version of future
# sudo pip install -U future

# get the data: https://www.kaggle.com/c/digit-recognizer
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

# model = SVC()
model = SVC(C=5., gamma=.05)

t0 = datetime.now()
model.fit(Xtrain, Ytrain)
print("train duration:", datetime.now() - t0)
t0 = datetime.now()
print("train score:", model.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
t0 = datetime.now()
print("test score:", model.score(Xtest, Ytest), "duration:", datetime.now() - t0)
