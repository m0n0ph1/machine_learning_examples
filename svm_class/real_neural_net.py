# https://deeplearningcourses.com/c/support-vector-machines-in-python
# https://www.udemy.com/support-vector-machines-in-python
from __future__ import division, print_function

from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from util import getKaggleMNIST

# Note: you may need to update your version of future
# sudo pip install -U future

# get the data: https://www.kaggle.com/c/digit-recognizer
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

# scale first
pipeline = Pipeline([
    # ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(500,), activation='tanh')),
    # ('lr', LogisticRegression()),
])

t0 = datetime.now()
pipeline.fit(Xtrain, Ytrain)
print("train duration:", datetime.now() - t0)
t0 = datetime.now()
print("train score:", pipeline.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
t0 = datetime.now()
print("test score:", pipeline.score(Xtest, Ytest), "duration:", datetime.now() - t0)
