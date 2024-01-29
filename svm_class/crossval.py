# https://deeplearningcourses.com/c/support-vector-machines-in-python
# https://www.udemy.com/support-vector-machines-in-python
from __future__ import division, print_function

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Note: you may need to update your version of future
# sudo pip install -U future

# load the data
data = load_breast_cancer()

for C in (0.5, 1.0, 5.0, 10.0):
    pipeline = Pipeline([ ('scaler', StandardScaler()), ('svm', SVC(C=C)) ])
    scores = cross_val_score(pipeline, data.data, data.target, cv=5)
    print("C:", C, "mean:", scores.mean(), "std:", scores.std())
