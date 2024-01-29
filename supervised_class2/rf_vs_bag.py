# https://deeplearningcourses.com/c/machine-learning-in-python-random-forest-adaboost
# https://www.udemy.com/machine-learning-in-python-random-forest-adaboost
from __future__ import division, print_function

from builtins import range

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from util import BaggedTreeRegressor

# Note: you may need to update your version of future
# sudo pip install -U future

# make simple regression data
N = 15
D = 100
X = (np.random.random((N, D)) - 0.5) * 10
Y = X.sum(axis=1) ** 2 + 0.5 * np.random.randn(N)
Ntrain = N // 2
Xtrain = X[ :Ntrain ]
Ytrain = Y[ :Ntrain ]
Xtest = X[ Ntrain: ]
Ytest = Y[ Ntrain: ]

# from rf_classification import get_data
# X, Y = get_data()
# Ntrain = int(0.8*len(X))
# Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
# Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

# from rf_regression import get_data
# Xtrain, Ytrain, Xtest, Ytest = get_data()

T = 300
test_error_rf = np.empty(T)
test_error_bag = np.empty(T)
for num_trees in range(T):
    if num_trees == 0:
        test_error_rf[ num_trees ] = None
        test_error_bag[ num_trees ] = None
    else:
        rf = RandomForestRegressor(n_estimators=num_trees)
        # rf = RandomForestClassifier(n_estimators=num_trees)
        rf.fit(Xtrain, Ytrain)
        test_error_rf[ num_trees ] = rf.score(Xtest, Ytest)
        
        bg = BaggedTreeRegressor(n_estimators=num_trees)
        # bg = BaggedTreeClassifier(n_estimators=num_trees)
        bg.fit(Xtrain, Ytrain)
        test_error_bag[ num_trees ] = bg.score(Xtest, Ytest)
    
    if num_trees % 10 == 0:
        print("num_trees:", num_trees)

plt.plot(test_error_rf, label='rf')
plt.plot(test_error_bag, label='bag')
plt.legend()
plt.show()
