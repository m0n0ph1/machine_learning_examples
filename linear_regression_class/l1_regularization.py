# notes for this course can be found at:
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# https://www.udemy.com/data-science-linear-regression-in-python

from __future__ import division, print_function

from builtins import range

import matplotlib.pyplot as plt
import numpy as np

# Note: you may need to update your version of future
# sudo pip install -U future

N = 50
D = 50

# uniformly distributed numbers between -5, +5
X = (np.random.random((N, D)) - 0.5) * 10

# true weights - only the first 3 dimensions of X affect Y
true_w = np.array([ 1, 0.5, -0.5 ] + [ 0 ] * (D - 3))

# generate Y - add noise
Y = X.dot(true_w) + np.random.randn(N) * 0.5

# perform gradient descent to find w
costs = [ ]  # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D)  # randomly initialize w
learning_rate = 0.001
l1 = 10.0  # Also try 5.0, 2.0, 1.0, 0.1 - what effect does it have on w?
for t in range(500):
    # update w
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate * (X.T.dot(delta) + l1 * np.sign(w))
    
    # find and store the cost
    mse = delta.dot(delta) / N
    costs.append(mse)

# plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

# plot our w vs true w
plt.plot(true_w, label='true w')
plt.plot(w, label='w_map')
plt.legend()
plt.show()
