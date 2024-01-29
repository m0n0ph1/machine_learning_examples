# https://deeplearningcourses.com/c/deep-learning-prerequisites-the-numpy-stack-in-python
# https://www.udemy.com/deep-learning-prerequisites-the-numpy-stack-in-python

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np

# Note: you may need to update your version of future
# sudo pip install -U future

# generate unlabeled data
N = 2000
X = np.random.random((N, 2)) * 2 - 1

# generate labels
Y = np.zeros(N)
Y[ (X[ :, 0 ] < 0) & (X[ :, 1 ] > 0) ] = 1
Y[ (X[ :, 0 ] > 0) & (X[ :, 1 ] < 0) ] = 1

# plot it
plt.scatter(X[ :, 0 ], X[ :, 1 ], c=Y)
plt.show()
