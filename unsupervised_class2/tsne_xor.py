# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Note: you may need to update your version of future
# sudo pip install -U future


def get_xor_data():
    X1 = np.random.random((100, 2))
    X2 = np.random.random((100, 2)) - np.array([ 1, 1 ])
    X3 = np.random.random((100, 2)) - np.array([ 1, 0 ])
    X4 = np.random.random((100, 2)) - np.array([ 0, 1 ])
    X = np.vstack((X1, X2, X3, X4))
    Y = np.array([ 0 ] * 200 + [ 1 ] * 200)
    return X, Y

def main():
    X, Y = get_xor_data()
    
    plt.scatter(X[ :, 0 ], X[ :, 1 ], s=100, c=Y, alpha=0.5)
    plt.show()
    
    tsne = TSNE(perplexity=40)
    Z = tsne.fit_transform(X)
    plt.scatter(Z[ :, 0 ], Z[ :, 1 ], s=100, c=Y, alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()
