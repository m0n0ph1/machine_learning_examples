# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python
from __future__ import division, print_function

from builtins import range

import matplotlib.pyplot as plt
import numpy as np

from kmeans import cost, get_simple_data, plot_k_means

# Note: you may need to update your version of future
# sudo pip install -U future


def main():
    X = get_simple_data()
    
    plt.scatter(X[ :, 0 ], X[ :, 1 ])
    plt.show()
    
    costs = np.empty(10)
    costs[ 0 ] = None
    for k in range(1, 10):
        M, R = plot_k_means(X, k, show_plots=False)
        c = cost(X, R, M)
        costs[ k ] = c
    
    plt.plot(costs)
    plt.title("Cost vs K")
    plt.show()

if __name__ == '__main__':
    main()
