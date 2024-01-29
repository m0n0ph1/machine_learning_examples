# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
from __future__ import division, print_function

import matplotlib.pyplot as plt

from knn import KNN
from util import get_donut

# Note: you may need to update your version of future
# sudo pip install -U future

if __name__ == '__main__':
    X, Y = get_donut()
    
    # display the data
    plt.scatter(X[ :, 0 ], X[ :, 1 ], s=100, c=Y, alpha=0.5)
    plt.show()
    
    # get the accuracy
    model = KNN(3)
    model.fit(X, Y)
    print("Accuracy:", model.score(X, Y))
