# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python

# data is from https://www.kaggle.com/c/digit-recognizer
# each image is a D = 28x28 = 784 dimensional vector
# there are N = 42000 samples
# you can plot an image by reshaping to (28,28) and using plt.imshow()
from __future__ import division, print_function

# from kmeans import plot_k_means, get_simple_data
# from gmm import gmm
from sklearn.mixture import GaussianMixture

from kmeans_mnist import DBI, get_data, purity

# Note: you may need to update your version of future
# sudo pip install -U future


def main():
    X, Y = get_data(10000)
    print("Number of data points:", len(Y))
    
    model = GaussianMixture(n_components=10)
    model.fit(X)
    M = model.means_
    R = model.predict_proba(X)
    
    print("Purity:", purity(Y, R))  # max is 1, higher is better
    print("DBI:", DBI(X, M, R))  # lower is better

if __name__ == "__main__":
    main()
