# https://deeplearningcourses.com/c/unsupervised-deep-learning-in-python
# https://www.udemy.com/unsupervised-deep-learning-in-python
from __future__ import division, print_function

from builtins import input, range

import matplotlib.pyplot as plt

from unsupervised import DBN
from util import getKaggleMNIST

# Note: you may need to update your version of future
# sudo pip install -U future


def main(loadfile=None, savefile=None):
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    if loadfile:
        dbn = DBN.load(loadfile)
    else:
        dbn = DBN([ 1000, 750, 500, 10 ])  # AutoEncoder is default
        # dbn = DBN([1000, 750, 500, 10], UnsupervisedModel=RBM)
        dbn.fit(Xtrain, pretrain_epochs=2)
    
    if savefile:
        dbn.save(savefile)
    
    # first layer features
    # initial weight is D x M
    W = dbn.hidden_layers[ 0 ].W.eval()
    for i in range(dbn.hidden_layers[ 0 ].M):
        imgplot = plt.imshow(W[ :, i ].reshape(28, 28), cmap='gray')
        plt.show()
        should_quit = input("Show more? Enter 'n' to quit\n")
        if should_quit == 'n':
            break
    
    # features learned in the last layer
    for k in range(dbn.hidden_layers[ -1 ].M):
        # activate the kth node
        X = dbn.fit_to_input(k)
        imgplot = plt.imshow(X.reshape(28, 28), cmap='gray')
        plt.show()
        if k < dbn.hidden_layers[ -1 ].M - 1:
            should_quit = input("Show more? Enter 'n' to quit\n")
            if should_quit == 'n':
                break

if __name__ == '__main__':
    # to load a saved file
    # main(loadfile='rbm15.npz')
    
    # to neither load nor save
    main()
    
    # to save a trained unsupervised deep network
    # main(savefile='rbm15.npz')
