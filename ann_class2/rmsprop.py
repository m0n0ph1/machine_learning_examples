# Compare RMSprop vs. constant learning rate
# For the class Data Science: Practical Deep Learning Concepts in Theano and TensorFlow
# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import division, print_function

from builtins import range

import matplotlib.pyplot as plt
import numpy as np

from mlp import derivative_b1, derivative_b2, derivative_w1, derivative_w2, forward
from util import cost, error_rate, get_normalized_data, y2indicator

# Note: you may need to update your version of future
# sudo pip install -U future


def main():
    max_iter = 20  # make it 30 for sigmoid
    print_period = 10
    
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    lr = 0.00004
    reg = 0.01
    
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    
    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz
    
    M = 300
    K = 10
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)
    
    # 1. const
    # cost = -16
    LL_batch = [ ]
    CR_batch = [ ]
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[ j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Ytrain_ind[ j * batch_sz:(j * batch_sz + batch_sz), ]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
            # print "first batch cost:", cost(pYbatch, Ybatch)
            
            # gradients
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg * W2
            gb2 = derivative_b2(Ybatch, pYbatch) + reg * b2
            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg * W1
            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1
            
            # updates
            W2 -= lr * gW2
            b2 -= lr * gb2
            W1 -= lr * gW1
            b1 -= lr * gb1
            
            if j % print_period == 0:
                # calculate just for LL
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                # print "pY:", pY
                ll = cost(pY, Ytest_ind)
                LL_batch.append(ll)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll))
                
                err = error_rate(pY, Ytest)
                CR_batch.append(err)
                print("Error rate:", err)
    
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final error rate:", error_rate(pY, Ytest))
    
    # 2. RMSprop
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)
    LL_rms = [ ]
    CR_rms = [ ]
    lr0 = 0.001  # if you set this too high you'll get NaN!
    cache_W2 = 1
    cache_b2 = 1
    cache_W1 = 1
    cache_b1 = 1
    decay_rate = 0.999
    eps = 1e-10
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[ j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Ytrain_ind[ j * batch_sz:(j * batch_sz + batch_sz), ]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
            # print "first batch cost:", cost(pYbatch, Ybatch)
            
            # gradients
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg * W2
            gb2 = derivative_b2(Ybatch, pYbatch) + reg * b2
            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg * W1
            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1
            
            # caches
            cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2 * gW2
            cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2 * gb2
            cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1 * gW1
            cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1
            
            # updates
            W2 -= lr0 * gW2 / (np.sqrt(cache_W2) + eps)
            b2 -= lr0 * gb2 / (np.sqrt(cache_b2) + eps)
            W1 -= lr0 * gW1 / (np.sqrt(cache_W1) + eps)
            b1 -= lr0 * gb1 / (np.sqrt(cache_b1) + eps)
            
            if j % print_period == 0:
                # calculate just for LL
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                # print "pY:", pY
                ll = cost(pY, Ytest_ind)
                LL_rms.append(ll)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll))
                
                err = error_rate(pY, Ytest)
                CR_rms.append(err)
                print("Error rate:", err)
    
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final error rate:", error_rate(pY, Ytest))
    
    plt.plot(LL_batch, label='const')
    plt.plot(LL_rms, label='rms')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
