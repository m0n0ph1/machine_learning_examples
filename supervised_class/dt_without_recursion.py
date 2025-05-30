# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# Decision Tree for continuous-vector input, binary output
from __future__ import division, print_function

from builtins import range
from datetime import datetime

import numpy as np

from util import get_data

# Note: you may need to update your version of future
# sudo pip install -U future


def entropy(y):
    # assume y is binary - 0 or 1
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    # return -p0*np.log2(p0) - p1*np.log2(p1)
    return 1 - p0 * p0 - p1 * p1

class DecisionTree:
    def __init__(self, depth=0, max_depth=None):
        # print 'depth:', depth
        # self.depth = depth
        self.max_depth = max_depth
        self.root = {}  # is a tree node
        # each node will have the attributes (k-v pairs):
        # - col
        # - split
        # - left
        # - right
        # - prediction
    
    def fit(self, X, Y):
        
        current_node = self.root
        depth = 0
        queue = [ ]
        # origX = X
        # origY = Y
        while True:
            
            if len(Y) == 1 or len(set(Y)) == 1:
                # base case, only 1 sample
                # another base case
                # this node only receives examples from 1 class
                # we can't make a split
                # self.col = None
                # self.split = None
                # self.left = None
                # self.right = None
                # self.prediction = Y[0]
                current_node[ 'col' ] = None
                current_node[ 'split' ] = None
                current_node[ 'left' ] = None
                current_node[ 'right' ] = None
                current_node[ 'prediction' ] = Y[ 0 ]
            
            else:
                D = X.shape[ 1 ]
                cols = range(D)
                
                max_ig = 0
                best_col = None
                best_split = None
                for col in cols:
                    ig, split = self.find_split(X, Y, col)
                    # print "ig:", ig
                    if ig > max_ig:
                        max_ig = ig
                        best_col = col
                        best_split = split
                
                if max_ig == 0:
                    # nothing we can do
                    # no further splits
                    # self.col = None
                    # self.split = None
                    # self.left = None
                    # self.right = None
                    # self.prediction = np.round(Y.mean())
                    current_node[ 'col' ] = None
                    current_node[ 'split' ] = None
                    current_node[ 'left' ] = None
                    current_node[ 'right' ] = None
                    current_node[ 'prediction' ] = np.round(Y.mean())
                else:
                    # self.col = best_col
                    # self.split = best_split
                    current_node[ 'col' ] = best_col
                    current_node[ 'split' ] = best_split
                    
                    # if self.depth == self.max_depth:
                    if depth == self.max_depth:
                        # self.left = None
                        # self.right = None
                        # self.prediction = [
                        #     np.round(Y[X[:,best_col] < self.split].mean()),
                        #     np.round(Y[X[:,best_col] >= self.split].mean()),
                        # ]
                        current_node[ 'left' ] = None
                        current_node[ 'right' ] = None
                        current_node[ 'prediction' ] = [
                            np.round(Y[ X[ :, best_col ] < self.split ].mean()),
                            np.round(Y[ X[ :, best_col ] >= self.split ].mean()),
                        ]
                    else:
                        # print "best split:", best_split
                        left_idx = (X[ :, best_col ] < best_split)
                        # print "left_idx.shape:", left_idx.shape, "len(X):", len(X)
                        # TODO: bad but I can't figure out a better way atm
                        Xleft = X[ left_idx ]
                        Yleft = Y[ left_idx ]
                        # self.left = TreeNode(self.depth + 1, self.max_depth)
                        # self.left.fit(Xleft, Yleft)
                        new_node = {}
                        current_node[ 'left' ] = new_node
                        left_data = {
                            'node': new_node,
                            'X': Xleft,
                            'Y': Yleft,
                        }
                        queue.insert(0, left_data)
                        
                        right_idx = (X[ :, best_col ] >= best_split)
                        Xright = X[ right_idx ]
                        Yright = Y[ right_idx ]
                        # self.right = TreeNode(self.depth + 1, self.max_depth)
                        # self.right.fit(Xright, Yright)
                        new_node = {}
                        current_node[ 'right' ] = new_node
                        right_data = {
                            'node': new_node,
                            'X': Xright,
                            'Y': Yright,
                        }
                        queue.insert(0, right_data)
            
            # setup for the next iteration of the loop
            # idea is, queue stores list of work to be done
            if len(queue) == 0:
                break
            
            next_data = queue.pop()
            current_node = next_data[ 'node' ]
            X = next_data[ 'X' ]
            Y = next_data[ 'Y' ]
    
    def find_split(self, X, Y, col):
        # print "finding split for col:", col
        x_values = X[ :, col ]
        sort_idx = np.argsort(x_values)
        x_values = x_values[ sort_idx ]
        y_values = Y[ sort_idx ]
        
        # Note: optimal split is the midpoint between 2 points
        # Note: optimal split is only on the boundaries between 2 classes
        
        # if boundaries[i] is true
        # then y_values[i] != y_values[i+1]
        # nonzero() gives us indices where arg is true
        # but for some reason it returns a tuple of size 1
        boundaries = np.nonzero(y_values[ :-1 ] != y_values[ 1: ])[ 0 ]
        best_split = None
        max_ig = 0
        last_ig = 0
        for b in boundaries:
            split = (x_values[ b ] + x_values[ b + 1 ]) / 2
            ig = self.information_gain(x_values, y_values, split)
            if ig < last_ig:
                break
            last_ig = ig
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split
    
    def information_gain(self, x, y, split):
        # assume classes are 0 and 1
        # print "split:", split
        y0 = y[ x < split ]
        y1 = y[ x >= split ]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0  # float(len(y1)) / N
        # print "entropy(y):", entropy(y)
        # print "p0:", p0
        # print "entropy(y0):", entropy(y0)
        # print "p1:", p1
        # print "entropy(y1):", entropy(y1)
        return entropy(y) - p0 * entropy(y0) - p1 * entropy(y1)
    
    def predict_one(self, x):
        # use "is not None" because 0 means False
        # if self.col is not None and self.split is not None:
        #     feature = x[self.col]
        #     if feature < self.split:
        #         if self.left:
        #             p = self.left.predict_one(x)
        #         else:
        #             p = self.prediction[0]
        #     else:
        #         if self.right:
        #             p = self.right.predict_one(x)
        #         else:
        #             p = self.prediction[1]
        # else:
        #     # corresponds to having only 1 prediction
        #     p = self.prediction
        p = None
        current_node = self.root
        while True:
            if current_node[ 'col' ] is not None and current_node[ 'split' ] is not None:
                feature = x[ current_node[ 'col' ] ]
                if feature < current_node[ 'split' ]:
                    if current_node[ 'left' ]:
                        current_node = current_node[ 'left' ]
                    else:
                        p = current_node[ 'prediction' ][ 0 ]
                        break
                else:
                    if current_node[ 'right' ]:
                        current_node = current_node[ 'right' ]
                    else:
                        p = current_node[ 'prediction' ][ 1 ]
                        break
            else:
                # corresponds to having only 1 prediction
                p = current_node[ 'prediction' ]
                break
        return p
    
    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[ i ] = self.predict_one(X[ i ])
        return P
    
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

if __name__ == '__main__':
    X, Y = get_data()
    
    # try donut and xor
    # from sklearn.utils import shuffle
    # X, Y = get_xor()
    # # X, Y = get_donut()
    # X, Y = shuffle(X, Y)
    
    # only take 0s and 1s since we're doing binary classification
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[ idx ]
    Y = Y[ idx ]
    
    # split the data
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[ :Ntrain ], Y[ :Ntrain ]
    Xtest, Ytest = X[ Ntrain: ], Y[ Ntrain: ]
    
    model = DecisionTree()
    # model = DecisionTree(max_depth=7)
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))
    
    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0))
    
    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0))
    
    # test SKLearn
    from sklearn.tree import DecisionTreeClassifier
    
    model = DecisionTreeClassifier()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("SK: Training time:", (datetime.now() - t0))
    
    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("SK: Time to compute train accuracy:", (datetime.now() - t0))
    
    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("SK: Time to compute test accuracy:", (datetime.now() - t0))
