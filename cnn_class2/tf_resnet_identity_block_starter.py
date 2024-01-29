# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
from __future__ import division, print_function

import numpy as np

import tensorflow as tf

# Note: you may need to update your version of future
# sudo pip install -U future


class IdentityBlock:
    def __init__(self):
        # TODO
        pass
    
    def predict(self, X):
        # TODO
        pass

if __name__ == '__main__':
    identity_block = IdentityBlock()
    
    # make a fake image
    X = np.random.random((1, 224, 224, 256))
    
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        identity_block.set_session(session)
        session.run(init)
        
        output = identity_block.predict(X)
        print("output.shape:", output.shape)
