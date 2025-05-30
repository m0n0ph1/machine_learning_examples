# https://deeplearningcourses.com/c/unsupervised-machine-learning-hidden-markov-models-in-python
# https://udemy.com/unsupervised-machine-learning-hidden-markov-models-in-python
# http://lazyprogrammer.me
# theano scan example: calculate fibonacci
from __future__ import division, print_function

import theano
import theano.tensor as T

# Note: you may need to update your version of future
# sudo pip install -U future

N = T.iscalar('N')

def recurrence(n, fn_1, fn_2):
    return fn_1 + fn_2, fn_1

outputs, updates = theano.scan(
    fn=recurrence,
    sequences=T.arange(N),
    n_steps=N,
    outputs_info=[ 1., 1. ]
)

fibonacci = theano.function(
    inputs=[ N ],
    outputs=outputs,
)

o_val = fibonacci(8)

print("output:", o_val)
