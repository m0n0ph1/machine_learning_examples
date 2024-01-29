from __future__ import division, print_function

from builtins import input

import matplotlib.pyplot as plt
import numpy as np
# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
import requests

from util import get_data

# Note: you may need to update your version of future
# sudo pip install -U future

# make a prediction from our own server!
# in reality this could be coming from any client

X, Y = get_data()
N = len(Y)
while True:
    i = np.random.choice(N)
    r = requests.post("http://localhost:8888/predict", data={'input': X[ i ]})
    print("RESPONSE:")
    print(r.content)
    j = r.json()
    print(j)
    print("target:", Y[ i ])
    
    plt.imshow(X[ i ].reshape(28, 28), cmap='gray')
    plt.title("Target: %d, Prediction: %d" % (Y[ i ], j[ 'prediction' ]))
    plt.show()
    
    response = input("Continue? (Y/n)\n")
    if response in ('n', 'N'):
        break
