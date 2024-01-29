# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision

from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model

# Note: you may need to update your version of future
# sudo pip install -U future


# helper
def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    I = np.zeros((N, K))
    I[ np.arange(N), Y ] = 1
    return I

# get the data
# https://www.kaggle.com/zalando-research/fashionmnist
data = pd.read_csv('../large_files/fashionmnist/fashion-mnist_train.csv')
data = data.values
np.random.shuffle(data)

X = data[ :, 1: ].reshape(-1, 28, 28, 1) / 255.0
Y = data[ :, 0 ].astype(np.int32)

# get shapes
# N = len(Y)
K = len(set(Y))

# by default Keras wants one-hot encoded labels
# there's another cost function we can use
# where we can just pass in the integer labels directly
# just like Tensorflow / Theano
Y = y2indicator(Y)

# make the CNN
i = Input(shape=(28, 28, 1))
x = Conv2D(filters=32, kernel_size=(3, 3))(i)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=64, kernel_size=(3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dense(units=100)(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=K)(x)
x = Activation('softmax')(x)

model = Model(inputs=i, outputs=x)

# list of losses: https://keras.io/losses/
# list of optimizers: https://keras.io/optimizers/
# list of metrics: https://keras.io/metrics/
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=[ 'accuracy' ]
)

# note: multiple ways to choose a backend
# either theano, tensorflow, or cntk
# https://keras.io/backend/


# gives us back a <keras.callbacks.History object at 0x112e61a90>
r = model.fit(X, Y, validation_split=0.33, epochs=15, batch_size=32)
print("Returned:", r)

# print the available keys
# should see: dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
print(r.history.keys())

# plot some data
plt.plot(r.history[ 'loss' ], label='loss')
plt.plot(r.history[ 'val_loss' ], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history[ 'accuracy' ], label='acc')
plt.plot(r.history[ 'val_accuracy' ], label='val_acc')
plt.legend()
plt.show()
