# https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow
# https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow
from __future__ import division, print_function

import matplotlib.pyplot as plt
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from keras.models import Model

from util import getCIFAR10

# Note: you may need to update your version of future
# sudo pip install -U future

# get the data
Xtrain, Ytrain, Xtest, Ytest = getCIFAR10()

# get shapes
N, H, W, C = Xtrain.shape
K = len(set(Ytrain))

# make the CNN
i = Input(shape=(H, W, C))
x = Lambda(lambda x: (x - 127.5) / 127.5)(i)
x = Conv2D(filters=32, kernel_size=(3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=64, kernel_size=(3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(filters=128, kernel_size=(3, 3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(units=300)(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
x = Dense(units=K)(x)
x = Activation('softmax')(x)

model = Model(inputs=i, outputs=x)

# list of losses: https://keras.io/losses/
# list of optimizers: https://keras.io/optimizers/
# list of metrics: https://keras.io/metrics/
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=[ 'accuracy' ]
)

# note: multiple ways to choose a backend
# either theano, tensorflow, or cntk
# https://keras.io/backend/


# gives us back a <keras.callbacks.History object at 0x112e61a90>
r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=80, batch_size=32)
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
plt.plot(r.history[ 'acc' ], label='acc')
plt.plot(r.history[ 'val_acc' ], label='val_acc')
plt.legend()
plt.show()
