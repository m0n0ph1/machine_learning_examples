# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision

from __future__ import division, print_function

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing import image

from style_transfer1 import scale_img, VGG16_AvgPool
from style_transfer2 import minimize, style_loss

# Note: you may need to update your version of future
# sudo pip install -U future
# In this script, we will focus on generating an image
# that attempts to match the content of one input image
# and the style of another input image.
#
# We accomplish this by balancing the content loss
# and style loss simultaneously.


# load the content image
def load_img_and_preprocess(path, shape=None):
    img = image.load_img(path, target_size=shape)
    
    # convert image to array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    return x

content_img = load_img_and_preprocess(
    # '../large_files/caltech101/101_ObjectCategories/elephant/image_0002.jpg',
    # 'batman.jpg',
    'content/sydney.jpg',
    # (225, 300),
)

# resize the style image
# since we don't care too much about warping it
h, w = content_img.shape[ 1:3 ]
style_img = load_img_and_preprocess(
    # 'styles/starrynight.jpg',
    # 'styles/flowercarrier.jpg',
    # 'styles/monalisa.jpg',
    'styles/lesdemoisellesdavignon.jpg',
    (h, w)
)

# we'll use this throughout the rest of the script
batch_shape = content_img.shape
shape = content_img.shape[ 1: ]

# we want to make only 1 VGG here
# as you'll see later, the final model needs
# to have a common input
vgg = VGG16_AvgPool(shape)

# create the content model
# we only want 1 output
# remember you can call vgg.summary() to see a list of layers
# 1,2,4,5,7-9,11-13,15-17
content_model = Model(vgg.input, vgg.layers[ 13 ].get_output_at(0))
content_target = K.variable(content_model.predict(content_img))

# create the style model
# we want multiple outputs
# we will take the same approach as in style_transfer2.py
symbolic_conv_outputs = [
    layer.get_output_at(1) for layer in vgg.layers \
    if layer.name.endswith('conv1')
]

# make a big model that outputs multiple layers' outputs
style_model = Model(vgg.input, symbolic_conv_outputs)

# calculate the targets that are output at each layer
style_layers_outputs = [ K.variable(y) for y in style_model.predict(style_img) ]

# we will assume the weight of the content loss is 1
# and only weight the style losses
style_weights = [ 0.2, 0.4, 0.3, 0.5, 0.2 ]

# create the total loss which is the sum of content + style loss
loss = K.mean(K.square(content_model.output - content_target))

for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
    # gram_matrix() expects a (H, W, C) as input
    loss += w * style_loss(symbolic[ 0 ], actual[ 0 ])

# once again, create the gradients and loss + grads function
# note: it doesn't matter which model's input you use
# they are both pointing to the same keras Input layer in memory
grads = K.gradients(loss, vgg.input)

# just like theano.function
get_loss_and_grads = K.function(
    inputs=[ vgg.input ],
    outputs=[ loss ] + grads
)

def get_loss_and_grads_wrapper(x_vec):
    l, g = get_loss_and_grads([ x_vec.reshape(*batch_shape) ])
    return l.astype(np.float64), g.flatten().astype(np.float64)

final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
plt.imshow(scale_img(final_img))
plt.show()
