# smartflow_cnn.py is part of SmartFlow
#
# SmartFlow is free software; you may redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. You should have received a copy of the GNU
# General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.
#
# (C) 2021 Athanasios Mattas
# ======================================================================
"""Creates Residual Networks.

  Reference papers:
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385)
  - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on
     Learning](https://arxiv.org/abs/1602.07261)
  - [Batch normalization: Accelerating deep network training by reducing
     internal covariate shift](https://arxiv.org/abs/1502.03167)
"""

from itertools import count

import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation,  # noqa: F401
                                     add,
                                     BatchNormalization,
                                     concatenate,
                                     Conv2D,
                                     Dense,
                                     Flatten,
                                     Input,
                                     MaxPool2D,
                                     Reshape)

from smartflow import backend as S, smartflow_pre, utils


def residual_layer(x, residual, identity=True, scale=0.3, name=None):
  """Adds the activations of a resudual layer to the activations of the current
  layer.

  There is also an option to scale the residuals as suggested by Szegedy-Ioffe,
  Section 3.3. (Suggested range: 0.1-0.3)
  """
  if x.shape != residual.shape or not identity:
    x_filters = x.shape[S.channel_axis()]
    # Scenarios:
    # ----------------------------
    # valid  /1 : h - 2
    #        /2 : (h - 3) // 2 + 1
    #        /3 : (h - 3) // 3 + 1
    # same   /1 : h
    #        /2 : ceil(h / 2)
    #        /3 : ceil(h / 3)
    # (shape[2] will always be the h or w of the grid.)
    if x.shape[2] == residual.shape[2] - 2:
      residual = conv2d_bn(residual,
                           x_filters,
                           (3, 3),
                           padding="valid")
    elif x.shape[2] == (residual.shape[2] - 3) // 2 + 1:
      residual = conv2d_bn(residual,
                           x_filters,
                           (3, 3),
                           strides=(2, 2),
                           padding="valid")
    elif x.shape[2] == (residual.shape[2] - 3) // 3 + 1:
      residual = conv2d_bn(residual,
                           x_filters,
                           (3, 3),
                           strides=(3, 3),
                           padding="valid")

    elif x.shape[2] == residual.shape[2]:
      residual = conv2d_bn(residual,
                           x_filters,
                           (1, 1))
    elif x.shape[2] == -(-residual.shape[2] // 2):
      residual = conv2d_bn(residual,
                           x_filters,
                           (3, 3),
                           strides=(2, 2))
    elif x.shape[2] == -(-residual.shape[2] // 3):
      residual = conv2d_bn(residual,
                           x_filters,
                           (3, 3),
                           strides=(3, 3))
    else:
      raise ValueError("The current tensor must have the same size with the"
                       " residual tensor or the output size after applying"
                       " a 3x3 convolution at the residual tensor.\n"
                       f"Current tesnor shape: {x.shape[1:]}\n"
                       f"Residual tensor shape: {residual.shape[1:]}")

  if scale is not None:
    residual = tf.math.multiply(residual, scale)

  x = add([x, residual], name=name)
  return x


def conv2d_bn(x,
              filters,
              kernel,
              strides=(1, 1),
              padding="same",
              activation=None,
              residual=None,
              identity=True,
              scale=0.3,
              **kwargs):
  """Implements: conv --> batch-normalization --> + residual --> non-linearity

  This technique addresses 'internal covariate shift', where the distribution
  of the input of a layer is changing during training, while the parameters of
  the previous layer change, resulting in slowing down the training process,
  because lower learning rates are required.
  (See Ioffe and Szegedy, 2015)

  - In case of a residual tensor is passed, it is added at the current output,
    before applying the non-linearity (See He et al., Section 3).
  - The identity argument is considered when residual is not None and determi-
    nes whether to use identity shortcuts or projection shortcuts.
    (See He et al., Section 4, "Identity vs. Projection Shortcuts")
  """
  if not hasattr(conv2d_bn, "counter"):
    conv2d_bn.counter = count()
  conv_number = next(conv2d_bn.counter)
  name = kwargs.pop("name", False)
  if not name:
    conv_name = f"Conv_{conv_number}"
    bn_name = f"BN_{conv_number}"
    residual_name = f"Residual_{conv_number}"
    name = f"Conv_bn{conv_number}"
  else:
    conv_name = name + "_conv"
    bn_name = name + "_bn"
    residual_name = name + "_residual"

  if activation is None:
    activation = S.activation()

  x = Conv2D(filters,
             kernel_size=kernel,
             strides=strides,
             padding=padding,
             use_bias=False,
             kernel_initializer=S.kernel_initializer(),
             name=conv_name,
             **kwargs)(x)
  x = BatchNormalization(axis=S.channel_axis(),
                         scale=False,
                         name=bn_name
                         )(x)
  if residual is not None:
    x = residual_layer(x, residual, identity, scale=scale, name=residual_name)
  x = Activation(activation, name=name)(x)
  return x


def resnet50_50x50():
  """Creates the ResNet50 architecture."""
  inshape, outshape = utils.inoutshapes()
  conv2d_bn.counter = count(0)

  input_layer = Input(shape=inshape)                           # 54x54x6
  x = conv2d_bn(input_layer, 64, (7, 7), strides=(2, 2))       # 27x27x64
  # x = MaxPool2D((3, 3), strides=(2, 2))(x)

  for _ in range(3):
    x1 = conv2d_bn(x, 64, (1, 1))
    x = conv2d_bn(x1, 64, (3, 3))
    x = conv2d_bn(x, 256, (1, 1), residual=x1)                 # 27x27x256

  x1 = conv2d_bn(x, 128, (1, 1))
  x = conv2d_bn(x1, 128, (3, 3), strides=(2, 2))
  x = conv2d_bn(x, 512, (1, 1), residual=x1)                   # 14x14x512
  for _ in range(3):
    x1 = conv2d_bn(x, 128, (1, 1))
    x = conv2d_bn(x1, 128, (3, 3))
    x = conv2d_bn(x, 512, (1, 1), residual=x1)                 # 14x14x512

  x1 = conv2d_bn(x, 256, (1, 1))
  x = conv2d_bn(x1, 256, (3, 3), strides=(2, 2))
  x = conv2d_bn(x, 1024, (1, 1), residual=x1)                  # 7x7x1024
  for _ in range(5):
    x1 = conv2d_bn(x, 256, (1, 1))
    x = conv2d_bn(x1, 256, (3, 3))
    x = conv2d_bn(x, 1024, (1, 1), residual=x1)                # 7x7x1024

  x1 = conv2d_bn(x, 512, (1, 1))
  x = conv2d_bn(x1, 512, (3, 3), strides=(2, 2))
  x = conv2d_bn(x, 2048, (1, 1), residual=x1)                  # 4x4x2048
  for _ in range(2):
    x1 = conv2d_bn(x, 512, (1, 1))
    x = conv2d_bn(x1, 512, (3, 3))
    x = conv2d_bn(x, 2048, (1, 1), residual=x1)                # 4x4x2048

  x1 = conv2d_bn(x, 1280, (1, 1))                              # 4x4x1280
  x = conv2d_bn(x1, 2540, (4, 4), padding="valid")             # 1x1x2540
  x = conv2d_bn(x, 2048, (1, 1), residual=x1)                  # 1x1x2048
  x = Flatten()(x)                                             # 2048
  x = Dense(4096, activation=S.activation())(x)                # 5120
  x = Dense(np.prod(outshape))(x)                              # 7500 (50x50x3)

  model = Model(input_layer, x)
  return model


def resnet50_90x90():
  """Creates the ResNet50 architecture."""
  inshape, outshape = utils.inoutshapes()
  conv2d_bn.counter = count(0)

  input_layer = Input(shape=inshape)                           # 94x94x6

  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)
    x = conv2d_bn(x, 64, (7, 7), strides=(2, 2))       # 47x47x64
  else:
    x = conv2d_bn(input_layer, 64, (7, 7), strides=(2, 2))       # 47x47x64
  # x = MaxPool2D((3, 3), strides=(2, 2))(x)

  for _ in range(3):
    x1 = conv2d_bn(x, 64, (1, 1))
    x = conv2d_bn(x1, 64, (3, 3))
    x = conv2d_bn(x, 256, (1, 1), residual=x1)                 # 47x47x256

  x1 = conv2d_bn(x, 128, (1, 1))
  x = conv2d_bn(x1, 128, (3, 3), strides=(2, 2))
  x = conv2d_bn(x, 512, (1, 1), residual=x1)                   # 24x24x512
  for _ in range(3):
    x1 = conv2d_bn(x, 128, (1, 1))
    x = conv2d_bn(x1, 128, (3, 3))
    x = conv2d_bn(x, 512, (1, 1), residual=x1)                 # 24x24x512

  x1 = conv2d_bn(x, 256, (1, 1))
  x = conv2d_bn(x1, 256, (3, 3), strides=(2, 2))
  x = conv2d_bn(x, 1024, (1, 1), residual=x1)                  # 12x12x1024
  for _ in range(5):
    x1 = conv2d_bn(x, 256, (1, 1))
    x = conv2d_bn(x1, 256, (3, 3))
    x = conv2d_bn(x, 1024, (1, 1), residual=x1)                # 12x12x1024

  x1 = conv2d_bn(x, 512, (1, 1))
  x = conv2d_bn(x1, 512, (3, 3), strides=(2, 2))
  x = conv2d_bn(x, 2048, (1, 1), residual=x1)                  # 6x6x2048
  for _ in range(2):
    x1 = conv2d_bn(x, 512, (1, 1))
    x = conv2d_bn(x1, 512, (3, 3))
    x = conv2d_bn(x, 2048, (1, 1), residual=x1)                # 6x6x2048

  x = conv2d_bn(x1, 2536, (3, 3), padding="valid")             # 4x4x2048
  x = conv2d_bn(x, 4092, (4, 4), padding="valid")              # 1x1x3584
  x = Flatten()(x)                                             # 3584
  x = Dense(5120, activation=S.activation())(x)                # 5120
  x = Dense(np.prod(outshape))(x)                              # 24300 (90x90x3)

  model = Model(input_layer, x)
  return model


def simple_cnn():

  inshape, outshape = utils.inoutshapes()
  S.set_activation(S.activation())
  conv2d_bn.counter = count(0)

  input_layer = Input(shape=inshape)                           # 94x94x6

  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)
    x1 = conv2d_bn(x, 48, (7, 7))                              # 94x94x48
    x2 = conv2d_bn(x, 32, (5, 5))                              # 94x94x32
  else:
    x1 = conv2d_bn(input_layer, 48, (7, 7))                    # 94x94x48
    x2 = conv2d_bn(input_layer, 32, (5, 5))                    # 94x94x32
  x = concatenate([x1, x2], axis=S.channel_axis())             # 94x94x80
  x = conv2d_bn(x, 80, (5, 5))                                 # 94x94x80
  x = conv2d_bn(x, 192, (3, 3), strides=(2, 2))                # 47x47x192
  x = conv2d_bn(x, 192, (3, 3))                                # 47x47x192
  x = conv2d_bn(x, 192, (3, 3))                                # 47x47x192
  x = conv2d_bn(x, 312, (3, 3), strides=(2, 2))                # 24x24x312
  x = conv2d_bn(x, 312, (3, 3))                                # 24x24x312
  x = conv2d_bn(x, 312, (3, 3))                                # 24x24x312
  x = conv2d_bn(x, 512, (3, 3), strides=(2, 2))                # 12x12x512
  x = conv2d_bn(x, 620, (3, 3))                                # 12x12x620
  x = conv2d_bn(x, 620, (3, 3))                                # 12x12x620
  x = conv2d_bn(x, 1280, (3, 3), strides=(2, 2), padding="valid")  # 5x5x1280
  x = conv2d_bn(x, 1280, (3, 3))                               # 5x5x1280
  x = conv2d_bn(x, 2048, (3, 3), padding="valid")              # 3x3x2048
  x = conv2d_bn(x, 2048, (3, 3))                               # 3x3x2048
  x = conv2d_bn(x, 3072, (3, 3), padding="valid")              # 1x1x3072
  x = Flatten()(x)                                             # 4608
  x = Dense(9192, activation=S.activation())(x)                # 9192
  x = Dense(np.prod(outshape))(x)                              # 24300 (90x90x3)

  model = Model(input_layer, x)
  return model
