# cnn.py is part of SmartFlow
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
"""Creates simple CNN's."""

from itertools import count

import numpy as np

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (concatenate,  # noqa: F401
                                     Conv2D,
                                     Dense,
                                     Dropout,
                                     Flatten,
                                     Input,
                                     MaxPool2D,
                                     Reshape)

from smartflow import backend as S, utils, smartflow_pre  # noqa: F401
from smartflow.archs.resnet import conv2d_bn


def dummy_model():
  """Creates a light yet somehow representative  dummy model."""
  inshape, outshape = utils.inoutshapes()

  input_layer = Input(inshape)                            # 54x54x32
  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)  # 54x54x32
    x = Conv2D(32, (3, 3), activation=S.activation(), padding="same")(x)            # 54x54x32
  else:
    x = Conv2D(48, (3, 3), activation=S.activation(), padding="same")(input_layer)  # 54x54x32
  x = Conv2D(48, (3, 3), activation=S.activation(), padding="same")(x)              # 54x54x32
  x = Conv2D(64, (3, 3), activation=S.activation(), strides=(2, 2))(x)              # 26x26x32
  x = Conv2D(128, (3, 3), activation=S.activation(), strides=(2, 2))(x)             # 12x12x64
  x = Conv2D(256, (3, 3), activation=S.activation(), strides=(2, 2))(x)             # 5x5x128
  x = Conv2D(388, (3, 3), activation=S.activation(), strides=(2, 2))(x)             # 2x2x128
  x = Conv2D(896, (3, 3), activation=S.activation())(x)  # 2x2x128
  x = Flatten()(x)                                       # 1024
  # x = Dense(3072, activation=S.activation())(x)          # 3072
  x = Dense(5120, activation=S.activation())(x)          # 4096
  x = Dense(np.prod(outshape))(x)                        # 7500
  return Model(input_layer, x)


def dummy_model_bn():
  """Creates a light yet somehow representative  dummy model."""
  inshape, outshape = utils.inoutshapes()

  input_layer = Input(inshape)                            # 54x54x32
  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)  # 54x54x32
    x = conv2d_bn(x, 48, (3, 3))
  else:
    x = conv2d_bn(input_layer, 48, (3, 3))

  x = conv2d_bn(x, 48, (3, 3))
  x = conv2d_bn(x, 64, (3, 3), strides=(2, 2), padding="valid")
  x = conv2d_bn(x, 128, (3, 3), strides=(2, 2), padding="valid")
  x = conv2d_bn(x, 256, (3, 3), strides=(2, 2), padding="valid")
  x = conv2d_bn(x, 388, (3, 3), strides=(2, 2), padding="valid")
  x = conv2d_bn(x, 896, (3, 3), strides=(2, 2))
  x = Flatten()(x)                                        # 1024
  # x = Dense(3072, activation=S.activation())(x)           # 3072
  x = Dense(5120, activation=S.activation())(x)           # 4096
  x = Dropout(0.2)(x)
  x = Dense(np.prod(outshape))(x)                         # 7500
  return Model(input_layer, x)


def cnn_50x50():
  inshape, outshape = utils.inoutshapes()

  input_layer = Input(inshape)                                    # 0  54x54x6
  x = Conv2D(
    48, (3, 3), padding="same", activation="relu"
  )(input_layer)                                                  # 1  54x54x48
  x = Conv2D(80, (3, 3), padding="same", activation="relu")(x)    # 2  54x54x80
  x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)   # 3  54x54x128
  x = Conv2D(288, (3, 3), strides=(2, 2), activation="relu")(x)   # 4  26x26x288
  x = Conv2D(288, (3, 3), padding="same", activation="relu")(x)   # 5  26x26x256
  x = Conv2D(784, (3, 3), strides=(2, 2), activation="relu")(x)   # 6  12x12x784
  x = Conv2D(784, (3, 3), padding="same", activation="relu")(x)   # 7  12x12x784
  x = Conv2D(1280, (3, 3), strides=(2, 2), activation="relu")(x)  # 6  5x5x1280
  x = Conv2D(2048, (3, 3), activation="relu")(x)                  # 9  3x3x1536
  x = Conv2D(3328, (3, 3), activation="relu")(x)                  # 10 1x1x3072
  x = Flatten()(x)                                                # 12 3072
  x = Dense(4096, activation="relu")(x)                           # 13 5120
  x = Dense(np.prod(outshape))(x)                                 # 14 7500 (50x50x3)

  model = Model(input_layer, x)
  return model


def cnn_80x80():
  inshape, outshape = utils.inoutshapes()

  input_layer = Input(inshape)                                    # 0  84x84x6
  # x = smartflow_pre.normalization_layer()(input_layer)
  x = Conv2D(
    64, (3, 3), padding="same", activation="relu"
  )(input_layer)                                                  # 1  84x84x48
  x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)    # 2  84x84x80
  x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)   # 3  84x84x128
  x = Conv2D(380, (3, 3), strides=(2, 2), activation="relu")(x)   # 4  41x41x288
  x = Conv2D(380, (3, 3), padding="same", activation="relu")(x)   # 5  41x41x256
  x = Conv2D(784, (3, 3), strides=(2, 2), activation="relu")(x)   # 6  20x20x784
  x = Conv2D(784, (3, 3), padding="same", activation="relu")(x)   # 7  20x20x784
  x = Conv2D(1280, (3, 3), strides=(2, 2), activation="relu")(x)  # 6  9x9x784
  x = Conv2D(1472, (3, 3), activation="relu")(x)                  # 6  7x7x1280
  x = Conv2D(1728, (3, 3), activation="relu")(x)                  # 6  5x5x1280
  x = Conv2D(2048, (3, 3), activation="relu")(x)                  # 9  3x3x1536
  x = Conv2D(2588, (3, 3), activation="relu")(x)                  # 10 1x1x3072
  x = Flatten()(x)                                                # 12 3072
  x = Dense(4096, activation="relu")(x)                           # 13 5120
  x = Dense(np.prod(outshape))(x)                                 # 14 7500 (50x50x3)

  model = Model(input_layer, x)
  return model


def cnn_80x80_bn():
  inshape, outshape = utils.inoutshapes()
  conv2d_bn.counter = count(0)

  input_layer = Input(inshape)                     # 0  84x84x6

  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)
    x = conv2d_bn(x, 64, (3, 3))                   # 1  84x84x48
  else:
    x = conv2d_bn(input_layer, 64, (3, 3))         # 1  84x84x48
  x = conv2d_bn(x, 64, (3, 3))                     # 2  84x84x80
  x = conv2d_bn(x, 128, (3, 3))                    # 3  84x84x128
  x = conv2d_bn(x, 380, (3, 3), strides=(2, 2))    # 4  41x41x288
  x = conv2d_bn(x, 380, (3, 3))                    # 5  41x41x256
  x = conv2d_bn(x, 784, (3, 3), strides=(2, 2))    # 6  20x20x784
  x = conv2d_bn(x, 784, (3, 3))                    # 7  20x20x784
  x = conv2d_bn(x, 1280, (3, 3), strides=(2, 2))   # 6  9x9x784
  x = conv2d_bn(x, 1472, (3, 3), padding="valid")  # 6  7x7x1280
  x = conv2d_bn(x, 1728, (3, 3), padding="valid")  # 6  5x5x1280
  x = conv2d_bn(x, 2048, (3, 3), padding="valid")  # 9  3x3x1536
  x = conv2d_bn(x, 2588, (3, 3), padding="valid")  # 10 1x1x3072
  x = Flatten()(x)                                 # 12 3072
  x = Dense(4096, activation="relu")(x)            # 13 5120
  x = Dense(np.prod(outshape))(x)                  # 14 7500 (50x50x3)

  model = Model(input_layer, x)
  return model


def cnn_90x90_bn():
  inshape, outshape = utils.inoutshapes()
  conv2d_bn.counter = count(0)

  input_layer = Input(inshape)                                     # 94x94x6

  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)
    x = conv2d_bn(x, 64, (3, 3))                                    # 94x94x64
  else:
    x = conv2d_bn(input_layer, 64, (3, 3))                          # 94x94x64
  x = conv2d_bn(x, 64, (3, 3))                                      # 94x94x64
  x = conv2d_bn(x, 96, (3, 3))                                      # 94x94x80

  x1 = conv2d_bn(x, 224, (3, 3), strides=(2, 2))                    # 47x47x256
  x = conv2d_bn(x1, 224, (3, 3), residual=x1)                       # 47x47x256

  x1 = conv2d_bn(x, 702, (3, 3), strides=(2, 2), padding="valid")   # 23x23x896
  x = conv2d_bn(x1, 702, (3, 3), residual=x1)                       # 23x23x896

  x1 = conv2d_bn(x, 1476, (3, 3), strides=(2, 2), padding="valid")  # 11x112048
  x = conv2d_bn(x1, 1476, (3, 3), residual=x1)                      # 11x112048

  x = conv2d_bn(x, 1024, (1, 1))                                    # 12x12x1536
  x1 = conv2d_bn(x, 2048, (3, 3), strides=(2, 2), padding="valid")  # 5x5x2560
  x = conv2d_bn(x1, 2048, (3, 3), residual=x1)                      # 5x5x2560

  x = conv2d_bn(x, 1664, (1, 1))                                    # 12x12x1536
  x = conv2d_bn(x, 2560, (3, 3), padding="valid")                   # 3x3x1768
  x = conv2d_bn(x, 2048, (1, 1))                                    # 5x5x2560
  x = conv2d_bn(x, 3072, (3, 3), padding="valid")                   # 1x1x4096

  x = Flatten()(x)                                                  # 4096
  x = Dense(5120, activation="relu")(x)                             # 5120
  x = Dense(np.prod(outshape))(x)                                   # 24300 (90x90x3)

  model = Model(input_layer, x)
  return model


def cnn_80x80_legacy():
  inshape, outshape = utils.inoutshapes()

  model = keras.Sequential([
    Conv2D(64, (3, 3), padding="same", activation="relu",
           input_shape=inshape),                                  # 84x84x64
    MaxPool2D(),                                                  # 42x24x64
    Conv2D(64, (3, 3), padding="same", activation="relu"),        # 42x42x64
    MaxPool2D(),                                                  # 21x21x64
    Conv2D(128, (3, 3), padding="same", activation="relu"),       # 21x21x128
    MaxPool2D(),                                                  # 10x10x128
    Conv2D(128, (3, 3), padding="same", activation="relu"),       # 10x10x128
    MaxPool2D(),                                                  # 5x5x128
    Conv2D(256, (3, 3), padding="same", activation="relu"),       # 5x5x256
    MaxPool2D(),                                                  # 2x2x256
    Flatten(),                                                    # 1024
    Dense(2048, activation="relu"),                               # 2048
    Dense(4096, activation="relu"),                               # 4096
    Dense(np.prod(outshape))                                      # 6400
  ], name="cnn")
  return model


def cnn_90x90_legacy():
  """~37M parameters"""
  inshape, outshape = utils.inoutshapes()

  model = keras.Sequential([
    # smartflow_pre.normalization_layer(),                                    # 8100
    Conv2D(64, (3, 3), padding="same", activation="relu",
           input_shape=inshape),                                  # 90x90x64
    MaxPool2D(),                                                  # 45X45X64
    Conv2D(64, (3, 3), padding="same", activation="relu"),        # 45X45x64
    MaxPool2D(),                                                  # 22x22x64
    Conv2D(128, (3, 3), padding="same", activation="relu"),       # 22x22x128
    MaxPool2D(),                                                  # 11x11x128
    Conv2D(128, (3, 3), padding="same", activation="relu"),       # 11x11x128
    MaxPool2D(),                                                  # 5x5x128
    Conv2D(256, (3, 3), padding="same", activation="relu"),       # 5x5x256
    MaxPool2D(),                                                  # 2x2x256
    Flatten(),                                                    # 1024
    Dense(2048, activation="relu"),                               # 2048
    Dense(4096, activation="relu"),                               # 4096
    Dense(np.prod(outshape))                                      # 8100
  ], name="cnn")
  return model


def cnn_100x100():
  inshape, outshape = utils.inoutshapes()

  same_kwargs = {
    "kernel_size": (3, 3),
    "padding": "same",
    "activation": S.activation()
  }
  valid_kwargs = {
    "kernel_size": (3, 3),
    "padding": "valid",
    "activation": S.activation()
  }
  reducer_kwargs = {
    "kernel_size": (1, 1),
    "padding": "same",
    "activation": S.activation()
  }

  input_layer = Input(inshape)                         # 0  104x104x2c

  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)
    x = Conv2D(64, **same_kwargs)(x)           # 1  104x104x64
  else:
    x = Conv2D(64, **same_kwargs)(input_layer)           # 1  104x104x64
  x = Conv2D(64, **same_kwargs)(x)                     # 2  104x104x64

  x = Conv2D(128, strides=(2, 2), **same_kwargs)(x)    # 3  52x52x128
  x = Conv2D(128, **same_kwargs)(x)                    # 4  52x52x128

  x = Conv2D(380, strides=(2, 2), **same_kwargs)(x)    # 5  26x26x380
  x = Conv2D(380, **same_kwargs)(x)                    # 6  26x26x380

  x = Conv2D(784, strides=(2, 2), **valid_kwargs)(x)   # 7  12x12x784
  x = Conv2D(784, **same_kwargs)(x)                    # 8  12x12x784

  x = Conv2D(1280, strides=(2, 2), **valid_kwargs)(x)  # 9  5x5x1280
  x = Conv2D(1280, **same_kwargs)(x)                   # 10 5x5x1280

  x = Conv2D(1728, **valid_kwargs)(x)                  # 11 3x3x1728
  x = Conv2D(1472, **reducer_kwargs)(x)                # 12 3x3x1472
  x = Conv2D(2588, **valid_kwargs)(x)                  # 13 1x1x2588
  x = Flatten()(x)                                     # 16 3072
  x = Dense(4096, activation="relu")(x)                # 17 4096
  # x = Dense(6144, activation="relu")(x)                # 18 6144
  x = Dense(np.prod(outshape))(x)                      # 19 10000c (100x100xc)

  model = Model(input_layer, x)
  return model
