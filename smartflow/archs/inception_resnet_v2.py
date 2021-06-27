# inception_resnet.py is part of SmartFlow
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
"""Creates the Inception-ResNet-v2 architecture, adjusting it to the SmartFlow
  input.

  Reference paper:
  - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on
     Learning](https://arxiv.org/abs/1602.07261)
"""

from itertools import count

import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation,  # noqa: F401
                                     Add,
                                     BatchNormalization,
                                     GlobalAveragePooling2D,
                                     concatenate,
                                     Conv2D,
                                     Dropout,
                                     Dense,
                                     Flatten,
                                     Input,
                                     MaxPool2D,
                                     Reshape,
                                     Softmax)

from smartflow import backend as S, smartflow_pre, utils
from smartflow.archs.resnet import conv2d_bn


def stem(x):                                                                 # 94x94x6
  x = conv2d_bn(x, 32, (3, 3), name="Stem_1")                                # 94x94x32
  x = conv2d_bn(x, 32, (3, 3), name="Stem_2")                                # 94x94x32
  x = conv2d_bn(x, 64, (3, 3), name="Stem_3")                                # 94x94x64

  x1 = conv2d_bn(x, 96, (3, 3), strides=(2, 2), name="Stem_4a")              # 47x47x96
  x2 = MaxPool2D((2, 2), strides=(2, 2), name="Stem_4b")(x)                  # 47x47x64
  x = concatenate([x1, x2], axis=S.channel_axis(), name="Stem_4")            # 47x47x160

  x1 = conv2d_bn(x, 64, (1, 1), name="Stem_5a1")                             # 47x47x64
  x1 = conv2d_bn(x1, 64, (7, 1), name="Stem_5a2")                            # 47x47x64
  x1 = conv2d_bn(x1, 64, (1, 7), name="Stem_5a3")                            # 47x47x64
  x1 = conv2d_bn(x1, 96, (3, 3), name="Stem_5a4")                            # 47x47x96
  x2 = conv2d_bn(x, 64, (1, 1), name="Stem_5b1")                             # 47x47x64
  x2 = conv2d_bn(x2, 96, (3, 3), name="Stem_5b2")                            # 47x47x96
  x = concatenate([x1, x2], axis=S.channel_axis(), name="Stem_5")            # 47x47x192

  x1 = conv2d_bn(x, 192, (3, 3), strides=(2, 2), name="Stem_6a")             # 24x24x192
  x2 = MaxPool2D((3, 3), strides=(2, 2), padding="same", name="Stem_6b")(x)  # 24x24x192
  x = concatenate([x1, x2], axis=S.channel_axis(), name="Stem")              # 24x24x384
  return x


def inception_resnet_a(x, scale=0.3, name=None):                             # 24x24x384
  a_block_number = next(inception_resnet_a.counter)
  if name is None:
    name_x1 = f"A{a_block_number}_3x3_3x3_"
    name_x2 = f"A{a_block_number}_3x3_"
    name_x3 = f"A{a_block_number}_1x1_"
    name_residual = f"A{a_block_number}_residual_"
    name = f"InceptionResNet_A{a_block_number}"
  else:
    name_x1 = name + "_3x3_3x3_"
    name_x2 = name + "_3x3_"
    name_x3 = name + "_1x1_"
    name_residual = name + "_residual_"

  x1 = conv2d_bn(x, 32, (1, 1), name=name_x1 + 'a')                          # 24x24x32
  x1 = conv2d_bn(x1, 48, (3, 3), name=name_x1 + 'b')                         # 24x24x48
  x1 = conv2d_bn(x1, 64, (3, 3), name=name_x1 + 'c')                         # 24x24x64

  x2 = conv2d_bn(x, 32, (1, 1), name=name_x2 + 'a')                          # 24x24x32
  x2 = conv2d_bn(x2, 32, (3, 3), name=name_x2 + 'b')                         # 24x24x32

  x3 = conv2d_bn(x, 32, (1, 1), name=name_x3)                                # 24x24x32

  residual = concatenate([x1, x2, x3],
                         axis=S.channel_axis(),
                         name=name_residual + 'a')                           # 24x24x96
  # NOTE: Doesn't this look like a representational bottleneck? (384 -> 96)
  residual = Conv2D(384, (1, 1), name=name_residual + 'b')(residual)         # 24x24x384
  residual = tf.math.multiply(residual, scale, name=name_residual + 'c')     # 24x24x384

  x = Add(name=name_residual + 'd')([residual, x])                           # 24x24x384
  x = Activation(S.activation(), name=name)(x)                               # 24x24x384
  return x


def reduction_a(x, name=None):                                               # 24x24x384
  if name is None:
    name_x1 = "Red_A_3x3_3x3_"
    name_x2 = "Red_A_3x3_"
    name_x3 = "Red_A_MP_"
    name = "Reduction_A"
  else:
    name_x1 = name + "_3x3_3x3_"
    name_x2 = name + "_3x3_"
    name_x3 = name + "_MP_"

  x1 = conv2d_bn(x, 256, (1, 1), name=name_x1 + 'a')                         # 24x24x256
  x1 = conv2d_bn(x1, 256, (3, 3), name=name_x1 + 'b')                        # 24x24x256
  x1 = conv2d_bn(x1, 384, (3, 3), strides=(2, 2), name=name_x1 + 'c')        # 12x12x384

  x2 = conv2d_bn(x, 384, (3, 3), strides=(2, 2), name=name_x2)               # 12x12x384

  x3 = MaxPool2D((3, 3), strides=(2, 2), padding="same", name=name_x3)(x)    # 12x12x384

  x = concatenate([x1, x2, x3], axis=S.channel_axis(), name=name)            # 12x12x1152
  return x


def inception_resnet_b(x, scale=0.3, name=None):                             # 12x12x1152
  b_block_number = next(inception_resnet_b.counter)
  if name is None:
    name_x1 = f"B{b_block_number}_7x1_1x7_"
    name_x2 = f"B{b_block_number}_1x1_"
    name_residual = f"B{b_block_number}_residual_"
    name = f"InceptionResNet_B{b_block_number}"
  else:
    name_x1 = name + "_7x1_1x7_"
    name_x2 = name + "_1x1_"
    name_residual = name + "_residual_"

  x1 = conv2d_bn(x, 128, (1, 1), name=name_x1 + 'a')                         # 12x12x128
  x1 = conv2d_bn(x1, 160, (7, 1), name=name_x1 + 'b')                        # 12x12x160
  x1 = conv2d_bn(x1, 192, (1, 7), name=name_x1 + 'c')                        # 12x12x192

  x2 = conv2d_bn(x, 192, (1, 1), name=name_x2)                               # 12x12x192

  residual = concatenate([x1, x2],
                         axis=S.channel_axis(),
                         name=name_residual + 'a')                           # 12x12x384
  residual = Conv2D(1152, (1, 1), name=name_residual + 'b')(residual)        # 12x12x1152
  residual = tf.math.multiply(residual, scale, name=name_residual + 'c')     # 12x12x1152

  x = Add(name=name_residual + 'd')([residual, x])                           # 12x12x1152
  x = Activation(S.activation(), name=name)(x)                               # 12x12x1152
  return x


def reduction_b(x, name=None):                                               # 12x12x1152
  if name is None:
    name_x1 = "Red_B_3x3_3x3_"
    name_x2 = "Red_B_3x3_256"
    name_x3 = "Red_B_3x3_384"
    name_x4 = "Red_B_MP_"
    name = "Reduction_B"
  else:
    name_x1 = name + "_3x3_3x3_"
    name_x2 = name + "_3x3_256"
    name_x3 = name + "_3x3_384"
    name_x4 = name + "_MP_"

  x1 = conv2d_bn(x, 256, (1, 1), name=name_x1 + 'a')                         # 12x12x256
  x1 = conv2d_bn(x1, 256, (3, 3), name=name_x1 + 'b')                        # 12x12x256
  x1 = conv2d_bn(x1, 256, (3, 3), strides=(2, 2), padding="valid",
                 name=name_x1 + 'c')                                         # 5x5x256

  x2 = conv2d_bn(x, 256, (1, 1), name=name_x2 + 'a')                         # 12x12x256
  x2 = conv2d_bn(x2, 256, (3, 3), strides=(2, 2), padding="valid",
                 name=name_x2 + 'b')                                         # 5x5x256

  x3 = conv2d_bn(x, 256, (1, 1), name=name_x3 + 'a')                         # 12x12x256
  x3 = conv2d_bn(x3, 384, (3, 3), strides=(2, 2), padding="valid",
                 name=name_x3 + 'b')                                         # 5x5x384

  x4 = MaxPool2D((3, 3), strides=(2, 2), name=name_x4)(x)                    # 5x5x1152

  x = concatenate([x1, x2, x3, x4], axis=S.channel_axis(), name=name)        # 5x5x2048
  return x


def inception_resnet_c(x, scale=0.3, name=None):                             # 5x5x2048
  c_block_number = next(inception_resnet_c.counter)
  if name is None:
    name_x1 = f"C{c_block_number}_3x1_1x3_"
    name_x2 = f"C{c_block_number}_1x1_"
    name_residual = f"C{c_block_number}_residual_"
    name = f"InceptionResNet_C{c_block_number}"
  else:
    name_x1 = name + "_3x1_1x3_"
    name_x2 = name + "_1x1_"
    name_residual = name + "_residual_"

  x1 = conv2d_bn(x, 192, (1, 1), name=name_x1 + 'a')                         # 5x5x192
  x1 = conv2d_bn(x1, 224, (3, 1), name=name_x1 + 'b')                        # 5x5x224
  x1 = conv2d_bn(x1, 256, (1, 3), name=name_x1 + 'c')                        # 5x5x256

  x2 = conv2d_bn(x, 192, (1, 1), name=name_x2)                               # 5x5x192

  residual = concatenate([x1, x2],
                         axis=S.channel_axis(),
                         name=name_residual + 'a')                           # 5x5x444
  residual = Conv2D(2048, (1, 1), name=name_residual + 'b')(residual)        # 5x5x2048
  residual = tf.math.multiply(residual, scale, name=name_residual + 'c')     # 5x5x2048

  x = Add(name=name_residual + 'd')([residual, x])                           # 5x5x2048
  x = Activation(S.activation(), name=name)(x)                               # 5x5x2048
  return x


def inception_resnet_v2(include_top=True, scale_a=1, scale_b=0.3, scale_c=0.1):
  # Counters as function attributes, used for incremental naming.
  inception_resnet_a.counter = count(0)
  inception_resnet_b.counter = count(0)
  inception_resnet_c.counter = count(0)
  conv2d_bn.counter = count(0)
  inshape, outshape = utils.inoutshapes()

  input_layer = Input(shape=inshape)                                         # 94x94x6

  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)                     # 94x94x6
    x = stem(x)                                                              # 24x24x384
  else:
    x = stem(input_layer)                                                    # 24x24x384

  for _ in range(2):
    x = inception_resnet_a(x, scale=scale_a)                                 # 24x24x384
  x = reduction_a(x)                                                         # 12x12x1152

  for _ in range(2):
    x = inception_resnet_b(x, scale=scale_b)                                 # 12x12x1152
  x = reduction_b(x)                                                         # 5x5x2048

  for _ in range(2):
    x = inception_resnet_c(x, scale=scale_c)                                 # 5x5x2048

  # x = GlobalAveragePooling2D()(x)
  # x = Dropout(0.2)(x)
  # x = Softmax(axis=S.channel_axis())
  x = conv2d_bn(x, 1754, (1, 1), padding="same", name="Head_0")             # 3x3x2560
  x = conv2d_bn(x, 2048, (3, 3), padding="valid", name="Head_1")             # 3x3x2560
  x = conv2d_bn(x, 2560, (3, 3), padding="valid", name="Head_2")             # 1x1x3072
  if include_top:
    x = Flatten(name="Head_3")(x)                                            # 3072
    x = Dense(4096,
              activation=S.activation(),
              kernel_initializer=S.kernel_initializer(),
              name="Head_4"
              )(x)                                                           # 4096
    # x = Dense(5120,
    #           activation=S.activation(),
    #           kernel_initializer=S.kernel_initializer(),
    #           name="Head_5"
    #           )(x)                                                           # 5120
    x = Dense(np.prod(outshape),
              kernel_initializer=S.kernel_initializer(),
              name="Output"                                                  # (90x90x3)
              )(x)                                                           # 24300

  model = Model(input_layer, x, name="Inception-ResNet-v2")
  return model


###############################################################################
# Inception-ResNet-v2-modified


def inception_resnet_a2(x, scale=0.3, name=None):                            # 24x24x384
  a_block_number = next(inception_resnet_a2.counter)
  if name is None:
    name_x1 = f"A{a_block_number}_3x3_3x3_"
    name_x2 = f"A{a_block_number}_3x3_"
    name_x3 = f"A{a_block_number}_1x1_"
    name_residual = f"A{a_block_number}_residual_"
    name_concat = f"A{a_block_number}_concat"
    name_bn = f"A{a_block_number}_bn"
    name = f"InceptionResNet_A{a_block_number}"
  else:
    name_x1 = name + "_3x3_3x3_"
    name_x2 = name + "_3x3_"
    name_x3 = name + "_1x1_"
    name_residual = name + "_residual_"
    name_concat = name + "_concat"
    name_bn = name + "_bn"

  residual = tf.math.multiply(x, scale, name=name_residual + "scale")          # 24x24x384

  x1 = conv2d_bn(x, 128, (1, 1), name=name_x1 + 'a')                           # 24x24x32
  x1 = conv2d_bn(x1, 160, (3, 3), name=name_x1 + 'b')                          # 24x24x48
  x1 = Conv2D(160, (3, 3), padding="same", name=name_x1 + 'c')(x1)             # 24x24x64

  x2 = conv2d_bn(x, 128, (1, 1), name=name_x2 + 'a')                           # 24x24x32
  x2 = Conv2D(160, (3, 3), padding="same", name=name_x2 + 'b')(x2)             # 24x24x32

  x3 = Conv2D(64, (1, 1), padding="same", name=name_x3)(x)                     # 24x24x32

  x = concatenate([x1, x2, x3], axis=S.channel_axis(), name=name_concat)       # 24x24x96

  x = Add(name=name_residual + 'add')([residual, x])                           # 24x24x384
  x = Activation(S.activation(), name=name)(x)                                 # 24x24x384
  x = BatchNormalization(axis=S.channel_axis(), scale=False, name=name_bn)(x)
  return x


def reduction_a2(x, name=None):                                                # 24x24x384
  if name is None:
    name_x1 = "Red_A_3x3_3x3_"
    name_x2 = "Red_A_3x3_"
    name_x3 = "Red_A_MP_"
    name = "Reduction_A"
  else:
    name_x1 = name + "_3x3_3x3_"
    name_x2 = name + "_3x3_"
    name_x3 = name + "_MP_"

  x1 = conv2d_bn(x, 256, (1, 1), name=name_x1 + 'a')                           # 24x24x256
  x1 = conv2d_bn(x1, 256, (3, 3), name=name_x1 + 'b')                          # 24x24x256
  x1 = conv2d_bn(x1, 288, (3, 3), strides=(2, 2), name=name_x1 + 'c')          # 12x12x384

  x2 = conv2d_bn(x, 288, (3, 3), strides=(2, 2), name=name_x2)                 # 12x12x384

  x3 = conv2d_bn(x, 256, (1, 1), name=name_x3 + 'a')                           # 24x24x256
  x3 = MaxPool2D((3, 3), strides=(2, 2), padding="same", name=name_x3 + 'b')(x3)  # 12x12x384

  x = concatenate([x1, x2, x3], axis=S.channel_axis(), name=name)              # 12x12x832
  return x


def inception_resnet_b2(x, scale=0.3, name=None):                      # 12x12x832
  b_block_number = next(inception_resnet_b2.counter)
  if name is None:
    name_x1 = f"B{b_block_number}_7x1_1x7_"
    name_x2 = f"B{b_block_number}_1x1_"
    name_residual = f"B{b_block_number}_residual_"
    name_concat = f"B{b_block_number}_concat"
    name = f"InceptionResNet_B{b_block_number}"
  else:
    name_x1 = name + "_7x1_1x7_"
    name_x2 = name + "_1x1_"
    name_residual = name + "_residual_"
    name_concat = name + "_concat"

  residual = tf.math.multiply(x, scale, name=name_residual + 'scale')  # 12x12x1152

  x1 = conv2d_bn(x, 416, (1, 1), name=name_x1 + 'a')                   # 12x12x128
  x1 = conv2d_bn(x1, 416, (7, 1), name=name_x1 + 'b')                  # 12x12x160
  x1 = Conv2D(448, (1, 7), padding="same", name=name_x1 + 'c')(x1)     # 12x12x192

  x2 = Conv2D(384, (1, 1), padding="same", name=name_x2)(x)            # 12x12x192

  x = concatenate([x1, x2], axis=S.channel_axis(), name=name_concat)   # 12x12x384

  x = Add(name=name_residual + 'add')([residual, x])                   # 12x12x832
  x = Activation(S.activation(), name=name)(x)                         # 12x12x832
  return x


def reduction_b2(x, name=None):                                              # 12x12x832
  if name is None:
    name_x1 = "Red_B_3x3_3x3_"
    name_x2 = "Red_B_3x3_256"
    name_x3 = "Red_B_3x3_384"
    name_x4 = "Red_B_MP_"
    name = "Reduction_B"
  else:
    name_x1 = name + "_3x3_3x3_"
    name_x2 = name + "_3x3_256"
    name_x3 = name + "_3x3_384"
    name_x4 = name + "_MP_"

  x1 = conv2d_bn(x, 384, (1, 1), name=name_x1 + 'a')                         # 12x12x256
  x1 = conv2d_bn(x1, 384, (3, 3), name=name_x1 + 'b')                        # 12x12x256
  x1 = conv2d_bn(x1, 416, (3, 3), strides=(2, 2), padding="valid",
                 name=name_x1 + 'c')                                         # 5x5x256

  x2 = conv2d_bn(x, 416, (1, 1), name=name_x2 + 'a')                         # 12x12x256
  x2 = conv2d_bn(x2, 416, (3, 3), strides=(2, 2), padding="valid",
                 name=name_x2 + 'b')                                         # 5x5x256

  x3 = conv2d_bn(x, 416, (1, 1), name=name_x3 + 'a')                         # 12x12x256
  x3 = conv2d_bn(x3, 444, (3, 3), strides=(2, 2), padding="valid",
                 name=name_x3 + 'b')                                         # 5x5x384

  x4 = conv2d_bn(x, 384, (1, 1), name=name_x4 + 'a')                         # 12x12x256
  x4 = MaxPool2D((3, 3), strides=(2, 2), name=name_x4 + 'b')(x4)                    # 5x5x1152

  x = concatenate([x1, x2, x3, x4], axis=S.channel_axis(), name=name)        # 5x5x1660
  return x


def inception_resnet_c2(x, scale=0.3, name=None):                             # 5x5x1660
  c_block_number = next(inception_resnet_c2.counter)
  if name is None:
    name_x1 = f"C{c_block_number}_3x1_1x3_"
    name_x2 = f"C{c_block_number}_1x1_"
    name_residual = f"C{c_block_number}_residual_"
    name_concat = f"C{c_block_number}_concat_"
    name = f"InceptionResNet_C{c_block_number}"
  else:
    name_x1 = name + "_3x1_1x3_"
    name_x2 = name + "_1x1_"
    name_residual = name + "_residual_"
    name_concat = name + "_concat_"

  residual = tf.math.multiply(x, scale, name=name_residual + 'scale')     # 5x5x2048

  x1 = conv2d_bn(x, 768, (1, 1), name=name_x1 + 'a')                         # 5x5x192
  x1 = conv2d_bn(x1, 832, (3, 1), name=name_x1 + 'b')                        # 5x5x224
  x1 = Conv2D(976, (1, 3), padding="same", name=name_x1 + 'c')(x1)                           # 5x5x256

  x2 = Conv2D(684, (1, 1), padding="same", name=name_x2)(x)                               # 5x5x192

  residual = concatenate([x1, x2], axis=S.channel_axis(), name=name_concat)  # 5x5x444

  x = Add(name=name_residual + 'add')([residual, x])                           # 5x5x2048
  x = Activation(S.activation(), name=name)(x)                               # 5x5x2048
  return x


def inception_resnet_v2_modified(include_top=True,
                                 scale_a=1,
                                 scale_b=0.3,
                                 scale_c=0.3):
  # Counters as function attributes, used for incremental naming.
  inception_resnet_a2.counter = count(0)
  inception_resnet_b2.counter = count(0)
  inception_resnet_c2.counter = count(0)
  conv2d_bn.counter = count(0)
  inshape, outshape = utils.inoutshapes()

  input_layer = Input(shape=inshape)                                         # 94x94x6

  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)                     # 94x94x6
    x = stem(x)                                                              # 24x24x384
  else:
    x = stem(input_layer)                                                    # 24x24x384

  for _ in range(3):
    x = inception_resnet_a2(x, scale=scale_a)                                 # 24x24x384
  x = reduction_a2(x)                                                         # 12x12x832

  for _ in range(2):
    x = inception_resnet_b2(x, scale=scale_b)                                 # 12x12x822
  x = reduction_b2(x)                                                         # 5x5x1660

  for _ in range(2):
    x = inception_resnet_c2(x, scale=scale_c)                                 # 5x5x1660

  x = conv2d_bn(x, 1280, (1, 1), padding="same", name="Head_0")              # 3x3x2560
  x = conv2d_bn(x, 1840, (3, 3), padding="valid", name="Head_1")             # 3x3x2560
  x = conv2d_bn(x, 2560, (3, 3), padding="valid", name="Head_2")             # 1x1x3072
  if include_top:
    x = Flatten(name="Head_3")(x)                                            # 3072
    x = Dense(4096,
              activation=S.activation(),
              kernel_initializer=S.kernel_initializer(),
              name="Head_4"
              )(x)                                                           # 4096
    # x = Dense(5120,
    #           activation=S.activation(),
    #           kernel_initializer=S.kernel_initializer(),
    #           name="Head_5"
    #           )(x)                                                           # 5120
    x = Dense(np.prod(outshape),
              kernel_initializer=S.kernel_initializer(),
              name="Output"                                                  # (90x90x3)
              )(x)                                                           # 24300

  model = Model(input_layer, x, name="Inception-ResNet-v3")
  return model

###############################################################################
# Inception-ResNet-remodified


def inception_resnet_a3(x, scale=0.3, name=None):                            # 24x24x384
  a_block_number = next(inception_resnet_a3.counter)
  if name is None:
    name_x1 = f"A{a_block_number}_3x3_3x3_"
    name_x2 = f"A{a_block_number}_3x3_"
    name_x3 = f"A{a_block_number}_1x1_"
    name_residual = f"A{a_block_number}_residual_"
    name_concat = f"A{a_block_number}_concat"
    name = f"InceptionResNet_A{a_block_number}"
  else:
    name_x1 = name + "_3x3_3x3_"
    name_x2 = name + "_3x3_"
    name_x3 = name + "_1x1_"
    name_residual = name + "_residual_"
    name_concat = name + "_concat"

  residual = tf.math.multiply(x, scale, name=name_residual + "scale")                  # 24x24x384

  x1 = conv2d_bn(x, 160, (1, 1), name=name_x1 + 'a')                         # 24x24x32
  x1 = conv2d_bn(x1, 160, (3, 3), name=name_x1 + 'b')                        # 24x24x48
  x1 = Conv2D(160, (3, 3), padding="same", name=name_x1 + 'c')(x1)          # 24x24x64
  x1 = BatchNormalization(axis=S.channel_axis(), scale=False, name=name_x1 + 'd')(x1)

  x2 = conv2d_bn(x, 160, (1, 1), name=name_x2 + 'a')                         # 24x24x32
  x2 = Conv2D(160, (3, 3), padding="same", name=name_x2 + 'b')(x2)          # 24x24x32
  x2 = BatchNormalization(axis=S.channel_axis(), scale=False, name=name_x2 + 'c')(x2)

  x3 = Conv2D(64, (1, 1), padding="same", name=name_x3 + 'a')(x)              # 24x24x32
  x3 = BatchNormalization(axis=S.channel_axis(), scale=False, name=name_x3 + 'b')(x3)

  x = concatenate([x1, x2, x3], axis=S.channel_axis(), name=name_concat)                                  # 24x24x96

  x = Add(name=name_residual + 'add')([residual, x])                           # 24x24x384
  x = Activation(S.activation(), name=name)(x)                               # 24x24x384
  return x


def reduction_a3(x, name=None):                                               # 24x24x384
  if name is None:
    name_x1 = "Red_A_3x3_3x3_"
    name_x2 = "Red_A_3x3_"
    name_x3 = "Red_A_MP_"
    name = "Reduction_A"
  else:
    name_x1 = name + "_3x3_3x3_"
    name_x2 = name + "_3x3_"
    name_x3 = name + "_MP_"

  x1 = conv2d_bn(x, 288, (1, 1), name=name_x1 + 'a')                         # 24x24x256
  x1 = conv2d_bn(x1, 320, (3, 3), name=name_x1 + 'b')                        # 24x24x256
  x1 = conv2d_bn(x1, 320, (3, 3), strides=(2, 2), name=name_x1 + 'c')        # 12x12x384

  x2 = conv2d_bn(x, 352, (3, 3), strides=(2, 2), name=name_x2)               # 12x12x384

  x3 = conv2d_bn(x, 256, (1, 1), name=name_x3 + 'a')                         # 24x24x256
  x3 = MaxPool2D((3, 3), strides=(2, 2), padding="same", name=name_x3 + 'b')(x3)    # 12x12x384

  x = concatenate([x1, x2, x3], axis=S.channel_axis(), name=name)            # 12x12x928
  return x


def inception_resnet_b3(x, scale=0.3, name=None):                            # 12x12x928
  b_block_number = next(inception_resnet_b3.counter)
  if name is None:
    name_x1 = f"B{b_block_number}_7x1_1x7_"
    name_x2 = f"B{b_block_number}_1x1_"
    name_residual = f"B{b_block_number}_residual_"
    name_concat = f"B{b_block_number}_concat"
    name = f"InceptionResNet_B{b_block_number}"
  else:
    name_x1 = name + "_7x1_1x7_"
    name_x2 = name + "_1x1_"
    name_residual = name + "_residual_"
    name_concat = name + "_concat"

  residual = tf.math.multiply(x, scale, name=name_residual + 'scale')     # 12x12x928

  x1 = conv2d_bn(x, 448, (1, 1), name=name_x1 + 'a')                         # 12x12x128
  x1 = conv2d_bn(x1, 480, (7, 1), name=name_x1 + 'b')                        # 12x12x160
  x1 = Conv2D(480, (1, 7), padding="same", name=name_x1 + 'c')(x1)                        # 12x12x192
  x1 = BatchNormalization(axis=S.channel_axis(), scale=False, name=name_x1 + 'd')(x1)

  x2 = Conv2D(448, (1, 1), padding="same", name=name_x2 + 'a')(x)  # 12x12x192
  x2 = BatchNormalization(axis=S.channel_axis(), scale=False, name=name_x2 + 'b')(x2)

  x = concatenate([x1, x2], axis=S.channel_axis(), name=name_concat)         # 12x12x384

  x = Add(name=name_residual + 'add')([residual, x])                         # 12x12x928
  x = Activation(S.activation(), name=name)(x)                               # 12x12x928
  return x


def reduction_b3(x, name=None):                                              # 12x12x928
  if name is None:
    name_x1 = "Red_B_3x3_3x3_"
    name_x2 = "Red_B_3x3_256"
    name_x3 = "Red_B_3x3_384"
    name_x4 = "Red_B_MP_"
    name = "Reduction_B"
  else:
    name_x1 = name + "_3x3_3x3_"
    name_x2 = name + "_3x3_256"
    name_x3 = name + "_3x3_384"
    name_x4 = name + "_MP_"

  x1 = conv2d_bn(x, 512, (1, 1), name=name_x1 + 'a')                         # 12x12x256
  x1 = conv2d_bn(x1, 640, (3, 3), name=name_x1 + 'b')                        # 12x12x256
  x1 = conv2d_bn(x1, 640, (3, 3), strides=(2, 2), padding="valid",
                 name=name_x1 + 'c')                                         # 5x5x256

  x2 = conv2d_bn(x, 448, (1, 1), name=name_x2 + 'a')                         # 12x12x256
  x2 = conv2d_bn(x2, 512, (3, 3), strides=(2, 2), padding="valid",
                 name=name_x2 + 'b')                                         # 5x5x256

  x3 = conv2d_bn(x, 640, (1, 1), name=name_x3 + 'a')                         # 12x12x256
  x3 = conv2d_bn(x3, 896, (3, 3), strides=(2, 2), padding="valid",
                 name=name_x3 + 'b')                                         # 5x5x384

  x4 = conv2d_bn(x, 512, (1, 1), name=name_x4 + 'a')                         # 12x12x256
  x4 = MaxPool2D((3, 3), strides=(2, 2), name=name_x4 + 'b')(x4)                    # 5x5x1152

  x = concatenate([x1, x2, x3, x4], axis=S.channel_axis(), name=name)        # 5x5x2560
  return x


def inception_resnet_c3(x, scale=0.3, name=None):                             # 5x5x2560
  c_block_number = next(inception_resnet_c3.counter)
  if name is None:
    name_x1 = f"C{c_block_number}_3x1_1x3_"
    name_x2 = f"C{c_block_number}_1x1_"
    name_residual = f"C{c_block_number}_residual_"
    name_concat = f"C{c_block_number}_concat_"
    name = f"InceptionResNet_C{c_block_number}"
  else:
    name_x1 = name + "_3x1_1x3_"
    name_x2 = name + "_1x1_"
    name_residual = name + "_residual_"
    name_concat = name + "_concat_"

  residual = tf.math.multiply(x, scale, name=name_residual + 'scale')     # 5x5x2048

  x1 = conv2d_bn(x, 1280, (1, 1), name=name_x1 + 'a')                         # 5x5x192
  x1 = conv2d_bn(x1, 1536, (3, 1), name=name_x1 + 'b')                        # 5x5x224
  x1 = Conv2D(1536, (1, 3), padding="same", name=name_x1 + 'c')(x1)                           # 5x5x256
  x1 = BatchNormalization(axis=S.channel_axis(), scale=False, name=name_x1 + 'd')(x1)

  x2 = Conv2D(1024, (1, 1), padding="same", name=name_x2 + 'a')(x)                               # 5x5x192
  x2 = BatchNormalization(axis=S.channel_axis(), scale=False, name=name_x2 + 'b')(x2)

  residual = concatenate([x1, x2], axis=S.channel_axis(), name=name_concat)  # 5x5x444

  x = Add(name=name_residual + 'add')([residual, x])                           # 5x5x2560
  x = Activation(S.activation(), name=name)(x)                               # 5x5x2560
  return x


def inception_resnet_v2_remodified(include_top=True,
                                   scale_a=1,
                                   scale_b=0.3,
                                   scale_c=0.3):
  # Counters as function attributes, used for incremental naming.
  inception_resnet_a3.counter = count(0)
  inception_resnet_b3.counter = count(0)
  inception_resnet_c3.counter = count(0)
  conv2d_bn.counter = count(0)
  inshape, outshape = utils.inoutshapes()

  input_layer = Input(shape=inshape)                                         # 94x94x6

  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)                     # 94x94x6
    x = stem(x)                                                              # 24x24x384
  else:
    x = stem(input_layer)                                                    # 24x24x384

  for _ in range(5):
    x = inception_resnet_a3(x, scale=scale_a)                                 # 24x24x384
  x = reduction_a3(x)                                                         # 12x12x832

  for _ in range(10):
    x = inception_resnet_b3(x, scale=scale_b)                                 # 12x12x822
  x = reduction_b3(x)                                                         # 5x5x1660

  for _ in range(5):
    x = inception_resnet_c3(x, scale=scale_c)                                 # 5x5x2560

  x = GlobalAveragePooling2D(data_format=S.image_data_format(), name="Head_1")(x)
  if include_top:
    # x = Flatten(name="Head_2")(x)                                            # 3072
    x = Dense(4096,
              activation=S.activation(),
              kernel_initializer=S.kernel_initializer(),
              name="Head_3"
              )(x)                                                           # 4096
    x = Dense(5120,
              activation=S.activation(),
              kernel_initializer=S.kernel_initializer(),
              name="Head_4"
              )(x)                                                           # 5120
    x = Dense(np.prod(outshape),
              kernel_initializer=S.kernel_initializer(),
              name="Output"                                                  # (90x90x3)
              )(x)                                                           # 24300

  model = Model(input_layer, x, name="Inception-ResNet-v3")
  return model
