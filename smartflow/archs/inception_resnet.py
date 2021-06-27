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
"""Creates a deep CNN architecture, using elements of Inception-v3 and ResNet.

  This module contains some custon experimentations on Inception-ResNet archi-
  tectures. For the initial implementation, introduced by Szegedy, Ioffe,
  Vanhoucke and Alemi, see the inception_resnet_v3.py module.

  Reference papers:
  - [Rethinking the Inception Architecture for Computer Vision](
      http://arxiv.org/abs/1512.00567) (CVPR 2016)
  - [Batch normalization: Accelerating deep network training by reducing
     internal covariate shift](https://arxiv.org/abs/1502.03167)
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385)
"""

from itertools import count

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import (concatenate,
                                     Conv2D,
                                     Dense,
                                     Flatten,
                                     Input,
                                     MaxPool2D,
                                     Reshape)

from smartflow import backend as S, smartflow_pre, utils
from smartflow.archs.resnet import conv2d_bn


def stem(x):
  """starting block of the architecture

  output shape: - padding: same
                - depth: filters_3x3 (defaults to 196)
  """
  x = conv2d_bn(x, 32, (3, 3), name="Stem_1")
  x = conv2d_bn(x, 48, (3, 3), name="Stem_2")
  x = conv2d_bn(x, 64, (3, 3), name="Stem_3")

  x = conv2d_bn(x, 80, (3, 3), strides=(2, 2), name="Stem_4")
  x = conv2d_bn(x, 96, (3, 3), name="Stem_5")
  return x


def inception_res_scanner(x,
                          filters_3x3_3x3_3x3_1=64,
                          filters_3x3_3x3_3x3_2=76,
                          filters_3x3_3x3_3x3_3=64,
                          filters_3x3_3x3_1=32,
                          filters_3x3_3x3_2=32,
                          filters_3x3=32,
                          filters_pool_1x1=32,
                          **kwargs):
  """The starting block of the architecture.

  - Factorizing: 7x7 --> 3x3_3x3_3x3 & 5x5 --> 3x3_3x3
    (see Inception paper, Section 3 and Figure 5)
  - Using strides=(1, 1), same padding and no pool, since the input grid is
    very small and, thus, the model should not introduce any representation
    loss early on.
  - Increasing network width and depth in parallel
    (see Inception paper, principle 4 in Section 2)
  """
  name = kwargs.pop("name", False)
  if not name:
    name_3x3_3x3_3x3 = "Sc_3x3_3x3_3x3_"
    name_3x3_3x3 = "Sc_3x3_3x3_"
    name_3x3 = "Sc_3x3_"
    name_pool = "Sc_MaxPool_"
    name = "Scanner"
  else:
    name_3x3_3x3_3x3 = name + "_3x3_3x3_3x3_"
    name_3x3_3x3 = name + "_3x3_3x3_"
    name_3x3 = name + "_3x3_"
    name_pool = name + "_MaxPool_"

  # ---------------------------------------------------------------------------
  conv3x3_3x3_3x3_1 = conv2d_bn(x,
                                filters_3x3_3x3_3x3_1,
                                kernel=(3, 3),
                                name=name_3x3_3x3_3x3 + 'a',
                                **kwargs)
  conv3x3_3x3_3x3_2 = conv2d_bn(conv3x3_3x3_3x3_1,
                                filters_3x3_3x3_3x3_2,
                                kernel=(3, 3),
                                name=name_3x3_3x3_3x3 + 'b')
  conv3x3_3x3_3x3_3 = conv2d_bn(conv3x3_3x3_3x3_2,
                                filters_3x3_3x3_3x3_3,
                                kernel=(3, 3),
                                residual=conv3x3_3x3_3x3_1,
                                name=name_3x3_3x3_3x3 + 'c')
  # ---------------------------------------------------------------------------
  conv3x3_3x3 = conv2d_bn(x,
                          filters_3x3_3x3_1,
                          kernel=(3, 3),
                          name=name_3x3_3x3 + 'a')
  conv3x3_3x3 = conv2d_bn(conv3x3_3x3,
                          filters_3x3_3x3_2,
                          kernel=(3, 3),
                          name=name_3x3_3x3 + 'b')
  # ---------------------------------------------------------------------------
  conv3x3 = conv2d_bn(x,
                      filters_3x3,
                      kernel=(3, 3),
                      name=name_3x3)
  # ---------------------------------------------------------------------------
  maxpool = MaxPool2D((3, 3),
                      strides=(1, 1),
                      padding="same",
                      name=name_pool + 'a'
                      )(x)
  maxpool = conv2d_bn(maxpool,
                      filters_pool_1x1,
                      kernel=(1, 1),
                      name=name_pool + 'b')
  # ---------------------------------------------------------------------------

  iblock = concatenate([conv3x3_3x3_3x3_3,
                        conv3x3_3x3,
                        conv3x3,
                        maxpool],
                       axis=S.channel_axis(),
                       name=name)
  return iblock


def inception_res_block(x,
                        filters_3x3_3x3_1=64,
                        filters_3x3_3x3_2=96,
                        filters_3x3_3x3_3=96,
                        filters_3x3_1=48,
                        filters_3x3_2=64,
                        filters_pool_1x1=64,
                        padding="valid",
                        strides=(2, 2),
                        name=None):
  """Basic Inception-Residual block"""
  inc_number = next(inception_res_block.counter)
  if name is None:
    name_3x3_3x3 = f"3x3_3x3_{inc_number}"
    name_3x3 = f"3x3_{inc_number}"
    name_pool = f"MaxPool_{inc_number}"
    name = f"InceRes_{inc_number}"
  else:
    conv3x3_3x3 = name + "_3x3_3x3_"
    name_3x3 = name + "_3x3_"
    name_pool = name + "_MaxPool_"

  # ---------------------------------------------------------------------------
  conv3x3_3x3_1 = conv2d_bn(x,
                            filters_3x3_3x3_1,
                            kernel=(1, 1),
                            name=name_3x3_3x3 + 'a')
  conv3x3_3x3_2 = conv2d_bn(conv3x3_3x3_1,
                            filters_3x3_3x3_2,
                            kernel=(3, 3),
                            name=name_3x3_3x3 + 'b')
  conv3x3_3x3_3 = conv2d_bn(conv3x3_3x3_2,
                            filters_3x3_3x3_3,
                            kernel=(3, 3),
                            strides=strides,
                            padding=padding,
                            residual=conv3x3_3x3_1,
                            name=name_3x3_3x3 + 'c')
  # ---------------------------------------------------------------------------
  conv3x3 = conv2d_bn(x,
                      filters_3x3_1,
                      kernel=(1, 1),
                      name=name_3x3 + 'a')
  conv3x3 = conv2d_bn(conv3x3,
                      filters_3x3_2,
                      kernel=(3, 3),
                      strides=strides,
                      padding=padding,
                      residual=x,
                      name=name_3x3 + 'b')
  # ---------------------------------------------------------------------------
  maxpool = MaxPool2D((3, 3),
                      strides=strides,
                      padding=padding,
                      name=name_pool + 'a'
                      )(x)
  maxpool = conv2d_bn(maxpool,
                      filters_pool_1x1,
                      kernel=(1, 1),
                      name=name_pool + 'b')
  # ---------------------------------------------------------------------------

  iblock = concatenate([conv3x3_3x3_3,
                        conv3x3,
                        maxpool],
                       axis=S.channel_axis(),
                       name=name)
  return iblock


def inception_res_block_1xn_nx1(x,
                                n=5,
                                filters_1xn_nx1_1xn_nx1_1=640,
                                filters_1xn_nx1_1xn_nx1_2=640,
                                filters_1xn_nx1_1xn_nx1_3=640,
                                filters_1xn_nx1_1xn_nx1_4=640,
                                filters_1xn_nx1_1xn_nx1_5=640,
                                filters_1xn_nx1_1=640,
                                filters_1xn_nx1_2=640,
                                filters_1xn_nx1_3=640,
                                filters_pool_1x1=388,
                                filters_1x1=388,
                                name=None):
  """Inception block, factorizing into asymmetric convolutions, significantly
  decreasing the number of computations, while resulting in the same volume.
  (see Inception paper, Figure 6 and principle 3 in Section 2)

  NOTE: Keep coherent filter size to result to identity shortcuts and, thus,
        less computations.
  """
  inc_number = next(inception_res_block_1xn_nx1.counter)
  if name is None:
    name_1xn_nx1_1xn_nx1 = f"ASYM_1nn11nn1_{inc_number}"
    name_1xn_nx1 = f"ASYM_1nn1_{inc_number}"
    name_pool = f"ASYM_Pool_{inc_number}"
    name_1x1 = f"ASYM_1x1_{inc_number}"
    name = f"IncRes_Assymetric_{inc_number}"
  else:
    name_1xn_nx1_1xn_nx1 = name + "_Conv_1nn11nn1_"
    name_1xn_nx1 = name + "_Conv_1nn1_"
    name_pool = name + "_AvgPool_"
    name_1x1 = name + "_Conv_1x1_"

  # ---------------------------------------------------------------------------
  conv_1xn_nx1_1xn_nx1_1 = conv2d_bn(x,
                                     filters_1xn_nx1_1xn_nx1_1,
                                     kernel=(1, 1),
                                     name=name_1xn_nx1_1xn_nx1 + 'a')
  conv_1xn_nx1_1xn_nx1_2 = conv2d_bn(conv_1xn_nx1_1xn_nx1_1,
                                     filters_1xn_nx1_1xn_nx1_2,
                                     kernel=(1, n),
                                     name=name_1xn_nx1_1xn_nx1 + 'b')
  conv_1xn_nx1_1xn_nx1_3 = conv2d_bn(conv_1xn_nx1_1xn_nx1_2,
                                     filters_1xn_nx1_1xn_nx1_3,
                                     kernel=(n, 1),
                                     name=name_1xn_nx1_1xn_nx1 + 'c')
  conv_1xn_nx1_1xn_nx1_4 = conv2d_bn(conv_1xn_nx1_1xn_nx1_3,
                                     filters_1xn_nx1_1xn_nx1_4,
                                     kernel=(1, n),
                                     residual=conv_1xn_nx1_1xn_nx1_1,
                                     name=name_1xn_nx1_1xn_nx1 + 'd')
  conv_1xn_nx1_1xn_nx1_5 = conv2d_bn(conv_1xn_nx1_1xn_nx1_4,
                                     filters_1xn_nx1_1xn_nx1_5,
                                     kernel=(n, 1),
                                     residual=conv_1xn_nx1_1xn_nx1_2,
                                     name=name_1xn_nx1_1xn_nx1 + 'e')
  # ---------------------------------------------------------------------------
  conv_1xn_nx1_1 = conv2d_bn(x,
                             filters_1xn_nx1_1,
                             kernel=(1, 1),
                             name=name_1xn_nx1 + 'a')
  conv_1xn_nx1_2 = conv2d_bn(conv_1xn_nx1_1,
                             filters_1xn_nx1_2,
                             kernel=(1, n),
                             name=name_1xn_nx1 + 'b')
  conv_1xn_nx1_3 = conv2d_bn(conv_1xn_nx1_2,
                             filters_1xn_nx1_3,
                             kernel=(n, 1),
                             residual=conv_1xn_nx1_1,
                             name=name_1xn_nx1 + 'c')
  # ---------------------------------------------------------------------------
  maxpool = MaxPool2D((2, 2),
                      strides=(1, 1),
                      padding="same",
                      name=name_pool + 'a'
                      )(x)
  maxpool = conv2d_bn(maxpool,
                      filters_pool_1x1,
                      kernel=(1, 1),
                      name=name_pool + 'b')
  # ---------------------------------------------------------------------------
  # conv1x1 = conv2d_bn(x,
  #                     filters_1x1,
  #                     kernel=(1, 1),
  #                     name=name_1x1)
  # ---------------------------------------------------------------------------

  iblock = concatenate([conv_1xn_nx1_1xn_nx1_5,
                        conv_1xn_nx1_3,
                        maxpool
                        # conv1x1
                        ],
                       axis=S.channel_axis(),
                       name=name)
  return iblock


def head(x, shape,
         filters_3x3=1024,
         filters_1x1=196,
         name=None):
  """Creates an output block.

  default input shape: <same>x256
  """
  pistil_num = next(head.counter)
  if name is None:
    name = f"OUT_{pistil_num}"

  x = MaxPool2D((2, 2))(x)                        # 17x17x256
  x = Conv2D(filters_3x3, (3, 3), activation="elu")(x)  # 15x15x512
  x = MaxPool2D((2, 2))(x)                               # 7x7x512
  # x = Conv2D(filters_3x3, (3, 3), **same_relu)(x)        # 7x7x1024
  x = MaxPool2D((2, 2))(x)                               # 3x3x1024
  x = Flatten()(x)                                       # 9604
  # x = Dense(4096, activation="elu")(x)                # 4096
  x = Dense(np.prod(shape), activation="elu")(x)        # 4225 12675
  x = Reshape(shape, name=name)(x)                       # 65x65x1
  return x


def inception_res_scanner2(x,
                           filters_7x7=64,
                           filters_5x5=48,
                           filters_3x3=32,
                           filters_pool_1x1=24,
                           **kwargs):
  name = kwargs.pop("name", False)
  if not name:
    name_7x7 = "Scan_7x7"
    name_5x5 = "Scan_5x5"
    name_3x3 = "Scan_3x3"
    name_pool = "Scan_MaxPool_"
    name = "Scanner"
  else:
    name_7x7 = f"{name}_7x7"
    name_5x5 = f"{name}_5x5"
    name_3x3 = f"{name}_3x3"
    name_pool = f"{name}_MaxPool_"

  # ---------------------------------------------------------------------------
  conv7x7 = conv2d_bn(x,
                      filters_7x7,
                      kernel=(7, 7),
                      strides=(2, 2),
                      name=name_7x7,
                      **kwargs)
  # ---------------------------------------------------------------------------
  conv5x5 = conv2d_bn(x,
                      filters_5x5,
                      kernel=(3, 3),
                      strides=(2, 2),
                      name=name_5x5)
  # ---------------------------------------------------------------------------
  conv3x3 = conv2d_bn(x,
                      filters_3x3,
                      kernel=(3, 3),
                      strides=(2, 2),
                      name=name_3x3)
  # ---------------------------------------------------------------------------
  maxpool = MaxPool2D((3, 3),
                      strides=(2, 2),
                      padding="same",
                      name=name_pool + 'a'
                      )(x)
  maxpool = conv2d_bn(maxpool,
                      filters_pool_1x1,
                      kernel=(1, 1),
                      name=name_pool + 'b')
  # ---------------------------------------------------------------------------

  iblock = concatenate([conv7x7,
                        conv5x5,
                        conv3x3,
                        maxpool],
                       axis=S.channel_axis(),
                       name=name)
  return iblock


def inception_res_block2(x,
                         filters_5x5_1=64,
                         filters_5x5_2=96,
                         filters_5x5_3=96,
                         filters_3x3_1=48,
                         filters_3x3_2=64,
                         filters_pool_1x1=64,
                         padding="same",
                         strides=None,
                         name=None):
  inc_number = next(inception_res_block.counter)
  if name is None:
    name_5x5 = f"5x5_{inc_number}"
    name_3x3 = f"3x3_{inc_number}"
    name_pool = f"MaxPool_{inc_number}"
    name = f"InceRes_{inc_number}"
  else:
    name_5x5 = f"{name}_3x3_3x3_"
    name_3x3 = f"{name}_3x3_"
    name_pool = f"{name}_MaxPool_"

  # ---------------------------------------------------------------------------
  conv5x5 = conv2d_bn(x,
                      filters_5x5_1,
                      kernel=(1, 1),
                      name=name_5x5 + 'a')
  conv5x5 = conv2d_bn(conv5x5,
                      filters_5x5_2,
                      kernel=(5, 5),
                      strides=strides,
                      name=name_5x5 + 'b')
  conv5x5 = conv2d_bn(conv5x5,
                      filters_5x5_3,
                      kernel=(1, 1),
                      padding=padding,
                      residual=x,
                      name=name_5x5 + 'c')
  # ---------------------------------------------------------------------------
  conv3x3 = conv2d_bn(x,
                      filters_3x3_1,
                      kernel=(3, 3),
                      strides=strides,
                      name=name_3x3 + 'a')
  conv3x3 = conv2d_bn(conv3x3,
                      filters_3x3_2,
                      kernel=(3, 3),
                      padding=padding,
                      residual=x,
                      name=name_3x3 + 'b')
  # ---------------------------------------------------------------------------
  maxpool = MaxPool2D((3, 3),
                      strides=strides,
                      padding=padding,
                      name=name_pool + 'a'
                      )(x)
  maxpool = conv2d_bn(maxpool,
                      filters_pool_1x1,
                      kernel=(1, 1),
                      name=name_pool + 'b')
  # ---------------------------------------------------------------------------

  iblock = concatenate([conv5x5,
                        conv3x3,
                        maxpool],
                       axis=S.channel_axis(),
                       name=name)
  return iblock


def inception_resnet_50x50(include_top=True):
  """A deep CNN architecture, using elements of Inception-v3 and ResNet

  - Residual networks address the degradation of the training accuracy that
    occures at a certain network depth onwards. The phenomenon is not correla-
    ted with vanishing/exploding gradients or overfitting; rather, it is an
    accuracy saturation related with the depth of the network.
    (See He et al.)
  """
  # counters as function attributes (used for layer naming)
  inception_res_block.counter = count(0)
  inception_res_block_1xn_nx1.counter = count(0)
  head.counter = count(0)
  conv2d_bn.counter = count(0)

  inshape, outshape = utils.inoutshapes()

  input_layer = Input(inshape)                                # 0  54x54x6
  x = inception_res_scanner(input_layer)                      # 1  54x54x176
  x = inception_res_block(x,
                          filters_3x3_3x3_1=128,
                          filters_3x3_3x3_2=128,
                          filters_3x3_3x3_3=128,
                          filters_3x3_1=136,
                          filters_3x3_2=136,
                          filters_pool_1x1=96,
                          padding="same",
                          strides=(1, 1)
                          )                                   # 2  54x54x360
  x = inception_res_block(x,
                          filters_3x3_3x3_1=196,
                          filters_3x3_3x3_2=196,
                          filters_3x3_3x3_3=196,
                          filters_3x3_1=196,
                          filters_3x3_2=196,
                          filters_pool_1x1=160)               # 3  26x26x552
  x = inception_res_block(x,
                          filters_3x3_3x3_1=256,
                          filters_3x3_3x3_2=256,
                          filters_3x3_3x3_3=256,
                          filters_3x3_1=256,
                          filters_3x3_2=256,
                          filters_pool_1x1=212,
                          padding="same",
                          strides=(1, 1))                     # 4  26x26x724
  x = inception_res_block(x,
                          filters_3x3_3x3_1=404,
                          filters_3x3_3x3_2=404,
                          filters_3x3_3x3_3=404,
                          filters_3x3_1=404,
                          filters_3x3_2=404,
                          filters_pool_1x1=324)               # 5  12x12x1132
  x = inception_res_block_1xn_nx1(x,
                                  n=5,
                                  filters_1xn_nx1_1xn_nx1_1=712,
                                  filters_1xn_nx1_1xn_nx1_2=448,
                                  filters_1xn_nx1_1xn_nx1_3=448,
                                  filters_1xn_nx1_1xn_nx1_4=448,
                                  filters_1xn_nx1_1xn_nx1_5=448,
                                  filters_1xn_nx1_1=712,
                                  filters_1xn_nx1_2=512,
                                  filters_1xn_nx1_3=712,
                                  filters_pool_1x1=512,
                                  filters_1x1=0)              # 6  12x12x1672
  x = inception_res_block(x,
                          filters_3x3_3x3_1=896,
                          filters_3x3_3x3_2=896,
                          filters_3x3_3x3_3=896,
                          filters_3x3_1=896,
                          filters_3x3_2=896,
                          filters_pool_1x1=896)               # 7  5x5x2688
  x1 = conv2d_bn(x, 2048, (1, 1))                             # 8  5x5x2048
  x = conv2d_bn(x1, 2658, (3, 3), padding="valid")            # 9  3x3x2658
  x = conv2d_bn(x, 2048, (1, 1), residual=x1)                 # 10 3x3x2048
  x = conv2d_bn(x, 3072, (3, 3), padding="valid")             # 11 1x1x3072
  if include_top:
    x = Flatten()(x)                                          # 12 3072
    x = Dense(5120,
              activation=S.activation(),
              kernel_initializer=S.kernel_initializer()
              )(x)                                            # 13 5120
    x = Dense(np.prod(outshape),
              kernel_initializer=S.kernel_initializer()
              )(x)                                            # 14 7500 (50x50x3)

  model = Model(input_layer, x)
  return model


def inception_resnet_50x50_2(include_top=True):
  """A deep CNN architecture, using elements of Inception-v3 and ResNet

  - Residual networks address the degradation of the training accuracy that
    occures at a certain network depth onwards. The phenomenon is not
    correlated with vanishing/exploding gradients or overfitting; rather, it is
    an ccuracy saturat1on related with the depth of the network.
    (See He et al.)
  """
  # counters as function attributes (used for layer naming)
  inception_res_block.counter = count(0)
  inception_res_block_1xn_nx1.counter = count(0)
  head.counter = count(0)
  conv2d_bn.counter = count(0)

  inshape, outshape = utils.inoutshapes()

  input_layer = Input(inshape)                                  # 0  54x54x6
  x = inception_res_scanner2(input_layer)                       # 1  27x27x168
  x = inception_res_block2(x,
                           filters_5x5_1=96,
                           filters_5x5_2=28,
                           filters_5x5_3=96,
                           filters_3x3_1=128,
                           filters_3x3_2=168,
                           filters_pool_1x1=80,
                           strides=(1, 1),
                           )                                    # 2  27x27x344
  x = inception_res_block2(x,
                           filters_5x5_1=176,
                           filters_5x5_2=176,
                           filters_5x5_3=192,
                           filters_3x3_1=176,
                           filters_3x3_2=256,
                           filters_pool_1x1=176,
                           strides=(2, 2),
                           )                                    # 2  14x14x624
  x = inception_res_block2(x,
                           filters_5x5_1=356,
                           filters_5x5_2=384,
                           filters_5x5_3=356,
                           filters_3x3_1=356,
                           filters_3x3_2=416,
                           filters_pool_1x1=320,
                           strides=(2, 2),
                           )                                    # 2  7x7x1092
  x = inception_res_block2(x,
                           filters_5x5_1=512,
                           filters_5x5_2=40,
                           filters_5x5_3=512,
                           filters_3x3_1=512,
                           filters_3x3_2=640,
                           filters_pool_1x1=480,
                           strides=(2, 2),
                           )                                    # 2  4x4x1632
  x1 = conv2d_bn(x, 1024, (1, 1))                               # 8  4x4x1024
  x = conv2d_bn(x1, 2048, (4, 4), padding="valid")              # 10 1x1x2560
  x = conv2d_bn(x, 2560, (1, 1), residual=x1)                   # 10 1x1x2560
  if include_top:
    x = Flatten()(x)                                            # 12 2048
    x = Dense(4096,
              activation=S.activation(),
              kernel_initializer=S.kernel_initializer()
              )(x)                                              # 13 5120
    x = Dense(np.prod(outshape),
              kernel_initializer=S.kernel_initializer()
              )(x)                                              # 14 7500 (50x50x3)

  model = Model(input_layer, x)
  return model


def inception_resnet_80x80(include_top=True):
  """A deep CNN architecture, using elements of Inception-v3 and ResNet

  - Residual networks address the degradation of the training accuracy that
    occures at a certain network depth onwards. The phenomenon is not correla-
    ted with vanishing/exploding gradients or overfitting; rather, it is an
    accuracy saturation related with the depth of the network.
    (See He et al.)
  """
  # counters as function attributes (used for layer naming)
  inception_res_block.counter = count(0)
  inception_res_block_1xn_nx1.counter = count(0)
  head.counter = count(0)
  conv2d_bn.counter = count(0)

  inshape, outshape = utils.inoutshapes()

  input_layer = Input(inshape)                                # 0  84x84x6
  x = inception_res_scanner(input_layer)                      # 1  84x84x176
  x = inception_res_block(x,
                          filters_3x3_3x3_1=128,
                          filters_3x3_3x3_2=128,
                          filters_3x3_3x3_3=128,
                          filters_3x3_1=128,
                          filters_3x3_2=128,
                          filters_pool_1x1=96,
                          # padding="same",
                          # strides=(1, 1)
                          )                                   # 2  41x41x352
  x = inception_res_block(x,
                          filters_3x3_3x3_1=212,
                          filters_3x3_3x3_2=212,
                          filters_3x3_3x3_3=212,
                          filters_3x3_1=212,
                          filters_3x3_2=212,
                          filters_pool_1x1=196,                # 3  21x21x620
                          padding="same")
  # x = inception_res_block(x,
  #                         filters_3x3_3x3_1=256,
  #                         filters_3x3_3x3_2=256,
  #                         filters_3x3_3x3_3=256,
  #                         filters_3x3_1=256,
  #                         filters_3x3_2=320,
  #                         filters_pool_1x1=256)               # 4  20x20x832
  #                         # padding="same",
  #                         # strides=(1, 1))
  x = inception_res_block(x,
                          filters_3x3_3x3_1=412,
                          filters_3x3_3x3_2=412,
                          filters_3x3_3x3_3=412,
                          filters_3x3_1=412,
                          filters_3x3_2=412,
                          filters_pool_1x1=324,               # 5  11x11x1148
                          padding="same")
  x = inception_res_block_1xn_nx1(x,
                                  n=5,
                                  filters_1xn_nx1_1xn_nx1_1=608,
                                  filters_1xn_nx1_1xn_nx1_2=412,
                                  filters_1xn_nx1_1xn_nx1_3=412,
                                  filters_1xn_nx1_1xn_nx1_4=412,
                                  filters_1xn_nx1_1xn_nx1_5=412,
                                  filters_1xn_nx1_1=608,
                                  filters_1xn_nx1_2=448,
                                  filters_1xn_nx1_3=608,
                                  filters_pool_1x1=480,
                                  filters_1x1=480)            # 6  11x11x1980
  x = inception_res_block(x,
                          filters_3x3_3x3_1=886,
                          filters_3x3_3x3_2=886,
                          filters_3x3_3x3_3=886,
                          filters_3x3_1=886,
                          filters_3x3_2=886,
                          filters_pool_1x1=886)               # 7  5x5x2658
  # x = inception_res_block(x,
  #                         filters_3x3_3x3_1=1024,
  #                         filters_3x3_3x3_2=1024,
  #                         filters_3x3_3x3_3=1024,
  #                         filters_3x3_1=1024,
  #                         filters_3x3_2=1024,
  #                         filters_pool_1x1=1024,
  #                         padding="same")                     # 7  5x5x3072
  x1 = conv2d_bn(x, 1792, (1, 1))                             # 8  5x5x2048
  x = conv2d_bn(x1, 2560, (3, 3), padding="valid")            # 9  3x3x3072
  x = conv2d_bn(x, 2048, (1, 1), residual=x1)                 # 10 3x3x2560
  x = conv2d_bn(x, 3072, (3, 3), padding="valid")             # 11 1x1x4096
  if include_top:
    x = Flatten()(x)                                          # 12 4096
    x = Dense(5120,
              activation=S.activation(),
              kernel_initializer=S.kernel_initializer()
              )(x)                                            # 13 8096
    x = Dense(np.prod(outshape),
              kernel_initializer=S.kernel_initializer()
              )(x)                                            # 14 19200 (80x80x3)

  model = Model(input_layer, x)
  return model


def inception_resnet_90x90(include_top=True):
  """A deep CNN architecture, using elements of Inception-v3 and ResNet

  - Residual networks address the degradation of the training accuracy that
    occures at a certain network depth onwards. The phenomenon is not correla-
    ted with vanishing/exploding gradients or overfitting; rather, it is an
    accuracy saturation related with the depth of the network.
    (See He et al.)
  """
  # counters as function attributes (used for layer naming)
  inception_res_block.counter = count(0)
  inception_res_block_1xn_nx1.counter = count(0)
  head.counter = count(0)
  conv2d_bn.counter = count(0)

  inshape, outshape = utils.inoutshapes()

  input_layer = Input(inshape)                                # 0  94x94x6
  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)
    x = inception_res_scanner(x)                              # 1  94x94x160
  else:
    x = inception_res_scanner(input_layer)                    # 1  94x94x160
  x = inception_res_block(x,
                          filters_3x3_3x3_1=112,
                          filters_3x3_3x3_2=112,
                          filters_3x3_3x3_3=112,
                          filters_3x3_1=112,
                          filters_3x3_2=112,
                          filters_pool_1x1=96,
                          padding="same"
                          # strides=(1, 1)
                          )                                   # 2  47x47x320
  x = inception_res_block(x,
                          filters_3x3_3x3_1=196,
                          filters_3x3_3x3_2=196,
                          filters_3x3_3x3_3=196,
                          filters_3x3_1=196,
                          filters_3x3_2=196,
                          filters_pool_1x1=180,               # 3  24x24x572
                          padding="same"
                          )
  # x = inception_res_block(x,
  #                         filters_3x3_3x3_1=256,
  #                         filters_3x3_3x3_2=256,
  #                         filters_3x3_3x3_3=256,
  #                         filters_3x3_1=256,
  #                         filters_3x3_2=320,
  #                         filters_pool_1x1=256)               # 4  20x20x832
  #                         # padding="same",
  #                         # strides=(1, 1))
  x = inception_res_block(x,
                          filters_3x3_3x3_1=372,
                          filters_3x3_3x3_2=372,
                          filters_3x3_3x3_3=372,
                          filters_3x3_1=372,
                          filters_3x3_2=372,
                          filters_pool_1x1=324,               # 5  12x12x1086
                          padding="same"
                          )
  x = inception_res_block_1xn_nx1(x,
                                  n=5,
                                  filters_1xn_nx1_1xn_nx1_1=604,
                                  filters_1xn_nx1_1xn_nx1_2=396,
                                  filters_1xn_nx1_1xn_nx1_3=396,
                                  filters_1xn_nx1_1xn_nx1_4=396,
                                  filters_1xn_nx1_1xn_nx1_5=396,
                                  filters_1xn_nx1_1=604,
                                  filters_1xn_nx1_2=512,
                                  filters_1xn_nx1_3=512,
                                  filters_pool_1x1=480
                                  # filters_1x1=0
                                  )             # 6  12x12x1388
  x = inception_res_block(x,
                          filters_3x3_3x3_1=800,
                          filters_3x3_3x3_2=800,
                          filters_3x3_3x3_3=800,
                          filters_3x3_1=822,
                          filters_3x3_2=822,
                          filters_pool_1x1=608)               # 7  5x5x2220
  # x = inception_res_block(x,
  #                         filters_3x3_3x3_1=1024,
  #                         filters_3x3_3x3_2=1024,
  #                         filters_3x3_3x3_3=1024,
  #                         filters_3x3_1=1024,
  #                         filters_3x3_2=1024,
  #                         filters_pool_1x1=1024,
  #                         padding="same")                     # 7  5x5x3072
  x1 = conv2d_bn(x, 1536, (1, 1))                             # 8  5x5x1788
  x = conv2d_bn(x1, 2048, (3, 3), padding="valid")            # 9  3x3x2560
  x = conv2d_bn(x, 1792, (1, 1), residual=x1)                 # 10 3x3x1536
  x = conv2d_bn(x, 3072, (3, 3), padding="valid")             # 11 1x1x3072
  if include_top:
    x = Flatten()(x)                                          # 12 3072
    # x = Dense(4096,
    #           activation=S.activation(),
    #           kernel_initializer=S.kernel_initializer()
    #           )(x)                                            # 13 4096
    x = Dense(5120,
              activation=S.activation(),
              kernel_initializer=S.kernel_initializer()
              )(x)                                            # 13 5120
    x = Dense(np.prod(outshape),
              kernel_initializer=S.kernel_initializer()
              )(x)                                            # 14 24300 (90x90x3)

  model = Model(input_layer, x)
  return model


def inception_resnet_100x100(include_top=True):
  """A deep CNN architecture, using elements of Inception-v3 and ResNet

  - Residual networks address the degradation of the training accuracy that
    occures at a certain network depth onwards. The phenomenon is not correla-
    ted with vanishing/exploding gradients or overfitting; rather, it is an
    accuracy saturation related with the depth of the network.
    (See He et al.)
  """
  # counters as function attributes (used for layer naming)
  inception_res_block.counter = count(0)
  inception_res_block_1xn_nx1.counter = count(0)
  head.counter = count(0)
  conv2d_bn.counter = count(0)

  inshape, outshape = utils.inoutshapes()

  input_layer = Input(inshape)                                 # 0  100x100x6
  if smartflow_pre.normalization_layer_weights() is not None:
    x = smartflow_pre.normalization_layer()(input_layer)
    x = inception_res_scanner(x)                               # 1  100x100x176
  else:
    x = inception_res_scanner(input_layer)                     # 1  100x100x176
  x = inception_res_block(x,
                          filters_3x3_3x3_1=128,
                          filters_3x3_3x3_2=128,
                          filters_3x3_3x3_3=128,
                          filters_3x3_1=128,
                          filters_3x3_2=128,
                          filters_pool_1x1=96,
                          # padding="same",
                          # strides=(1, 1)
                          )                                    # 2  50x50x352
  x = inception_res_block(x,
                          filters_3x3_3x3_1=212,
                          filters_3x3_3x3_2=212,
                          filters_3x3_3x3_3=212,
                          filters_3x3_1=212,
                          filters_3x3_2=212,
                          filters_pool_1x1=196,                # 3  25x25x620
                          padding="same")
  # x = inception_res_block(x,
  #                         filters_3x3_3x3_1=256,
  #                         filters_3x3_3x3_2=256,
  #                         filters_3x3_3x3_3=256,
  #                         filters_3x3_1=256,
  #                         filters_3x3_2=320,
  #                         filters_pool_1x1=256)               # 4  20x20x832
  #                         # padding="same",
  #                         # strides=(1, 1))
  x = inception_res_block(x,
                          filters_3x3_3x3_1=412,
                          filters_3x3_3x3_2=412,
                          filters_3x3_3x3_3=412,
                          filters_3x3_1=412,
                          filters_3x3_2=412,
                          filters_pool_1x1=324)               # 5  12x12x1148
  x = inception_res_block_1xn_nx1(x,
                                  n=5,
                                  filters_1xn_nx1_1xn_nx1_1=608,
                                  filters_1xn_nx1_1xn_nx1_2=412,
                                  filters_1xn_nx1_1xn_nx1_3=412,
                                  filters_1xn_nx1_1xn_nx1_4=412,
                                  filters_1xn_nx1_1xn_nx1_5=412,
                                  filters_1xn_nx1_1=608,
                                  filters_1xn_nx1_2=448,
                                  filters_1xn_nx1_3=608,
                                  filters_pool_1x1=480,
                                  filters_1x1=480)            # 6  12x12x1980
  x = inception_res_block(x,
                          filters_3x3_3x3_1=886,
                          filters_3x3_3x3_2=886,
                          filters_3x3_3x3_3=886,
                          filters_3x3_1=886,
                          filters_3x3_2=886,
                          filters_pool_1x1=886)               # 7  5x5x2658
  # x = inception_res_block(x,
  #                         filters_3x3_3x3_1=1024,
  #                         filters_3x3_3x3_2=1024,
  #                         filters_3x3_3x3_3=1024,
  #                         filters_3x3_1=1024,
  #                         filters_3x3_2=1024,
  #                         filters_pool_1x1=1024,
  #                         padding="same")                     # 7  5x5x3072
  x1 = conv2d_bn(x, 1792, (1, 1))                             # 8  5x5x2048
  x = conv2d_bn(x1, 2560, (3, 3), padding="valid")            # 9  3x3x3072
  x = conv2d_bn(x, 2048, (1, 1), residual=x1)                 # 10 3x3x2560
  x = conv2d_bn(x, 3072, (3, 3), padding="valid")             # 11 1x1x4096
  if include_top:
    x = Flatten()(x)                                          # 12 4096
    x = Dense(5120,
              activation=S.activation(),
              kernel_initializer=S.kernel_initializer()
              )(x)                                            # 13 8096
    x = Dense(np.prod(outshape),
              kernel_initializer=S.kernel_initializer()
              )(x)                                            # 14 19200 (80x80x3)

  model = Model(input_layer, x)
  return model
