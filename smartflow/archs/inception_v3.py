# inceptionv3.py is part of SmartFlow
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
"""Implements the Inception-v3 architecture

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

from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import (Activation,
                                     AveragePooling2D,
                                     BatchNormalization,
                                     concatenate,
                                     Dense,
                                     Flatten,
                                     Input,
                                     MaxPool2D,
                                     Reshape)
from tensorflow.keras.models import Model

from smartflow import backend as S, utils
from smartflow.archs.resnet import conv2d_bn


def inception_v3_built_in(lr=0.01):
  """Creates a model, using the tf built-in Inception-v3 architecture."""
  inshape, outshape = S.inoutshapes()

  input_layer = Input(shape=inshape)                                  # 69x69xc
  x = InceptionV3(include_top=False, weights=None)(input_layer)       # 8x8x2048
  x = conv2d_bn(x, 2950, (3, 3), padding="valid"),                    # 5x5x2950
  x = conv2d_bn(x, 4096, (3, 3), padding="valid"),                    # 3x3x4096
  x = conv2d_bn(x, np.prod(outshape), (3, 3), padding="valid"),       # 1x1x4225
  x = Reshape(outshape)(x)                                            # 65x65xc

  model = Model(input_layer, x, name="Inceptionv3_built_in")
  return model


def stem(x):
  """The starting block of the architecture (7x7 factorized into 3 3x3's)

  - Using strides=(1, 1), same padding and no pool, since the input grid is
    very small and, thus, the model should not introduce any representation
    loss early on.
  - Increasing network width and depth in parallel
    (see Inception paper, principle 4 in Section 2)
  """                                                           # 69x69xc
  x = conv2d_bn(x, 32, (3, 3), strides=(1, 1), name="Stem_1")   # 69x69x32
  x = conv2d_bn(x, 32, (3, 3), strides=(1, 1), name="Stem_2")   # 69x69x32
  x = conv2d_bn(x, 64, (3, 3), strides=(1, 1), name="Stem_3")   # 69x69x64

  x = conv2d_bn(x, 96, (1, 1), strides=(1, 1), name="Stem_4")   # 69x69x96
  x = conv2d_bn(x, 128, (3, 3), strides=(2, 2), name="Stem_5")  # 35x35x128
  x = conv2d_bn(x, 192, (3, 3), strides=(1, 1), name="Stem_6")  # 35x35x192
  return x


def inception_block_basic(x,
                          filters_3x3_3x3_compress=64,
                          filters_3x3_3x3_1=96,
                          filters_3x3_3x3_2=96,
                          filters_3x3_compress=48,
                          filters_3x3=64,
                          filters_pool_1x1=64,
                          filters_1x1=64,
                          name=None):
  """Basic Inception block, factorizing 5x5 into 2 3x3's, resulting to 28%
  computations reduction.

  (see Inception paper, Figure 5 and principle 3 of Section 2)
  """
  inc_number = next(inception_block_basic.counter)
  if name is None:
    name_3x3_fact = f"IB_3x3_fact_{inc_number}"
    name_3x3 = f"IB_3x3_{inc_number}"
    name_pool = f"IB_pool_{inc_number}"
    name_1x1 = f"IB_1x1_{inc_number}"
    name = f"Inc_Basic_{inc_number}"
  else:
    name_3x3_fact = name + "_Conv_3x3_fact_"
    name_3x3 = name + "_Conv_3x3_"
    name_pool = name + "_AvgPool_"
    name_1x1 = name + "_Conv_1x1_"

  # ---------------------------------------------------------------------------
  conv3x3_3x3 = conv2d_bn(x,
                          filters_3x3_3x3_compress,
                          kernel=(1, 1),
                          name=name_3x3_fact + 'a')
  conv3x3_3x3 = conv2d_bn(conv3x3_3x3,
                          filters_3x3_3x3_1,
                          kernel=(3, 3),
                          name=name_3x3_fact + 'b',)
  conv3x3_3x3 = conv2d_bn(conv3x3_3x3,
                          filters_3x3_3x3_2,
                          kernel=(3, 3),
                          name=name_3x3_fact + 'c')
  # ---------------------------------------------------------------------------
  conv3x3 = conv2d_bn(x,
                      filters_3x3_compress,
                      kernel=(1, 1),
                      name=name_3x3 + 'a')
  conv3x3 = conv2d_bn(conv3x3,
                      filters_3x3,
                      kernel=(3, 3),
                      name=name_3x3 + 'b')
  # ---------------------------------------------------------------------------
  avgpool = AveragePooling2D((3, 3),
                             strides=(1, 1),
                             padding="same",
                             name=name_pool + 'a'
                             )(x)
  avgpool = conv2d_bn(avgpool,
                      filters_pool_1x1,
                      kernel=(1, 1),
                      name=name_pool + 'b')
  # ---------------------------------------------------------------------------
  conv1x1 = conv2d_bn(x,
                      filters_1x1,
                      kernel=(1, 1),
                      name=name_1x1)
  # ---------------------------------------------------------------------------

  iblock = concatenate([conv3x3_3x3, conv3x3, avgpool, conv1x1],
                       axis=S.channel_axis(),
                       name=name)
  return iblock


def inception_block_grid_compressor_1(x,
                                      filters_3x3_3x3_compress=64,
                                      filters_3x3_3x3_1=96,
                                      filters_3x3_3x3_2=96,
                                      filters_3x3_compress=332,
                                      filters_3x3=384,
                                      name=None):
  """Inception block that reduces the grid-size, while expands the filter size
  avoiding representation bottleneckes upon grid compression.

  - Here, padding is same instead of valid, in order to result in a 17x17 grid.

  (see Inception paper, Figure 10 and principle 1 in Section 2)
  """
  inc_number = next(inception_block_grid_compressor_1.counter)
  if name is None:
    name_3x3_3x3 = f"IGC1_3333_{inc_number}"
    name_3x3 = f"IGC1_33_{inc_number}"
    name_pool = f"IGC1_Pool_{inc_number}"
    name = f"Inc_Grid_Compressor_1_{inc_number}"
  else:
    name_3x3_3x3 = name + "_Conv_3333_"
    name_3x3 = name + "_Conv_33_"
    name_pool = name + "_MaxPool_"

  # ---------------------------------------------------------------------------
  conv3x3_3x3 = conv2d_bn(x,
                          filters_3x3_3x3_compress,
                          kernel=(1, 1),
                          name=name_3x3_3x3 + 'a')
  conv3x3_3x3 = conv2d_bn(conv3x3_3x3,
                          filters_3x3_3x3_1,
                          kernel=(3, 3),
                          name=name_3x3_3x3 + 'b')
  conv3x3_3x3 = conv2d_bn(conv3x3_3x3,
                          filters_3x3_3x3_2,
                          kernel=(3, 3),
                          strides=(2, 2),
                          name=name_3x3_3x3 + 'c')
  # ---------------------------------------------------------------------------
  conv3x3 = conv2d_bn(x,
                      filters_3x3_compress,
                      kernel=(1, 1),
                      name=name_3x3 + 'a')
  conv3x3 = conv2d_bn(conv3x3,
                      filters_3x3,
                      kernel=(3, 3),
                      strides=(2, 2),
                      name=name_3x3 + 'b')
  # ---------------------------------------------------------------------------
  maxpool = MaxPool2D((3, 3),
                      strides=(2, 2),
                      padding="same",
                      name=name_pool + 'a'
                      )(x)
  # ---------------------------------------------------------------------------

  iblock = concatenate(
    [conv3x3_3x3, conv3x3, maxpool],
    axis=S.channel_axis(),
    name=name
  )
  return iblock


def inception_block_1xn_nx1(x,
                            n=7,
                            filters_1nn11nn1_compress=128,
                            filters_1xn=128,
                            filters_1xn_nx1=128,
                            filters_1xn_nx1_1xn=128,
                            filters_1xn_nx1_1xn_nx1=192,
                            filters_1nn1_compress=128,
                            filters_1xn_single=128,
                            filters_1xn_nx1_single=192,
                            filters_pool_1x1=192,
                            filters_1x1=192,
                            name=None):
  """Inception block, factorizing into asymmetric convolutions, significantly
  decreasing the number of computations, while resulting in the same volume.

  (see Inception paper, Figure 6 and principle 3 in Section 2)"""
  inc_number = next(inception_block_1xn_nx1.counter)
  if name is None:
    name_1xn_nx1_1xn_nx1 = f"IAF_1nn11nn1_{inc_number}"
    name_1xn_nx1 = f"IAF_1nn1_{inc_number}"
    name_pool = f"IAF_Pool_{inc_number}"
    name_1x1 = f"IAF_1x1_{inc_number}"
    name = f"Inc_Assym_Fact_{inc_number}"
  else:
    name_1xn_nx1_1xn_nx1 = name + "_Conv_1nn11nn1_"
    name_1xn_nx1 = name + "_Conv_1nn1_"
    name_pool = name + "_AvgPool_"
    name_1x1 = name + "_Conv_1x1_"

  # ---------------------------------------------------------------------------
  conv_1xn_nx1_1xn_nx1 = conv2d_bn(x,
                                   filters_1nn11nn1_compress,
                                   kernel=(1, 1),
                                   name=name_1xn_nx1_1xn_nx1 + 'a')
  conv_1xn_nx1_1xn_nx1 = conv2d_bn(conv_1xn_nx1_1xn_nx1,
                                   filters_1xn,
                                   kernel=(1, n),
                                   name=name_1xn_nx1_1xn_nx1 + 'b')
  conv_1xn_nx1_1xn_nx1 = conv2d_bn(conv_1xn_nx1_1xn_nx1,
                                   filters_1xn_nx1,
                                   kernel=(n, 1),
                                   name=name_1xn_nx1_1xn_nx1 + 'c')
  conv_1xn_nx1_1xn_nx1 = conv2d_bn(conv_1xn_nx1_1xn_nx1,
                                   filters_1xn_nx1_1xn,
                                   kernel=(1, n),
                                   name=name_1xn_nx1_1xn_nx1 + 'd')
  conv_1xn_nx1_1xn_nx1 = conv2d_bn(conv_1xn_nx1_1xn_nx1,
                                   filters_1xn_nx1_1xn_nx1,
                                   kernel=(n, 1),
                                   name=name_1xn_nx1_1xn_nx1 + 'e')
  # ---------------------------------------------------------------------------
  conv_1xn_nx1 = conv2d_bn(x,
                           filters_1nn1_compress,
                           kernel=(1, 1),
                           name=name_1xn_nx1 + 'a')
  conv_1xn_nx1 = conv2d_bn(conv_1xn_nx1,
                           filters_1xn_single,
                           kernel=(1, n),
                           name=name_1xn_nx1 + 'b')
  conv_1xn_nx1 = conv2d_bn(conv_1xn_nx1,
                           filters_1xn_nx1_single,
                           kernel=(n, 1),
                           name=name_1xn_nx1 + 'c')
  # ---------------------------------------------------------------------------
  avgpool = AveragePooling2D((3, 3),
                             strides=(1, 1),
                             padding="same",
                             name=name_pool + 'a'
                             )(x)
  avgpool = conv2d_bn(avgpool,
                      filters_pool_1x1,
                      kernel=(1, 1),
                      name=name_pool + 'b')
  # ---------------------------------------------------------------------------
  conv1x1 = conv2d_bn(x,
                      filters_1x1,
                      kernel=(1, 1),
                      name=name_1x1)
  # ---------------------------------------------------------------------------

  iblock = concatenate([conv_1xn_nx1_1xn_nx1, conv_1xn_nx1, avgpool, conv1x1],
                       axis=S.channel_axis(),
                       name=name)
  return iblock


def inception_block_grid_compressor_2(x,
                                      n=7,
                                      filters_1xn_nx1_3x3_compress=192,
                                      filters_1xn=192,
                                      filters_1xn_nx1=192,
                                      filters_1xn_nx1_3x3=192,
                                      filters_3x3_compress=192,
                                      filters_3x3=320,
                                      name=None):
  """Inception block that reduces the grid-size, while expands the filter size
  avoiding representation bottleneckes upon grid compression.

  - The difference with the compressor_1 is that here the first stage of the
    factorization branch is factorized further, using 1x7 and 7x1 convolutions.

  (see Inception paper, Figure 10 and principle 1 in Section 2)
  """
  inc_number = next(inception_block_grid_compressor_2.counter)
  if name is None:
    name_nxn_3x3 = f"IGC2_1nn133_{inc_number}"
    name_3x3 = f"IGC2_33_{inc_number}"
    name_pool = f"IGC2_Pool_{inc_number}"
    name = f"Inc_Grid_Compressor_2_{inc_number}"
  else:
    name_nxn_3x3 = name + "_Conv_1nn133_"
    name_3x3 = name + "_Conv_33_"
    name_pool = name + "_MaxPool_"

  # ---------------------------------------------------------------------------
  conv_1xn_nx1_3x3 = conv2d_bn(x,
                               filters_1xn_nx1_3x3_compress,
                               kernel=(1, 1),
                               name=name_nxn_3x3 + 'a')
  conv_1xn_nx1_3x3 = conv2d_bn(conv_1xn_nx1_3x3,
                               filters_1xn,
                               kernel=(3, 3),
                               name=name_nxn_3x3 + 'b')
  conv_1xn_nx1_3x3 = conv2d_bn(conv_1xn_nx1_3x3,
                               filters_1xn_nx1,
                               kernel=(3, 3),
                               name=name_nxn_3x3 + 'c')
  conv_1xn_nx1_3x3 = conv2d_bn(conv_1xn_nx1_3x3,
                               filters_1xn_nx1_3x3,
                               kernel=(3, 3),
                               strides=(2, 2),
                               padding="valid",
                               name=name_nxn_3x3 + 'd')
  # ---------------------------------------------------------------------------
  conv3x3 = conv2d_bn(x,
                      filters_3x3_compress,
                      kernel=(1, 1),
                      name=name_3x3 + 'a')
  conv3x3 = conv2d_bn(conv3x3,
                      filters_3x3,
                      kernel=(3, 3),
                      strides=(2, 2),
                      padding="valid",
                      name=name_3x3 + 'b')
  # ---------------------------------------------------------------------------
  maxpool = MaxPool2D((3, 3),
                      strides=(2, 2),
                      name=name_pool
                      )(x)
  # ---------------------------------------------------------------------------

  iblock = concatenate(
    [conv_1xn_nx1_3x3, conv3x3, maxpool],
    axis=S.channel_axis(),
    name=name
  )
  return iblock


def inception_block_filter_expander(x,
                                    filters_3x3_1x3_3x1_compress=448,
                                    filters_3x3=318,
                                    filters_3x3_1x3=318,
                                    filters_3x3_3x1=318,
                                    filters_1x3_3x1_compress=384,
                                    filters_1x3=318,
                                    filters_3x1=318,
                                    filters_pool_1x1=128,
                                    filters_1x1=256,
                                    name=None):
  """Inception block that expands the filter size of the output resulting to
  more disentangled features.

  - Avoids the deterioration in expressiveness of the traditional pooling and,
    at the same time, decreases the amount of computations performed.
  - Used at the coarsest grid (8x8)

  (see Inception paper, Figure 7 and principle 2 of Section 2)
  """
  inc_number = next(inception_block_filter_expander.counter)
  if name is None:
    name_3x3_1x3_3x1 = f"IFE_331331_{inc_number}"
    name_1x3_3x1 = f"IFE_1331_{inc_number}"
    name_pool = f"IFE_Pool_{inc_number}"
    name_1x1 = f"IFE_1x1_{inc_number}"
    name = f"Inc_Filter_Exp_{inc_number}"
  else:
    name_3x3_1x3_3x1 = name + "_Conv_331331_"
    name_1x3_3x1 = name + "_Conv_1331_"
    name_pool = name + "_MaxPool_"
    name_pool = name + "_AvgPool_"
    name_1x1 = name + "_Conv_1x1_"

  # ---------------------------------------------------------------------------
  conv3x3 = conv2d_bn(x,
                      filters_3x3_1x3_3x1_compress,
                      kernel=(1, 1),
                      name=name_3x3_1x3_3x1 + 'a')
  conv3x3 = conv2d_bn(conv3x3,
                      filters_3x3,
                      kernel=(3, 3),
                      name=name_3x3_1x3_3x1 + 'b')
  conv3x3_1x3 = conv2d_bn(conv3x3,
                          filters_3x3_1x3,
                          kernel=(1, 3),
                          name=name_3x3_1x3_3x1 + 'c')
  conv3x3_3x1 = conv2d_bn(conv3x3,
                          filters_3x3_3x1,
                          kernel=(3, 1),
                          name=name_3x3_1x3_3x1 + 'd')
  conv_3x3_1x3_3x1 = concatenate([conv3x3_1x3, conv3x3_3x1],
                                 axis=S.channel_axis(),
                                 name=name_3x3_1x3_3x1)
  # ---------------------------------------------------------------------------
  conv1x3_3x1 = conv2d_bn(x,
                          filters_1x3_3x1_compress,
                          kernel=(1, 1),
                          name=name_1x3_3x1 + 'a')
  conv1x3 = conv2d_bn(conv1x3_3x1,
                      filters_1x3,
                      kernel=(1, 3),
                      name=name_1x3_3x1 + 'b')
  conv3x1 = conv2d_bn(conv1x3_3x1,
                      filters_3x1,
                      kernel=(3, 1),
                      name=name_1x3_3x1 + 'c')
  conv_1x3_3x1 = concatenate([conv1x3, conv3x1],
                             axis=S.channel_axis(),
                             name=name_1x3_3x1)
  # ---------------------------------------------------------------------------
  avgpool = AveragePooling2D((3, 3),
                             strides=(1, 1),
                             padding="same",
                             name=name_pool + 'a'
                             )(x)
  avgpool = conv2d_bn(avgpool,
                      filters_pool_1x1,
                      kernel=(1, 1),
                      name=name_pool + 'b')
  # ---------------------------------------------------------------------------
  conv1x1 = conv2d_bn(x,
                      filters_1x1,
                      kernel=(1, 1),
                      name=name_1x1)
  # ---------------------------------------------------------------------------

  iblock = concatenate(
    [conv_3x3_1x3_3x1, conv_1x3_3x1, avgpool, conv1x1],
    axis=S.channel_axis(),
    name=name
  )
  return iblock


def inception_v3_custom(include_top=True,
                        include_side_head=True,
                        training=None):
  """Creates the Inception-v3 architecture"""
  # counters as function attributes (used for layer naming)
  inception_block_basic.counter = count(0)
  inception_block_1xn_nx1.counter = count(0)
  inception_block_filter_expander.counter = count(0)
  inception_block_grid_compressor_1.counter = count(0)
  inception_block_grid_compressor_2.counter = count(0)
  conv2d_bn.counter = count(0)

  inshape, outshape = utils.inoutshapes()

  input_layer = Input(shape=inshape)                             # 54x54x6
  x = stem(input_layer)                                          # 25x25x192
  x = inception_block_basic(x, filters_1x1=32)                   # 25x25x256
  x = inception_block_basic(x)                                   # 25x25x288
  x = inception_block_basic(x)                                   # 25x25x288
  x = inception_block_grid_compressor_1(x)                       # 13x13x768
  x = inception_block_1xn_nx1(x,
                              n=7,
                              filters_1nn11nn1_compress=128,
                              filters_1xn=128,
                              filters_1xn_nx1=128,
                              filters_1xn_nx1_1xn=128,
                              filters_1xn_nx1_1xn_nx1=192,
                              filters_1nn1_compress=128,
                              filters_1xn_single=128,
                              filters_1xn_nx1_single=192)        # 13x13x768
  for _ in range(2):
    x = inception_block_1xn_nx1(x,
                                n=7,
                                filters_1nn11nn1_compress=160,
                                filters_1xn=160,
                                filters_1xn_nx1=160,
                                filters_1xn_nx1_1xn=160,
                                filters_1xn_nx1_1xn_nx1=192,
                                filters_1nn1_compress=160,
                                filters_1xn_single=160,
                                filters_1xn_nx1_single=192)      # 13x13x768
  for _ in range(2):
    x = inception_block_1xn_nx1(x,
                                n=7,
                                filters_1nn11nn1_compress=192,
                                filters_1xn=192,
                                filters_1xn_nx1=192,
                                filters_1xn_nx1_1xn=192,
                                filters_1xn_nx1_1xn_nx1=192,
                                filters_1nn1_compress=192,
                                filters_1xn_single=192,
                                filters_1xn_nx1_single=192)      # 13x13x768
  # ---------------------------------------------------------------------------
  # Auxiliary side-head
  if include_top and include_side_head:
    x1 = conv2d_bn(x, 1160, (1, 7), padding="valid")             # 13x7x1160
    x1 = conv2d_bn(x1, 1800, (7, 1), padding="valid")            # 7x7x1800
    x1 = conv2d_bn(x1, 3072, (3, 3), padding="valid")            # 5x5x3072
    x1 = conv2d_bn(x1, 2048, (1, 1), padding="valid")            # 5x5x2048
    x1 = conv2d_bn(x1, 4096, (3, 3), padding="valid")            # 3x3x4096
    x1 = conv2d_bn(x1, 3072, (1, 1), padding="valid")            # 3x3x3072
    x1 = conv2d_bn(x1, 8192, (3, 3), padding="valid")            # 1x1x8192
    x1 = Flatten()(x1)                                           # 8192
    x1 = Dense(10240)(x1)                                        # 10240
    x1 = BatchNormalization()(x1, training=training)
    x1 = Activation("relu")(x1)
    x1 = Dense(np.prod(outshape))(x1)                            # 7500
    x1 = Reshape(outshape)(x1)                                   # 50x50x3
  # ---------------------------------------------------------------------------
  x = inception_block_grid_compressor_2(x)                       # 6x6x1280
  x = inception_block_filter_expander(x)                         # 6x6x2048
  x = inception_block_filter_expander(x)                         # 6x6x2048
  x = conv2d_bn(x, 1540, (1, 1), padding="valid")                # 6x6x1540
  x = conv2d_bn(x, 4096, (3, 3), padding="valid")                # 3x3x4096
  x = conv2d_bn(x, 3072, (1, 1), padding="valid")                # 3x3x3072
  x = conv2d_bn(x, 8192, (4, 4), padding="valid")                # 3x3x8192
  if include_top:
    x = Flatten()(x)                                             # 8192
    x = Dense(10240, activation="relu")(x)                       # 10240
    x = Dense(np.prod(outshape), activation="linear")(x)         # 7500
    x = Reshape(outshape)(x)                                     # 50x50x3

  if include_top and include_side_head:
    x = [x1, x]

  model = Model(input_layer, x)
  return model


###############################################################################
class InceptionV3CustomExperimental(keras.Model):
  """Creates the Inception-V3 architecture.

  - Subclasses keras.Model, in order to use BatchNormalization with moving
    statistics.

  More details:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
  https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class
  """
  def __init__(self, include_top=True,
               include_side_head=True,
               activation="elu"):
    super(InceptionV3, self).__init__(name="inception_v3")
    self.include_top = include_top
    self.include_side_head = include_side_head
    self.inshape, self.outshape = utils.inoutshapes()
    self.stem = stem
    self.inception_block_basic = inception_block_basic
    self.inception_block_grid_compressor_1 = \
        inception_block_grid_compressor_1
    self.inception_block_1xn_nx1 = inception_block_1xn_nx1
    self.inception_block_grid_compressor_2 = \
        inception_block_grid_compressor_2
    self.inception_block_filter_expander = inception_block_filter_expander

  def call(self, inputs, training=True):                                 # 50x50x6
    """Creates the Inception-v3 architecture"""
    # counters as function attributes (used for layer naming)
    inception_block_basic.counter = count(0)
    inception_block_1xn_nx1.counter = count(0)
    inception_block_filter_expander.counter = count(0)
    inception_block_grid_compressor_1.counter = count(0)
    inception_block_grid_compressor_2.counter = count(0)
    conv2d_bn.counter = count(0)
    conv2d_bn.training = training

    x = self.stem(inputs, shape=self.inshape)                            # 25x25x192
    x = self.inception_block_basic(x, filters_1x1=32)                    # 25x25x256
    x = self.inception_block_basic(x)                                    # 25x25x288
    x = self.inception_block_basic(x)                                    # 25x25x288
    x = self.inception_block_grid_compressor_1(x)                        # 13x13x768
    x = self.inception_block_1xn_nx1(x,
                                     n=7,
                                     filters_1nn11nn1_compress=128,
                                     filters_1xn=128,
                                     filters_1xn_nx1=128,
                                     filters_1xn_nx1_1xn=128,
                                     filters_1xn_nx1_1xn_nx1=192,
                                     filters_1nn1_compress=128,
                                     filters_1xn_single=128,
                                     filters_1xn_nx1_single=192)         # 13x13x768
    for _ in range(2):
      x = self.inception_block_1xn_nx1(x,
                                       n=7,
                                       filters_1nn11nn1_compress=160,
                                       filters_1xn=160,
                                       filters_1xn_nx1=160,
                                       filters_1xn_nx1_1xn=160,
                                       filters_1xn_nx1_1xn_nx1=192,
                                       filters_1nn1_compress=160,
                                       filters_1xn_single=160,
                                       filters_1xn_nx1_single=192)       # 13x13x768
    for _ in range(2):
      x = self.inception_block_1xn_nx1(x,
                                       n=7,
                                       filters_1nn11nn1_compress=192,
                                       filters_1xn=192,
                                       filters_1xn_nx1=192,
                                       filters_1xn_nx1_1xn=192,
                                       filters_1xn_nx1_1xn_nx1=192,
                                       filters_1nn1_compress=192,
                                       filters_1xn_single=192,
                                       filters_1xn_nx1_single=192)       # 13x13x768
    # -------------------------------------------------------------------------
    # Auxiliary side-head
    if self.include_top and self.include_side_head:
      x1 = conv2d_bn(x, 1160, (1, 7), padding="valid")                   # 13x7x1160
      x1 = conv2d_bn(x1, 1800, (7, 1), padding="valid")                  # 7x7x1800
      x1 = conv2d_bn(x1, 3072, (3, 3), padding="valid")                  # 5x5x3072
      x1 = conv2d_bn(x1, 2048, (1, 1), padding="valid")                  # 5x5x2048
      x1 = conv2d_bn(x1, 4096, (3, 3), padding="valid")                  # 3x3x4096
      x1 = conv2d_bn(x1, 3072, (1, 1), padding="valid")                  # 3x3x3072
      x1 = conv2d_bn(x1, 8192, (3, 3), padding="valid")                  # 1x1x8192
      x1 = Flatten()(x1)                                                 # 8192
      x1 = Dense(10240)(x1)                                              # 10240
      x1 = BatchNormalization()(x1, training=training)
      x1 = Activation("relu")(x1)
      x1 = Dense(np.prod(self.outshape), activation="linear")(x1)        # 7500
      x1 = Reshape(self.outshape)(x1)                                    # 50x50x3
    # -------------------------------------------------------------------------
    x = self.inception_block_grid_compressor_2(x)                        # 6x6x1280
    x = self.inception_block_filter_expander(x)                          # 6x6x2048
    x = self.inception_block_filter_expander(x)                          # 6x6x2048
    x = conv2d_bn(x, 1540, (1, 1), padding="valid")                      # 6x6x1540
    x = conv2d_bn(x, 4096, (3, 3), strides=(2, 2), padding="valid")      # 3x3x4096
    x = conv2d_bn(x, 3072, (1, 1), padding="valid")                      # 3x3x3072
    x = conv2d_bn(x, 8192, (3, 3), padding="valid")                      # 3x3x8192
    if self.include_top:
      x = Flatten()(x)                                                   # 8192
      x = Dense(10240, activation="relu")(x)                             # 10240
      x = Dense(np.prod(self.outshape), activation="linear")(x)          # 7500
      x = Reshape(self.outshape)(x)                                      # 50x50x3
    if self.include_top and self.include_side_head:
      return [x1, x]
    return x
