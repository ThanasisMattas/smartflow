# experimental.py is part of SmartFlow
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
"""Subclassing keras.Model and keras.layers.Layer, in order to use
BatchNormalization with moving statistics.

  - Currently unused.
"""

from tensorflow import keras
from tensorflow.keras.layers import (Activation,
                                     add,
                                     BatchNormalization,
                                     Conv2D)

import smartflow.backend as S
from smartflow.archs.resnet import conv2d_bn


class InceptionResnet(keras.Model):
  """A deep CNN architecture, using elements of Inception-v3 and ResNet

  - Residual networks address the degradation of the training accuracy that
    occures at a certain network depth onwards. The phenomenon is not
    correlated with vanishing/exploding gradients or overfitting; rather, it is
    an accuracy saturation related with the depth of the network.
    (See He et al.)
  - Subclasses keras.Model, in order to use BatchNormalization with moving
    statistics. More details:
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
    https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class

  NOTE: Keep same padding for leading layers because the boundary
        conditions constitute very important information.
  """
  def __init__(self, include_top=True, **kwargs):
    super(InceptionResnet, self).__init__(**kwargs)
    self.include_top = include_top

  def call(self, inputs, training=True):
    pass

class Conv2DBN(keras.layers.Layer):

  def __init__(self,
               conv2d_config=None,
               bn_config=None,
               activation_config=None,
               **kwargs):
    super(Conv2DBN, self).__init__(**kwargs)
    self.conv2d = Conv2D(conv2d_config)
    self.bn = BatchNormalization(bn_config)
    self.actitvation = Activation(activation_config)

  def call(self, inputs, training=None):
    pass


class Residual(keras.layers.Layer):
  """Adds a tensor (residual) to the current layer, addressing the degradation
  of learning accuracy in deep networks. (See He et al.)

  Args:
    identity (bool) :  Whether the residual is added as it is or a 'projection'
                       of it on the receiving tensor. (See He et al., Section
                       4, "Identity vs. Projection Shortcuts")
  Returns:
    output (tensor) :  Same shape with the receiving tensor
  """
  def __init__(self, identity=True, **kargs):
    super(Residual, self).__init__(**kargs)
    self.identity = identity

  def call(self, inputs):
    if not isinstance(inputs, (list, tuple)) and len(inputs) != 2:
      raise ValueError('A `Residual` layer should be called '
                       'on exactly 2 inputs')
    x = inputs[0]
    residual = inputs[1]
    if x.shape != residual.shape or not self.identity:
      x_filters = x.shape[S.channel_axis()]
      # (shape[2] will always be the h or w of the grid)
      if x.shape[2] == residual.shape[2]:
        residual = conv2d_bn(residual, x_filters, (1, 1))
      elif x.shape[2] == residual.shape[2] - 2:
        residual = conv2d_bn(residual,
                             x_filters,
                             (3, 3))
      elif x.shape[2] == residual.shape[2] // 2:
        residual = conv2d_bn(residual,
                             x_filters,
                             (3, 3),
                             strides=(2, 2))
      elif x.shape[2] == (residual.shape[2] - 3) // 2 + 1:
        residual = conv2d_bn(residual,
                             x_filters,
                             (3, 3),
                             strides=(2, 2),
                             padding="valid")
      elif x.shape[2] == residual.shape[2] // 3:
        residual = conv2d_bn(residual,
                             x_filters,
                             (3, 3),
                             strides=(3, 3))
      elif x.shape[2] == (residual.shape[2] - 3) // 3 + 1:
        residual = conv2d_bn(residual,
                             x_filters,
                             (3, 3),
                             strides=(3, 3),
                             padding="valid")
      else:
        raise ValueError("The current tensor must have the same size with the"
                         " residual tensor or the output size after applying"
                         " a 3x3 convolution at the residual tensor.")
    output = add([x, residual])
    return output
