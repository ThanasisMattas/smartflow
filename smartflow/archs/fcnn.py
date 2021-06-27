# smartflow_fcnn.py is part of SmartFlow
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
"""Creates Fully Connected NN models"""

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from smartflow import backend as S, utils

class Normalizer(keras.layers.Layer):
  """normalizes the input: (x - mean) / std"""

  def __init__(self, **kwargs):
    super(Normalizer, self).__init__(**kwargs)
    self.norm_layer = preprocessing.Normalization
    self.channelwise_mean, self.channelwise_std = S.channelwise_train_stats()

  def call(self, inputs):
    normlayer = self.norm_layer()
    normlayer.mean = self.channelwise_mean.sum()
    normlayer.variance = self.channelwise_std.sum()
    return normlayer(inputs)


def fcnn():
  """creates a Fully Connected NN"""
  inshape, outshape = utils.inoutshapes()

  model = keras.Sequential([
    layers.Reshape((1, np.prod(inshape)), input_shape=inshape),
    # Normalizer(),
    layers.Dense(5120, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(5120, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(np.prod(outshape)),
  ], name="fcnn")

  return model
