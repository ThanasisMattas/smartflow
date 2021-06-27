# backend.py is part of SmartFlow
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
"""SmartFlow API backend"""

import logging
import os

from mattflow import config as conf, utils as matt_utils
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from smartflow import utils


_DTYPE = None
_BATCH_SIZE = None
_SPLIT_RATIO = None
# Both NCHW & NHWC are supported
_IMAGE_DATA_FORMAT = None
_CHANNEL_AXIS = None
# 1 channel  : height
# 3 channels : height, flux_x, flux_y
_NUM_CHANNELS = None
# (iters, channels, Nx + 2 * Ng, Nx + 2 * Ng)
_DATA_SHAPE = None
# Number of x frames, used as input
_NUM_X_FRAMES = None
# Used with BatchNormalization layers to trigger moving statistics
_TRAINING = None
_ACTIVATION = None
_KERNEL_INITIALIZER = None
# With respect to the dataset type (e.g. {"train_ds": 400, "val_ds": 80}).
_STEPS_PER_EPOCH = {}
_USE_BOUNDARY_CONDITIONS = None
_NUM_EXAMPLES = None


def set_steps_per_epoch(entry: dict):
  global _STEPS_PER_EPOCH
  _STEPS_PER_EPOCH = dict(_STEPS_PER_EPOCH, **entry)


def steps_per_epoch(ds_type: str) -> int:
  return _STEPS_PER_EPOCH[ds_type]


def set_dtype(value: str):
  allowed_dtypes = ["float32", "float64"]
  if value in allowed_dtypes:
    global _DTYPE
    _DTYPE = value
  else:
    raise ValueError(f"dtype should be one of {allowed_dtypes}")


def dtype(platform: str):
  if platform == "numpy":
    return np.dtype(_DTYPE)
  elif platform == "tensorflow":
    if _DTYPE == "float32":
      return tf.dtypes.float32
    elif _DTYPE == "float64":
      return tf.dtypes.float64
  else:
    raise ValueError("'platform' argument should be on of"
                     " ['numpy', 'tensorflow'].")


def num_examples():
  return _NUM_EXAMPLES


def set_num_examples(value):
  global _NUM_EXAMPLES
  _NUM_EXAMPLES = value


def batch_size():
  return _BATCH_SIZE


def split_ratio():
  return _SPLIT_RATIO


def channel_axis():
  return _CHANNEL_AXIS


def image_data_format():
  return _IMAGE_DATA_FORMAT


def data_shape():
  return _DATA_SHAPE


def num_x_frames():
  return _NUM_X_FRAMES


def num_channels():
  return _NUM_CHANNELS


def set_batch_size(value):
  global _BATCH_SIZE
  _BATCH_SIZE = value


def set_split_ratio(value):
  global _SPLIT_RATIO
  _SPLIT_RATIO = value


def set_training(value):
  global _TRAINING
  _TRAINING = value


def training():
  return _TRAINING


def set_activation(value):
  global _ACTIVATION
  _ACTIVATION = value


def activation():
  return _ACTIVATION


def kernel_initializer():
  return _KERNEL_INITIALIZER


def set_kernel_initializer(value):
  global _KERNEL_INITIALIZER
  _KERNEL_INITIALIZER = value


def set_use_boundary_conditions(value: bool):
  global _USE_BOUNDARY_CONDITIONS
  _USE_BOUNDARY_CONDITIONS = value


def use_boundary_conditions() -> bool:
  return _USE_BOUNDARY_CONDITIONS


def set_mattflow_config(nx, max_x, **kwargs):
  """Sets both mattflow and smartflow backends."""
  allowed_kwargs = {
    "ny",
    "max_y",
    "max_iters",
    "ng",
    "dtype",
    "mode",
    "surface_level",
    "rotation",
    "fps",
    "frame_save_freq",
    "consecutive_frames",
    "image_data_format",
    "num_x_frames",
    "num_channels"
  }
  utils.validate_kwargs(kwargs, allowed_kwargs)

  set_dtype(kwargs.pop("dtype", "float32"))
  conf.DTYPE = dtype("numpy")

  matt_utils.preprocessing(Nx=nx,
                           Ny=kwargs.pop("ny", nx),
                           Ng=kwargs.pop("ng", 2),
                           max_x=max_x,
                           min_x=-max_x,
                           max_y=kwargs.get("max_y", max_x),
                           min_y=-kwargs.pop("max_y", max_x))

  conf.MAX_ITERS = kwargs.pop("max_iters", 20000)
  set_num_examples(conf.MAX_ITERS)
  global _DATA_SHAPE
  _DATA_SHAPE = (num_examples(), 3, conf.Nx, conf.Ny)
  configure_channels(**kwargs)

  conf.MODE == kwargs.pop("mode", "drops")
  conf.SURFACE_LEVEL = kwargs.pop("surface_level", 1)
  conf.ROTATION = kwargs.pop("rotation", False)
  conf.FPS = kwargs.pop("fps", 15)
  conf.FRAME_SAVE_FREQ = kwargs.pop("frame_save_freq", 1)
  conf.CONSECUTIVE_FRAMES = kwargs.pop("consecutive_frames", 1)


def configure_channels(image_data_format="channels_first",
                       num_x_frames=2,
                       num_channels=3):
  """Configures the channelwise info.

  Args:
    image_data_format (str) :  Both NCHW and NHWC formats are supported.
                               Defaults to NCHW, for better performance with
                               NVIDIA cuDNN. To convert to NHWC:
                               data = np.moveaxis(data, 1, 3)
    num_x_frames (int)          :  Number of input frames (defaults to 2)
    num_channes (int)       :  1: only height channel
                               3: height, flux_x, flux_y channels (default)
  """
  global _CHANNEL_AXIS
  global _NUM_CHANNELS
  global _NUM_X_FRAMES
  global _IMAGE_DATA_FORMAT
  keras.backend.set_image_data_format(image_data_format)
  print(f"\nSetting image data format: {keras.backend.image_data_format()}\n")

  if image_data_format == "channels_first":
    _CHANNEL_AXIS = 1
  else:
    _CHANNEL_AXIS = 3
  _IMAGE_DATA_FORMAT = image_data_format
  _NUM_X_FRAMES = num_x_frames
  _NUM_CHANNELS = num_channels


def set_tf_loglevel(level):
  """Toggle between logging.INFO and logging.FATAL."""
  if level >= logging.FATAL:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  elif level >= logging.ERROR:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  elif level >= logging.WARNING:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
  else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
  logging.getLogger('tensorflow').setLevel(level)


def set_print_options():
  np.set_printoptions(suppress=True, formatter={"float": "{: .3f}".format})
  pd.options.display.float_format = "{:,.3f}".format


def set_tf_on_gpu(memory_limit=4040):
  tf.config.list_physical_devices("GPU")
  physical_devices = tf.config.list_physical_devices("GPU")
  tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
  tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(
      memory_limit=memory_limit
    )]
  )
