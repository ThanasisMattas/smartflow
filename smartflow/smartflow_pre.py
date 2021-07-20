# smartflow_pre.py is part of SmartFlow
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
"""Creates and pre-processes train-val-test datasets.

Types of datasets:
 1. DSequence
    Derives from keras.utils.Sequence, in order to load one batch at a time,
    reading from a numpy memmap. This is prefered when the dataset cannot fit
    in the memory.
    [guide](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence)
 2. DSet
    Derives from keras.data.Dataset, in order to utilize dataset performance
    optimization (required for TPU empoyment on the google colab cloud).
    [TPU guide](https://www.tensorflow.org/guide/tpu)
    [Dataset guide](https://www.tensorflow.org/guide/data_performance)

Types of normalization:
  1. On-device, using normalization_layer().
  2. At dataset creation, using the Normalizer class.
"""

import abc
# from itertools import count
from functools import partial
import gc
import os
import random

from mattflow import bcmanager, config as conf
import numpy as np
# from sklearn.model_selection import train_test_split_np

import tensorflow as tf
from tensorflow import keras

from smartflow import (backend as S,  # noqa: F401
                       io,
                       smartflow_post,
                       utils)
from smartflow.utils import time_this


# [channelwise_mean, channelwise_var]
_NORMALIZATION_LAYER_WEIGHTS = None


def set_normalization_layer_weights(data_mean, data_var):
  """Repeats mean and var arrays for the multiple input frames per example.

  final num channels = frame_num_channels x frames_per_example

  Example:
  data_mean = array([1.5, 12.3, 11.7])
  input frames per example (num_x_frames()) = 2
  Then, the channelwise mean of the input whould be:
  array([1.5, 12.3, 11.7, 1.5, 12.3, 11.7])

  Args:
    data_mean (ndarray) : channelwise mean of the data
    data_var (ndarray)  : channelwise variance of the data
  """
  global _NORMALIZATION_LAYER_WEIGHTS
  extended_data_mean = np.array([])
  extended_data_var = np.array([])
  for _ in range(S.num_x_frames()):
    extended_data_mean = np.append(extended_data_mean, data_mean)
    extended_data_var = np.append(extended_data_var, data_var)
  _NORMALIZATION_LAYER_WEIGHTS = [extended_data_mean, extended_data_var]


def normalization_layer_weights():
  return _NORMALIZATION_LAYER_WEIGHTS


def normalization_layer(**kwargs):
  """Adapted to the train dataset on the channel axis."""
  input_shape = kwargs.get("input_shape", utils.inoutshapes()[0])

  NormalizationLayer = keras.layers.experimental.preprocessing.Normalization(
    axis=kwargs.pop("axis", S.channel_axis()),
    **kwargs
  )
  # The layer's weights must be instantiated, before explicitly setting them.
  dummy_adapter = np.full(
      input_shape,
      1,
      dtype=S.dtype("numpy")
  )[np.newaxis, :]
  NormalizationLayer.adapt(dummy_adapter)
  NormalizationLayer.set_weights(normalization_layer_weights())
  return NormalizationLayer


class Normalizer:
  """Normalizes a dataset or a batch, using mean-std normalization

  NOTE: 2nd and 3rd channels have max~=65 and by definition mean=0 and std=1
        across the dataset, but they heavily vary between individual frames.
  TODO: When tf supports python 3.9, @property can be chained with @classmethod.

  Usage:
  >>> # The distribution to adpat to
  >>> train_x = np.random.rand(10, 3, 2, 2)
  >>> batch_x = np.random.rand(2, 3, 2, 2)
  >>> Normalizer.adapt(train_x)
  >>> batch_x = Normalizer()(batch_x)

  Alternatively, the statistics can be set explititly as attributes:
  >>> nomralizer = Normalizer()
  >>> nomralizer._mean = 1.5
  >>> nomralizer._std = 0.2
  >>> nomralizer._max = 3.5
  >>> batch_x = normalizer(batch_x)

  Args:
    x (ndarray)      : dataset or batch to normalize
    stats_type (str) : [optional] the statistics type to use (Overrides the
                       existing ones, if adapt was already used.)
                       Options:
                         - frame
                         - batch
                         - channelwise_frame
                         - channelwise_batch
                         - channelwise_train_ds
                         - zero_surface_level (Brings surface level to zero)
                         - custom (Using train_ds statistics):
                           - (height - height.mean) / 0.8
                             #           ~-> [- 1.9, 12.0], mean=0, std=0.25
                           - flux_x * 15 / (flux_x.max + e)
                             # (~ / 3.9) ~-> [-13.1, 15,0], mean=0, std=0.25
                           - flux_y * 15 / (flux_y.max + e)
                             # (~ / 4.3) ~-> [-15.0, 14,7], mean=0, std=0.24

  Returns:
    x (ndarray)      : normalized
  """
  _epsilon = 1e-3

  @property
  def _mean(self):
    return self.mean_

  @_mean.setter
  def _mean(self, value):
    self.mean_ = value

  @property
  def _std(self):
    return self.std_

  @_std.setter
  def _std(self, value):
    self.std_ = value

  @property
  def _max(self):
    return self.max_

  @_max.setter
  def _max(self, value):
    self.max_ = value

  @classmethod
  def adapt(cls, x, stats_type=None):
    """Assigns the weights of Normalizer with the statistics of the input."""
    cls.stats_type = stats_type
    aggr_axes = [0, 1, 2, 3]
    aggr_axes.remove(S.channel_axis())
    aggr_axes = tuple(aggr_axes)

    cls._mean = x.mean(axis=aggr_axes)
    cls._std = x.std(axis=aggr_axes)
    cls._max = np.abs(x).max(axis=aggr_axes)

  def _per_frame_normalization(self, x, stats_type):
    """Applies per-frame mean/std normalization."""
    if stats_type == "channelwise_frame":
      aggr_axes = [0, 1, 2]
      # S.channel_axis() refers to the dataset shape, so the examples dimension
      # is subtracted.
      aggr_axes.remove(S.channel_axis() - 1)
      aggr_axes = tuple(aggr_axes)
    elif stats_type == "frame":
      aggr_axes = None
    else:
      raise ValueError(
        "'stats_type' should be set to one of ['frame', 'channelwise_frame']"
      )

    for i in range(len(x)):
      x[i] = np.apply_along_axis(np.subtract,
                                 S.channel_axis() - 1,
                                 x[i],
                                 x[i].mean(axis=aggr_axes))
      x[i] = np.apply_along_axis(np.divide,
                                 S.channel_axis() - 1,
                                 x[i],
                                 x[i].std(axis=aggr_axes) + self._epsilon)
    return x

  def _per_channel_normalization(self, x):
    """Applies per-channel mean/std normalization."""
    for c in range(len(x[0])):
      try:
        x[:, c] = (x[:, c] - self._mean[c]) / (self._std[c] + self._epsilon)
      except IndexError:
        __import__('ipdb').set_trace(context=9)
    return x

  def __call__(self, x, stats_type=None):
    allowed_stats_types = [
      None,
      "frame",
      "batch",
      "train_ds",
      "channelwise_frame",
      "channelwise_batch",
      "channelwise_train_ds",
      "zero_surface_level",
      "custom"
    ]
    if stats_type is None:
      stats_type = self.stats_type

    if stats_type in [None, "train_ds", "channelwise_train_ds", "custom"]:
      if (self._mean is None) or (self._std is None) or (self._max is None):
        raise ValueError("Normalizer must get adapted on the train_ds, using"
                         " Normalizer.adapt(), before using it as callable"
                         f" with stats_type={stats_type}.")

    if stats_type in [None, "train_ds"]:
      train_ds_mean = self._mean.mean()
      train_ds_std = np.sqrt((self._std ** 2).sum() / S.num_channels())
      return (x - train_ds_mean) / (train_ds_std + self._epsilon)
    elif stats_type in ["frame", "channelwise_frame"]:
      return self._per_frame_normalization(x, stats_type=stats_type)
    elif stats_type == "batch":
      return (x - x.mean()) / (x.std() + self._epsilon)
    elif stats_type == "channelwise_batch":
      Normalizer.adapt(x)
    elif stats_type == "zero_surface_level":
      x[:, 0, :, :] -= conf.SURFACE_LEVEL
      return x
    elif stats_type == "custom":
      self._std[0] = 0.8
      self._std[1:] = self._max[1:] / 15
    elif stats_type == "channelwise_train_ds":
      # The statistics are already set.
      pass
    else:
      raise ValueError(f"'stats_type' must be one of: {allowed_stats_types}")

    return self._per_channel_normalization(x)


class SmartFlowDS:
  """Base class for SmartFlow datasets.

  - Preprocessing occurs at this stage, in order to relieve the GPU while trai-
  ning, sacrifizing the portability of the model. Thus, all necessary processes
  will be implemented again at inference level (using the staticmethods).
  - Normalization is done using train_dataset, per batch or frame statistics.
  - Both NCHW and NHWC formats are supported.
  - Input frames have updated ghost cells, but labels dont't (those cells will
  not be predicted). Therefore, after inference the prediction will be padded
  with ghost cells, before it is fed back to the model for the next prediction.
  Boundary conditions are required by the numerical scheme, they can be easily
  calculated upon inference and they will hopefully provide some valuable
  information to the model.
  - Time-steps at which a drop fell cannot be used as labels, because there is
  no way to infer when and where a new drop will fall, using information from
  the previous state of the fluid. However, those frames can perfectly be used
  as input and the next ones as labels.
  """
  def __init__(self,
               data,
               drop_iters,
               batch_size,
               ds_type,
               preprocessing_vis=True):
    self.data = data
    self.drop_iters = drop_iters
    self.batch_size = batch_size
    self.ds_type = ds_type
    self.preprocessing_vis = preprocessing_vis

  @staticmethod
  def augment(x, y, rate=0.25, set_seed=None):
    """Randomly flips, rotates and shuffles a train batch.

    - Supports both NCHW and NHWC formats.
    - x and y are stacked together, in order to have the same flip status.

    TODO: shuffle the whole train ds and disable shuffling here
    NOTE: Don't apply on validation and test datasets, in order to preserve a
          coherent reference.

    Args:
      x (3D ndarray) : features
      y (3D ndarray) : labels
      rate (float)   : the rate at wich random flips occure (defaults to 0.5)
      set_seed (int) : defaults to None

    Returns:
      x, y (tuple)   : features and labels augmented

    """
    if set_seed:
      random.seed(set_seed)
    # x and y are stacked together, in order to have the same flip status.
    channel_stack = np.concatenate((x, y), axis=S.channel_axis())
    # Set h and w axes for a training example.
    if S.channel_axis() == 1:
      h = 1
      w = 2
    else:
      h = 0
      w = 1

    for i in range(len(channel_stack)):
      # random vertical flip
      if random.random() < rate:
        channel_stack[i] = np.flip(channel_stack[i], axis=h)
      # random horizontal flip
      if random.random() < rate:
        channel_stack[i] = np.flip(channel_stack[i], axis=w)
      # random rotation (90 or -90 deg)
      if random.random() < rate:
        channel_stack[i] = np.rot90(channel_stack[i],
                                    k=random.choice([1, 3]),
                                    axes=(h, w))
    # split back
    x[...], y[...] = np.split(channel_stack,
                              [x.shape[S.channel_axis()]],
                              axis=S.channel_axis())
    return x, y

  def _visualize_preprocessing(self,
                               x,
                               y,
                               idx=1,
                               show_fig=False,
                               save_fig=True):
    """Visualize that the dataset is shuffled, xy pairs have the same flip and
    rotation status, and drop-frame labels are removed. (Close each pop-up
    figure, to plot the next one.)
    """
    # Force it to run only on the 1st epoch.
    vis_dir = "preprocessing_visualizations"
    if os.path.isdir("preprocessing_visualizations"):
      return

    channels = {0: "Height", 1: "Flux-x", 2: "Flux-y"}

    if self.ds_type == "train" and idx == 1:
      random_frames = random.choices(range(len(x)), k=5)
      for frame in random_frames:
        for channel in range(S.num_channels()):
          Zx = x[frame,
                 channel,
                 conf.Ng: -conf.Ng,
                 conf.Ng: -conf.Ng]
          if S.num_x_frames() == 1:
            Zprev = None
          else:
            Zprev = x[frame,
                      S.num_channels() + channel,
                      conf.Ng: -conf.Ng,
                      conf.Ng: -conf.Ng]

          smartflow_post.plot_example(frame,
                                      Zx=Zx,
                                      Zprev=Zprev,
                                      Zgt=y[frame, channel],
                                      show_fig=show_fig,
                                      save_fig=save_fig,
                                      title=f"{channels[channel]} channel",
                                      save_dir=vis_dir)

  @staticmethod
  def update_height_ghost_cells(U):
    """Applies boundary conditions only on height channel.

    See mattflow.bcmanager.update_ghost_cells() for more info.

    Args:
      U (ndarray) :  (1, Nx + 2 * Ng, Ny + 2 * Ng)

    Returns:
      U (ndarray) :  updated
    """
    # left wall (0 <= x < Ng)
    U[0, :, :conf.Ng] = np.flip(U[0, :, conf.Ng: 2 * conf.Ng], 1)
    # right wall (Nx + Ng <= x < Nx + 2Ng)
    U[0, :, conf.Nx + conf.Ng: conf.Nx + 2 * conf.Ng] \
        = np.flip(U[0, :, conf.Nx: conf.Nx + conf.Ng], 1)
    # top wall (0 <= y < Ng)
    U[0, :conf.Ng, :] = np.flip(U[0, conf.Ng: 2 * conf.Ng, :], 0)
    # bottom wall (Ny + Ng <= y < Ny + 2Ng)
    U[0, conf.Ny + conf.Ng: conf.Ny + 2 * conf.Ng, :] \
        = np.flip(U[0, conf.Ny: conf.Ny + conf.Ng, :], 0)
    return U

  @staticmethod
  def apply_bc(x):
    if S.num_channels() == 1:
      return SmartFlowDS.update_height_ghost_cells(x)
    return bcmanager.update_ghost_cells(x)

  @staticmethod
  def apply_boundary_conditions(x):
    """Pads the mesh with the ghost cells and updates them.

    - If num_x_frames > 1, the x frames are already concatenated.
    """
    Ng = conf.Ng
    c = S.num_channels()
    x = np.pad(x, ((0, 0), (0, 0), (Ng, Ng), (Ng, Ng)))

    for example in range(len(x)):
      for frame in range(S.num_x_frames()):
        x[example, frame * c: (frame + 1) * c, :, :] = SmartFlowDS.apply_bc(
          x[example, frame * c: (frame + 1) * c, :, :]
        )
    return x

  def _concatenate_multiple_x_frames(self, x):
    """Concatenates the consecutive num_x_frames, which constitute the features
    of each example.

    - Every training example will have concatenated <num_x_frames> frames.
    - The oldest frame goes last.

    Example:
    x = [0, 1, 2, 3, 4]
    num_x_frames = 2
    --> x_concat = [[1, 0], [2, 1], [3, 2], [4, 3]]
    (and the correspoding labels: [2, 3, 4, 5])
    """
    x_concat = x[: - S.num_x_frames() + 1]
    window = len(x_concat)
    for frame in range(1, S.num_x_frames()):
      x_concat = np.concatenate(
        [x[frame: frame + window],
         x_concat],
        axis=S.channel_axis()
      )
    return x_concat

  def _remove_new_drop_labels(self,
                              x,
                              y,
                              y_start=None,
                              batch_drop_iters=None):
    """Removes the xy pairs for which the label is a new-drop frame.

    Args:
      x (ndarray)             :  features (1-batch or mini-batch)
      y (ndarray)             :  labels (1-batch or mini-batch)
      y_start(int)            :  absolute dataset index of the 1st label
      batch_drop_iters (list) :  drop_iters that fall into the current batch

    Returns:
      x, y (tuple)            :  new-drop labels removed
    """
    if batch_drop_iters == []:
      # It is called from a batch with no new-drop labels.
      return x, y

    if batch_drop_iters is None:
      drop_iters = self.drop_iters
    else:
      drop_iters = batch_drop_iters
    if y_start is None:
      # It is called on an 1-batch.
      y_start = S.num_x_frames()

    # Offset drop_iters with the absolute starting index of the labels.
    drop_iters_offset = [i - y_start for i in drop_iters]

    mask = np.ones(len(y), dtype=bool)
    mask[drop_iters_offset] = False
    x = x[mask]
    y = y[mask]
    # Uncomment this to print the drop-frames excluded by each batch
    #   print(f"batch: {idx:2d}"
    #         f"  drop_count: {next(self._remove_new_drop_labels.counter)}"
    #         f"  drop_it: {batch_drop_iters[0] + y_start}")
    gc.collect()
    return x, y

  @abc.abstractmethod
  def _allocate_xy(self, *args, **kwargs):
    """Allocates features and labels from the dataset.

    - Every example (frame) constitutes the label for the previous one and the
      features for the next.
    - Normalization occurs here before concatenation, in case that the input
      frames are more than one (smartflow.backend.num_x_frames() > 1).
    """
    raise NotImplementedError

  def _set_steps_per_epoch(self, x):
    """Sets steps_per_epoch for each dataset type, stored at the backend."""
    S.set_steps_per_epoch({self.ds_type: int(-(-len(x) // self.batch_size))})

  def _preprocess(self, x, y, idx=1):
    """Wrapper for data augmentation and boundary condition update.

    Args:
      x (ndarray)   :  features (1-batch or mini-batch)
      y (ndarray)   :  labels (1-batch or mini-batch)
      idx (int)     :  the index of the current batch in the Sequence
      y_start (int) :  the absolute starting raw-data index of the batch

    Returns:
      x, y (tuple)  :
        Sizes:
          x: (ds/batch_size, c * num_x_frames, Nx + 2 * Ng, Nx + 2 * Ng)
          y: (ds/batch_size, c, Nx, Ny)
    """
    # 1. Random flip & rotation.
    if self.ds_type == "train":
      x[...], y[...] = self.augment(x, y)

    # 2. Pad and udtate ghost cells.
    if S.use_boundary_conditions():
      x = self.apply_boundary_conditions(x)

    # 4. Visualize that the batch is shuffled, xy pairs have the
    # same flip and rotation status, and drop-frame labels are removed.
    if self.preprocessing_vis:
      self._visualize_preprocessing(x, y, idx)

    return x, y


class DSequence(SmartFlowDS, keras.utils.Sequence):
  """Creates a dataset as a Sequence, loading only one batch at a time.

  - This subclass is prefered when the dataset cannot fit into the memory and,
  therefore, only one batch at a time is loaded from a numpy memmap.
  - Each batch is a tuple: (batch_x, batch_y) with sizes:
      batch_x: (batch_size, c * num_x_frames, Nx + 2 * Ng, Nx + 2 * Ng)
      batch_y: (batch_size, c, Nx, Ny)
  """
  def __init__(self, *args, **kwargs):
    super(DSequence, self).__init__(*args, **kwargs)
    self._set_steps_per_epoch(self.data)

  def __len__(self):
    return -(-len(self.data) // self.batch_size)

  def _xy_batches_absolute_limits(self, idx):
    num_x_frames = S.num_x_frames()

    # Absolute start and stop dataset indexes of the batch_x.
    # If num_x_frames > 1, x_start and x_stop refer to the 1st x_frame.
    x_start = idx * self.batch_size
    x_stop = (idx + 1) * self.batch_size

    # In case that the last batch doesn't have enough frames, return.
    if len(self.data) - x_start < num_x_frames:
      return
    # In case that the last batch is smaller than the batch_size, trim it.
    x_stop = min(x_stop, len(self.data) - num_x_frames - 1)
    # x_stop = min(x_stop, len(self.data))

    # Absolute start and stop dataset indexes of the batch_y.
    y_start = x_start + num_x_frames
    y_stop = x_stop + num_x_frames
    return (x_start, x_stop), (y_start, y_stop)

  def _allocate_xy(self, idx):
    num_x = S.num_x_frames()
    num_c = S.num_channels()
    (x_start, x_stop), (y_start, y_stop) = \
        self._xy_batches_absolute_limits(idx)

    # 1. Bring batch_x, batch_y chunks into the memory.
    # - batch_x will have num_x_frames concatenated for each entry
    #   (if num_x_frames > 1)
    # - num_channels: 1: height
    #                 3: height, flux_x, flux_y
    batch_x = self.data[x_start: x_stop + num_x - 1, 0: num_c].copy()
    batch_y = self.data[y_start: y_stop, 0: num_c].copy()

    # 2. Normalize the batch (before concatenation)
    if normalization_layer_weights() is None:
      batch_x[...] = Normalizer()(batch_x)

    # 3. Concatenate the multiple frames that constitute the input of each example.
    if num_x > 1:
      batch_x = self._concatenate_multiple_x_frames(batch_x)

    # 4. Remove pairs for which the label is a new-drop frame.
    #    (There is no way to infer that a drop will fall.)
    batch_x, batch_y = self._remove_new_drop_labels(batch_x,
                                                    batch_y,
                                                    y_start=y_start)

    return (x_start, x_stop), (y_start, y_stop), (batch_x, batch_y)

  def _batch_drop_iters(self, y_start, y_stop):
    """Allocates the new-drop iterations that fall into the batch.

    Args:
      y_start (int) :  absolute dataset index of the start of the y_batch
      y_stop (int)  :  absolute dataset index of the end of the y_batch

    Returns:
      batch_drop_iters (list)
    """
    batch_drop_iters = []
    # Check whether a drop-frames falls into the batch.
    start_stop_idxs = np.searchsorted(self.drop_iters, (y_start, y_stop))
    if start_stop_idxs[1] == start_stop_idxs[0]:
      # There is no drop in this batch, we 're fine.
      pass
    else:
      # Between these indexes there is at least one drop_iter.
      # When min iters between drops > batch_size, it should be just one frame.
      for i in range(start_stop_idxs[0], start_stop_idxs[1]):
        batch_drop_iters.append(self.drop_iters[i])
    return batch_drop_iters

  def _remove_new_drop_labels(self,
                              batch_x,
                              batch_y,
                              y_start=None,
                              batch_drop_iters=None):
    y_stop = y_start + len(batch_y)
    batch_drop_iters = self._batch_drop_iters(y_start, y_stop)
    return super(DSequence, self)._remove_new_drop_labels(batch_x,
                                                          batch_y,
                                                          y_start,
                                                          batch_drop_iters)

  def __getitem__(self, idx):
    """Generates a batch-tuple, (batch_x, batch_y).

    Args:
      idx (int) :  the index of the current batch in the Sequence

    Returns:
      batch_x, batch_y (tuple) :
        Sizes:
          batch_x: (batch_size, c * num_x_frames, Nx + 2 * Ng, Nx + 2 * Ng)
          batch_y: (batch_size, c, Nx, Ny)
    """
    # 1. Allocate x, y batches.
    (x_start, x_stop), (y_start, y_stop), (batch_x, batch_y) = \
        self._allocate_xy(idx)
    # 2. Preprocessing
    batch_x, batch_y = self._preprocess(batch_x, batch_y, idx, y_start)

    # 3. Flatten each example here, to relieve the training process.
    batch_y = np.reshape(batch_y, (len(batch_y), -1))

    # Uncomment these to print batch info, during training.
    # if idx % 15 == 0:
    #   print("batch mean  std   max")
    # print(f"{idx:>5}"
    #       f" {batch_x.mean():.3f}"
    #       f" {batch_x.std():.3f}"
    #       f" {batch_x.max():.3f}")

    return batch_x, batch_y


class DSet(SmartFlowDS, tf.data.Dataset):
  """Allocates and pre-processes a train, val or test dataset.

  - This subclass is prefered when the dataset does fit into the memory or TPUs
  will be deployed on the google colab cloud.
  """
  def __init__(self, *args, **kwargs):
    # Used for big datasets (>20k examples)
    self.dump_memmap = kwargs.pop("dump_memmap", False)
    super(DSet, self).__init__(*args, **kwargs)

  def _inputs(self):
    return [self.data, self.drop_iters]

  @property
  def element_spec(self):
    return self._structure

  def _allocate_xy(self):
    # 1. Bring x, y chunks in the memory.
    # The last frame is the last label.
    x = self.data[: -1].copy()
    # The first <num_x_frames> frames constitute the input of the first example.
    y = self.data[S.num_x_frames():].copy()

    # 2. Normalize x (before concatenation)
    if normalization_layer_weights() is None:
      x[...] = Normalizer()(x)

    # 3. Concatenate the multiple num_x_frames.
    if S.num_x_frames() > 1:
      x = self._concatenate_multiple_x_frames(x)

    # 4. Remove pairs for which the label is a new-drop frame.
    #    (There is no way to infer that a drop will fall.)
    x, y = self._remove_new_drop_labels(x, y)
    return x, y

  def _set_steps_per_epoch(self, x):
    super(DSet, self)._set_steps_per_epoch(x)
    print(f"{self.ds_type.capitalize():<5} dataset built."
          f"  Length: {len(x):<5},"
          f" steps-per-epoch: {S.steps_per_epoch(self.ds_type):>3}")

  def _allocate_xy_and_preprocess(self):
    """Wrapper used to get out of scope and release the memory.

    In case of a big dataset that doesn't fit into the memory, x and y arrays
    are saved locally to .npy files, in order to be loaded as memmap. Thus the
    corresponding memory can be released, when this function gets out of scope.
    """
    # 1. Allocate x, y batches + normalization.
    x, y = self._allocate_xy()

    # 2. Preprocessing
    x, y = self._preprocess(x, y)

    # 3. Flatten each example here, to relieve the training process.
    y = np.reshape(y, (len(y), -1))

    self._set_steps_per_epoch(x)

    if self.dump_memmap:
      np.save(f"{self.ds_type}_x.npy", x)
      np.save(f"{self.ds_type}_y.npy", y)
    else:
      return x, y

  def __new__(cls, *args, **kwargs):
    dset = super(DSet, cls).__new__(cls)
    dset.__init__(*args, **kwargs)

    # Allocate and preprocess x and y, as numpy arrays.
    if dset.dump_memmap:
      dset._allocate_xy_and_preprocess()
      x = np.load(f"{dset.ds_type}_x.npy", mmap_mode='r')
      y = np.load(f"{dset.ds_type}_y.npy", mmap_mode='r')
      os.remove(f"{dset.ds_type}_x.npy")
      os.remove(f"{dset.ds_type}_y.npy")
    else:
      x, y = dset._allocate_xy_and_preprocess()

    # Dataset creation
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if dset.ds_type == "train":
      ds = ds.shuffle(conf.MAX_ITERS, reshuffle_each_iteration=True)
    ds = ds.batch(dset.batch_size)
    if conf.MAX_ITERS <= 12000:
      ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


@time_this
def preprocess(batch_size=32,
               split_ratio=0.8,
               dataset_as_sequence=False,
               normalize_at_training=True,
               stats_type=None,
               use_boundary_conditions=True,
               save_datasets=False,
               preprocessing_vis=True,
               dump_memmap=False,
               datasets=["train", "val", "test"],
               **kwargs):
  """Generates and preprocesses all train, val and test datasets.

  A dataset is a collection of 2-tuples, (features, labels).

  Args:
    batch_size (int)               : raw simulation data
    split_ratio (float)            : the train_ds portion of the data
    dataset_as_sequence (bool)     : whether to use the DSequence (True) or the
                                     DSet subclass (False)
    normalize_at_training (bool)   : whether to use normalization_layer() (True)
                                     or the Normalizer class (False)
    stats_type (str)               : the statistics to use at normalization
    use_boundary_conditions (bool) : pad and update the domain with ghost cells
    preprocessing_vis (bool)       : plot some examples to visualize that input
                                     and label frames share the same flip and
                                     rotation orientation
    save_datasets (bool)           : using tf.data.experimental.save()
    dump_memmap (bool)             : used for big datasets (>20k examples)
    datasets (list)                : the dataset types to create
                                     default: ["train", "val", "test"]
    kwargs (dict)                  : passing kwargs to io.read_data()

  Returns:
    train_ds, val_ds, test_ds (tuple) : DSequence or DSet objects
  """
  # SmartFlowDS._remove_new_drop_labels.counter = count(0)
  allowed_kwargs = {
    "data_path",
    "drop_iters_path",
    "frame_freq",
    "data_subset"
  }
  utils.validate_kwargs(kwargs, allowed_kwargs)

  S.set_batch_size(batch_size)
  S.set_split_ratio(split_ratio)
  S.set_use_boundary_conditions(use_boundary_conditions)

  data, drop_iters = io.read_data(**kwargs)

  len_ds = conf.MAX_ITERS
  len_train = int(split_ratio * split_ratio * len_ds)
  len_val = int((1 - split_ratio) * split_ratio * len_ds)
  len_test = len_ds - len_train - len_val - S.num_x_frames() - 1

  if normalize_at_training:
    # In case of on-device normalization with a Normalization Layer
    # - Currently, the drawback is that some augmentation processes, such as
    #   flip and rotation are not implemented for on-device normalization and
    #   they still have to run at the dataset creation stage.
    # - In case of multiple input frames for each example, the channelwise
    #   statistics will be repeated accordingly.
    aggr_axes = [0, 1, 2, 3]
    aggr_axes.remove(S.channel_axis())
    aggr_axes = tuple(aggr_axes)

    data_mean = data[:len_train].mean(axis=aggr_axes)
    data_var = data[:len_train].var(axis=aggr_axes)
    set_normalization_layer_weights(data_mean, data_var)
  else:
    # In case of normalization upon dataset generation
    Normalizer.adapt(x=data[:len_train], stats_type=stats_type)

  tvt = {
    "train": {
      "limits": (0, len_train),
      "dataset": None,
    },
    "val": {
      "limits": (len_train, len_train + len_val),
      "dataset": None,
    },
    "test": {
      "limits": (len_train + len_val, len_train + len_val + len_test),
      "dataset": None,
    }
  }

  ds_subclass = DSequence if dataset_as_sequence else DSet
  crop_data = partial(utils.adjust_data_to_subset,
                      data=data,
                      drop_iters=drop_iters,
                      update_max_iters=False)
  for ds in tvt:
    if ds in datasets:
      tvt[ds]["dataset"] = ds_subclass(*crop_data(subset=tvt[ds]["limits"]),
                                       batch_size=batch_size,
                                       ds_type=ds,
                                       preprocessing_vis=preprocessing_vis,
                                       dump_memmap=dump_memmap
                                       )
      if save_datasets:
        ds_path = utils.child_dir("saved_datasets", f"{ds}_dataset")
        tf.data.experimental.save(tvt[ds]["dataset"], ds_path)

  return (tvt["train"]["dataset"],
          tvt["val"]["dataset"],
          tvt["test"]["dataset"])
