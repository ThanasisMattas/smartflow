# io.py is part of SmartFlow
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
"""Handles io operations."""

import os

import numpy as np
import tensorflow as tf

from smartflow import backend as S, utils
# from numpy.lib.format import open_memmap


def read_data(data_path=None,
              drop_iters_path=None,
              frame_freq=None,
              data_subset=None):
  if (data_path is None) and (drop_iters_path is None):
    data_path, drop_iters_path = utils.data_paths()

  # The consecutive frames, evaluated via the numerical simulation.
  data = np.load(data_path, mmap_mode='r')
  # An array with the simulation iterations at which a new drop fell.
  drop_iters = np.load(drop_iters_path, mmap_mode='r')

  # Keep 1st (height) out of the 3 channels (height, flux_x, flux_y).
  if S.num_channels() == 1:
    channel_slicing_obj = [slice(None)] * 4
    channel_slicing_obj[S.channel_axis()] = slice(0, S.num_channels())
    channel_slicing_obj = tuple(channel_slicing_obj)
    data = data[channel_slicing_obj]

  # Omit (1 or 2) intermediate frames, to result in bigger differences from
  # frame to frame.
  # Example:
  # data = [0, 1, 2, 3, 4, 5, 6, 7] |    data = [0, 3, 6]
  # drop_iters = [0, 3, 4, 7]       | => drop_iters = [0, 3]
  # frame_freq = 3                  |
  if frame_freq is not None:
    data = data[::frame_freq]
    drop_iters = drop_iters[drop_iters % frame_freq == 0]

  # Slice a subset, if one is provided.
  if data_subset is not None:
    data, drop_iters = utils.adjust_data_to_subset(data,
                                                   drop_iters,
                                                   data_subset)
  return data, drop_iters


def read_datasets():
  """Reads train, val and test datasets from file."""
  inshape, outshape = utils.inoutshapes()
  element_spec = (tf.TensorSpec(shape=(None,) + inshape,
                                dtype=tf.float32,
                                name=None),
                  tf.TensorSpec(shape=(None,) + (np.prod(outshape),),
                                dtype=tf.float32,
                                name=None))

  train_ds = tf.data.experimental.load(
    os.path.join(os.getcwd(), "saved_datasets", "train_dataset"),
    element_spec
  )
  val_ds = tf.data.experimental.load(
    os.path.join(os.getcwd(), "saved_datasets", "val_dataset"),
    element_spec
  )
  test_ds = tf.data.experimental.load(
    os.path.join(os.getcwd(), "saved_datasets", "test_dataset"),
    element_spec
  )
  return train_ds, val_ds, test_ds


def save_predictions(h_hist, t_hist, its, file_format="memmap"):
  """supported formats: ["memmap", "csv"]"""
  try:
    right_now = utils.today_and_now()
    pred_dir = utils.child_dir("predictions")
    pred_h_name = f"predicted_h_{right_now}"
    pred_h_path = os.path.join(pred_dir, pred_h_name)
    pred_t_name = f"predicted_t_{right_now}"
    pred_t_path = os.path.join(pred_dir, pred_t_name)
    if file_format == "memmap":
      np.save(f"{pred_h_path}.npy", h_hist)
      np.save(f"{pred_t_path}.npy", t_hist)
    elif file_format == "csv":
      np.savetxt(f"{pred_h_path}.csv",
                 h_hist.reshape(h_hist.shape[0], -1),
                 delimiter=',')
      np.savetxt(f"{pred_t_path}.csv",
                 t_hist.reshape(t_hist.shape[0], -1),
                 delimiter=',')
  except FileNotFoundError:
    print("FileNotFoundError: Could not save predictions")
