# utils.py is part of SmartFlow
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
"""Some auxiliary stuff."""

from datetime import datetime, timedelta
from functools import wraps
import os
import shutil
from timeit import default_timer as timer

from mattflow import config as conf
import numpy as np
import tensorflow as tf

from smartflow import backend as S, io


def data_paths():
  """Generates data_path and drop_iters_path."""
  s = S.data_shape()
  data_path = os.path.join(
    os.getcwd(),
    "mattflow_data",
    f"mattflow_data_{s[0]}x{s[1]}x{s[2]}x{s[3]}.npy"
  )
  drop_iters_path = os.path.join(
    os.getcwd(),
    "mattflow_data",
    f"drop_iters_list_{s[0]}x{s[1]}x{s[2]}x{s[3]}.npy"
  )
  return data_path, drop_iters_path


def validate_kwargs(kwargs, allowed_kwargs):
  """Checks that all keyword arguments are in the set of allowed keys."""
  for kwarg in kwargs:
    if kwarg not in allowed_kwargs:
      raise TypeError(f"Keyward argument '{kwarg}' is not understood.")


def checkpoint_path(epoch: int = None) -> str:
  """Returns the checkpoing file path.

  It should be a "format"able string, which takes the parameter "epoch", so
  that tensorflow can internally name the checkpoint appropriately.
  """
  checkpoint_file_name = "cp-{epoch:04d}.ckpt"
  if epoch is not None:
    checkpoint_file_name = checkpoint_file_name.format(epoch=epoch)
  checkpoint_dir = "training_checkpoints"
  child_dir(checkpoint_dir)
  cp_path = os.path.join(os.getcwd(), checkpoint_dir, checkpoint_file_name)
  return cp_path


def today_and_now():
  """Formats datetime.now() to be used in file names."""
  # replace : with - for cross-os file name format
  tan = str(datetime.now())[:19].replace(':', '-').replace(' ', '_')
  return tan


def delete_prev_runs_data(models=True,
                          checkpoints=True,
                          cache=True,
                          saved_datasets=True,
                          predictions=True,
                          videos=True,
                          logs=True,
                          figs=True,
                          input_pred_gt=True,
                          preprocessing_visualizations=True):
    """Deletes data of previous models (for debugging)."""
    # prompt user to press enter
    input("Deleting data from previous runs. Press ENTER to continue...")
    cwd = os.getcwd()

    directories = {
      "saved_models": models,
      "training_checkpoints": checkpoints,
      "predictions": predictions,
      "cache": cache,
      "saved_datasets": saved_datasets,
      "logs": logs,
      "input-pred-gt_visualizations": input_pred_gt,
      "preprocessing_visualizations": preprocessing_visualizations
    }
    for directory, dir_condition in directories.items():
      dir_path = os.path.join(cwd, directory)
      if dir_condition and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

    extensions = {
      ".png": figs,
      ".jpg": figs,
      ".mp4": videos,
      ".gif": videos,
      ".log": logs
    }
    extensions_to_remove = []
    for exte, exte_condition in extensions.items():
      if exte_condition:
        extensions_to_remove.append(exte)
    extensions_to_remove = tuple(extensions_to_remove)
    for f in os.listdir(cwd):
        if f.endswith(extensions_to_remove):
            os.remove(f)


def child_dir(*consecutive_dirs):
  """Creates a directory under the current working directory."""
  dirname = os.path.join(*consecutive_dirs)
  try:
      if os.path.isdir(os.path.join(os.getcwd(), dirname)):
          pass
      else:
          os.makedirs(os.path.join(os.getcwd(), dirname))
  except OSError:
      print(f"Unable to create ./{dirname} directory")
  return dirname


def print_duration(start, end, process):
  """Prints the duration of a process."""
  process_name = {
    "main": "Total",
    "_train": "Training",
    "preprocessing": "Pre-processing",
    "simulate": "Prediction",
    "animate": "Post-processing",
    "create_animation": "Post-processing",
  }
  if process in process_name:
    process = process_name[process]
  prefix = f"{process.capitalize()} duration"
  duration = timedelta(seconds=end - start)
  print(f"{prefix:-<30}{duration}"[:40])


def norm_and_train_test_val_split(ds):
  """Currently unused

    - the label of each case (frame) is the next case (next frame)

    Args:
      ds (Dataset) : all simulated states of the fluid

    Returns
      train_ds, test_ds, val_ds (tuple of Dataset's)
  """
  split_ratio = S.split_ratio()

  len_ds = conf.MAX_ITERS
  len_train = int(split_ratio * split_ratio * len_ds)
  len_test = len_ds - len_train - 1
  len_val = int((1 - split_ratio) * split_ratio * len_ds)

  train_input = ds.take(len_train)
  train_labels = ds.skip(1).take(len_train)
  # .map(tf.image.per_image_standardization)
  # .mpa(labda x: tf.numpy_function
  train_ds = tf.data.Dataset.zip((train_input, train_labels))

  val_input = ds.skip(len_train).take(len_val)
  val_labels = ds.skip(len_train + 1).take(len_val)
  val_ds = tf.data.Dataset.zip((val_input, val_labels))

  test_input = ds.skip(len_train + len_val).take(len_test)
  test_labels = ds.skip(len_train + len_val + 1).take(len_test)
  test_ds = tf.data.Dataset.zip((test_input, test_labels))

  return train_ds, test_ds, val_ds


def train_test_split_pandas(df):
  split_ratio = S.split_ratio()

  num_train_cases = int(split_ratio * len(df))
  train_ds = df.iloc[: num_train_cases]
  train_labels = df.iloc[1: num_train_cases + 1]

  test_ds = df.iloc[num_train_cases: -1]
  test_labels = df.iloc[num_train_cases + 1:]
  return (train_ds, train_labels), (test_ds, test_labels)


def train_test_split_numpy(data):
  split_ratio = S.split_ratio()

  num_train_cases = int(split_ratio * len(data))
  train_ds = data[: num_train_cases].reshape(num_train_cases, 1, conf.Nx, conf.Ny)
  train_labels = data[1: num_train_cases + 1].reshape(num_train_cases, conf.Nx * conf.Ny)

  test_ds = data[num_train_cases: -1]
  test_ds = test_ds.reshape(len(test_ds), 1, conf.Nx, conf.Ny)
  test_labels = data[num_train_cases + 1:].reshape(len(test_ds), conf.Nx * conf.Ny)
  return (train_ds, train_labels), (test_ds, test_labels)


def time_this(f):
    """function timer decorator

    - Uses wraps to preserve the metadata of the decorated function
      (__name__ and __doc__)
    - prints the duration

    Args:
        f(funtion)      : the function to be decorated

    Returns:
        wrap (callable) : returns the result of the decorated function
    """
    assert callable(f)

    @wraps(f)
    def wrap(*args, **kwargs):
        start = timer()
        result = f(*args, **kwargs)
        end = timer()
        print_duration(start, end, f.__name__)
        return result
    return wrap


def adjust_data_to_subset(data, drop_iters, subset, update_max_iters=True):
  """Crops the data to te subset limits."""
  start = subset[0]
  stop = subset[1]
  if update_max_iters:
    conf.MAX_ITERS = stop - start
  # crop data
  data = data[start: stop]
  # crop drop_iters
  limits = np.searchsorted(drop_iters, (start, stop))
  drop_iters = drop_iters[limits[0]: limits[1]]
  # copy, in order to write into
  drop_iters = drop_iters.copy()
  # offset with the start of the subset
  drop_iters = drop_iters - start
  # print(f"len: {len(drop_iters)}  last: {drop_iters[-1]}")
  return data, drop_iters


def inoutshapes():
  """Generates input and output shapes."""
  if S.use_boundary_conditions():
    inshape = [conf.Nx + 2 * conf.Ng, conf.Ny + 2 * conf.Ng]
  else:
    inshape = [conf.Nx, conf.Ny]
  inshape.insert(S.channel_axis() - 1,
                 S.num_channels() * S.num_x_frames())
  outshape = [conf.Nx, conf.Ny]
  outshape.insert(S.channel_axis() - 1,
                  S.num_channels())
  return tuple(inshape), tuple(outshape)


def describe(x):
  """Prints the statistics of the data.

  Example:
  >>> describe("mattflow_data/mattflow_data_30000x3x80x80.npy")
            min    max    mean    std
  height:   0.039 14.567  1.321  0.303
  flux_x: -92.233 87.368 -0.008  0.746
  flux_y: -88.906 78.308 -0.000  0.734
  total : -92.233 87.368  0.437  0.887
  >>> describe("mattflow_data/mattflow_data_5000x1x90x90.npy")
            min    max    mean    std
  height:   0.000 20.505  0.675  0.778
  """
  if isinstance(x, str):
    x = io.read_data(x)
  elif isinstance(x, np.ndarray):
    pass
  else:
    # x.shape     : NCHW
    # x_list.shape: N2CHW (every example is a 2-tuple of features and labels)
    x_list = list(x.as_numpy_iterator())
    x_list = [list(i) for j in x_list for i in j]
    x = np.array(x_list)[:, 0]

  print("          min    max    mean    std")
  print(f"height: {x[:, 0].min():>{7}.{3}f} {x[:, 0].max():.3f}"
        f" {x[:, 0].mean():>{6}.{3}f} {x[:, 0].std():>{6}.{3}f}")
  if len(x[0]) == 3:
    print(f"flux_x: {x[:, 1].min():>{7}.{3}f} {x[:, 1].max():.3f}"
          f" {x[:, 1].mean():>{6}.{3}f} {x[:, 1].std():>{6}.{3}f}")
    print(f"flux_y: {x[:, 2].min():>{7}.{3}f} {x[:, 2].max():.3f}"
          f" {x[:, 2].mean():>{6}.{3}f} {x[:, 2].std():>{6}.{3}f}")
    print(f"total : {x.min():>{7}.{3}f} {x.max():.3f}"
          f" {x.mean():>{6}.{3}f} {x.std():>{6}.{3}f}")
