# smartflow_post.py is part of SmartFlow
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
"""Handles the post processing of the model's predictions"""

import os

import numpy as np
import matplotlib.pyplot as plt
from mattflow import config as conf, mattflow_post

from smartflow import backend as S, smartflow_pre, utils
from smartflow.utils import time_this


def plot_loss(hist, save_fig=True, show_fig=False):
  """plots loss vs epochs for train and val datasets

  Args:
    hist (History) : the returned History object of Model.fit()
  """
  plt.plot(hist.history['loss'], label='train_loss')
  try:
    plt.plot(hist.history['val_loss'], label='val_loss')
  except KeyError:
    pass
  plt.ylim([0, 1.5])
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  if save_fig:
    plt.savefig("loss.png")
  if show_fig:
    plt.show()
  plt.close()


def subplot(X, Y, Z, fig, fig_config, sub_title="height", style="wireframe"):
    sub = fig.add_subplot(fig_config, projection="3d")
    sub.view_init(30, 55)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    sub.set_zlim([-0.5, 3.5])
    plt.title(sub_title, y=0.91, fontsize=10)
    if style == "water":
      sub.plot_surface(X, Y, Z,
                       rstride=1, cstride=1, linewidth=0,
                       color=(0.251, 0.643, 0.875, 0.95),
                       shade=True, antialiased=False)
    elif style == "wireframe":
      sub.plot_wireframe(X, Y, Z, rstride=2, cstride=2, linewidth=1)
    return sub


def plot_example(it,
                 Zx,
                 Zprev=None,
                 Zpred=None,
                 Zgt=None,
                 style="wireframe",
                 show_fig=False,
                 save_fig=True,
                 title=None,
                 save_dir=None):
  """Plots input, predicted and/or ground truth frames (for debugging).

  It supports plotting just Zx, or any combination of Zx with Zprev, Zx_pred
  and Zgt.

  Args:
    Zx (2D ndarray)    :  one height state on a xy grid
    Zprev (2D ndarray) :  the previous state
                          (defaults to None)
    Zpred (2D ndarray) :  the predicted height state (on the next time-step)
                          (defaults to None)
    Zgt (2D ndarray)   :  the ground truth height state (defaults to None)
    stlye (str)        :  options: ["water", "wireframe"]
    show_fig (bool)
    save_fig (bool)
  """
  X, Y = np.meshgrid(np.arange(Zx.shape[0]), np.arange(Zx.shape[1]))
  fig = plt.figure(figsize=(10, 5.625), dpi=160)  # 1600x900
  if title is None:
    fig.suptitle(f"iter: {it}", y=0.93)
  else:
    fig.suptitle(title, y=0.93)
  # plt.axis('off')

  # Set figure configuration
  sub_x_config = 111
  if not ((Zpred is not None) or (Zgt is not None)):
    if Zprev is not None:
      sub_prev_config = 121
      sub_x_config = 122
  elif (Zpred is not None) and (Zgt is not None):
    if Zprev is not None:
      sub_prev_config = 221
      sub_x_config = 222
      sub_pred_config = 223
      sub_gt_config = 224
    else:
      sub_x_config = 131
      sub_pred_config = 132
      sub_gt_config = 133
  else:
    if Zprev is not None:
      sub_prev_config = 131
      sub_x_config = 132
      last_sub_config = 133
    else:
      sub_x_config = 121
      last_sub_config = 122
    if Zpred is not None:
      sub_pred_config = last_sub_config
    elif Zgt is not None:
      sub_gt_config = last_sub_config

  # Generate the subplots
  sub_x = subplot(X, Y, Zx, fig, sub_x_config, f"Current Frame\niter: {it}")
  if Zprev is not None:
    sub_prev = subplot(X, Y, Zprev, fig, sub_prev_config,
                       f"Previous Frame\niter: {it - 1}")
  if Zpred is not None:
    sub_pred = subplot(X, Y, Zpred, fig, sub_pred_config,
                       f"Prediction\niter: {it + 1}")
  if Zgt is not None:
    sub_gt = subplot(X, Y, Zgt, fig, sub_gt_config,
                     f"Ground Truth\niter: {it + 1}")

  if save_fig:
    if save_dir is None:
      save_dir = "input-pred-gt_visualizations"
    utils.child_dir(save_dir)
    if title is None:
      fig_name = f"it_{it:0>5}.png"
    else:
      fig_name = f"it_{it:0>5}_{title}.png"
    plt.savefig(os.path.join(os.getcwd(), save_dir, fig_name))
  if show_fig:
    plt.show()
  plt.close(fig)


def plot_input_pred_gt(model,
                       idx,
                       data=None,
                       show_fig=False,
                       save_fig=True):
  """Plots input, predicted and ground truth frames of a given time-step.

  Args:
    data (memmap, str or pathlike) :  (if path, use '.npy' format)
    model (Sequential)             :  the trained model
    idx (int or tuple)             :  the example idx or range of idxes
    show_fig (bool)
    save_fig (bool)
  """
  if data is None:
    data_path = utils.data_paths()[0]
    data = np.load(data_path, mmap_mode='r')
  elif isinstance(data, str):
    data = np.load(data, mmap_mode='r')

  if isinstance(idx, int):
    if idx == len(data) - 1:
      print(f"There is no next frame to predict for input: {idx}"
            f" [len(data): {len(data)}].")
      return

    # Allocate & preprocess frames
    x = data[idx, 0: S.num_channels(), :, :].copy()
    gt = data[idx + 1, 0: S.num_channels(), :, :]
    Zx = x[0].copy()
    Zprev = None

    x = np.expand_dims(x, axis=0)
    if smartflow_pre.normalization_layer_weights() is None:
      x = smartflow_pre.Normalizer()(x)

    if S.num_x_frames() > 1:
      if idx == 0:
        print(f"There is no previous frame for iteration {idx}.")
        return

      x_prev = data[idx - 1, 0: S.num_channels(), :, :].copy()
      Zprev = x_prev[0].copy()
      x_prev = np.expand_dims(x_prev, axis=0)

      if smartflow_pre.normalization_layer_weights() is None:
        x_prev = smartflow_pre.Normalizer()(x_prev)

      x = np.concatenate([x, x_prev], axis=S.channel_axis())

    if S.use_boundary_conditions():
      x = smartflow_pre.SmartFlowDS.apply_boundary_conditions(x)

    # Predict
    pred = model.predict(x)
    # In case of multiple output heads, take the output of the last one.
    if isinstance(pred, list):
      pred = pred[-1]

    # Un-flatten the prediction
    pred = np.reshape(pred, (1, S.num_channels(), conf.Nx, conf.Ny))

    # Plot
    plot_example(it=idx,
                 Zx=Zx,
                 Zprev=Zprev,
                 Zpred=pred[0][0],
                 Zgt=gt[0],
                 show_fig=show_fig,
                 save_fig=save_fig)

  elif isinstance(idx, (list, tuple, np.ndarray)):
    for i in (idx):
      plot_input_pred_gt(model=model,
                         idx=i,
                         data=data,
                         show_fig=show_fig,
                         save_fig=save_fig)


def peek_data(data=None, channel=0, **kwargs):
  """Genarates an intermittent animation of the data.

  It shows <frames_per_interval> frames every <frequency> frames.
  """
  allowed_kwargs = {
    "num_intervals",
    "frames_per_interval",
    "dpi",
    "fps",
    "show_animation",
    "save_animation"
  }
  num_intervals = kwargs.pop("num_intervals", 10)
  frames_per_interval = kwargs.pop("frames_per_interval", 30)
  # Store previous settings.
  prev_fps = conf.FPS
  prev_dpi = conf.DPI
  kwargs["dpi"] = kwargs.get("dpi", 50)
  kwargs["fps"] = kwargs.get("fps", 35)
  kwargs["show_animation"] = kwargs.get("show_animation", True)
  kwargs["save_animation"] = kwargs.get("save_animation", False)

  mattflow_configured = ((conf.Nx is not None)
                         and (conf.dx is not None)
                         and (conf.CX is not None))
  if data is None:
    if not mattflow_configured:
      raise TypeError("Please configure MattFlow,"
                      " using 'smartflow.backend.set_config()'.")
    data_path = utils.data_paths()[0]
    data = np.load(data_path, mmap_mode='r')
  if isinstance(data, str):
    data = np.load(data, mmap_mode='r')

  if not mattflow_configured:
    data_shape = data.shape
    S.set_mattflow_config(nx=data_shape[2],
                          ny=data_shape[3],
                          max_x=0.0094 * data_shape[2],
                          max_y=0.0094 * data_shape[3],
                          max_iters=data_shape[0],
                          ng=1)

  # Keep only the channel to get a peek at.
  data = data[:, channel]

  frequency = len(data) // num_intervals

  if frequency <= frames_per_interval:
    data_peek = data
  else:
    mask = np.arange(frames_per_interval)
    for i in range(1, num_intervals):
      mask = np.concatenate(
        (mask, np.arange(i * frequency, i * frequency + frames_per_interval)),
        axis=0
      )
    data_peek = data[mask]

  ani = create_animation(h_hist=data_peek, **kwargs)
  # Restore previous settings.
  conf.FPS = prev_fps
  conf.DPI = prev_dpi
  return ani


@time_this
def create_animation(h_hist=None, t_hist=None,
                     show_animation=False, save_animation=True,
                     load_predictions=False, predictions_path=None,
                     iters=None, dpi=conf.DPI, fps=conf.FPS):
  """Creates an animation of the successive predicted frames.

  Args:
    h_hist (ndarray)      : the array of the predicted states
    t_hist (ndarray)      : the cumulative time-steps of the states
    show_animation (bool) : generates a pop-up player
    save_animation (bool) : save the animation in mp4
  """
  conf.DPI = dpi
  conf.FPS = fps
  conf.SHOW_ANIMATION = show_animation
  conf.SAVE_ANIMATION = save_animation
  conf.ROTATION = True
  # conf.PLOTTING_STYLE = "water"
  if load_predictions:
    h_hist = np.load(predictions_path, mmap_mode='r')
  return mattflow_post.animate(h_hist, t_hist)
