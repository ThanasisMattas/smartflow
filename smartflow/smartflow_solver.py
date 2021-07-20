# smartflow_solver.py is part of SmartFlow
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
"""Handles the iterative prediction of the state of the fluid."""

from mattflow import (config as conf,
                      initializer,
                      mattflow_solver)
import numpy as np

from smartflow import backend as S, io, smartflow_pre
from smartflow.utils import time_this


def _predict(U, model, it, drops_count, Uprev=None):
  """Predicts the state of the fluid at a new time-step.

  It can be used in a for loop, iterating through each time-step.

  Args:
    U (ndarray)        : (1, c, Nx + 2 * Ng, Ny + 2 * Ng)
    model (Sequential) : the trained model
    it (int)           : current iteration
    drops_count (int)  : drops counter
    Uprev (ndarray)    : When the input comprises by 2 frames, this holds the
                         oldest time-step (1, c, Nx + 2 * Ng, Ny + 2 * Ng)

  Returns:
    Upred (ndarray)    : (1, c, Nx, Ny)
    drops_count (int)
  """
  # Generate a random drop every 105 iterations (and don't make a prediction).
  if it % 105 == 0:
    if Uprev is not None:
      Uprev = U
    # !!!
    if not S.use_boundary_conditions():
      U = np.pad(U, ((0, 0), (0, 0), (conf.Ng, conf.Ng), (conf.Ng, conf.Ng)))
    U[0, 0, :, :] = initializer.drop(U[0, 0, :, :], drops_count)
    # Strip the ghost cells, just like predict would do.
    Upred = U[0: 1,
              0: S.num_channels(),
              conf.Ng: -conf.Ng,
              conf.Ng: -conf.Ng]
    drops_count += 1

  else:
    if Uprev is None:
      # !!!
      # U = U.reshape(1, -1)
      Upred = model.predict(U)
    else:
      Upred = model.predict(np.concatenate([U, Uprev], axis=S.channel_axis()))
      Uprev = U
    # In case of multiple outputs
    if isinstance(U, list):
      Upred = Upred[-1]
    # Un-flatten the prediction.
    Upred = np.reshape(Upred, (1, S.num_channels(), conf.Nx, conf.Ny))

  return Upred, Uprev, drops_count


@time_this
def simulate(model, iters=conf.MAX_ITERS, save_predictions=True):
  """Iteretively uses the current frame to predict the next one.

  Args:
    model (Model)           : the trained model
    iters (int)             : the consecutive number of frames to predict
    save_predictions (bool) : save locally the array of the predictions

  Returns:
    h_hist (3D ndarray)     : the array of the predicted frames
                              (#iters, Nx, Ny)
    t_hist (ndarray)        : the array of the calculated timestamps of the
                              frames (CFL condition)
  """
  iters = min(iters, conf.MAX_ITERS)
  time = 0
  drops_count = 1

  # Initialization with the 1st drop (only on height channel)
  U, h_hist, t_hist, _ = initializer.initialize()

  # If num_x_frames > 1, then all frames will correspond to a single label, except
  # from the leading num_x_frames - 1 (label) frames, which are needed for the
  # first prediction. Thus they hove to be included in h_hist, too.
  h_hist = h_hist[:iters + S.num_x_frames() - 1]
  t_hist = t_hist[:iters + S.num_x_frames() - 1]
  # (c, h, w) --> (1, c, h, w)
  U = U[np.newaxis, :]

  # Keep only the channels used at the current analysis.
  U = U[:, 0: S.num_channels(), :, :]

  if not S.use_boundary_conditions():
    U = U[..., conf.Ng: -conf.Ng, conf.Ng: -conf.Ng]

  if S.num_x_frames() == 2:
    # upon initialization, move the new-drop frame to the second iteration
    h_hist[1] = h_hist[0]
    h_hist[0] = h_hist[2]
    # this will hold the previous frame
    Uprev = np.full(U.shape, 0, dtype=S.dtype("numpy"))
  else:
    Uprev = None

  for it in range(1, iters):
    # Calculate and insert the timestamp of the frame (it requires the fluxes)
    if S.num_channels() == 3:
      delta_t = mattflow_solver.dt(U[0])
      time += delta_t
      t_hist[it] = time * 10
    # Udtate ghost cells
    if S.use_boundary_conditions():
      U[0] = smartflow_pre.SmartFlowDS.apply_bc(U[0])
    # Normalization
    if smartflow_pre.normalization_layer_weights() is None:
      U = smartflow_pre.Normalizer()(U)
    # Predict the next frame
    U, Uprev, drops_count = _predict(U, model, it, drops_count, Uprev=Uprev)
    # Insert current frame to the array, to be animated at post-processing
    h_hist[it] = U[0][0]
    # Pad back the ghost cells
    if S.use_boundary_conditions():
      U = np.pad(U, ((0, 0), (0, 0), (conf.Ng, conf.Ng), (conf.Ng, conf.Ng)))

  if save_predictions:
    io.save_predictions(h_hist, t_hist, iters, file_format="memmap")

  return h_hist, t_hist
