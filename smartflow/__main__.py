# __main__.py is part of SmartFlow
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
"""Calls model construction, training, prediction and post-processing"""

import logging
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from mattflow.config import MAX_ITERS

from smartflow import (backend as S,
                       core,
                       smartflow_pre,
                       smartflow_post,
                       smartflow_solver,
                       utils)
from smartflow.utils import time_this

S.set_tf_on_gpu(memory_limit=4040)
S.set_print_options()


@time_this
def main():
  # Uncoment this to delete data from previous runs (for debugging).
  utils.delete_prev_runs_data(preprocessing_visualizations=False)

  # 0. Configuration of the numerical simulation
  #    - features size: (MAX_ITERS, 3, (Nx + 2 * Ng), (Ny + 2 * Ng))
  #    - labels size  : (MAX_ITERS, 3, Nx, Ny)
  #
  #    Data examples:
  #      - N:  50,  max_x: 0.4
  #      - N:  80,  max_x: 0.65
  #      - N:  90,  max_x: 0.75
  #      - N: 100,  max_x: 0.85
  S.set_mattflow_config(nx=90, max_x=1, ng=2, max_iters=40000,
                        num_x_frames=2, num_channels=1)
  # S.set_mattflow_config(nx=100, max_x=0.85, max_iters=20000,
  #                       num_x_frames=2, num_channels=3)
  # S.set_mattflow_config(nx=80, max_x=0.75, max_iters=30000,
  #                       num_x_frames=2, num_channels=3)

  # 1 Raw data visualization
  # smartflow_post.peek_data()

  # 2. Pre-processing
  preprocessing_config = {
    "data_subset": (5000, 15000),
    "batch_size": 32,
    # "preprocessing_vis": False,
    "normalize_at_training": False,
    "stats_type": "channelwise_train_ds",
    # "dump_memmap": True
    # "dataset_as_sequence": True
  }
  train_ds, val_ds, test_ds = smartflow_pre.preprocess(**preprocessing_config)
  # smartflow_pre.preprocess(**preprocessing_config)
  # train_ds, val_ds, test_ds = io.read_datasets()

  # 3. Model creation and training
  model_config = {
    "arch": "inception_resnet_v2_remodified",
    # "load_saved_model": True,
    # "model_path": "saved_models/functional_1_2021-04-14_00-18-20",
    # "load_weights_only": True,
    # "checkpoint_path": utils.checkpoint_path().format(epoch=9999),
    # "train_model": False,
    # "evaluate_model": False
  }
  train_config = {
    "epochs": 20,
    "cp_epochs_freq": 5,
    "validation_freq": 4
  }
  model = core.trained_model(train_ds, val_ds, test_ds,
                             model_config,
                             train_config)

  # 4. Visualization of random input-pred-gt examples
  examples = random.sample(range(MAX_ITERS - S.num_x_frames() - 1), 15)
  smartflow_post.plot_input_pred_gt(model,
                                    examples,
                                    show_fig=False,
                                    save_fig=True)

  # 5. Prediction of consecutive fluid-states
  # (Each prediction is the input for the next one.)
  h_hist, t_hist = smartflow_solver.simulate(model,
                                             iters=250,
                                             save_predictions=False)

  # 6. Post-processing
  smartflow_post.create_animation(h_hist=h_hist,
                                  t_hist=t_hist,
                                  show_animation=True,
                                  dpi=80,
                                  fps=20)


if __name__ == "__main__":
  main()
