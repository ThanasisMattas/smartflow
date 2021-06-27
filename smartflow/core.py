# core.py is part of SmartFlow
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
"""Creates and trains DL models on Shallow Water simulation data."""

import gc
import os

import pandas as pd

from mattflow import config as conf
import tensorflow as tf
from tensorflow import keras

from smartflow import (backend as S,
                       smartflow_post,
                       smartflow_pre,
                       utils)
from smartflow.archs import (cnn,
                             fcnn,
                             inception_resnet,
                             inception_resnet_v2,
                             inception_v3,
                             resnet)
from smartflow.utils import time_this


class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


def _cp_steps_freq(cp_epochs_freq: int) -> int:
  """Used at keras.callbacks.ModelCheckpoint."""
  return cp_epochs_freq * S.steps_per_epoch("train")


def _save_options(operation: str):
  """Sets tf.train.CheckpointOptions or tf.saved_model.SaveOptions, in case of
  running on ipython without writing access to CPU:0 physical device.

  operation (str) : the underlying operation ("checkpoint" or "save_model")
  """
  try:
    get_ipython()  # type: ignore
    if operation == "checkpoint":
      options = tf.train.CheckpointOptions(
        experimental_io_device='/job:localhost'
      )
    elif operation == "save_model":
      options = tf.saved_model.SaveOptions(
        experimental_io_device='/job:localhost'
      )
  except NameError:
    options = None
  return options


def _callbacks(cp_epochs_freq, monitor, **callbacks):
  allowed_callbacks = {
    "checkpoint",
    "earlystopping",
    "tensorboard",
    "lr_schedule",
    "GarbageCollectorCallback"
  }
  utils.validate_kwargs(callbacks, allowed_callbacks)

  cbs = []

  if callbacks.pop("checkpoint", True):
    cp_cb = keras.callbacks.ModelCheckpoint(
      filepath=utils.checkpoint_path(),
      monitor=monitor,
      save_best_only=True,
      save_weights_only=True,
      save_freq=_cp_steps_freq(cp_epochs_freq),
      options=_save_options("checkpoint")
    )
    cbs.append(cp_cb)

  if callbacks.pop("earlystopping", True):
    # TODO: https://github.com/tensorflow/tensorflow/issues/44107
    # EarlyStopping callback (stop before model starts to overfit)
    es_cb = keras.callbacks.EarlyStopping(
      monitor=monitor,
      min_delta=0.0001,
      patience=8,
      restore_best_weights=True
    )
    cbs.append(es_cb)

  if callbacks.pop("tensorboard", False):
    utils.child_dir("logs")
    tb_cb = keras.callbacks.TensorBoard(
      log_dir="logs",
      histogram_freq=1,
      write_graph=True,
      write_images=True,
      update_freq="epoch",
      profile_batch=2,
    )
    cbs.append(tb_cb)

  if callbacks.pop("lr_schedule", True):
    def scheduler(epoch, lr):
      """Exponational decay: lr = lr * base ^ epoch"""
      # if epoch < 20:
      #   return lr * 0.97
      # elif epoch < 90:
      #   return lr * 0.99
      if epoch < 50:
        return lr * 0.995
      else:
        return lr

    lr_scheduler_cb = keras.callbacks.LearningRateScheduler(
      scheduler,
      # verbose=verbose
    )
    cbs.append(lr_scheduler_cb)

  if callbacks.pop("GarbageCollectorCallback", False):
    cbs.append(GarbageCollectorCallback)

  return cbs


def _lr_scheduler(initial_lr=0.005,
                  decay_steps=3,
                  decay_rate=0.9,
                  staircase=True):
  return keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
  )


def compiled_model(arch=None, **kwargs):
  """Factory for compiled models"""
  allowed_kwargs = {
    "activation",
    "initial_lr",
    "kernel_initializer",
    "experimental_steps_per_execution",
    "decay_steps",
    "decay_rate",
    "staircase"
  }
  utils.validate_kwargs(kwargs, allowed_kwargs)
  S.set_activation(kwargs.pop("activation", "relu"))
  S.set_kernel_initializer(kwargs.pop("kernel_initializer", "he_normal"))

  architectures = {
    "inception_v3_built_in": inception_v3.inception_v3_built_in,
    "inception_v3_custom": inception_v3.inception_v3_custom,

    "inception_resnet_50x50_2": inception_resnet.inception_resnet_50x50_2,
    "inception_resnet_50x50": inception_resnet.inception_resnet_50x50,
    "resnet50_50x50": resnet.resnet50_50x50,
    "cnn_50x50": cnn.cnn_50x50,

    "inception_resnet_80x80": inception_resnet.inception_resnet_80x80,
    "cnn_80x80_legacy": cnn.cnn_80x80_legacy,
    "cnn_80x80_bn": cnn.cnn_80x80_bn,
    "cnn_80x80": cnn.cnn_80x80,

    "inception_resnet_v2": inception_resnet_v2.inception_resnet_v2,
    "inception_resnet_v2_modified": inception_resnet_v2.inception_resnet_v2_modified,
    "inception_resnet_v2_remodified": inception_resnet_v2.inception_resnet_v2_remodified,
    "inception_resnet_90x90": inception_resnet.inception_resnet_90x90,
    "resnet50_90x90": resnet.resnet50_90x90,
    "cnn_90x90_bn": cnn.cnn_90x90_bn,

    "inception_resnet_100x100": inception_resnet.inception_resnet_100x100,
    "cnn_100x100": cnn.cnn_100x100,

    "fcnn": fcnn.fcnn,
    "dummy": cnn.dummy_model,
    "crazy_ass_model": cnn.crazy_ass_model,
    "dummy_bn": cnn.dummy_model_bn
  }

  compile_config = {
    "loss": [keras.losses.MeanSquaredError()],
    "metrics": [keras.metrics.MeanAbsoluteError(name="mae")],
    "experimental_steps_per_execution": kwargs.get(
      "experimental_steps_per_execution", 1
    ),
    "optimizer": keras.optimizers.Adam(learning_rate=0.001)
    # "optimizer": keras.optimizers.Adam(learning_rate=_lr_scheduler(**kwargs))
  }

  # Architectures that incorporate multiple heads.
  if arch in ["inception_v3_custom"]:
    compile_config["loss"].append(keras.losses.MeanSquaredError())
    compile_config["loss_weights"] = [0.2, 0.8]
    compile_config["metrics"].append(
      keras.metrics.MeanAbsoluteError(name="mae")
    )

  model = architectures[arch]()
  model.compile(**compile_config)
  return model


@time_this
def _train(model, **kwargs):
  """Wrapper of Model.fit()"""
  allowed_kwargs = {
    "x",
    "validation_data",
    "epochs",
    "cp_epochs_freq",
    "verbose",
    "save_model",
    "save_weights_only",
    "steps_per_epoch",
    "validation_steps",
    "validation_freq",
    "shuffle",
    "workers",
    "use_multiprocessing"
  }
  utils.validate_kwargs(kwargs, allowed_kwargs)

  save_weights_only = kwargs.pop("save_weights_only", True)
  save_model = kwargs.pop("save_model", True)

  if kwargs.get("validation_freq", 1) == 1:
    monitor = "val_loss"
  else:
    monitor = "loss"

  # NOTE: Don't set batch_size, if it is already set at dataset creation.
  hist = model.fit(
    callbacks=_callbacks(kwargs.pop("cp_epochs_freq", 20), monitor),
    **kwargs
  )

  if save_weights_only:
    cp_path = utils.checkpoint_path()
    model.save_weights(cp_path.format(epoch=9999),
                       options=_save_options("checkpoint"))
  if save_model:
    model_dir = (f"{model.name}_{utils.today_and_now()}")
    models_dir = "saved_models"
    utils.child_dir(models_dir)
    model_path = os.path.join(os.getcwd(), models_dir, model_dir)
    model.save(model_path, options=_save_options("save_model"))
  if kwargs.get("verbose", True) and kwargs.get("validation_freq", 1) == 1:
    hist_df = pd.DataFrame(hist.history)
    hist_df["epoch"] = hist.epoch
    print()
    print(hist_df.tail(15))
    print()
  smartflow_post.plot_loss(hist, save_fig=True)
  return hist, model


def trained_model(train_ds,
                  val_ds,
                  test_ds,
                  model_config,
                  train_config):
  """Constructs and trains a model.

  Args:
    train_ds, val_ds, test_ds (DSequence or Dset)
    model_config (dict) :  kwargs regarding the model creation
    train_config (dict) :  kwargs regarding the model training

  Returns:
    model (Model)       :  the model after training
  """
  allowed_model_kwargs = {
    "strategy",
    "arch",
    "activation",
    "kernel_initializer",
    "initial_lr",
    "decay_steps",
    "decay_rate",
    "load_weights_only",
    "checkpoint_path",
    "load_saved_model",
    "model_path",
    "train_model",
    "evaluate_model"
  }
  allowed_train_kwargs = {
    "epochs",
    "cp_epochs_freq",
    "validation_freq",
    "verbose",
    "shuffle",
    "workers",
    "use_multiprocessing",
    "save_weights_only",
    "save_model",
  }
  utils.validate_kwargs(model_config, allowed_model_kwargs)
  utils.validate_kwargs(train_config, allowed_train_kwargs)

  load_weights_only = model_config.pop("load_weights_only", False)
  cp_path = model_config.pop("checkpoint_path", utils.checkpoint_path())

  train_model = model_config.pop("train_model", True)
  evaluate_model = model_config.pop("evaluate_model", True)

  # 1. Load/Create the model
  if model_config.pop("load_saved_model", False):
    model = keras.models.load_model(model_config.pop("model_path"), None)
    try:
      num_channels = model.layers[-1].output_shape[1] // conf.Nx // conf.Ny
      num_x_frames = model.layers[0].input_shape[0][1] // num_channels
    except TypeError:
      # input and output shapes are flattened.
      num_channels = 1
      num_x_frames = 1
      print(f"Setting num_channels = {num_channels}"
            f" and num_x_frames = {num_x_frames}")
    S.configure_channels(num_channels=num_channels, num_x_frames=num_x_frames)
  else:
    strategy = model_config.pop("strategy", None)
    if isinstance(strategy, tf.distribute.Strategy):
      # Run multiple batches inside a single tf.function call in case of a
      # google colab on TPUs session.Guide:
      # https://www.tensorflow.org/guide/tpu#train_a_model_using_keras_high_level_apis
      model_config["experimental_steps_per_execution"] = \
          S.steps_per_epoch("train")
      with strategy.scope():
        model = compiled_model(**model_config)
    else:
      model = compiled_model(**model_config)
    if load_weights_only:
      # The model must have the same architecture with the loaded weights.
      # In order to retrieve the latest checkpoint:
      # latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
      model.load_weights(cp_path)
  model.summary()
  print(f"#layers: {len(model.layers)}\n")

  # 2. Training
  if isinstance(train_ds, smartflow_pre.DSequence):
    train_config["steps_per_epoch"] = S.steps_per_epoch("train")
    train_config["validation_steps"] = S.steps_per_epoch("val")

  default_train_config = {
    'x': train_ds,
    "validation_data": val_ds,
    "verbose": 2,
    "shuffle": True,
    "workers": 6,
    "use_multiprocessing": True
  }
  # TODO: python 3.9: default_train_config |= train_config
  train_config = dict(default_train_config, **train_config)

  if train_model:
    hist, model = _train(model, **train_config)

  # 3. Evaluation
  if evaluate_model:
    if isinstance(test_ds, (tf.data.Dataset,
                            keras.utils.Sequence,
                            smartflow_pre.DSequence,
                            smartflow_pre.DSet)):
      evaluation = model.evaluate(test_ds, steps=S.steps_per_epoch("test"))
    else:
      evaluation = model.evaluate(x=test_ds[0],
                                  y=test_ds[1],
                                  batch_size=S.batch_size())
  return model
