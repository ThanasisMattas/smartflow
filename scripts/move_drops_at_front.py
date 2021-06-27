#!/usr/bin/env python
"""This scipt moves iteration groups of <interval>, starting from new-drop
frames, to the front of the dataset.
"""

import os

import click
import numpy as np
from numpy.lib.format import open_memmap


@click.command
@click.argument("ds_path", type=click.Path(), help="format: .npy")
@click.argument("drop_iters_path", type=click.Path(), help="format: .npy")
@click.option("--interval", "-i", type=click.INT,
              default=20, show_default=True, help="group lenght")
@click.option("--overwrite", type=click.BOOL,
              default=False, show_default=True,
              help=r"loads ~20% of the ds to memory")
@click.option("--new_ds_path", type=click.Path(),
              default=None, show_default=True, help="format: .npy")
def main(ds_path, drop_iters_path, interval, overwrite, new_ds_path):
  ds = np.load(ds_path, mmap_mode='r')
  drop_iters = np.load(drop_iters_path, mmap_mode='r')
  print(f"ds.shape: {ds.shape}")
  print(f"drop_iters.shape: {drop_iters.shape}")

  if overwrite:
    new_ds = np.load(ds_path, mmap_mode='r+')
  else:
    if new_ds_path is None:
      p, e = os.path.splitext(ds_path)
      new_ds_path = f"{p}_drops_moved{e}"

    new_ds = open_memmap(new_ds_path,
                         mode="w+",
                         dtype=np.dtype("float64"),
                         shape=ds.shape)

  # for every new-drop frame keep the next 19 iterations
  intervals_stack = []
  for drop_it in drop_iters:
    for i in range(interval):
      intervals_stack.append(drop_it + i)
  print(f"len(intervals_stack): {len(intervals_stack)}")

  # fill the front of the new_ds
  if overwrite:
    ds_front = ds[intervals_stack].copy()  # loads ~20% of the ds to memory
  else:
    new_ds[:len(intervals_stack)] = ds[intervals_stack]
  # fill the rest of the new_ds
  mask = np.ones(len(ds), dtype=bool)
  mask[intervals_stack] = False
  new_ds[len(intervals_stack):] = ds[mask]
  if overwrite:
    new_ds[:len(intervals_stack)] = ds_front
