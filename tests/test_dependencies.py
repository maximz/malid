#!/usr/bin/env python

import pytest
import gc

# test that libraries are installed -- all these imports should work regardless of CPU or GPU
import numpy as np
import umap
import torch
import tensorflow as tf
from tensorflow import keras


@pytest.mark.gpu
def test_tensorflow_can_see_gpus():
    assert len(tf.config.list_physical_devices("GPU")) > 0


@pytest.mark.skip_if_gpu
def test_tensorflow_does_not_see_gpus_in_cpu_mode():
    assert len(tf.config.list_physical_devices("GPU")) == 0


@pytest.mark.gpu
def test_torch_can_see_gpus():
    assert torch.cuda.is_available()


@pytest.mark.skip_if_gpu
def test_torch_does_not_see_gpus_in_cpu_mode():
    assert not torch.cuda.is_available()


@pytest.mark.gpu
def test_gpu_nearest_neighbors_works():
    import cudf
    import cugraph
    import dask
    import cupy
    import dask_cudf
    import rmm

    import cuml
    from cuml.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=15, metric="euclidean")
    nn.fit(np.array([[0, 0, 1], [1, 0, 0]]))

    del nn
    gc.collect()
