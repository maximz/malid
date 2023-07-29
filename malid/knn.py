"""
note: before importing this, make sure GPU is configured
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import malid.external.genetools_arrays

if TYPE_CHECKING:
    # Same API to satisfy the type checker without having CuML installed (GPU required?)
    from sklearn.neighbors import NearestNeighbors
else:
    # Rapids CuML wrapper around FAISS with GPU support. Used by scanpy under the hood.
    from cuml.neighbors import NearestNeighbors
# You can leverage the above logic by importing NearestNeighbors type from this module directly,
# which will use CuML by default but falls back to sklearn for type checker (i.e. pretends CuML is not available)

logger = logging.getLogger(__name__)

k_neighbors = 15


def _fit_knn_index(
    X: np.ndarray,
    n_neighbors: int = 5,
    metric: str = "euclidean",
) -> NearestNeighbors:
    return NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X)


def _get_neighbors(
    knn_index: NearestNeighbors,
    data_X_contiguous: np.ndarray,
    n_neighbors: int,
    max_row_count_per_batch=100000,
) -> pd.DataFrame:
    # **knn_indices**, **knn_dists** : np.arrays of shape (n_observations, n_neighbors)
    # knn_dists, knn_indices = nn.kneighbors(test_X_contiguous)
    knn_indices_all = []

    # split into chunks and run one chunk at a time
    # https://numpy.org/doc/stable/reference/generated/numpy.array_split.html#numpy.array_split
    # https://stackoverflow.com/a/14406661/130164
    for chunk_from_data_X_contiguous in np.array_split(
        data_X_contiguous,
        np.ceil(data_X_contiguous.shape[0] / max_row_count_per_batch).astype(int),
    ):
        knn_dists, knn_indices = knn_index.kneighbors(
            chunk_from_data_X_contiguous, n_neighbors=n_neighbors
        )
        knn_indices_all.append(knn_indices)
    knn_indices_all = np.vstack(knn_indices_all)
    if knn_indices_all.shape[0] != data_X_contiguous.shape[0]:
        raise ValueError("Failed to run kNN neighbors - wrong shape")

    # unpack knn_indices matrix into adjacency lists, so each element is one row of a dataframe
    # row_id is test sequence ID
    # col_id is ranking of how close the neighbor is, from 0 to K-1
    # value is ID of the neighbor in training set
    # uses cell ID (not obsname) for center node and for each neighboring node of a specific center node

    dfnonzero = (
        malid.external.genetools_arrays.convert_matrix_to_one_element_per_row(
            knn_indices_all
        )
        .rename(columns={"row_id": "center_id", "value": "neighbor_id"})
        .drop("col_id", axis=1)
    )
    return dfnonzero
