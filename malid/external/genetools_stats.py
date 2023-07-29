import numpy as np
import pandas as pd
import sklearn.utils
from typing import Optional, List


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_label: str,
    pred_label: str,
    label_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    # rows ground truth label - columns predicted label
    cm = pd.crosstab(
        np.array(y_true),
        np.array(y_pred),
        rownames=[true_label],
        colnames=[pred_label],
    )

    # reorder so columns and index match
    if label_order is None:
        label_order = cm.index.union(cm.columns)

    resulting_row_order, resulting_col_order = [
        pd.Index(label_order).intersection(source_list).tolist()
        for source_list in [
            cm.index,
            cm.columns,
        ]
    ]

    cm = cm.loc[resulting_row_order][resulting_col_order]

    return cm


def softmax(arr: np.ndarray) -> np.ndarray:
    """softmax a 1d vector or softmax all rows of a 2d array. ensures result sums to 1:

    - if input was a 1d vector, returns a 1d vector summing to 1.
    - if input was a 2d array, returns a 2d array where every row sums to 1.
    """
    orig_ndim = arr.ndim
    if orig_ndim == 1:
        # input array is a M-dim vector that we want to make sum to 1
        # but sklearn softmax expects a matrix NxM, so pass a 1xM matrix.
        # convert single vector into a single-row matrix, e.g. [1,2,3] to [[1, 2, 3]]:
        arr = np.reshape(arr, (1, -1))

    # Convert to probabilities with softmax
    probabilities = sklearn.utils.extmath.softmax(arr.astype(float), copy=False)

    if orig_ndim == 1:
        # Finish workaround for softmax when X has a single row: extract relevant softmax result (first row of matrix)
        probabilities = probabilities[0, :]

    return probabilities
