import numpy as np
import pandas as pd
from malid.supervised_embedding import (
    _stack_background_and_foreground_points_into_line_segments,
)


def test_stack_background_and_foreground_points_into_line_segments():
    # This is used after querying kNN to get nearest background neighbor for each foreground point
    # Foreground point 0's nearest neighbor is background point 1: want line segment from (0,1) to (1,0)
    # Foreground point 1's nearest neighbor is background point 0: want line segment from (1,1) to (0,0)
    background_points = pd.DataFrame({"X_umap1": [0, 1], "X_umap2": [0, 0]})
    foreground_points = pd.DataFrame({"X_umap1": [0, 1], "X_umap2": [1, 1]})
    neighbor_links_df = pd.DataFrame({"center_id": [0, 1], "neighbor_id": [1, 0]})

    background_points_in_order = background_points.iloc[
        neighbor_links_df["neighbor_id"].values
    ]
    foreground_points_in_order = foreground_points.iloc[
        neighbor_links_df["center_id"].values
    ]
    line_points = _stack_background_and_foreground_points_into_line_segments(
        background_points_in_order=background_points_in_order[["X_umap1", "X_umap2"]],
        foreground_points_in_order=foreground_points_in_order[["X_umap1", "X_umap2"]],
    )
    assert np.array_equal(line_points, [((0, 1), (1, 0)), ((1, 1), (0, 0))])
