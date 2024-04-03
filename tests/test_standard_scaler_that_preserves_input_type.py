import numpy as np
import pandas as pd
from malid.external.standard_scaler_that_preserves_input_type import (
    StandardScalerThatPreservesInputType,
)


def test_extended_scaler_with_numpy():
    # Using numpy array
    data_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    scaler = StandardScalerThatPreservesInputType().fit(data_array)
    scaled_array = scaler.transform(data_array)
    assert isinstance(
        scaled_array, np.ndarray
    ), "Expected numpy.ndarray output for numpy input"

    # repeat with fit_transform
    scaled_array = StandardScalerThatPreservesInputType().fit_transform(data_array)
    assert isinstance(
        scaled_array, np.ndarray
    ), "Expected numpy.ndarray output for numpy input"


def test_extended_scaler_with_pandas():
    # Using pandas DataFrame
    df = pd.DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})

    def _test(scaled_df):
        assert isinstance(
            scaled_df, pd.DataFrame
        ), "Expected pandas DataFrame output for DataFrame input"
        assert scaled_df.shape == df.shape, "Expected same shape as input DataFrame"
        assert scaled_df.columns.equals(
            df.columns
        ), "Expected same columns as input DataFrame"
        assert scaled_df.index.equals(
            df.index
        ), "Expected same index as input DataFrame"

    scaler = StandardScalerThatPreservesInputType().fit(df)
    _test(scaler.transform(df))

    # repeat with fit_transform
    _test(StandardScalerThatPreservesInputType().fit_transform(df))
