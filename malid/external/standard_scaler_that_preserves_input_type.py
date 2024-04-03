import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import overload
from numpy.typing import ArrayLike


class StandardScalerThatPreservesInputType(StandardScaler):
    """
    Extend sklearn.preprocessing.StandardScaler for "pandas in, pandas out" behavior:
    If initial fit() input is a pandas DataFrame, return DataFrames from transform().
    By default, StandardScaler returns numpy arrays from transform() even if the initial fit() input was a pandas DataFrame.

    This uses the set_output API (https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html).
    But rather than fixing a single output type as the set_output examples would have you do, this class determines the output type based on the input type.
    """

    def fit(self, X: ArrayLike, y=None, sample_weight=None):
        # Use the parent's fit method
        super().fit(X, y=y, sample_weight=sample_weight)

        # Determine the output type for future transform calls based on input type
        if isinstance(X, pd.DataFrame):
            self.set_output(transform="pandas")
        else:
            self.set_output(transform="default")

        return self

    ###

    # Syntactic sugar / type hinting:

    @overload
    def fit_transform(
        self, X: pd.DataFrame, y=None, sample_weight=None
    ) -> pd.DataFrame:
        pass

    @overload
    def fit_transform(self, X: np.ndarray, y=None, sample_weight=None) -> np.ndarray:
        pass

    def fit_transform(self, X: ArrayLike, y=None, sample_weight=None) -> ArrayLike:
        return super().fit_transform(X=X, y=y, sample_weight=sample_weight)

    @overload
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    @overload
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def transform(self, X: ArrayLike) -> ArrayLike:
        return super().transform(X=X)
