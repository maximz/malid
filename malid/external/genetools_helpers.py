from typing import Callable, Union
import numpy as np
import pandas as pd
import sklearn.pipeline
from joblib import Parallel, delayed
from pandas.core.groupby.generic import DataFrameGroupBy


def parallel_groupby_apply(
    df_grouped: DataFrameGroupBy, func: Callable, **kwargs
) -> pd.Series:
    """
    Parallelize apply() on a pandas groupby object.

    Each subprocesses is given one group to process. This approach isn't appropriate if your applied function is very fast but you have many, many groups. In that scenario, the parallelization of groups will simply introduce a lot of unnecessary overhead. Make sure to benchmark with and without parallelization. May want to first split full dataframe into big chunks containing many groups, then run groupby-apply on each chunk in parallel.

    Also, transferring big groups to subprocesses can be slow. Again consider chunking the dataframe first.

    Func cannot be a lambda, since lambda functions can't be pickled for subprocesses.

    Kwargs are passed to joblib.Parallel(...)

    TODO: allow apply returning dataframe not just series -- see `_wrap_applied_output` in `pandas/core/groupby/generic.py`

    TODO: wrap with progress bar: https://stackoverflow.com/a/58936697/130164
    """

    # Get values
    values = Parallel(**kwargs)(delayed(func)(group) for name, group in df_grouped)

    # Get index
    group_keys = [name for name, group in df_grouped]

    # Assign index column name(s)

    # https://github.com/pandas-dev/pandas/blob/059c8bac51e47d6eaaa3e36d6a293a22312925e6/pandas/core/groupby/groupby.py#L1202
    if df_grouped.grouper.nkeys > 1:
        index = pd.MultiIndex.from_tuples(group_keys, names=df_grouped.grouper.names)
    else:
        index = pd.Index(group_keys, name=df_grouped.grouper.names[0])

    # Package up as series
    return pd.Series(
        values,
        index=index,
    )


def apply_pipeline_transforms(
    fit_pipeline: sklearn.pipeline.Pipeline, data: Union[np.ndarray, pd.DataFrame]
) -> Union[np.ndarray, pd.DataFrame]:
    """
    apply all transformations in an already-fit sklearn Pipeline, except the final estimator

    only use for pipelines that have an estimator (e.g. classifier) at the last step.
    in these cases, we cannot call pipeline.transform(). use this method instead to apply all steps except the final classifier/regressor/estimator.

    otherwise, if you have a pipeline of all transformers, you should just call .transform() on the pipeline.
    """
    data_transformed = data
    # apply all but last step
    for _, _, step in fit_pipeline._iter(with_final=False, filter_passthrough=True):
        data_transformed = step.transform(data_transformed)
    return data_transformed
