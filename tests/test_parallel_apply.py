#!/usr/bin/env python

import pandas as pd
import pytest

from malid.external.genetools_helpers import parallel_groupby_apply


# Can't pickle lambda functions
def parallel_func(grp):
    return grp.shape[0]


@pytest.fixture
def df():
    num_groups = 10
    return pd.DataFrame(
        {
            "name": ["Tom", "Mary", "Albert"] * num_groups,
            "age": [20, 30, 40] * num_groups,
            "group_id": sorted([0, 1, 2] * num_groups),
            "second_groupby_column": "A",
        }
    )


def test_parallel_apply_single_index(df):
    grpby = df.groupby("group_id")
    conventional = grpby.apply(parallel_func)
    parallelized = parallel_groupby_apply(
        grpby, parallel_func, n_jobs=2, backend="multiprocessing"
    )
    pd.testing.assert_series_equal(conventional, parallelized)


def test_parallel_apply_multi_index(df):
    grpby = df.groupby(["group_id", "second_groupby_column"], observed=True)
    conventional = grpby.apply(parallel_func)
    parallelized = parallel_groupby_apply(
        grpby, parallel_func, n_jobs=2, backend="multiprocessing"
    )
    pd.testing.assert_series_equal(conventional, parallelized)
