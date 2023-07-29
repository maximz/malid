import numpy as np
import pandas as pd
import pytest
import scipy.stats

import malid.external


def test_convert_matrix_to_one_element_per_row():
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    df = malid.external.genetools_arrays.convert_matrix_to_one_element_per_row(arr)
    df_expected = pd.DataFrame(
        np.array(
            [
                [0, 0, 1],
                [1, 0, 5],
                [0, 1, 2],
                [1, 1, 6],
                [0, 2, 3],
                [1, 2, 7],
                [0, 3, 4],
                [1, 3, 8],
            ]
        ),
        columns=["row_id", "col_id", "value"],
    )
    pd.testing.assert_frame_equal(df, df_expected)


def test_get_trim_both_sides_mask():
    arr = np.array([10, 20, 0, 30, 40, 50])
    weights = np.array([5, 6, 4, 7, 8, 9])
    proportiontocut = 0.2

    mask = malid.external.genetools_arrays.get_trim_both_sides_mask(
        arr, proportiontocut=proportiontocut
    )
    assert np.array_equal(
        scipy.stats.trimboth(arr, proportiontocut=proportiontocut), arr[mask]
    )
    assert np.array_equal(arr[mask], [10, 20, 30, 40])
    assert np.array_equal(weights[mask], [5, 6, 7, 8])

    ###
    # bigger. but even
    arr = np.random.randint(low=1, high=100, size=50)
    proportiontocut = 0.1
    mask = malid.external.genetools_arrays.get_trim_both_sides_mask(
        arr, proportiontocut=proportiontocut
    )
    assert np.array_equal(
        scipy.stats.trimboth(arr, proportiontocut=proportiontocut), arr[mask]
    )

    ###
    # now odd.
    arr = np.random.randint(low=1, high=100, size=49)
    proportiontocut = 0.1
    mask = malid.external.genetools_arrays.get_trim_both_sides_mask(
        arr, proportiontocut=proportiontocut
    )
    assert np.array_equal(
        scipy.stats.trimboth(arr, proportiontocut=proportiontocut), arr[mask]
    )

    ###
    # now test a 2d matrix with a weight for each row
    arr = np.c_[
        np.array([0.8, 0.5, 0.6, 0.2, 0.3]), np.array([0.2, 0.5, 0.4, 0.3, 0.8])
    ]
    weights = np.array([3, 4, 1, 2, 5])

    mask = malid.external.genetools_arrays.get_trim_both_sides_mask(
        arr, proportiontocut=0.2, axis=0
    )

    weights_horizontally_cloned = np.tile(
        weights[np.newaxis, :].transpose(), arr.shape[1]
    )
    assert arr.shape == weights_horizontally_cloned.shape == (5, 2)

    column_weighted_averages = np.average(
        a=np.take_along_axis(arr, mask, axis=0),
        weights=np.take_along_axis(weights_horizontally_cloned, mask, axis=0),
        axis=0,
    )
    assert np.array_equal(
        column_weighted_averages,
        [
            np.average([0.3, 0.5, 0.6], weights=[5, 4, 1]),
            np.average([0.3, 0.4, 0.5], weights=[2, 1, 4]),
        ],
    )


def test_add_dummy_variables():
    isotype_groups = ["IGHG", "IGHA", "IGHD-M", "IGHD-M"]
    assert malid.external.genetools_arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=False,
    ).shape == (4, 3)


def test_add_dummy_variables_with_some_isotype_groups_missing():
    isotype_groups = ["IGHG", "IGHA", "IGHA", "IGHA"]
    assert malid.external.genetools_arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=True,
    ).shape == (4, 3)


def test_add_dummy_variables_with_some_isotype_groups_missing_categorical_input():
    # Categorical input case is special because pd.get_dummies(categorical_series) doesn't allow adding further columns.
    isotype_groups = pd.Series(["IGHG", "IGHA", "IGHA", "IGHA"]).astype("category")
    assert malid.external.genetools_arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=True,
    ).shape == (4, 3)


@pytest.mark.xfail(raises=ValueError)
def test_add_dummy_variables_with_some_isotype_groups_missing_disallowed():
    isotype_groups = ["IGHG", "IGHA", "IGHA", "IGHA"]
    malid.external.genetools_arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=False,
    )


@pytest.mark.xfail(raises=ValueError)
def test_add_dummy_variables_with_unexpected_isotype_groups_fails():
    isotype_groups = ["IGHG", "IGHA", "IGHE", "IGHD-M"]
    malid.external.genetools_arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=False,
    )


@pytest.mark.xfail(raises=ValueError)
def test_add_dummy_variables_with_unexpected_isotype_groups_fails_regardless_of_setting():
    isotype_groups = ["IGHG", "IGHA", "IGHE", "IGHD-M"]
    malid.external.genetools_arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=True,
    )
