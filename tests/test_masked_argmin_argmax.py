#!/usr/bin/env python

import numpy as np
import pytest

from malid.external.genetools_arrays import masked_argmin, masked_argmax


@pytest.fixture
def arr():
    arr = np.array([[0.5, 0.5, 0.1, 0.5], [0.5, 0.5, 0.3, 0.5], [0.05, 0.5, 0.1, 0.5]])
    arr = np.ma.MaskedArray(arr, arr > 0.2)
    return arr


def test_argmin(arr):
    np.testing.assert_array_equal(masked_argmin(arr, axis=1), [2.0, np.nan, 0.0])
    np.testing.assert_array_equal(
        masked_argmin(arr, axis=0), [2.0, np.nan, 0.0, np.nan]
    )
    np.testing.assert_equal(masked_argmin(arr), 8)
    np.testing.assert_equal(masked_argmin(arr[0, :]), 2)
    np.testing.assert_equal(arr[0, :].argmin(), 2)
    np.testing.assert_equal(masked_argmin(arr[1, :]), np.nan)


def test_numpy_argmin_has_wrong_behavior(arr):
    # the standard wrong numpy behavior for nan rows / columns
    np.testing.assert_array_equal(arr.argmin(axis=0), [2, 0, 0, 0])
    np.testing.assert_array_equal(arr.argmin(axis=1), [2, 0, 0])
    np.testing.assert_equal(arr[1, :].argmin(), 0)


def test_argmin_harder():
    # This was an earlier bug: returned [nan] instead of [0]
    arr = np.array([[0.0, 1.0]])  # shape (1,2)
    arr_masked = np.ma.MaskedArray(
        arr,
        arr > 0.5,
    )
    np.testing.assert_array_equal(masked_argmin(arr_masked, axis=1), [0.0])
    np.testing.assert_equal(masked_argmin(arr_masked), 0.0)


def test_argmax(arr):
    np.testing.assert_array_equal(masked_argmax(arr, axis=1), [2.0, np.nan, 2.0])
    np.testing.assert_array_equal(
        masked_argmax(arr, axis=0), [2.0, np.nan, 0.0, np.nan]
    )
    np.testing.assert_equal(masked_argmax(arr), 2)
    np.testing.assert_equal(masked_argmax(arr[0, :]), 2)
    np.testing.assert_equal(arr[0, :].argmax(), 2)
    np.testing.assert_equal(masked_argmax(arr[1, :]), np.nan)


def test_numpy_argmax_has_wrong_behavior(arr):
    # the standard wrong numpy behavior for nan rows / columns
    np.testing.assert_array_equal(arr.argmax(axis=0), [2, 0, 0, 0])
    np.testing.assert_array_equal(arr.argmax(axis=1), [2, 0, 2])
    np.testing.assert_equal(arr[1, :].argmax(), 0)


@pytest.mark.xfail(raises=TypeError)
def test_argmin_requires_masked_array():
    masked_argmin(np.array([1, 2, 3]))


@pytest.mark.xfail(raises=TypeError)
def test_argmax_requires_masked_array():
    masked_argmax(np.array([1, 2, 3]))
