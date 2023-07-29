import numpy as np
import pandas as pd
import pytest
from pandas import testing as tm
from malid.trained_model_wrappers.rollup_sequence_classifier import (
    _weighted_median,
    _trim_bottom_only,
    _entropy_threshold,
    _trimmed_mean,
)


@pytest.fixture
def uniform_probabilities_uniform_weights():
    p_base = np.arange(0.0, 1.01, 0.1)
    probs = pd.DataFrame(
        {
            "covid": p_base,
            "hiv": 1 - p_base,
        }
    )
    weights = pd.Series(1, index=probs.index)
    return probs, weights


# base case of uniform probabilities (linear sequence of probs from 0->1 by 0.1 increments) with no sequence weighting strategy
def test_weighted_median_uniform_probabilities_uniform_weights(
    uniform_probabilities_uniform_weights,
):
    probs, weights = uniform_probabilities_uniform_weights
    tested = _weighted_median(probs, weights)
    expected = pd.Series([0.5, 0.5], index=["covid", "hiv"])
    tm.assert_series_equal(tested, expected)


def test_trimmed_mean_uniform_probabilities_uniform_weights(
    uniform_probabilities_uniform_weights,
):
    probs, weights = uniform_probabilities_uniform_weights
    tested = _trimmed_mean(probs, weights)
    expected = pd.Series([0.5, 0.5], index=["covid", "hiv"])
    tm.assert_series_equal(tested, expected)


def test_weighted_median_uniform_probabilities_uniform_weights(
    uniform_probabilities_uniform_weights,
):
    probs, weights = uniform_probabilities_uniform_weights
    tested = _trim_bottom_only(probs, weights)
    expected = pd.Series([0.5, 0.5], index=["covid", "hiv"])
    tm.assert_series_equal(tested, expected)


def test_weighted_median_uniform_probabilities_uniform_weights(
    uniform_probabilities_uniform_weights,
):
    probs, weights = uniform_probabilities_uniform_weights
    tested = _entropy_threshold(probs, weights)
    expected = pd.Series([0.5, 0.5], index=["covid", "hiv"])
    tm.assert_series_equal(tested, expected)
