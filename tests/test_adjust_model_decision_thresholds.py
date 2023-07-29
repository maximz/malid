#!/usr/bin/env python

import numpy as np
import pytest
import sklearn.base

# import pytest
from numpy.testing import assert_array_equal
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from malid.external.model_evaluation_scores import roc_auc_score

from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)


class FakeModel:
    def __init__(self, y_pred_proba, classes):
        self.y_pred_proba = y_pred_proba
        self.classes_ = classes

    def predict_proba(self, X):
        # ignores X, always returns preprogrammed y_pred_proba
        return self.y_pred_proba


@pytest.fixture
def data1():
    # four classes: clear diagonal, final row muddy (all ground truth consistent, but predictions are a toss up)
    # first, clear diagonal
    # how many entries per class are clear diagonal
    n_diagonal_clear = 100
    labels = np.array([0, 1, 2, 3])
    clear_diagonal_probas = np.vstack(
        [
            np.tile([0.7, 0.1, 0.1, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.7, 0.1, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.1, 0.7, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.1, 0.1, 0.7], n_diagonal_clear).reshape(-1, 4),
        ]
    )
    clear_diagonal_trues = np.hstack(
        [np.tile([lbl], n_diagonal_clear) for lbl in labels]
    )

    # now, muddy up the final row:
    # all ground truth consistent, but predictions slightly all over the place - almost a toss up
    n_muddy = 100
    muddy_final_row_probas = np.vstack(
        [
            np.tile([0.34, 0.22, 0.22, 0.22], n_muddy).reshape(-1, 4),
            np.tile([0.22, 0.34, 0.22, 0.22], n_muddy).reshape(-1, 4),
            np.tile([0.22, 0.22, 0.34, 0.22], n_muddy).reshape(-1, 4),
            np.tile([0.22, 0.22, 0.22, 0.34], n_muddy).reshape(-1, 4),
        ]
    )
    muddy_final_row_trues = np.tile(labels[-1], n_muddy * 4)

    y_score = np.vstack([clear_diagonal_probas, muddy_final_row_probas])
    y_true = np.hstack([clear_diagonal_trues, muddy_final_row_trues])
    return (y_true, y_score, labels)


@pytest.fixture
def data2():
    # Same as above, but now muddied final column not row (consistent predictions, ground truth is a toss up)

    # first, clear diagonal
    # how many entries per class are clear diagonal
    n_diagonal_clear = 100
    labels = np.array([0, 1, 2, 3])
    clear_diagonal_probas = np.vstack(
        [
            np.tile([0.7, 0.1, 0.1, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.7, 0.1, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.1, 0.7, 0.1], n_diagonal_clear).reshape(-1, 4),
            np.tile([0.1, 0.1, 0.1, 0.7], n_diagonal_clear).reshape(-1, 4),
        ]
    )
    clear_diagonal_trues = np.hstack(
        [np.tile([lbl], n_diagonal_clear) for lbl in labels]
    )

    # now, muddy up the final column:
    # all predictions consistent, but ground truth is a toss up
    n_muddy = 100
    muddy_final_row_probas = np.tile([0.1, 0.1, 0.1, 0.7], n_muddy * 4).reshape(-1, 4)
    muddy_final_row_trues = np.hstack([np.tile([lbl], n_muddy) for lbl in labels])

    y_score = np.vstack([clear_diagonal_probas, muddy_final_row_probas])
    y_true = np.hstack([clear_diagonal_trues, muddy_final_row_trues])
    return (y_true, y_score, labels)


# Parametrize with fixtures: see:
# https://github.com/tvorog/pytest-lazy-fixture
# https://github.com/pytest-dev/pytest/issues/349#issuecomment-671900957
# https://github.com/pytest-dev/pytest/issues/349#issuecomment-501796965
# https://smarie.github.io/python-pytest-cases/
# https://miguendes.me/how-to-use-fixtures-as-arguments-in-pytestmarkparametrize
@pytest.mark.parametrize(
    "data,expected_original_accuracy,expected_adjusted_accuracy",
    [
        # This one improves
        (pytest.lazy_fixture("data1"), 0.625, 1.0),
        # This one does not improve any further
        (pytest.lazy_fixture("data2"), 0.625, 0.625),
    ],
)
def test_adjust_decision_threshold(
    data, expected_original_accuracy, expected_adjusted_accuracy
):
    # unpack
    y_true, y_score, labels = data

    n_classes = y_score.shape[1]
    X_test = np.random.randn(y_score.shape[0], 10)  # random
    model = FakeModel(y_score, labels)
    adjusted_model = AdjustedProbabilitiesDerivedModel.adjust_model_decision_thresholds(
        model=model, X_validation=X_test, y_validation_true=y_true
    )
    y_pred = y_score.argmax(axis=1)
    adjusted_y_pred = adjusted_model.predict(X_test)

    accuracy_original = accuracy_score(y_true, y_pred)
    accuracy_adjusted = accuracy_score(y_true, adjusted_y_pred)
    print(accuracy_original, accuracy_adjusted)
    assert accuracy_original == expected_original_accuracy
    assert accuracy_adjusted == expected_adjusted_accuracy
    assert accuracy_adjusted >= accuracy_original

    # confirm roc-auc unchanged
    original_rocauc = roc_auc_score(
        y_true,
        model.predict_proba(X_test),
        multi_class="ovo",
        labels=labels,
        average="macro",
    )
    new_rocauc_adjusted = roc_auc_score(
        y_true,
        adjusted_model.predict_proba(X_test),
        multi_class="ovo",
        labels=labels,
        average="macro",
    )
    assert original_rocauc == new_rocauc_adjusted


def test_sklearn_clonable():
    inner_clf = DummyClassifier()
    inner_clf.classes_ = ["A", "B"]
    inner_clf.constant = 5

    estimator = AdjustedProbabilitiesDerivedModel(
        inner_clf=inner_clf,
        class_weights=[1, 1, 1],
    )
    # properties of inner_clf are exposed
    assert estimator.constant == inner_clf.constant
    assert_array_equal(inner_clf.classes_, estimator.classes_)

    # Check that supports cloning with sklearn.base.clone
    estimator_clone = sklearn.base.clone(estimator)

    # Check that attributes have transferred, except for inner_clf's attributes that have _ at the end, e.g. `classes_`
    # sklearn specifically does not clone those. but the rest, like `constant`, should transfer
    assert_array_equal(
        estimator_clone.class_weights, estimator.class_weights
    ), "Derived class property failed to transfer"
    assert (
        estimator_clone.constant == estimator.constant == inner_clf.constant
    ), "Inner class property failed to transfer"
    assert not hasattr(estimator_clone, "classes_") and not hasattr(
        estimator_clone.inner_clf, "classes_"
    ), "Inner class underscore property should not be cloned by sklearn"


def test_choose_winning_label():
    # test behavior of _choose_winning_label when passed a 2d array of probabilities vs a 1d array
    # this corresponds to having multiple rows in X versus only a single data point in X
    class_names = ["classA", "classB"]
    adjusted_probabilities_1d = np.array([0.2, 0.8])
    adjusted_probabilities_2d = np.array([[0.2, 0.8], [0.7, 0.3]])

    # should not be an array; should return a single value
    assert (
        AdjustedProbabilitiesDerivedModel._choose_winning_label(
            class_names, adjusted_probabilities_1d
        )
        == "classB"
    )

    # should return an array
    np.testing.assert_array_equal(
        AdjustedProbabilitiesDerivedModel._choose_winning_label(
            class_names, adjusted_probabilities_2d
        ),
        np.array(["classB", "classA"]),
    )


def test_compute_adjusted_class_probabilites():
    # test behavior of _compute_adjusted_class_probabilities when passed a 2d array of probabilities vs a 1d array
    # this corresponds to having multiple rows in X versus only a single data point in X
    probabilities_1d = np.array([0.2, 0.8])
    probabilities_2d = np.array([[0.2, 0.8], [0.7, 0.3]])
    class_priors = np.array([1, 3])

    # check output shapes and that outputs don't sum to 1
    probabilities_2d_adjusted = (
        AdjustedProbabilitiesDerivedModel._compute_adjusted_class_probabilities(
            probabilities_2d, class_priors
        )
    )
    assert probabilities_2d_adjusted.shape == (2, 2)
    assert all(probabilities_2d_adjusted.sum(axis=1) != 1)
    assert np.array_equal(
        probabilities_2d_adjusted,
        np.array([[0.2, 0.8 * 3], [0.7, 0.3 * 3]]),
    )

    # This should still be 1d!
    probabilities_1d_adjusted = (
        AdjustedProbabilitiesDerivedModel._compute_adjusted_class_probabilities(
            probabilities_1d, class_priors
        )
    )
    assert probabilities_1d_adjusted.shape == (2,)
    assert probabilities_1d_adjusted.sum() != 1
    assert np.array_equal(probabilities_1d_adjusted, [0.2, 0.8 * 3])
