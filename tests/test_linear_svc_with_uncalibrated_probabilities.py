import numpy as np
from malid.external.linear_svc_with_uncalibrated_probabilities import (
    LinearSVCWithUncalibratedProbabilities,
)


def test_linearsvc_softmax_probabilities(sample_data):
    (X_train, y_train, X_test, y_test) = sample_data

    # test this classifier end-to-end to confirm it functions at all
    clf = LinearSVCWithUncalibratedProbabilities(
        dual=False, multi_class="ovr", random_state=0
    )
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    assert np.array_equal(y_pred, y_test)

    # test our new predict_proba function
    y_decision_function = clf.decision_function(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    assert (
        y_decision_function.shape
        == y_pred_proba.shape
        == (X_test.shape[0], len(clf.classes_))
    ), "In 2d case, decision_function and predict_proba should both have the same (n_samples, n_classes) shape"
    assert np.allclose(np.sum(y_pred_proba, axis=1), 1.0), "All rows must sum to 1"


def test_linearsvc_softmax_probabilities_binary(sample_data_binary):
    (X_train, y_train, X_test, y_test) = sample_data_binary

    # test this classifier end-to-end to confirm it functions at all
    clf = LinearSVCWithUncalibratedProbabilities(
        dual=False, multi_class="ovr", random_state=0
    )
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    assert np.array_equal(y_pred, y_test)

    # test our new predict_proba function
    y_decision_function = clf.decision_function(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    assert y_decision_function.shape == (
        X_test.shape[0],
    ), "In 1d case, only decision_function should have (n_samples, ) shape"
    assert y_pred_proba.shape == (
        X_test.shape[0],
        len(clf.classes_),
    ), "In 1d case, only predict_proba should have (n_samples, n_classes) shape"
    assert np.allclose(np.sum(y_pred_proba, axis=1), 1.0), "All rows must sum to 1"
