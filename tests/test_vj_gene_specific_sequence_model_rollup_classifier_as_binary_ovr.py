import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from malid.train.one_vs_rest_except_negative_class_classifier import OneVsRestClassifier
from malid.train.vj_gene_specific_sequence_model_rollup_classifier_as_binary_ovr import (
    BinaryOvRClassifierWithFeatureSubsettingByClass,
)
from StratifiedGroupKFoldRequiresGroups import (
    StratifiedGroupKFoldRequiresGroups,
)
from wrap_glmnet import GlmnetLogitNetWrapper
import pytest

"""
Goals:
- Run on some fake data with a simple LogisticRegression.
- Check overall feature names in and submodel feature names in.
- Add a feature that is always ignored (never present in submodels)
- Confirm that doesn't accept numpy inputs.
"""


@pytest.fixture
def data_binary():
    """Create a fake binary dataset for testing."""
    # Fake data specific to class A and class B
    X_A, y_A = make_classification(
        n_samples=100, n_features=3, n_informative=3, n_redundant=0, random_state=42
    )
    X_B, y_B = make_classification(
        n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=43
    )

    df_A = pd.DataFrame(X_A, columns=[f"A_{i}" for i in range(X_A.shape[1])])
    df_B = pd.DataFrame(X_B, columns=[f"B_{i}" for i in range(X_B.shape[1])])

    # Merge dataframes along axis=1 (i.e., add more columns) and add a dummy feature
    X = pd.concat([df_A, df_B], axis=1)
    X["dummy_feature"] = 1.0
    y = np.where(y_A == 1, "A", "B")

    weights = np.ones(len(y)) * 0.5
    groups = np.random.choice(
        ["patientA", "patientB", "patientC", "patientD"], size=len(y)
    )

    return X, y, weights, groups


@pytest.fixture
def data_multiclass():
    """Create a fake multiclass dataset for testing."""
    # Fake data specific to class A, class B, and class C
    X_A, y_A = make_classification(
        n_samples=100, n_features=3, n_informative=3, n_redundant=0, random_state=42
    )
    X_B, y_B = make_classification(
        n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=43
    )
    X_C, y_C = make_classification(
        n_samples=100, n_features=4, n_informative=4, n_redundant=0, random_state=44
    )

    df_A = pd.DataFrame(X_A, columns=[f"A_{i}" for i in range(X_A.shape[1])])
    df_B = pd.DataFrame(X_B, columns=[f"B_{i}" for i in range(X_B.shape[1])])
    df_C = pd.DataFrame(X_C, columns=[f"C_{i}" for i in range(X_C.shape[1])])

    # Merge dataframes along axis=1 (i.e., add more columns) and add a dummy feature
    X = pd.concat([df_A, df_B, df_C], axis=1)
    X["dummy_feature"] = 1.0

    # Create multiclass labels
    y = np.where(y_A == 1, "A", np.where(y_B == 1, "B", "C"))

    weights = np.ones(len(y)) * 0.5
    groups = np.random.choice(
        ["patientA", "patientB", "patientC", "patientD"], size=len(y)
    )

    return X, y, weights, groups


@pytest.fixture
def clf_basic():
    clf = BinaryOvRClassifierWithFeatureSubsettingByClass(
        estimator=LogisticRegression(), feature_name_class_delimeter="_"
    )
    return clf


@pytest.fixture
def clf_that_requires_groups():
    # We'll make sure that groups parameter is passed through.
    clf = BinaryOvRClassifierWithFeatureSubsettingByClass(
        # repurpose the GlmnetLogitNetWrapper from model_definitions.py
        estimator=GlmnetLogitNetWrapper(
            alpha=1,
            internal_cv=StratifiedGroupKFoldRequiresGroups(
                n_splits=2, shuffle=True, random_state=0
            ),
            n_lambda=3,
            standardize=False,
            scoring=GlmnetLogitNetWrapper.rocauc_scorer,
            n_jobs=1,
            verbose=True,
            random_state=0,
            use_lambda_1se=True,
            class_weight="balanced",
            # This is the key extra check!
            # We will check for error:
            # "GlmnetLogitNetWrapper requires groups parameter in fit() call because require_cv_group_labels was set to True."
            require_cv_group_labels=True,
        ),
        feature_name_class_delimeter="_",
    )
    return clf


@pytest.mark.parametrize(
    "clf",
    [pytest.lazy_fixture("clf_basic"), pytest.lazy_fixture("clf_that_requires_groups")],
)
@pytest.mark.parametrize(
    "data,num_classes,num_estimators",
    [
        # binary problem: 2 classes, 1 estimator
        (pytest.lazy_fixture("data_binary"), 2, 1),
        # multiclass problem: 3 classes, 3 estimators
        (pytest.lazy_fixture("data_multiclass"), 3, 3),
    ],
)
def test_fit(
    clf: BinaryOvRClassifierWithFeatureSubsettingByClass,
    data,
    num_classes: int,
    num_estimators: int,
):
    """Test the fit method."""
    X, y, weights, groups = data
    clf.fit(X, y, sample_weight=weights, groups=groups)

    assert len(clf.classes_) == num_classes
    assert len(clf.estimators_) == num_estimators

    # Check feature names
    assert "A_1" in clf.feature_names_in_
    assert "B_1" in clf.feature_names_in_
    if num_classes == 3:
        assert "C_1" in clf.feature_names_in_
    assert "dummy_feature" in clf.feature_names_in_

    # Check individual classifiers
    for est in clf.estimators_:
        clf_features = est.clf.coef_.shape[1]
        if est.positive_class == "A":
            assert clf_features == 3
            assert "A_1" in est.clf.feature_names_in_
            assert "B_1" not in est.clf.feature_names_in_
            assert "C_1" not in est.clf.feature_names_in_
            assert "dummy_feature" not in est.clf.feature_names_in_
        elif est.positive_class == "B":
            assert clf_features == 2
            assert "A_1" not in est.clf.feature_names_in_
            assert "B_1" in est.clf.feature_names_in_
            assert "C_1" not in est.clf.feature_names_in_
            assert "dummy_feature" not in est.clf.feature_names_in_
        elif est.positive_class == "C":
            assert clf_features == 4
            assert "A_1" not in est.clf.feature_names_in_
            assert "B_1" not in est.clf.feature_names_in_
            assert "C_1" in est.clf.feature_names_in_
            assert "dummy_feature" not in est.clf.feature_names_in_
        else:
            raise ValueError(f"Unexpected positive_class: {est.positive_class}")

    assert isinstance(clf, OneVsRestClassifier)


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary"),
        pytest.lazy_fixture("data_multiclass"),
    ],
)
def test_fail_if_positive_class_has_no_features(clf_basic, data):
    """Test that the classifier rejects numpy inputs."""
    X, y, *_ = data

    # It's a ValueError wrapped inside a RuntimeError using exception chaining:
    # The OvR wrapper catches errors and raises RuntimeError with original error as the cause.
    # The original error message is appended so we can still match on it, but that's why we look for RuntimeError instead of the expected ValueError.
    with pytest.raises(RuntimeError, match="No features found for positive class A"):
        clf_basic.fit(X.rename(columns=lambda s: s.removeprefix("A_")), y)


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary"),
        pytest.lazy_fixture("data_multiclass"),
    ],
)
def test_reject_numpy_input(clf_basic, data):
    """Test that the classifier rejects numpy inputs."""
    X, y, *_ = data
    with pytest.raises(
        ValueError, match="X must be a pandas dataframe with feature names"
    ):
        clf_basic.fit(X.values, y)


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary"),
        pytest.lazy_fixture("data_multiclass"),
    ],
)
def test_dummy_feature_ignored(clf_basic, data):
    """Test that dummy_feature is ignored in individual classifiers."""
    X, y, *_ = data
    clf_basic.fit(X, y)

    for est in clf_basic.estimators_:
        assert "dummy_feature" not in est.clf.feature_names_in_


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_binary"),
        pytest.lazy_fixture("data_multiclass"),
    ],
)
def test_predict(clf_basic, data):
    """Test the predict method."""
    X, y, *_ = data
    clf_basic.fit(X, y)
    predictions = clf_basic.predict(X)
    assert len(predictions) == len(y)


@pytest.mark.parametrize(
    "data,num_classes",
    [
        (pytest.lazy_fixture("data_binary"), 2),
        (pytest.lazy_fixture("data_multiclass"), 3),
    ],
)
def test_predict_proba(clf_basic, data, num_classes: int):
    """Test the predict_proba method."""
    X, y, *_ = data
    clf_basic.fit(X, y)
    probabilities = clf_basic.predict_proba(X)
    assert probabilities.shape == (len(y), num_classes)


@pytest.mark.parametrize(
    "data,num_estimators",
    [
        (pytest.lazy_fixture("data_binary"), 1),
        (pytest.lazy_fixture("data_multiclass"), 3),
    ],
)
def test_decision_function(clf_basic, data, num_estimators: int):
    """Test the decision_function method."""
    X, y, *_ = data
    clf_basic.fit(X, y)
    decision_scores = clf_basic.decision_function(X)
    if num_estimators > 1:
        assert decision_scores.shape == (len(y), num_estimators)
    else:
        assert decision_scores.shape == (len(y),)
