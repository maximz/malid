import numpy as np
import pandas as pd
import sklearn.base
import pytest
from malid.train.one_vs_rest_except_negative_class_classifier import (
    OneVsRestExceptNegativeClassClassifier,
)
from sklearn.linear_model import LogisticRegression
from StratifiedGroupKFoldRequiresGroups import (
    StratifiedGroupKFoldRequiresGroups,
)
from sklearn.datasets import make_classification
from wrap_glmnet import GlmnetLogitNetWrapper


@pytest.fixture
def data():
    X, y = make_classification(
        n_samples=100,
        n_classes=3,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        random_state=42,
        n_clusters_per_class=1,
    )
    X = pd.DataFrame(X).rename(columns=lambda s: f"col{s}")
    y = pd.Series(y).replace(dict(enumerate(["Covid19", "Healthy", "HIV"]))).to_numpy()
    weights = np.ones(len(y)) * 0.5
    groups = np.random.choice(
        ["patientA", "patientB", "patientC", "patientD"], size=len(y)
    )
    return X, y, weights, groups


@pytest.fixture
def inner_clf_basic():
    return LogisticRegression(penalty="l1", solver="saga")


@pytest.fixture
def inner_clf_that_requires_groups():
    # We'll make sure that groups parameter is passed through.
    # repurpose the GlmnetLogitNetWrapper from model_definitions.py
    return GlmnetLogitNetWrapper(
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
    )


@pytest.mark.parametrize(
    "inner_clf",
    [
        pytest.lazy_fixture("inner_clf_basic"),
        pytest.lazy_fixture("inner_clf_that_requires_groups"),
    ],
)
@pytest.mark.parametrize("normalize_predicted_probabilities", [False, True])
def test_works_and_has_sklearn_properties(
    data, inner_clf: sklearn.base.BaseEstimator, normalize_predicted_probabilities: bool
):
    X, y, weights, groups = data
    clf = OneVsRestExceptNegativeClassClassifier(
        estimator=inner_clf,
        other_class_label="Healthy",
        normalize_predicted_probabilities=normalize_predicted_probabilities,
    )
    # not fitted yet
    assert not hasattr(clf, "classes_")

    # Fit with feature names first
    clf = clf.fit(X, y, sample_weight=weights, groups=groups)

    # make sure these attributes exist
    assert clf.n_features_in_ == 5
    assert np.array_equal(
        clf.feature_names_in_, ["col0", "col1", "col2", "col3", "col4"]
    )

    assert clf.predict(X).shape == (100,)

    # notice that Healthy class was removed
    assert len(clf.estimators_) == 2
    assert np.array_equal(clf.classes_, ["Covid19", "HIV"])
    assert clf.estimators_[0].positive_class == "Covid19"
    assert clf.estimators_[1].positive_class == "HIV"
    assert clf.predict_proba(X).shape == (100, 2)
    assert clf.decision_function(X).shape == (100, 2)

    # confirm normalize_predicted_probabilities was respected
    assert (
        np.allclose(clf.predict_proba(X).sum(axis=1), 1)
        == normalize_predicted_probabilities
    )

    # make sure the labels are encoded properly
    assert all(predicted_label in clf.classes_ for predicted_label in clf.predict(X))

    # Refit without feature names
    clf = clf.fit(X.values, y, sample_weight=weights, groups=groups)
    assert clf.n_features_in_ == 5
    assert not hasattr(clf, "feature_names_in_")

    # Confirm that cloning is supported
    clf = sklearn.base.clone(clf)
    assert not hasattr(clf, "n_features_in_")
    assert not hasattr(clf, "feature_names_in_")
    assert not hasattr(clf, "classes_")
    clf = clf.fit(
        X.values, y, sample_weight=weights, groups=groups
    )  # Can refit after cloning
