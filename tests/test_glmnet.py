import numpy as np
import pandas as pd
import sklearn.base
import pytest
from malid.external.glmnet_wrapper import GlmnetLogitNetWrapper
import glmnet.scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold


@pytest.fixture
def data():
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(15, 5)).rename(columns=lambda s: f"col{s}")
    y = np.array(["Covid19", "Healthy", "HIV"] * 5)
    participant_label = np.array(
        [
            "covid1",
            "healthy1",
            "hiv1",
            "covid2",
            "healthy2",
            "hiv2",
            "covid3",
            "healthy3",
            "hiv3",
            "covid1",
            "healthy1",
            "hiv1",
            "covid2",
            "healthy2",
            "hiv2",
        ]
    )
    return X, y, participant_label


def test_sklearn_clonable(data):
    X, y, groups = data
    estimator = GlmnetLogitNetWrapper(alpha=1, n_lambda=5, n_splits=3)
    # Check that supports cloning with sklearn.base.clone
    estimator_clone = sklearn.base.clone(estimator)

    # not fitted yet
    assert not hasattr(estimator, "classes_")
    assert not hasattr(estimator_clone, "classes_")

    # fit
    estimator = estimator.fit(X, y, groups=groups)

    # confirm fitted
    estimator.classes_ = np.array(["a", "b"])
    assert hasattr(estimator, "classes_")

    # confirm clone is not fitted
    estimator_clone_2 = sklearn.base.clone(estimator)
    assert not hasattr(estimator_clone_2, "classes_")


def test_scorer(data):
    for scorer in [
        GlmnetLogitNetWrapper.rocauc_scorer,
        glmnet.scorer.make_scorer(matthews_corrcoef),
    ]:
        X, y, groups = data
        clf = GlmnetLogitNetWrapper(alpha=1, n_lambda=5, n_splits=3, scoring=scorer)
        clf = clf.fit(X, y, groups=groups)
        assert clf.cv_mean_score_final_ is not None, f"scorer {scorer} failed"


def test_has_sklearn_properties(data):
    X, y, groups = data
    clf = GlmnetLogitNetWrapper(
        alpha=1, n_lambda=5, n_splits=3, scoring=GlmnetLogitNetWrapper.rocauc_scorer
    )

    # Fit with feature names first
    clf = clf.fit(X, y, groups=groups)
    # make sure these attributes exist
    assert clf.n_features_in_ == 5
    assert np.array_equal(
        clf.feature_names_in_, ["col0", "col1", "col2", "col3", "col4"]
    )
    assert np.array_equal(clf.classes_, ["Covid19", "HIV", "Healthy"])
    assert clf.predict(X).shape == (15,)
    assert clf.predict_proba(X).shape == (15, 3)
    # make sure the labels are encoded
    assert all(predicted_label in clf.classes_ for predicted_label in clf.predict(X))
    assert clf.coef_.shape == (3, 5)

    # Refit without feature names
    clf = clf.fit(X.values, y, groups=groups)
    assert clf.n_features_in_ == 5
    assert not hasattr(clf, "feature_names_in_")

    # Confirm again that cloning works, even after a real fit
    clf = sklearn.base.clone(clf)
    assert not hasattr(clf, "n_features_in_")
    assert not hasattr(clf, "feature_names_in_")
    assert not hasattr(clf, "classes_")


def test_lambda(data):
    X, y, groups = data
    clf = GlmnetLogitNetWrapper(
        alpha=1, n_lambda=100, n_splits=3, scoring=GlmnetLogitNetWrapper.rocauc_scorer
    ).fit(X, y, groups=groups)

    # lambda_max_ is always a scalar
    assert np.isscalar(clf.lambda_max_)
    assert np.isscalar(clf._inner.lambda_max_)
    assert clf._inner.lambda_max_ == clf.lambda_max_

    # lambda_best_ is wrapped in an array, but we unwrap it
    assert np.isscalar(clf.lambda_best_)
    assert not np.isscalar(clf._inner.lambda_best_)
    assert clf._inner.lambda_best_[0] == clf.lambda_best_

    # lambda_max_inx_ is always a scalar
    assert np.isscalar(clf.lambda_max_inx_)
    assert np.isscalar(clf._inner.lambda_max_inx_)
    assert clf._inner.lambda_max_inx_ == clf.lambda_max_inx_

    # lambda_best_inx_ is wrapped in an array, but we unwrap it
    assert np.isscalar(clf.lambda_best_inx_)
    assert not np.isscalar(clf._inner.lambda_best_inx_)
    assert clf._inner.lambda_best_inx_[0] == clf.lambda_best_inx_

    # test selection of the lambda we want to use
    lambda_best_performance: float = clf.lambda_max_
    lambda_best_performance_inx: int = clf.lambda_max_inx_
    lambda_1se_performance_simpler_model: float = clf.lambda_best_
    lambda_1se_performance_simpler_model_inx: int = clf.lambda_best_inx_
    assert (
        lambda_best_performance != lambda_1se_performance_simpler_model
    ), "lambda_best_performance should not equal lambda_1se_performance_simpler_model, otherwise our test is meaningless"
    sample_input = np.random.randn(1, 5)

    def test(correct_value, correct_index, incorrect_value, incorrect_index):
        assert clf._lambda_for_prediction_ == correct_value
        assert clf._lambda_inx_for_prediction_ == correct_index

        assert np.array_equal(
            clf.predict(sample_input), clf.predict(sample_input, lamb=correct_value)
        )

        assert np.array_equal(
            clf.predict_proba(sample_input),
            clf.predict_proba(sample_input, lamb=correct_value),
        )
        assert not np.array_equal(
            clf.predict_proba(sample_input),
            clf.predict_proba(sample_input, lamb=incorrect_value),
        )

        assert np.array_equal(
            clf.decision_function(sample_input),
            clf.decision_function(sample_input, lamb=correct_value),
        )
        assert not np.array_equal(
            clf.decision_function(sample_input),
            clf.decision_function(sample_input, lamb=incorrect_value),
        )

        assert np.array_equal(clf.coef_, clf.coef_path_[:, :, correct_index])
        assert not np.array_equal(clf.coef_, clf.coef_path_[:, :, incorrect_index])

        assert np.array_equal(clf.intercept_, clf.intercept_path_[:, correct_index])
        assert not np.array_equal(
            clf.intercept_, clf.intercept_path_[:, incorrect_index]
        )

        assert np.array_equal(
            clf.cv_mean_score_final_, clf.cv_mean_score_[correct_index]
        )
        assert not np.array_equal(
            clf.cv_mean_score_final_, clf.cv_mean_score_[incorrect_index]
        )

        assert np.array_equal(
            clf.cv_standard_error_final_, clf.cv_standard_error_[correct_index]
        )
        assert not np.array_equal(
            clf.cv_standard_error_final_, clf.cv_standard_error_[incorrect_index]
        )

    # test with default first
    assert clf.use_lambda_1se
    test(
        correct_value=lambda_1se_performance_simpler_model,
        correct_index=lambda_1se_performance_simpler_model_inx,
        incorrect_value=lambda_best_performance,
        incorrect_index=lambda_best_performance_inx,
    )

    # switch to non-default and retest
    clf.use_lambda_1se = False
    assert not clf.use_lambda_1se
    test(
        correct_value=lambda_best_performance,
        correct_index=lambda_best_performance_inx,
        incorrect_value=lambda_1se_performance_simpler_model,
        incorrect_index=lambda_1se_performance_simpler_model_inx,
    )


def test_require_cv_group_labels(data):
    # Confirm require_cv_group_labels is respected
    X, y, groups = data

    clf = GlmnetLogitNetWrapper(
        require_cv_group_labels=False,
        alpha=1,
        n_lambda=5,
        n_splits=3,
        scoring=GlmnetLogitNetWrapper.rocauc_scorer,
    )
    clf = clf.fit(X, y, groups=groups)
    clf = clf.fit(X, y)

    clf = GlmnetLogitNetWrapper(
        require_cv_group_labels=True,
        alpha=1,
        n_lambda=5,
        n_splits=3,
        scoring=GlmnetLogitNetWrapper.rocauc_scorer,
    )
    clf = clf.fit(X, y, groups=groups)
    with pytest.raises(
        ValueError,
        match="requires groups parameter in fit()",
    ):
        clf = clf.fit(X, y)


def test_accept_cv_splitter(data):
    # Confirms that we can pass a splitter to internal_cv and have it incorporated by glmnet,
    # i.e. our wrapper of _score_lambda_path is applied (tested in test_cv_split_wrapper_applied),
    # and the inner model's _cv is set to our provided CV splitter (tested here).

    X, y, groups = data
    splitter = StratifiedGroupKFold(n_splits=4)

    clf = GlmnetLogitNetWrapper(
        alpha=1,
        n_lambda=5,
        scoring=GlmnetLogitNetWrapper.rocauc_scorer,
        internal_cv=splitter
        # notice we are not passing n_splits. it gets defaulted to n_splits=3 at initialization
    )
    clf = clf.fit(X, y, groups=groups)
    assert clf.cv_mean_score_final_ is not None, f"no CV score generated"
    assert (
        clf._inner._cv == splitter
    ), "clf._inner._cv should be replaced by our splitter"
    assert (
        clf.n_splits == 4
    ), "clf.n_splits should be autofilled by our splitter's n_splits"
    assert (
        clf._inner.n_splits == 4
    ), "clf._inner.n_splits should be replaced by our splitter's n_splits"


def test_cv_split_wrapper_applied(data):
    # separating this out from the above test in case the imports change anything
    # __wrapped__ is set by functools.wraps
    import glmnet.util, glmnet.logistic

    assert hasattr(glmnet.util._score_lambda_path, "__wrapped__")
    assert hasattr(glmnet.logistic._score_lambda_path, "__wrapped__")

    X, y, groups = data
    splitter = StratifiedGroupKFold(n_splits=3)

    clf = GlmnetLogitNetWrapper(
        alpha=1,
        n_lambda=5,
        scoring=GlmnetLogitNetWrapper.rocauc_scorer,
        internal_cv=splitter
        # notice we are not passing n_splits
    )
    clf = clf.fit(X, y, groups=groups)

    # still wrapped
    assert hasattr(glmnet.util._score_lambda_path, "__wrapped__")
    assert hasattr(glmnet.logistic._score_lambda_path, "__wrapped__")


def test_nsplits_below_3_still_accepted(data):
    # glmnet special cases n_splits<3, but we override that

    X, y, groups = data
    splitter = StratifiedGroupKFold(n_splits=2)

    clf = GlmnetLogitNetWrapper(
        alpha=1,
        n_lambda=5,
        scoring=GlmnetLogitNetWrapper.rocauc_scorer,
        internal_cv=splitter
        # notice we are not passing n_splits. it gets defaulted to n_splits=3 at initialization
    )
    clf = clf.fit(X, y, groups=groups)
    assert clf.cv_mean_score_final_ is not None, f"no CV score generated"
    assert (
        clf.n_splits == 2
    ), "clf.n_splits should be autofilled by our splitter's n_splits"
    assert (
        clf._inner.n_splits == 3
    ), "clf._inner.n_splits should be falsely set to 3 to avoid glmnet not doing CV"


def test_nsplits_below_3_still_accepted_also_pass_nsplits_explicitly(data):
    # glmnet special cases n_splits<3, but we override that

    X, y, groups = data
    splitter = StratifiedGroupKFold(n_splits=2)

    clf = GlmnetLogitNetWrapper(
        alpha=1,
        n_lambda=5,
        scoring=GlmnetLogitNetWrapper.rocauc_scorer,
        internal_cv=splitter,
        n_splits=2,  # variant: pass explicitly
    )
    clf = clf.fit(X, y, groups=groups)
    assert clf.cv_mean_score_final_ is not None, f"no CV score generated"
    assert (
        clf.n_splits == 2
    ), "clf.n_splits should be autofilled by our splitter's n_splits"
    assert (
        clf._inner.n_splits == 3
    ), "clf._inner.n_splits should be falsely set to 3 to avoid glmnet not doing CV"


def test_plot(data):
    X, y, groups = data
    clf = GlmnetLogitNetWrapper(
        alpha=1, n_lambda=5, n_splits=3, scoring=GlmnetLogitNetWrapper.rocauc_scorer
    )
    clf = clf.fit(X, y, groups=groups)
    clf.plot_cross_validation_curve("ROC AUC")
