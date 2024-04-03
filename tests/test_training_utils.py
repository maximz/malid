import pytest
import numpy as np
import pandas as pd
import crosseval
from sklearn.datasets import make_classification
from wrap_glmnet import GlmnetLogitNetWrapper
from malid.train import training_utils
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from malid.external.standard_scaler_that_preserves_input_type import (
    StandardScalerThatPreservesInputType,
)
from StratifiedGroupKFoldRequiresGroups import (
    StratifiedGroupKFoldRequiresGroups,
)
from malid.train.one_vs_rest_except_negative_class_classifier import (
    OneVsRestClassifier,
    OneVsNegativeClassifier,
)


def test_prepend_scaler_if_not_present_with_estimator():
    result = training_utils.prepend_scaler_if_not_present(LogisticRegression())
    assert type(result) == Pipeline
    assert len(result.steps) == 2
    assert result.steps[0][0] == "standardscalerthatpreservesinputtype"
    assert isinstance(result.steps[0][1], StandardScalerThatPreservesInputType)
    assert result.steps[1][0] == "logisticregression"
    assert isinstance(result.steps[1][1], LogisticRegression)


def test_prepend_scaler_if_not_present_with_pipeline():
    result = training_utils.prepend_scaler_if_not_present(
        make_pipeline(LogisticRegression())
    )
    assert type(result) == Pipeline
    assert len(result.steps) == 2
    assert result.steps[0][0] == "standardscalerthatpreservesinputtype"
    assert isinstance(result.steps[0][1], StandardScalerThatPreservesInputType)
    assert result.steps[1][0] == "logisticregression"
    assert isinstance(result.steps[1][1], LogisticRegression)


def test_prepend_scaler_if_not_present_with_pipeline_that_already_has_standardscaler_standard():
    result = training_utils.prepend_scaler_if_not_present(
        make_pipeline(StandardScaler(), LogisticRegression())
    )
    assert type(result) == Pipeline
    assert len(result.steps) == 2
    # no change
    assert result.steps[0][0] == "standardscaler"
    assert isinstance(result.steps[0][1], StandardScaler)
    assert result.steps[1][0] == "logisticregression"
    assert isinstance(result.steps[1][1], LogisticRegression)


def test_prepend_scaler_if_not_present_with_pipeline_that_already_has_standardscaler_special():
    result = training_utils.prepend_scaler_if_not_present(
        make_pipeline(StandardScalerThatPreservesInputType(), LogisticRegression())
    )
    assert type(result) == Pipeline
    assert len(result.steps) == 2
    assert result.steps[0][0] == "standardscalerthatpreservesinputtype"
    assert isinstance(result.steps[0][1], StandardScalerThatPreservesInputType)
    assert result.steps[1][0] == "logisticregression"
    assert isinstance(result.steps[1][1], LogisticRegression)


# specify name format for test reporting, so it's clear which parameter is which
@pytest.mark.parametrize(
    "original_setting", [True, False], ids=lambda val: f"original_setting={val}"
)
@pytest.mark.parametrize(
    "wrap_in_pipeline", [True, False], ids=lambda val: f"wrap_in_pipeline={val}"
)
def test_lambda_swap(original_setting: bool, wrap_in_pipeline: bool):
    """Test swapping lambda on a glmnet model."""
    clf = GlmnetLogitNetWrapper(
        use_lambda_1se=original_setting,
    )
    assert clf.use_lambda_1se is original_setting
    if wrap_in_pipeline:
        clf = training_utils.prepend_scaler_if_not_present(clf)

    # Make data
    X, y = make_classification(
        n_samples=100, n_features=3, n_informative=3, n_redundant=0, random_state=42
    )
    X_train = X[:75, :]
    X_test = X[75:, :]
    y_train = y[:75]
    y_test = y[75:]

    # Fit
    clf, performance = training_utils.run_model_multiclass(
        model_name="lasso_cv",
        model_clf=clf,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        fold_id=0,
        output_prefix=None,
        fold_label_train="train",
        fold_label_test="test",
        fail_on_error=True,
        export=False,
    )
    # Sanity check that use_lambda_1se has not changed after run_model_multiclass
    # (This was broken before)
    assert clf is not None
    assert (
        crosseval._get_final_estimator_if_pipeline(clf).use_lambda_1se
        is original_setting
    )

    assert training_utils.does_fitted_model_support_lambda_setting_change(clf)

    assert performance is not None
    (
        modified_clf,
        modified_performance,
    ) = training_utils.modify_fitted_model_lambda_setting(
        fitted_clf=clf,
        performance=performance,
        X_test=X_test,
        output_prefix=None,
        export=False,
    )
    assert (
        "lambda1se" in modified_performance.model_name
        or "lambdamax" in modified_performance.model_name
    )
    assert training_utils.does_fitted_model_support_lambda_setting_change(modified_clf)
    final_clf = crosseval._get_final_estimator_if_pipeline(modified_clf)
    assert isinstance(final_clf, GlmnetLogitNetWrapper)
    assert final_clf.use_lambda_1se is not original_setting


@pytest.fixture
def data():
    diseases = ["Covid19", "Healthy", "HIV"]
    X, y = make_classification(
        n_samples=1000,
        n_classes=len(diseases),
        n_features=5,
        n_informative=5,
        n_redundant=0,
        random_state=42,
        n_clusters_per_class=1,
    )
    X = pd.DataFrame(X).rename(columns=lambda s: f"col{s}")
    weights = np.ones(len(y)) * 0.5

    y = pd.Series(y).replace(dict(enumerate(diseases))).to_numpy()

    # create patient IDs, ensuring consistent disease y label for each patient
    groups = np.empty(len(y), dtype=object)
    for disease in diseases:
        patients = [f"{disease}_patient{i}" for i in range(10)]
        groups[y == disease] = np.random.choice(patients, size=sum(y == disease))

    return X, y, weights, groups


@pytest.mark.parametrize(
    "clf",
    [
        GlmnetLogitNetWrapper(),
        make_pipeline(StandardScaler(), GlmnetLogitNetWrapper()),
        OneVsRestClassifier(
            estimator=GlmnetLogitNetWrapper(
                # this will confirm that groups are being passed through into the CV splitter
                internal_cv=StratifiedGroupKFoldRequiresGroups(
                    n_splits=3, shuffle=True, random_state=0
                ),
            ),
        ),
        make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                estimator=GlmnetLogitNetWrapper(
                    # this will confirm that groups are being passed through into the CV splitter
                    internal_cv=StratifiedGroupKFoldRequiresGroups(
                        n_splits=3, shuffle=True, random_state=0
                    ),
                ),
            ),
        ),
    ],
)
def test_glmnet_to_sklearn_conversion(clf, data):
    """We should be able to convert the following to Sklearn versions:
    - standalone Glmnet models
    - Glmnet models wrapped in a sklearn Pipeline
    - Glmnet models wrapped in OneVsRestClassifier or its child classes
    - Glmnet models wrapped in OneVsRestClassifier or its child classes and wrapped in a sklearn Pipeline
    """
    X, y, weights, groups = data

    # Should fail if not yet fitted
    with pytest.raises(ValueError, match="The model must be fitted before conversion."):
        assert (
            training_utils.does_fitted_model_support_conversion_from_glmnet_to_sklearn(
                clf
            )
        )

    # Use this wrapper around clf.fit() to customize name of sample_weight and groups parameter for Pipelines.
    clf, _ = crosseval.train_classifier(
        clf=clf, X_train=X, y_train=y, train_sample_weights=weights, train_groups=groups
    )

    assert training_utils.does_fitted_model_support_conversion_from_glmnet_to_sklearn(
        clf
    )
    # Prepare to train a sklearn model using the best lambda from the glmnet model.
    sklearn_clf = training_utils.convert_trained_glmnet_pipeline_to_untrained_sklearn_pipeline_at_best_lambda(
        clf
    )
    # Confirm we have no more glmnet
    new_final_clf = crosseval._get_final_estimator_if_pipeline(sklearn_clf)
    assert not isinstance(new_final_clf, GlmnetLogitNetWrapper)
    if isinstance(new_final_clf, OneVsRestClassifier):
        assert not isinstance(new_final_clf.estimator, GlmnetLogitNetWrapper)
