import numpy as np
import pandas as pd
import sklearn.base
import pytest
from malid.train.one_vs_rest_except_negative_class_classifier import (
    OneVsNegativeClassifier,
    OneVsRestClassifier,
)
from sklearn.linear_model import LogisticRegression
from StratifiedGroupKFoldRequiresGroups import (
    StratifiedGroupKFoldRequiresGroups,
)
from sklearn.datasets import make_classification
from wrap_glmnet import GlmnetLogitNetWrapper
from malid.train.vj_gene_specific_sequence_model_rollup_classifier_as_binary_ovr import (
    OneVsNegativeClassifierWithFeatureSubsettingByClass,
    BinaryOvRClassifierWithFeatureSubsettingByClass,
)
import re
import warnings

# see https://github.com/pytest-dev/pytest/pull/4682/files for an alternative:
from contextlib import nullcontext as does_not_raise


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
    clf = OneVsNegativeClassifier(
        estimator=inner_clf,
        negative_class="Healthy",
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
    # Healthy is always the negative class
    assert clf.estimators_[0].negative_class == "Healthy"
    assert clf.estimators_[1].negative_class == "Healthy"
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


@pytest.mark.parametrize(
    "inner_clf",
    [
        pytest.lazy_fixture("inner_clf_basic"),
        pytest.lazy_fixture("inner_clf_that_requires_groups"),
    ],
)
@pytest.mark.parametrize("allow_some_classes_to_fail_to_train", [False, True])
def test_wrap_in_binary_ovr(inner_clf, data, allow_some_classes_to_fail_to_train: bool):
    """Test OneVsNegativeClassifierWithFeatureSubsettingByClass, but first confirm why it's needed"""
    X, y, weights, groups = data
    X.columns = ["HIV:0", "Covid19:0", "Healthy:0", "HIV:1", "HIV_NotARealFeature"]

    # Test a wrapping approach that fails, because the outer wrapper converts class labels to [0 1] so the inner wrapper fails to find the negative class.

    if allow_some_classes_to_fail_to_train:
        # Special case:
        # We should have a failure regardless of allow_some_classes_to_fail_to_train.
        # But the error message will be different.`
        expected_error = ValueError
        excepted_error_message = (
            "Failed to train any classes: all _fit_binary calls failed"
        )
    else:
        # It's a ValueError wrapped inside a RuntimeError using exception chaining:
        # The OvR wrapper catches errors and raises RuntimeError with original error as the cause.
        # The original error message is appended so we can still match on it, but that's why we look for RuntimeError instead of the expected ValueError.
        expected_error = RuntimeError
        excepted_error_message = "The specified negative class 'Healthy' is not in y, which has classes [0 1]"

    with pytest.raises(
        expected_error,
        match=re.escape(
            # match expects a regex pattern and we have special characters, so use re.escape:
            excepted_error_message
        ),
    ):
        # Right before the error, this warning is thrown:
        # "FutureWarning: elementwise comparison failed; returning scalar, but in the future will perform elementwise comparison"
        # That happens at _filter_label_binarizer_classes() line "if self.negative_class not in label_binarizer_classes"
        # You can convert the warning into an error with the following code before the clf fit call below:
        # with warnings.catch_warnings():
        #     warnings.simplefilter(action="error", category=FutureWarning)
        #     clf.fit(...)
        clf = BinaryOvRClassifierWithFeatureSubsettingByClass(
            estimator=OneVsNegativeClassifier(
                estimator=inner_clf,
                negative_class="Healthy",
                allow_some_classes_to_fail_to_train=allow_some_classes_to_fail_to_train,
            ),
            # Our feature names are start with the class name and then a colon
            feature_name_class_delimeter=":",
            allow_some_classes_to_fail_to_train=allow_some_classes_to_fail_to_train,
        )
        clf = clf.fit(X, y, sample_weight=weights, groups=groups)

    # Confirm that our unified class does not have this issue.
    # Should not raise an error, and should not throw the FutureWarning either.
    with warnings.catch_warnings():
        # If we hit the same FutureWarning described above, throw an error
        warnings.simplefilter(action="error", category=FutureWarning)

        clf = OneVsNegativeClassifierWithFeatureSubsettingByClass(
            estimator=inner_clf,
            negative_class="Healthy",
            # Our feature names are start with the class name and then a colon
            feature_name_class_delimeter=":",
        )
        clf = clf.fit(X, y, sample_weight=weights, groups=groups)

        # Confirm that the behavior mixes both classes, as expected:
        assert len(clf.estimators_) == 2
        assert clf.estimators_[0].positive_class == "Covid19"
        assert clf.estimators_[1].positive_class == "HIV"
        # Healthy should be the negative class:
        assert clf.estimators_[0].negative_class == "Healthy"
        assert clf.estimators_[1].negative_class == "Healthy"
        # Submodels have fewer features than the total original
        assert clf.estimators_[0].clf.n_features_in_ == 1
        assert clf.estimators_[1].clf.n_features_in_ == 2
        assert clf.n_features_in_ == 5


@pytest.mark.parametrize("allow_some_classes_to_fail_to_train", [False, True])
def test_some_classes_fail_in_one_vs_rest_classifier(
    inner_clf_that_requires_groups, allow_some_classes_to_fail_to_train: bool
):
    """Confirm that OneVsRestClassifier does not fail when some classes fail to train, if allow_some_classes_to_fail_to_train enabled"""
    # Trigger a failure in the inner classifier for one class.
    # To do so, we will have only one sample for that class.
    # This will cause internal cross validation to fail for that class's binary OvR problem.

    ## Generate patient data
    # HIV is the odd class out, with a single patient.
    # Other classes have 3 patients, so 3-fold group cross validation shouldn't fail for them.
    patient_labels = {
        "patientA": "Covid19",
        "patientB": "Healthy",
        "patientC": "HIV",
        "patientD": "Covid19",
        "patientE": "Healthy",
        "patientF": "Covid19",
        "patientG": "Healthy",
    }

    X_list = []
    y_list = []
    weights_list = []
    groups_list = []

    # Generate data for each patient
    for patient, label in patient_labels.items():
        X_patient, y_patient = make_classification(
            n_samples=25,  # Adjust the number of samples per patient
            n_classes=3,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            random_state=42,
            n_clusters_per_class=1,
        )

        # Convert y_patient to the specific disease label
        y_patient = np.array([label] * len(y_patient))

        # Append the data to the lists
        X_list.append(X_patient)
        y_list.append(y_patient)
        weights_list.extend([0.5] * len(y_patient))
        groups_list.extend([patient] * len(y_patient))

    # Combine the data from all patients
    X_combined = np.vstack(X_list)
    y_combined = np.concatenate(y_list)
    weights_combined = np.array(weights_list)
    groups_combined = np.array(groups_list)

    # Convert X to DataFrame and rename columns
    X_combined = pd.DataFrame(X_combined).rename(columns=lambda s: f"col{s}")

    with does_not_raise() if allow_some_classes_to_fail_to_train else pytest.raises(
        RuntimeError,
        # We believe this is roughly the same situation that caused "Training data need to contain at least 2 classes"
        match="Only one class present in y_true",
    ):
        clf = OneVsRestClassifier(
            estimator=inner_clf_that_requires_groups,
            allow_some_classes_to_fail_to_train=allow_some_classes_to_fail_to_train,
        )
        clf = clf.fit(
            X_combined,
            y_combined,
            sample_weight=weights_combined,
            groups=groups_combined,
        )

        # We should have error'ed out if allow_some_classes_to_fail_to_train is False
        assert allow_some_classes_to_fail_to_train

        # HIV class should be missing:
        assert np.array_equal(clf.classes_, ["Covid19", "Healthy"])

        # HIV estimator should be missing:
        assert len(clf.estimators_) == 2
        assert clf.estimators_[0].positive_class == "Covid19"
        assert clf.estimators_[1].positive_class == "Healthy"


@pytest.mark.parametrize("allow_some_classes_to_fail_to_train", [False, True])
def test_all_classes_fail_in_one_vs_rest_classifier(
    inner_clf_that_requires_groups, allow_some_classes_to_fail_to_train: bool
):
    """Confirm that OneVsRestClassifier fail when all classes fail to train, regardless of allow_some_classes_to_fail_to_train"""
    # Trigger a failure in the inner classifier for all classes.
    # To do so, we will have only one sample for these class.
    # This will cause internal cross validation to fail for that class's binary OvR problem.

    ## Generate patient data: each class has a single patient, so 3-fold group cross validation should fail.
    patient_labels = {
        "patientA": "Covid19",
        "patientB": "Healthy",
        "patientC": "HIV",
    }

    X_list = []
    y_list = []
    weights_list = []
    groups_list = []

    # Generate data for each patient
    for patient, label in patient_labels.items():
        X_patient, y_patient = make_classification(
            n_samples=25,  # Adjust the number of samples per patient
            n_classes=3,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            random_state=42,
            n_clusters_per_class=1,
        )

        # Convert y_patient to the specific disease label
        y_patient = np.array([label] * len(y_patient))

        # Append the data to the lists
        X_list.append(X_patient)
        y_list.append(y_patient)
        weights_list.extend([0.5] * len(y_patient))
        groups_list.extend([patient] * len(y_patient))

    # Combine the data from all patients
    X_combined = np.vstack(X_list)
    y_combined = np.concatenate(y_list)
    weights_combined = np.array(weights_list)
    groups_combined = np.array(groups_list)

    # Convert X to DataFrame and rename columns
    X_combined = pd.DataFrame(X_combined).rename(columns=lambda s: f"col{s}")

    # Exceptions thrown regardless of allow_some_classes_to_fail_to_train, just different messages
    if allow_some_classes_to_fail_to_train:
        error_type = ValueError
        error_msg = "Failed to train any classes: all _fit_binary calls failed"
    else:
        error_type = RuntimeError
        error_msg = "Training data need to contain at least 2 classes"
    with pytest.raises(
        error_type,
        match=error_msg,
    ):
        clf = OneVsRestClassifier(
            estimator=inner_clf_that_requires_groups,
            allow_some_classes_to_fail_to_train=allow_some_classes_to_fail_to_train,
        )
        clf = clf.fit(
            X_combined,
            y_combined,
            sample_weight=weights_combined,
            groups=groups_combined,
        )
