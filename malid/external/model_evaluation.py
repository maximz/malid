import collections.abc
import glob
import logging
import time
import dataclasses
from dataclasses import InitVar, dataclass, fields, field
from functools import cache, cached_property
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, Union, Optional
from typing_extensions import Self

import genetools
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sentinels
from kdict import kdict
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline
from enum import Enum, auto

import malid.external.model_evaluation_scores

logger = logging.getLogger(__name__)

from malid.external.genetools_plots import plot_confusion_matrix
from malid.external.genetools_stats import make_confusion_matrix
from enum_mixins import ValidatableEnumMixin

# TODO: is there a type hint for all sklearn models? BaseEstimator is odd to put here: https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
Classifier = Union[Pipeline, BaseEstimator]


def is_clf_a_sklearn_pipeline(clf: Classifier) -> bool:
    # clf may be an individual estimator, or it may be a pipeline, in which case the estimator is the final pipeline step
    return type(clf) == Pipeline


def _get_final_estimator_if_pipeline(clf: Classifier) -> BaseEstimator:
    """If this is a pipeline, return final step (after any transformations). Otherwise pass through."""
    if is_clf_a_sklearn_pipeline(clf):
        return clf.steps[-1][1]
    else:
        return clf


def _get_feature_names(clf: Classifier) -> Union[List[str], List[int], None]:
    """Get feature names from classifier"""
    # If this is a pipeline, get feature names from the last step (after any transformations).
    final_estimator: BaseEstimator = _get_final_estimator_if_pipeline(clf)

    # Get number of features inputted to the final pipeline step,
    # which may be different than the number of features inputted to the whole pipeline originally (X_test.shape[1]).
    # However this can be None for DummyClassifier.
    n_features: Optional[int] = final_estimator.n_features_in_

    if n_features is None:
        return None

    feature_names = None
    if is_clf_a_sklearn_pipeline(clf):
        # for pipelines, this is an alternate way of getting feature names after all transformations
        # unfortunately hasattr(clf[:-1], "get_feature_names_out") does not guarantee no error,
        # can still hit estimator does not provide get_feature_names_out?
        try:
            feature_names = clf[:-1].get_feature_names_out()
        except AttributeError:
            pass

    if feature_names is not None:
        return feature_names

    # Above approach failed - still None
    if hasattr(final_estimator, "feature_names_in_"):
        # Get feature names from classifier
        # feature_names_in_ can be undefined, would throw AttributeError
        return final_estimator.feature_names_in_
    else:
        # Feature names are not available.
        return list(range(n_features))


def _extract_feature_importances(clf: Classifier) -> Optional[np.ndarray]:
    """
    get feature importances or coefficients from a classifier.
    does not support multiclass OvR/OvO.
    """
    # If this is a pipeline, get feature importances from the last step (after any transformations).
    final_estimator: BaseEstimator = _get_final_estimator_if_pipeline(clf)

    if hasattr(final_estimator, "feature_importances_"):
        # random forest
        # one feature importance vector per fold
        return np.ravel(final_estimator.feature_importances_)
    elif hasattr(final_estimator, "coef_") and final_estimator.coef_.shape[0] == 1:
        # linear model - access coef_
        # coef_ is ndarray of shape (1, n_features) if binary or (n_classes, n_features) if multiclass

        # Here we handle the case of a linear model with a single feature importance vector
        # Multiclass OvR/OvO will be handled separately

        # we will flatten each (1, n_features)-shaped vector to a (n_features,)-shaped vector
        return np.ravel(final_estimator.coef_)
    else:
        # Model has no attribute 'feature_importances_' or 'coef_',
        # or is a multiclass linear model
        return None


def _extract_multiclass_feature_importances(clf: Classifier) -> Optional[np.ndarray]:
    """get feature importances or coefficients from a multiclass OvR/OvO classifier."""
    # If this is a pipeline, use the final estimator
    final_estimator: BaseEstimator = _get_final_estimator_if_pipeline(clf)

    if hasattr(final_estimator, "coef_") and final_estimator.coef_.shape[0] > 1:
        # coef_ is ndarray of shape (1, n_features) if binary,
        # or (n_classes, n_features) if multiclass OvR,
        # or (n_classes * (n_classes - 1) / 2, n_features) if multiclass OvO.
        if final_estimator.coef_.shape[0] == len(final_estimator.classes_):
            return final_estimator.coef_

        else:
            # TODO: Handle multiclass OvO?
            # Note: sometimes n_classes * (n_classes - 1) / 2 == n_classes, e.g. for n_classes = 3.
            # So we might still fall into the case above, thinking it's OvR when really it's OvO.
            # Let's add a warning for the user.
            return None

    # Not a multiclass linear model
    return None


def train_classifier(
    clf: Classifier,
    model_name: str,
    fold_id: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_weights: Optional[np.ndarray] = None,
    train_groups: Optional[np.ndarray] = None,
    export_clf_fname: Optional[Union[str, Path]] = None,
):
    logger.info(f"Training {model_name} on fold {fold_id}...")

    itime = time.time()

    # clf may be an individual estimator, or it may be a pipeline, in which case the estimator is the final pipeline step
    is_pipeline = is_clf_a_sklearn_pipeline(clf)

    # check if the estimator (or final pipeline step, if pipeline) accepts sample weights
    fit_parameters = inspect.signature(
        clf[-1].fit if is_pipeline else clf.fit
    ).parameters
    estimator_supports_sample_weight = "sample_weight" in fit_parameters.keys()
    estimator_supports_groups = "groups" in fit_parameters.keys()
    estimator_supports_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in fit_parameters.values()
    )
    extra_kwargs_warning_message = " (Classifier does support kwargs, but we don't pass any because it may cause issues down the call stack.)"

    def make_kwarg_name(name):
        if is_pipeline:
            # Fitting a pipeline with sample weights requires this odd syntax.
            # https://stackoverflow.com/a/36224909/130164
            # https://github.com/scikit-learn/scikit-learn/issues/18159
            last_step_name = clf.steps[-1][0]
            return last_step_name + "__" + name
        else:
            # Just a plain-old estimator, not a pipeline.
            # No parameter renaming necessary.
            return name

    fit_kwargs = {}
    if train_sample_weights is not None:
        # User wants to use sample weights
        if not estimator_supports_sample_weight:
            # Classifier does not support sample weights
            msg = f"Classifier {model_name} does not support sample weights -- fitting without them."
            if estimator_supports_kwargs:
                # But classifier does support arbitrary kwargs, which may be used to pass sample weights, but we're not sure
                msg += extra_kwargs_warning_message
            logger.warning(msg)
        else:
            # Fit with sample weights.
            fit_kwargs[make_kwarg_name("sample_weight")] = train_sample_weights

    if train_groups is not None:
        # User wants to use groups
        if not estimator_supports_groups:
            # Classifier does not support sample weights
            msg = f"Classifier {model_name} does not support groups parameter -- fitting without it."
            if estimator_supports_kwargs:
                # But classifier does support arbitrary kwargs, which may be used to pass groups, but we're not sure
                msg += extra_kwargs_warning_message
            logger.warning(msg)
        else:
            # Fit with groups parameter.
            fit_kwargs[make_kwarg_name("groups")] = train_groups

    # Fit.
    clf = clf.fit(X_train, y_train, **fit_kwargs)

    elapsed_time = time.time() - itime
    logger.info(
        f"Finished training {model_name} on fold {fold_id} (input shape = {X_train.shape}) in {int(elapsed_time)} seconds."
    )

    if export_clf_fname is not None:
        # Save clf (or pipeline) to disk
        try:
            joblib.dump(clf, export_clf_fname)
        except Exception as err:
            logger.error(
                f"Error in saving {model_name} clf to disk for fold {fold_id}: {err}"
            )

    return clf, elapsed_time


@dataclass(frozen=True, order=True)
class Metric:
    """Store metric value along with friendly name. Comparison operators of Metric objects only compare the metric values, not the names."""

    value: float
    # don't use friendly_name in equality and inequality (e.g. greater than) comparisons
    friendly_name: str = field(compare=False)


def _compute_classification_scores(
    y_true: np.ndarray,
    y_preds: np.ndarray,
    y_preds_proba: Optional[np.ndarray] = None,
    y_preds_proba_classes: Optional[np.ndarray] = None,
    sample_weights: Optional[np.ndarray] = None,
    label_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
    probability_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
) -> Dict[str, Metric]:
    """Get classification scores.
    Pass in metrics: map output dictionary key name to (scoring function, friendly name) tuple.
    """
    if len(y_true) == 0:
        raise ValueError("Cannot compute scores when y_true is empty.")

    # Default metrics
    if label_scorers is None:
        label_scorers = {
            "accuracy": (accuracy_score, "Accuracy"),
            "mcc": (matthews_corrcoef, "MCC"),
        }
    if probability_scorers is None:
        probability_scorers = {
            "rocauc": (malid.external.model_evaluation_scores.roc_auc_score, "ROC-AUC"),
            "auprc": (malid.external.model_evaluation_scores.auprc, "au-PRC"),
        }

    output = {}
    for label_scorer_name, (
        label_scorer_func,
        label_scorer_friendly_name,
    ) in label_scorers.items():
        output[label_scorer_name] = Metric(
            value=label_scorer_func(y_true, y_preds),
            friendly_name=label_scorer_friendly_name,
        )
        if sample_weights is not None:
            # Compute classification metrics with test sample weights
            output[f"{label_scorer_name}_samples_weighted"] = Metric(
                value=label_scorer_func(y_true, y_preds, sample_weight=sample_weights),
                friendly_name=f"{label_scorer_friendly_name} with sample weights",
            )
    if y_preds_proba is not None:
        if y_preds_proba_classes is None:
            raise ValueError(
                "y_preds_proba_classes must be provided if y_preds_proba is provided"
            )

        # defensive cast to numpy array
        y_score = np.array(y_preds_proba)

        # handle binary classification case for roc-auc score
        # (multiclass_probabilistic_score_with_missing_labels handles this for us, but still doing in case we use other evaluation approaches.)
        y_score = y_score[:, 1] if len(y_preds_proba_classes) == 2 else y_score
        for probability_scorer_name, (
            probability_scorer_func,
            probability_scorer_friendly_name,
        ) in probability_scorers.items():
            try:
                # sample_weight is not supported for multiclass one-vs-one ROC AUC
                # use weighted average by default, but provide macro average just in case.
                for key_name, average in [
                    (probability_scorer_name, "weighted"),
                    (f"{probability_scorer_name}_macro_average", "macro"),
                ]:
                    output[key_name] = Metric(
                        value=probability_scorer_func(
                            y_true=y_true,
                            y_score=y_score,
                            labels=y_preds_proba_classes,
                            average=average,
                            multi_class="ovo",
                        ),
                        friendly_name=f"{probability_scorer_friendly_name} ({average} OvO)",
                    )

            except Exception as err:
                logger.error(
                    f"Error in evaluating predict-proba-based metric {probability_scorer_name}: {err}"
                )
    return output


@dataclass(eq=False)
class ModelSingleFoldPerformance:
    """Evaluate trained classifier. Gets performance of one model on one fold."""

    model_name: str
    fold_id: int
    y_true: Union[np.ndarray, pd.Series]
    fold_label_train: str
    fold_label_test: Optional[
        str
    ]  # Optional if we're evaluating on a dataset with no "fold label". But there should still be a fold label for the training set that was used.

    # InitVar means these variables are used in __post_init__ but not stored in the dataclass afterwards.
    # Mark these optional. If they aren't provided, the user should directly provide the fields normally filled in by post_init (see below).
    clf: InitVar[Classifier] = None
    X_test: InitVar[np.ndarray] = None

    # These fields are normally filled in by post_init.
    # However we are not going to do the conventional ` = field(init=False)`
    # because for backwards-compatability, we want ability to initialize with these directly, rather than providing clf and X_test.
    # So we provide them default values of None so that we can call init() without passing,
    # but we don't set their types to Optional because, by the time __post_init__ is done, we want these to be set.
    # (They are not truly optional!)
    y_pred: Union[np.ndarray, pd.Series] = None
    class_names: Union[List[str], List[int], np.ndarray] = None
    # These can be optional, if we avoid the standard path of passing clf and X_test:
    X_test_shape: Optional[Tuple[int, int]] = None
    y_decision_function: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None
    y_preds_proba: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None
    feature_names: Optional[Union[List[str], List[int]]] = None
    feature_importances: Optional[pd.Series] = None
    multiclass_feature_importances: Optional[
        pd.DataFrame
    ] = None  # n_classes x n_features, for one-vs-rest models.

    # Truly optional parameters:
    # TODO: Make these be default_factories to be consistent with FeaturizedData?
    test_metadata: Optional[pd.DataFrame] = None
    test_abstentions: Optional[
        np.ndarray
    ] = None  # ground truth labels for abstained examples
    test_abstention_metadata: Optional[
        pd.DataFrame
    ] = None  # metadata for abstained examples
    train_time: Optional[float] = None  # elapsed time for training
    test_sample_weights: Optional[np.ndarray] = None
    test_abstention_sample_weights: Optional[
        np.ndarray
    ] = None  # sample weights for abstained examples. must be provided if we have abstentions and if test_sample_weights are provided.
    export_clf_fname: Optional[
        Union[str, Path]
    ] = None  # this can record where the classifier was saved, so it can be reloaded later

    def __post_init__(self, clf: Classifier = None, X_test: np.ndarray = None):
        if clf is None or X_test is None:
            # If we are not provided with clf and X_test, the user must provide the following fields themselves.
            # Confirm that they have.
            if any(x is None for x in [self.y_pred, self.class_names]):
                raise ValueError(
                    "Must provide clf and X_test to initialize ModelSingleFoldPerformance, or initialize with pre-computed fields directly."
                )
        else:
            self.y_pred = clf.predict(X_test)
            self.class_names = clf.classes_
            self.X_test_shape = X_test.shape

            # Set optional properties
            if hasattr(clf, "decision_function"):
                self.y_decision_function = clf.decision_function(X_test)
            if hasattr(clf, "predict_proba"):
                # Get predicted class probabilities
                self.y_preds_proba = clf.predict_proba(X_test)

            self.feature_names = _get_feature_names(clf)
            if self.feature_names is not None:
                # Get feature importances - this may return None:
                feature_importances: Optional[
                    np.ndarray
                ] = _extract_feature_importances(clf)
                if feature_importances is not None:
                    # Sanity check the shape
                    n_features = len(self.feature_names)
                    if feature_importances.shape[0] != n_features:
                        raise ValueError(
                            f"Feature importances shape {feature_importances.shape} does not match expected n_features = {n_features}"
                        )

                    # Transform into Series
                    self.feature_importances = pd.Series(
                        feature_importances, index=self.feature_names
                    )
                else:
                    self.feature_importances = None

                # Special case: multiclass OvR feature importances
                # This too might return None
                multiclass_feature_importances: Optional[
                    np.ndarray
                ] = _extract_multiclass_feature_importances(clf)
                if multiclass_feature_importances is not None:
                    # Sanity check the shape
                    n_features = len(self.feature_names)
                    if multiclass_feature_importances.shape[1] != n_features:
                        raise ValueError(
                            f"Multiclass feature importances shape {multiclass_feature_importances.shape} does not match expected n_features = {n_features}"
                        )

                    n_classes = len(self.class_names)
                    if multiclass_feature_importances.shape[0] != n_classes:
                        raise ValueError(
                            f"Multiclass feature importances shape {multiclass_feature_importances.shape} does not match expected n_classes = {n_classes}"
                        )

                    # Transform into DataFrame
                    self.multiclass_feature_importances = pd.DataFrame(
                        multiclass_feature_importances,
                        index=self.class_names,
                        columns=self.feature_names,
                    )
                else:
                    self.multiclass_feature_importances = None

        # Validation
        if self.y_true.shape[0] != self.y_pred.shape[0]:
            raise ValueError(
                f"y_true shape {self.y_true.shape} does not match y_pred shape {self.y_pred.shape}"
            )
        if (
            self.test_metadata is not None
            and self.y_true.shape[0] != self.test_metadata.shape[0]
        ):
            raise ValueError(
                "Metadata was supplied but does not match y_true or y_pred length."
            )
        if self.n_abstentions > 0:
            if self.test_abstention_metadata is not None:
                if self.test_abstention_metadata.shape[0] != self.n_abstentions:
                    raise ValueError(
                        "If test_abstentions_metadata is provided, it must match the number of abstention ground truth labels (test_abstentions) provided."
                    )
            else:  # self.cv_abstentions_metadata is None
                if self.test_metadata is not None:
                    # test_abstention_metadata must be provided if test_abstentions and test_metadata provided
                    raise ValueError(
                        "If there are abstentions and metadata was provided for non-abstained examples, then test_abstentions_metadata (metadata dataframe) must be provided alongside test_abstentions (ground truth labels)"
                    )

            if self.test_abstention_sample_weights is not None:
                if self.test_abstention_sample_weights.shape[0] != self.n_abstentions:
                    raise ValueError(
                        "If test_abstention_sample_weights is provided, it must match the number of abstention ground truth labels (test_abstentions) provided."
                    )
            else:  # self.test_abstention_sample_weights is None
                if self.test_sample_weights is not None:
                    raise ValueError(
                        "If there are abstentions and sample weights was provided for non-abstained examples, then test_abstention_sample_weights must be provided alongside test_abstentions (ground truth labels)"
                    )

    def export(self, metadata_fname: Union[str, Path]):
        # Export this object
        joblib.dump(self, metadata_fname)

    def copy(self) -> Self:
        """Return a copy of this ModelSingleFoldPerformance."""
        # Running dataclasses.replace() without any changes gives:
        # ValueError: InitVar 'clf' must be specified with replace()
        return dataclasses.replace(self, clf=None, X_test=None)

    @cached_property
    def classifier(self) -> Classifier:
        """load original classifier from disk, if export_clf_fname was provided"""
        if self.export_clf_fname is None:
            raise ValueError(
                "Could not load classifier from disk because export_clf_fname is not set"
            )
        return joblib.load(self.export_clf_fname)

    @property
    def n_abstentions(self) -> int:
        if self.test_abstentions is None:
            return 0
        return len(self.test_abstentions)

    def scores(
        self,
        label_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        with_abstention=True,
        exclude_metrics_that_dont_factor_in_abstentions=False,
        abstain_label="Unknown",
    ) -> Dict[str, Metric]:
        # Compute probability scores without abstention.
        # (Probability-based scores do not support abstention.)
        # (We have to separate out computing of the probability scores from the label scores, because the probability scores require y_true without abstention, i.e. not exxpanded)
        computed_scores = _compute_classification_scores(
            y_true=self.y_true,
            y_preds=self.y_pred,
            y_preds_proba=self.y_preds_proba,
            y_preds_proba_classes=self.class_names,
            sample_weights=self.test_sample_weights,
            label_scorers={},  # Disable label scorers
            probability_scorers=probability_scorers,
        )

        # Compute label scores with abstention if supplied and requested.
        to_add = {}
        if with_abstention and self.n_abstentions > 0:
            if exclude_metrics_that_dont_factor_in_abstentions:
                # We are asked to discard probability-based scores that did not factor in abstentions
                computed_scores = {}

            # TODO: warn if abstain_label conflicts with name of existing class
            y_true = np.hstack([self.y_true, self.test_abstentions])
            y_pred = np.hstack([self.y_pred, [abstain_label] * self.n_abstentions])
            # We are guaranteed that test_sample_weights and test_abstention_sample_weights are both given or both null
            sample_weights = (
                np.hstack(
                    [self.test_sample_weights, self.test_abstention_sample_weights]
                )
                if self.test_sample_weights is not None
                else None
            )
            to_add["abstention_rate"] = Metric(
                value=self.n_abstentions / len(y_true),
                friendly_name="Unknown/abstention proportion",
            )
        else:
            y_true = self.y_true
            y_pred = self.y_pred
            sample_weights = self.test_sample_weights

        # Merge dictionaries
        return (
            computed_scores
            | _compute_classification_scores(
                y_true=y_true,
                y_preds=y_pred,
                sample_weights=sample_weights,
                label_scorers=label_scorers,
                # Explicitly disable probability_scorers and do not pass y_preds_proba
                probability_scorers={},
            )
            | to_add
        )

    def apply_abstention_mask(self, mask: np.ndarray) -> Self:
        """Pass a boolean mask. Returns a copy of self with the mask samples turned into abstentions."""
        # TODO: consider exposing this functionality on an ExperimentSetSummary, and on a ModelGlobalPerformance -> basically regenerates by modifying inner ModelSingleFoldPerformances?
        if mask.shape[0] != self.y_true.shape[0]:
            raise ValueError(
                f"Must supply boolean mask, but got mask.shape[0] ({mask.shape[0]}) != self.y_true.shape[0] ({self.y_true.shape[0]})"
            )
        return dataclasses.replace(
            self,
            # Pass in null InitVars to make a copy (see comments in `.copy()` method above)
            clf=None,
            X_test=None,
            # Make changes
            y_true=self.y_true[~mask],
            y_pred=self.y_pred[~mask],
            X_test_shape=(
                self.X_test_shape[0] - np.sum(mask),
                self.X_test_shape[1],
            )
            if self.X_test_shape is not None
            else None,
            y_decision_function=self.y_decision_function[~mask]
            if self.y_decision_function is not None
            else None,
            y_preds_proba=self.y_preds_proba[~mask]
            if self.y_preds_proba is not None
            else None,
            test_metadata=self.test_metadata[~mask]
            # TODO: can this line be removed when we switch test_metadata default to empty dataframe, or do we still need to check shape[0] == 0 edge case?
            if self.test_metadata is not None else None,
            test_sample_weights=self.test_sample_weights[~mask]
            if self.test_sample_weights is not None
            else None,
            test_abstentions=np.hstack(
                [
                    self.test_abstentions
                    # TODO: this line can be removed when we switch test_abstentions default to empty numpy array
                    if self.test_abstentions is not None else [],
                    self.y_true[mask],
                ]
            ),
            test_abstention_metadata=pd.concat(
                [
                    self.test_abstention_metadata
                    # TODO: this line can be removed when we switch test_abstention_metadata default to empty dataframe
                    if self.test_abstention_metadata is not None else pd.DataFrame(),
                    self.test_metadata[mask],
                ],
                axis=0,
            )
            # TODO: this line can be removed when we switch test_metadata default to empty dataframe
            if self.test_metadata is not None else None,
            test_abstention_sample_weights=np.hstack(
                [
                    self.test_abstention_sample_weights
                    if self.test_abstention_sample_weights is not None
                    else [],
                    self.test_sample_weights[mask],
                ]
            )
            if self.test_sample_weights is not None
            else self.test_abstention_sample_weights,
            # All other fields stay the same
        )


# Sentinel value
Y_TRUE_VALUES: sentinels.Sentinel = sentinels.Sentinel("default y_true column")


def _wrap_as_list(list_or_single_value: Union[Any, Iterable]) -> Iterable[Any]:
    """Wrap as list, even if given single value"""
    if isinstance(list_or_single_value, str) or not isinstance(
        list_or_single_value, collections.abc.Iterable
    ):
        return [list_or_single_value]

    # already iterable, and is not a string
    return list_or_single_value


def _stack_numpy_arrays_horizontally_into_string_array(
    arrs: Iterable[np.ndarray],
) -> np.ndarray:
    """create combined tuples of chosen columns (essentially zip)"""
    return np.array([", ".join(item) for item in np.column_stack(arrs).astype(str)])


@dataclass(eq=False)
class ModelGlobalPerformance:
    """Summarizes performance of one model across all CV folds."""

    model_name: str
    per_fold_outputs: Dict[
        int, ModelSingleFoldPerformance
    ]  # map fold ID -> ModelSingleFoldPerformance
    abstain_label: str
    # Optional override of the default column name for global scores.
    # Can pass multiple columns.
    # If provided, each column must either match a metadata column name, or must be model_evaluation.Y_TRUE_VALUES (a sentinel indicating use default y_true).
    # Example use case: override y_true to show confusion matrix to delineate disease groups further into "past exposure" or "active exposure" groups, even though classifier trained on full disease label.
    global_evaluation_column_name: Optional[
        Union[str, sentinels.Sentinel, List[Union[str, sentinels.Sentinel]]]
    ] = None

    def __post_init__(self):
        """Data validation on initialization.
        See https://stackoverflow.com/a/60179826/130164"""

        # Computing all these properties is expensive.
        # TODO: Should we disable most validation for higher performance through lazy evaluation?

        for fold_output in self.per_fold_outputs.values():
            if self.model_name != fold_output.model_name:
                raise ValueError(
                    "All folds must be from the same model. Model_name was different."
                )

        # sanity checks / validation
        # TODO: can we remove some of these since these are now being validated at ModelSingleFoldPerformance level?
        if self.has_abstentions:
            # TODO: warn if abstain_label conflicts with name of existing class
            if self.cv_abstentions_metadata is not None:
                if self.cv_abstentions_metadata.shape[0] != self.n_abstentions:
                    raise ValueError(
                        "If test_abstentions_metadata is provided, it must match the number of abstention ground truth labels (test_abstentions) provided."
                    )
            else:  # self.cv_abstentions_metadata is None
                if self.cv_metadata is not None:
                    raise ValueError(
                        "If there are abstentions and metadata was provided for non-abstained examples, then test_abstentions_metadata (metadata dataframe) must be provided alongside test_abstentions (ground truth labels)"
                    )

            if self._cv_abstentions_sample_weights is not None:
                if self._cv_abstentions_sample_weights.shape[0] != self.n_abstentions:
                    raise ValueError(
                        "If test_abstention_sample_weights is provided, it must match the number of abstention ground truth labels (test_abstentions) provided."
                    )
            else:  # self._cv_abstentions_sample_weights is None
                if self.cv_sample_weights_without_abstention is not None:
                    raise ValueError(
                        "If there are abstentions and sample weights was provided for non-abstained examples, then test_abstention_sample_weights must be provided alongside test_abstentions (ground truth labels)"
                    )

        if self.global_evaluation_column_name is not None:
            # loop over provided column names
            # wrap as list even if given a single value
            colnames = _wrap_as_list(self.global_evaluation_column_name)
            if len(colnames) == 0:
                raise ValueError(
                    "global_evaluation_column_name cannot be an empty list."
                )
            for colname in colnames:
                if colname == Y_TRUE_VALUES:
                    # skip any that match sentinel like model_evaluation.Y_TRUE_COLUMN
                    continue
                # validate that cv_metadata exists and column name is in there
                if self.cv_metadata is None:
                    raise ValueError(
                        "If a global_evaluation_column_name is provided (that is not Y_TRUE_VALUES), then cv_metadata must be provided."
                    )
                elif colname not in self.cv_metadata.columns:
                    raise ValueError(
                        f"global_evaluation_column_name {colname} is not a column in the metadata dataframe"
                    )

                if self.has_abstentions:
                    # and if we have abstentions, validate that cv_abstentions_metadata exists and column name in there
                    if self.cv_abstentions_metadata is None:
                        raise ValueError(
                            "If global_evaluation_column_name is provided (that is not Y_TRUE_VALUES) and there are abstentions, then cv_abstentions_metadata must be provided."
                        )
                    elif colname not in self.cv_abstentions_metadata.columns:
                        # validate that all provided column names are valid abstention metadata column names,
                        # except for a sentinel like model_evaluation.Y_TRUE_COLUMN
                        raise ValueError(
                            f"global_evaluation_column_name {colname} is not a column in the abstentions metadata dataframe"
                        )

        if (
            self.cv_y_true_without_abstention.shape[0]
            != self.cv_y_pred_without_abstention.shape[0]
        ):
            raise ValueError(f"cv_y_true and cv_y_pred must have same shape")

        if (
            self.cv_y_true_with_abstention.shape[0]
            != self.cv_y_pred_with_abstention.shape[0]
        ):
            raise ValueError("cv_y_true and cv_y_pred must have same shape")

        # If any metadata was supplied, make sure all entries (all folds) had metadata supplied.
        if (
            self.cv_metadata is not None
            and self.cv_y_true_without_abstention.shape[0] != self.cv_metadata.shape[0]
        ):
            raise ValueError(
                "Not all folds supplied metadata (cv_y_true_without_abstention and cv_metadata have different lengths)."
            )

        if self.cv_sample_weights_without_abstention is not None:
            if (
                self.cv_sample_weights_without_abstention.shape[0]
                != self.cv_y_pred_without_abstention.shape[0]
            ):
                raise ValueError(
                    "cv_sample_weights_without_abstention and cv_y_pred must have same shape"
                )
            if (
                self.has_abstentions
                and self.cv_sample_weights_without_abstention.shape[0]
                + self._cv_abstentions_sample_weights.shape[0]
                != self.cv_y_pred_with_abstention.shape[0]
            ):
                # sample weights are passed for abstained examples separately,
                # and we already validated above that if non-abstained examples had sample weights, then abstained examples do too.
                # here we check that total counts match.
                raise ValueError(
                    "Sample weights must have same shape as cv_y_pred + # of abstentions"
                )

    @cached_property
    def fold_order(self) -> List[int]:
        return sorted(self.per_fold_outputs.keys())

    @cache
    def aggregated_per_fold_scores(
        self,
        with_abstention=True,
        exclude_metrics_that_dont_factor_in_abstentions=False,
        label_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
    ) -> Dict[str, str]:
        """return dict mapping friendly-metric-name to "mean +/- std" formatted string (computed across folds)"""
        raw_metrics_per_fold: Dict[int, Dict[str, Metric]] = {
            fold_id: fold_output.scores(
                with_abstention=with_abstention,
                exclude_metrics_that_dont_factor_in_abstentions=exclude_metrics_that_dont_factor_in_abstentions,
                abstain_label=self.abstain_label,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            )
            for fold_id, fold_output in self.per_fold_outputs.items()
        }

        # Put everything in a pandas df,
        # aggregate by (metric_keyname, metric_friendlyname),
        # and run mean, std, and size (i.e. report n_folds present in case < total n_folds)

        # This dataframe has index = fold_id, columns = metric keyname, values = Metric object
        scores_per_fold = pd.DataFrame.from_dict(raw_metrics_per_fold, orient="index")
        # extract metric friendlynames (TODO(later): move away from separation of metric-keyname and metric-friendly-name. use friendly name only?)
        map_metric_keyname_to_friendly_name = {}
        for colname in scores_per_fold.columns:
            friendly_names = (
                scores_per_fold[colname]
                .dropna()
                .apply(lambda metric: metric.friendly_name)
                .values
            )
            # These should be identical between folds (though a metric might not appear in all folds)
            if len(set(friendly_names)) > 1:
                raise ValueError(
                    f"Metric friendly names must be unique for metric keyname {colname}"
                )
            map_metric_keyname_to_friendly_name[colname] = friendly_names[0]
        # now change df to have values = metric value rather than full Metric object (again, note that a metric might not appear in all folds)
        scores_per_fold = scores_per_fold.applymap(
            lambda metric: metric.value if isinstance(metric, Metric) else np.nan
        )

        # aggregate mean, standard deviation, and non-NaN count (columns) for each metric keyname (index)
        scores_per_fold_agg = scores_per_fold.describe().loc[["mean", "std", "count"]].T
        scores_per_fold_agg["std"].fillna(
            0, inplace=True
        )  # if a metric appeared in only one fold, it will have std NaN
        # Add metric friendlyname. (unlike replace()'s pass-through behavior, map() means that if not in the dict, will store NaN)
        scores_per_fold_agg[
            "metric_friendly_name"
        ] = scores_per_fold_agg.index.to_series().map(
            map_metric_keyname_to_friendly_name
        )
        if scores_per_fold_agg.isna().any().any():
            raise ValueError("Scores_per_fold_agg had NaNs")
        if scores_per_fold_agg["metric_friendly_name"].duplicated().any():
            raise ValueError("Some metrics had duplicate friendly names")

        # summarize a range of scores into strings: mean plus-minus one standard deviation (68% interval if normally distributed).
        return {
            row[
                "metric_friendly_name"
            ]: f"""{row['mean']:0.3f} +/- {row['std']:0.3f} (in {row['count']:n} folds)"""
            for _, row in scores_per_fold_agg.iterrows()
        }

    @staticmethod
    def _concatenate_in_fold_order(value_name, model_output, fold_order) -> np.ndarray:
        """Combine numpy array from all folds, in fold order, not original anndata.obs order.
        If data not present in folds, returns empty numpy array, instead of None."""
        vals = (getattr(model_output[fold_id], value_name) for fold_id in fold_order)
        vals = [v for v in vals if v is not None]
        if len(vals) == 0:
            return np.array([])
        return np.concatenate(vals).ravel()

    @staticmethod
    def _concatenate_dataframes_in_fold_order(
        value_name, model_output, fold_order
    ) -> pd.DataFrame:
        """Combine pandas dataframe from all folds, in fold order, not original anndata.obs order.
        Index gets reset (original index stored as a column).
        If data not present in folds, returns empty dataframe, instead of None."""
        vals = (getattr(model_output[fold_id], value_name) for fold_id in fold_order)
        return pd.concat(
            [
                # Concatenate in fold order
                v if v is not None
                # Concat empty dataframe in case all metadata dataframes are empty, since we can't pd.concat all Nones
                else pd.DataFrame()
                for v in vals
            ],
            axis=0,
        ).reset_index()  # original index gets stored as a column. keep index as a column in case there was valuable info in there.

    @staticmethod
    def _get_column_combination_or_pass_through(
        requested_column_names,
        func_get_metadata_df,
        func_get_default_ground_truth,
        default_ground_truth_sentinel_value=Y_TRUE_VALUES,
    ):
        """create synthetic column that combines all requested column names
        func_get_metadata_df and func_get_default_ground_truth should be functions for lazy evaluation, because they can be expensive to compute
        """
        if requested_column_names is None:
            return func_get_default_ground_truth()

        # wrap as list in case we were given a single value
        requested_column_names = _wrap_as_list(requested_column_names)

        metadata_df = None
        if any(
            colname != default_ground_truth_sentinel_value
            for colname in requested_column_names
        ):
            # lazy evaluation
            metadata_df = func_get_metadata_df()

        values_to_combine = []
        for colname in requested_column_names:
            if colname == default_ground_truth_sentinel_value:
                values_to_combine.append(func_get_default_ground_truth())
            else:
                values_to_combine.append(metadata_df[colname].values)

        if len(values_to_combine) == 1:
            # pass through if only one column
            return values_to_combine[0]

        # combine
        return _stack_numpy_arrays_horizontally_into_string_array(values_to_combine)

    @cached_property
    def cv_y_true_without_abstention(self) -> np.ndarray:
        # sub in self.global_evaluation_column_name if defined
        return self._get_column_combination_or_pass_through(
            requested_column_names=self.global_evaluation_column_name,
            func_get_metadata_df=lambda: self.cv_metadata,
            # standard output if not overriding with a global_evaluation_column_name
            func_get_default_ground_truth=lambda: self._concatenate_in_fold_order(
                "y_true", self.per_fold_outputs, self.fold_order
            ),
        )

    @cached_property
    def cv_y_pred_without_abstention(self) -> np.ndarray:
        return self._concatenate_in_fold_order(
            "y_pred", self.per_fold_outputs, self.fold_order
        )

    @cached_property
    def _model_output_with_abstention(self) -> pd.DataFrame:
        output = pd.DataFrame(
            {
                "y_true": self.cv_y_true_without_abstention,
                "y_pred": self.cv_y_pred_without_abstention,
            }
        )

        if self.has_abstentions:
            # prepare abstention values
            # sub in self.global_evaluation_column_name if defined
            abstention_values = self._get_column_combination_or_pass_through(
                requested_column_names=self.global_evaluation_column_name,
                func_get_metadata_df=lambda: self.cv_abstentions_metadata,
                # standard output if not overriding with a global_evaluation_column_name
                func_get_default_ground_truth=lambda: self.cv_abstentions,
            )

            # concatenate with abstentions
            output = pd.concat(
                [
                    output,
                    pd.DataFrame({"y_true": abstention_values}).assign(
                        y_pred=self.abstain_label
                    ),
                ],
                axis=0,
            ).reset_index(drop=True)

        return output

    def get_all_entries(self) -> pd.DataFrame:
        """Get all predicted (y_pred) and ground-truth (y_true) labels for each example from all folds, along with any metadata if provided.
        Abstentions are included at the end, if available.
        max_predicted_proba, second_highest_predicted_proba, and difference_between_top_two_predicted_probas included for all but abstentions."""

        true_vs_pred_labels = (
            self._model_output_with_abstention
        )  # abstentions included. default (0 to n-1) index

        if self.cv_y_preds_proba is not None:
            df_probas_top_two = pd.DataFrame(
                self.cv_y_preds_proba.apply(
                    lambda row: pd.Series(row.nlargest(2).values), axis=1
                )
            )
            df_probas_top_two.columns = [
                "max_predicted_proba",
                "second_highest_predicted_proba",
            ]

            df_probas_top_two["difference_between_top_two_predicted_probas"] = (
                df_probas_top_two["max_predicted_proba"]
                - df_probas_top_two["second_highest_predicted_proba"]
            )

            # combine horizontally, but note that this will have NaNs for abstentions
            true_vs_pred_labels = pd.concat(
                [true_vs_pred_labels, df_probas_top_two], axis=1
            )

        if self.cv_metadata is not None:
            metadata_compiled = pd.concat(
                [self.cv_metadata, self.cv_abstentions_metadata], axis=0
            ).reset_index(drop=True)
            # if these columns already exist, rename them
            metadata_compiled = metadata_compiled.rename(
                columns={"y_true": "metadata_y_true", "y_pred": "metadata_y_pred"}
            )
            # combine horizontally
            true_vs_pred_labels = pd.concat(
                [true_vs_pred_labels, metadata_compiled], axis=1
            )

        if true_vs_pred_labels.shape[0] != self._model_output_with_abstention.shape[0]:
            raise ValueError("Shape changed unexpectedly")

        return true_vs_pred_labels

    @property
    def cv_y_true_with_abstention(self) -> np.ndarray:
        """includes abstained ground truth labels"""
        return self._model_output_with_abstention["y_true"].values

    @property
    def cv_y_pred_with_abstention(self) -> np.ndarray:
        """includes "Unknown" or similar when abstained on an example"""
        return self._model_output_with_abstention["y_pred"].values

    @cached_property
    def cv_sample_weights_without_abstention(self) -> Union[np.ndarray, None]:
        """Combine test-set sample weights in fold order, if they were supplied"""
        if not all(
            fold_output.test_sample_weights is not None
            for fold_output in self.per_fold_outputs.values()
        ):
            return None
        return self._concatenate_in_fold_order(
            "test_sample_weights", self.per_fold_outputs, self.fold_order
        )

    @cached_property
    def _cv_abstentions_sample_weights(self) -> Union[np.ndarray, None]:
        """Concatenate sample weights (if supplied) of abstained test examples from each fold.
        (Abstentions don't necessarily need to occur in each fold, though.)
        """
        if not all(
            fold_output.test_abstention_sample_weights is not None
            for fold_output in self.per_fold_outputs.values()
        ):
            return None
        return self._concatenate_in_fold_order(
            "test_abstention_sample_weights", self.per_fold_outputs, self.fold_order
        )

    @cached_property
    def cv_sample_weights_with_abstention(self) -> Union[np.ndarray, None]:
        """Combine test-set sample weights in fold order, then add abstention sample weights if we had abstentions. Returns None if no sample weights supplied."""
        if not self.has_abstentions:
            return self.cv_sample_weights_without_abstention  # might be None
        if self.cv_sample_weights_without_abstention is not None:
            return np.hstack(
                [
                    self.cv_sample_weights_without_abstention,
                    self._cv_abstentions_sample_weights,
                ]
            )
        return None

    @cached_property
    def cv_metadata(self) -> Union[pd.DataFrame, None]:
        """If supplied, concatenate dataframes of metadata for each test example, in fold order, not in original adata.obs order.
        If supplied in any fold, must be supplied for all folds.
        If not supplied in any fold, returns None.
        """
        test_metadata_concat = self._concatenate_dataframes_in_fold_order(
            "test_metadata", self.per_fold_outputs, self.fold_order
        )

        # Return None if no metadata was supplied for any fold.
        if test_metadata_concat.shape[0] > 0:
            return test_metadata_concat
        return None

    @cached_property
    def cv_abstentions(self) -> np.ndarray:
        """Concatenate ground truth labels of abstained test examples from each fold.
        These don't necessarily need to be provided for each fold though.
        """
        return self._concatenate_in_fold_order(
            "test_abstentions", self.per_fold_outputs, self.fold_order
        )

    @cached_property
    def cv_abstentions_metadata(self) -> Union[pd.DataFrame, None]:
        """Concatenate metadata of abstained test examples from each fold.
        These don't necessarily need to be provided for each fold though.
        Returns None if not supplied for any fold.
        """
        test_metadata_concat = self._concatenate_dataframes_in_fold_order(
            "test_abstention_metadata", self.per_fold_outputs, self.fold_order
        )

        # Return None if no metadata was supplied for any fold.
        if test_metadata_concat.shape[0] > 0:
            return test_metadata_concat
        return None

    @cached_property
    def sample_size_without_abstentions(self):
        return self.cv_y_true_without_abstention.shape[0]

    @cached_property
    def sample_size_with_abstentions(self):
        return self.cv_y_true_with_abstention.shape[0]

    @cached_property
    def n_abstentions(self):
        return self.cv_abstentions.shape[0]

    @cached_property
    def has_abstentions(self) -> bool:
        return self.cv_abstentions.shape[0] > 0

    @cached_property
    def abstention_proportion(self):
        """abstention proportion: what percentage of predictions were unknown"""
        return self.n_abstentions / self.sample_size_with_abstentions

    @cached_property
    def cv_y_preds_proba(self) -> Union[pd.DataFrame, None]:
        """Concatenate y_preds_proba (if supplied), in fold order, not in original adata.obs order.
        Abstentions never included here."""
        if not all(
            fold_output.y_preds_proba is not None
            for fold_output in self.per_fold_outputs.values()
        ):
            return None
        # Confirm class names are in same order across all folds.
        for fold_id in self.fold_order:
            if not np.array_equal(
                self.per_fold_outputs[fold_id].class_names,
                self.per_fold_outputs[self.fold_order[0]].class_names,
            ):
                logger.warning(
                    f"Class names are not the same across folds: {fold_id} vs {self.fold_order[0]} for model {self.model_name}"
                )

        # Convert to dataframes with column names, and concatenate
        y_preds_proba_concat = pd.concat(
            [
                pd.DataFrame(
                    self.per_fold_outputs[fold_id].y_preds_proba,
                    columns=self.per_fold_outputs[fold_id].class_names,
                )
                for fold_id in self.fold_order
            ],
            axis=0,
        )

        # Confirm n_examples by n_classes shape.
        if self.cv_y_pred_without_abstention.shape[0] != y_preds_proba_concat.shape[0]:
            raise ValueError(
                "y_preds_proba has different number of rows than cv_y_pred_without_abstention"
            )

        if y_preds_proba_concat.isna().any().any():
            logger.warning(
                f"Model {self.model_name} has missing probabilities for some samples; may be because class names were not the same across folds. Filling with 0s."
            )
            y_preds_proba_concat.fillna(0.0, inplace=True)

        # So far we have included all class names ever predicted by any fold's model.
        # But it's possible there are other class names seen in the data.
        # Add any missing classes to the probability matrix.
        (
            y_preds_proba_concat,
            labels,
        ) = malid.external.model_evaluation_scores._inject_missing_labels(
            y_true=self.confusion_matrix_label_ordering,
            y_score=y_preds_proba_concat.values,
            labels=y_preds_proba_concat.columns,
        )
        # convert back to dataframe (default 0 to n-1 index)
        y_preds_proba_concat = pd.DataFrame(y_preds_proba_concat, columns=labels)

        # Arrange columns in same order as cm_label_order
        if set(y_preds_proba_concat.columns) != set(
            self.confusion_matrix_label_ordering
        ):
            raise ValueError(
                "y_preds_proba has different columns than confusion_matrix_label_ordering (without considering order)"
            )
        y_preds_proba_concat = y_preds_proba_concat[
            self.confusion_matrix_label_ordering
        ]

        return y_preds_proba_concat

    @cached_property
    def classification_report(self) -> str:
        """Classification report"""
        # zero_division=0 is same as "warn" but suppresses this warning for labels with no predicted samples:
        # `UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.``
        return classification_report(
            self.cv_y_true_with_abstention,
            self.cv_y_pred_with_abstention,
            zero_division=0,
        )

    def _full_report_scores(
        self,
        label_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
    ):
        scores = {
            "per_fold": self.aggregated_per_fold_scores(
                with_abstention=False,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            ),
            "global": self.global_scores(
                with_abstention=False,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            ),
        }
        if self.has_abstentions:
            scores["per_fold_with_abstention"] = self.aggregated_per_fold_scores(
                with_abstention=True,
                exclude_metrics_that_dont_factor_in_abstentions=True,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            )
            scores["global_with_abstention"] = self.global_scores(
                with_abstention=True,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            )
        return scores

    def full_report(
        self,
        label_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
    ) -> str:
        def formatted_scores(header, score_dict):
            return "\n".join(
                [header]
                + [
                    f"{metric_friendly_name}: {metric_value}"
                    for metric_friendly_name, metric_value in score_dict.items()
                ]
            )

        # Return per-fold scores, global scores without abstention, and global scores with abstention.
        scores = self._full_report_scores(
            label_scorers=label_scorers, probability_scorers=probability_scorers
        )
        global_column_usage = (
            f" using column name {self.global_evaluation_column_name}"
            if self.global_evaluation_column_name is not None
            else ""
        )
        pieces = [
            formatted_scores(
                f"Per-fold scores{' without abstention' if self.has_abstentions else ''}:",
                scores["per_fold"],
            ),
            formatted_scores(
                f"Global scores{' without abstention' if self.has_abstentions else ''}{global_column_usage}:",
                scores["global"],
            ),
        ]
        if self.has_abstentions:
            pieces.append(
                formatted_scores(
                    f"Per-fold scores with abstention (note that abstentions not included in probability-based scores):",
                    scores["per_fold_with_abstention"],
                ),
            )
            pieces.append(
                formatted_scores(
                    f"Global scores with abstention{global_column_usage}:",
                    scores["global_with_abstention"],
                ),
            )
        pieces.append(
            "\n".join(
                [
                    f"Global classification report{' with abstention' if self.has_abstentions else ''}{global_column_usage}:",
                    self.classification_report,
                ]
            )
        )
        return "\n\n".join(pieces)

    @cache
    def global_scores(
        self,
        with_abstention=True,
        label_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
    ) -> dict:
        """Calculate global scores with or without abstention. Global scores should not include probabilistic scores."""
        scores = _compute_classification_scores(
            y_true=self.cv_y_true_with_abstention
            if with_abstention
            else self.cv_y_true_without_abstention,
            y_preds=self.cv_y_pred_with_abstention
            if with_abstention
            else self.cv_y_pred_without_abstention,
            y_preds_proba=None,
            y_preds_proba_classes=None,
            sample_weights=self.cv_sample_weights_with_abstention
            if with_abstention
            else self.cv_sample_weights_without_abstention,
            label_scorers=label_scorers,
            probability_scorers=probability_scorers,
        )
        # Format
        scores_formatted = [
            (metric.friendly_name, f"{metric.value:0.3f}")
            for metric_keyname, metric in scores.items()
        ]
        # confirm all metric friendly names are unique
        all_metric_friendly_names = [v[0] for v in scores_formatted]
        if len(set(all_metric_friendly_names)) != len(all_metric_friendly_names):
            raise ValueError("Metric friendly names are not unique")

        # then convert to dict
        scores_formatted = dict(scores_formatted)

        if with_abstention:
            scores_formatted[
                "Unknown/abstention proportion"
            ] = f"{self.abstention_proportion:0.3f}"
            scores_formatted["Abstention label"] = self.abstain_label

        if self.global_evaluation_column_name is not None:
            scores_formatted[
                "Global evaluation column name"
            ] = self.global_evaluation_column_name

        return scores_formatted

    def _get_stats(
        self,
        label_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
    ):
        """Get overall stats for table"""
        scores = self._full_report_scores(
            label_scorers=label_scorers, probability_scorers=probability_scorers
        )

        # Combine all scores into single dictionary.
        scores_dict = {}
        for suffix, scores_dict_part in [
            ("per fold", "per_fold"),
            ("global", "global"),
            ("per fold with abstention", "per_fold_with_abstention"),
            ("global with abstention", "global_with_abstention"),
        ]:
            if scores_dict_part not in scores:
                continue
            for k, v in scores[scores_dict_part].items():
                scores_dict[f"{k} {suffix}"] = v

        # Other summary stats.
        nunique_predicted_labels = np.unique(self.cv_y_pred_without_abstention).shape[0]
        nunique_true_labels = np.unique(self.cv_y_true_without_abstention).shape[0]

        scores_dict.update(
            {
                "sample_size": self.sample_size_without_abstentions,
                "n_abstentions": self.n_abstentions,
                "sample_size including abstentions": self.sample_size_with_abstentions,
                "abstention_rate": self.abstention_proportion,
                # Flag if number of unique predicted labels is less than number of unique ground truth labels
                "missing_classes": nunique_predicted_labels < nunique_true_labels,
            }
        )
        return scores_dict

    @cached_property
    def confusion_matrix_label_ordering(self) -> List[str]:
        """Order of labels in confusion matrix"""
        return sorted(
            np.unique(
                np.hstack(
                    [
                        np.unique(
                            self.cv_y_true_with_abstention
                        ),  # includes self.cv_abstentions abstained ground truth labels, if any
                        np.unique(
                            self.cv_y_pred_with_abstention
                        ),  # may include self.abstain_label
                    ]
                )
            )
        )

    def confusion_matrix(
        self,
        confusion_matrix_true_label="Patient of origin",
        confusion_matrix_pred_label="Predicted label",
    ) -> pd.DataFrame:
        """Confusion matrix"""
        return make_confusion_matrix(
            y_true=self.cv_y_true_with_abstention,
            y_pred=self.cv_y_pred_with_abstention,
            true_label=confusion_matrix_true_label,
            pred_label=confusion_matrix_pred_label,
            label_order=self.confusion_matrix_label_ordering,
        )

    def confusion_matrix_fig(
        self,
        confusion_matrix_figsize: Optional[Tuple[float, float]] = None,
        confusion_matrix_true_label="Patient of origin",
        confusion_matrix_pred_label="Predicted label",
    ) -> plt.Figure:
        """Confusion matrix figure"""
        fig, ax = plot_confusion_matrix(
            self.confusion_matrix(
                confusion_matrix_true_label=confusion_matrix_true_label,
                confusion_matrix_pred_label=confusion_matrix_pred_label,
            ),
            figsize=confusion_matrix_figsize,
        )
        plt.close(fig)
        return fig

    def export(
        self,
        classification_report_fname: Union[str, Path],
        confusion_matrix_fname: Union[str, Path],
        confusion_matrix_figsize: Optional[Tuple[float, float]] = None,
        confusion_matrix_true_label="Patient of origin",
        confusion_matrix_pred_label="Predicted label",
        dpi=72,
        label_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
    ):
        with open(classification_report_fname, "w") as w:
            w.write(
                self.full_report(
                    label_scorers=label_scorers, probability_scorers=probability_scorers
                )
            )

        genetools.plots.savefig(
            self.confusion_matrix_fig(
                confusion_matrix_figsize=confusion_matrix_figsize,
                confusion_matrix_true_label=confusion_matrix_true_label,
                confusion_matrix_pred_label=confusion_matrix_pred_label,
            ),
            confusion_matrix_fname,
            dpi=dpi,
        )

    @property
    def per_fold_classifiers(self) -> Dict[int, Classifier]:
        """reload classifier objects from disk"""
        return {
            fold_id: model_single_fold_performance.classifier
            for fold_id, model_single_fold_performance in self.per_fold_outputs.items()
        }

    @cached_property
    def feature_importances(self) -> Union[pd.DataFrame, None]:
        """
        Get feature importances for each fold.
        Extracts feature importances or coefficients from sklearn pipelines' inner models.
        Returns fold_ids x feature_names DataFrame, or None if no feature importances are available.
        """
        feature_importances = {
            fold_id: model_single_fold_performance.feature_importances
            for fold_id, model_single_fold_performance in self.per_fold_outputs.items()
        }

        if all(fi is None for fi in feature_importances.values()):
            # Model had no feature importances in any fold.
            return None

        # Combine all
        return pd.DataFrame.from_dict(feature_importances, orient="index").rename_axis(
            index="fold_id"
        )

    @cached_property
    def multiclass_feature_importances(self) -> Union[Dict[int, pd.DataFrame], None]:
        """
        Get One-vs-Rest multiclass feature importances for each fold.
        Extracts feature importances or coefficients from sklearn pipelines' inner models.
        Returns dict mapping fold_id to a classes x features DataFrame,
        or returns None if no feature importances are available.
        """
        feature_importances = {
            fold_id: model_single_fold_performance.multiclass_feature_importances
            for fold_id, model_single_fold_performance in self.per_fold_outputs.items()
        }

        if all(fi is None for fi in feature_importances.values()):
            # Model had no feature importances in any fold.
            return None

        return feature_importances


@dataclass
class ExperimentSetGlobalPerformance:
    """Summarizes performance of many models across all CV folds."""

    model_global_performances: Dict[
        str, ModelGlobalPerformance
    ]  # map model name -> ModelGlobalPerformance

    def get_model_comparison_stats(
        self,
        label_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        sort=True,
    ):
        if len(self.model_global_performances) == 0:
            # Edge case: empty
            return pd.DataFrame()

        # Put all scores in one table, and optionally sort by first score column.
        combined_stats = pd.DataFrame.from_dict(
            {
                model_name: model_global_performance._get_stats(
                    label_scorers=label_scorers, probability_scorers=probability_scorers
                )
                for model_name, model_global_performance in self.model_global_performances.items()
            },
            orient="index",
        )

        if sort:
            combined_stats.sort_values(
                by=combined_stats.columns[0], ascending=False, inplace=True
            )

        return combined_stats

    def export_all_models(
        self,
        func_generate_classification_report_fname: Callable[[str], Union[str, Path]],
        func_generate_confusion_matrix_fname: Callable[[str], Union[str, Path]],
        confusion_matrix_figsize: Optional[Tuple[float, float]] = None,
        confusion_matrix_true_label="Patient of origin",
        confusion_matrix_pred_label="Predicted label",
        dpi=72,
        label_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
        probability_scorers: Optional[Dict[str, Tuple[Callable, str]]] = None,
    ):
        """Export global results for each model.
        func_generate_classification_report_fname and func_generate_confusion_matrix_fname should be functions that accept model_name str and return a file name.
        """
        for (
            model_name,
            model_global_performance,
        ) in self.model_global_performances.items():
            model_global_performance.export(
                classification_report_fname=func_generate_classification_report_fname(
                    model_name
                ),
                confusion_matrix_fname=func_generate_confusion_matrix_fname(model_name),
                confusion_matrix_figsize=confusion_matrix_figsize,
                confusion_matrix_true_label=confusion_matrix_true_label,
                confusion_matrix_pred_label=confusion_matrix_pred_label,
                dpi=dpi,
                label_scorers=label_scorers,
                probability_scorers=probability_scorers,
            )


class RemoveIncompleteStrategy(ValidatableEnumMixin, Enum):
    """
    How to handle incomplete ExperimentSets: remove incomplete models or remove incomplete folds.

    These represent two strategies for removing incomplete results to compare models apples-to-apples on the same data (i.e. same collection of cross validation folds):

    1. Keep only those models that have results for all folds (DROP_INCOMPLETE_MODELS; sensible default).
    (How this works: Find the maximum number of folds any model has results for; keep only models analyzed for that number of folds.)
    This is relevant when you have some finnicky models that may give up on a fold out of the blue.

    2. Keep only folds that have results for all models (DROP_INCOMPLETE_FOLDS).
    This is relevant when you have a finnicky fold that fails for some but not all models.
    For example, if one cross validation fold's training set somehow only has samples of a single class (perhaps the splits were stratified for one target variable, and now you are evaluating how well you can model another classification target without changing the cross validation structure),
    many models will fail and cite that there was only a single class in the data — but some models might still succeed.
    The right answer may be to drop this broken fold altogether, rather than restricting the analysis to only those models that happen to handle this edge case.
    """

    DROP_INCOMPLETE_MODELS = auto()
    DROP_INCOMPLETE_FOLDS = auto()


class ExperimentSet:
    """Store ModelSingleFoldPerformance objects for many (model_name, fold_id) combinations."""

    model_outputs: kdict  # map (model_name, fold_id) to ModelSingleFoldPerformance

    @classmethod
    def _unwrap_nested_dict(
        cls,
        model_outputs: Union[
            collections.abc.Sequence,
            collections.abc.Mapping,
            ModelSingleFoldPerformance,
        ],
    ):
        if isinstance(model_outputs, ModelSingleFoldPerformance):
            yield model_outputs
        elif isinstance(model_outputs, collections.abc.Mapping):
            if "model_name" in model_outputs.keys():
                # stop
                yield model_outputs
            else:
                for key, value in model_outputs.items():
                    # this is a nested dict, not a dict that should become a ModelSingleFoldPerformance dataclass
                    yield from cls._unwrap_nested_dict(value)
        elif isinstance(model_outputs, collections.abc.Sequence):
            for model_output in model_outputs:
                yield from cls._unwrap_nested_dict(model_output)

    def __init__(
        self,
        model_outputs: Union[
            collections.abc.Sequence, collections.abc.Mapping, None
        ] = None,
    ):
        """stores ModelSingleFoldPerformance objects for each fold and model name.
        accepts existing single-fold model outputs as a list, a kdict, or a nested dict."""
        if model_outputs is None:
            model_outputs = []
        self.model_outputs = kdict()
        for model_output in self._unwrap_nested_dict(model_outputs):
            self.add(model_output)

    def add(self, model_output: Union[ModelSingleFoldPerformance, Dict]) -> None:
        def cast_to_dataclass(model_output):
            if isinstance(model_output, ModelSingleFoldPerformance):
                return model_output
            if isinstance(model_output, collections.abc.Mapping):
                # If dict, wrap dict as dataclass,
                # but only the subset of dict keys that match dataclass field names (and are not non-init fields - which are forbidden to pass)
                all_valid_field_names = [
                    field.name
                    for field in fields(ModelSingleFoldPerformance)
                    if field.init
                ]
                return ModelSingleFoldPerformance(
                    **{
                        k: v
                        for k, v in model_output.items()
                        if k in all_valid_field_names
                    }
                )
            raise ValueError(f"Unrecognized model output type: {type(model_output)}")

        data: ModelSingleFoldPerformance = cast_to_dataclass(model_output)
        self.model_outputs[data.model_name, data.fold_id] = data

    @property
    def incomplete_models(self) -> List[str]:
        if len(self.model_outputs) == 0:
            # edge case: empty
            return []

        n_folds_per_model = {
            model_name: len(self.model_outputs[model_name, :])
            for model_name in self.model_outputs.keys(dimensions=0)
        }
        max_n_folds_per_model = max(n_folds_per_model.values())
        return [
            model_name
            for model_name, n_folds in n_folds_per_model.items()
            if n_folds != max_n_folds_per_model
        ]

    @property
    def incomplete_folds(experiment_set) -> List[int]:
        n_models_per_fold = {
            fold_id: len(experiment_set.model_outputs[:, fold_id])
            for fold_id in experiment_set.model_outputs.keys(dimensions=1)
        }
        max_n_models_per_fold = max(n_models_per_fold.values())
        return [
            fold_id
            for fold_id, n_models in n_models_per_fold.items()
            if n_models != max_n_models_per_fold
        ]

    def remove_incomplete(
        self,
        inplace: bool = True,
        remove_incomplete_strategy: RemoveIncompleteStrategy = RemoveIncompleteStrategy.DROP_INCOMPLETE_MODELS,
    ) -> Self:
        """
        Removes incomplete results, in-place by default (which can be disabled with inplace=False).
        remove_incomplete_strategy determines whether we remove incomplete models (default) or remove incomplete folds. See RemoveIncompleteStrategy for details.
        """
        RemoveIncompleteStrategy.validate(remove_incomplete_strategy)

        if not inplace:
            clone = self.copy()
            return clone.remove_incomplete(
                inplace=False, remove_incomplete_strategy=remove_incomplete_strategy
            )

        if (
            remove_incomplete_strategy
            == RemoveIncompleteStrategy.DROP_INCOMPLETE_MODELS
        ):
            for model_name in self.incomplete_models:
                # TODO: make kdict support: del self.model_outputs[model_name, :] (and vice versa for other branch below)
                for key in self.model_outputs[model_name, :].keys():
                    logger.info(
                        f"Removing {key} because model {model_name} is incomplete."
                    )
                    del self.model_outputs[key]
        elif (
            remove_incomplete_strategy == RemoveIncompleteStrategy.DROP_INCOMPLETE_FOLDS
        ):
            for fold_id in self.incomplete_folds:
                for key in self.model_outputs[:, fold_id].keys():
                    logger.info(f"Removing {key} because fold {fold_id} is incomplete.")
                    del self.model_outputs[key]
        else:
            raise NotImplementedError(
                f"remove_incomplete_strategy={remove_incomplete_strategy} not implemented."
            )

        return self

    def copy(self) -> Self:
        # TODO: write test to confirm the kdicts are the same when copy
        return self.__class__(self.model_outputs)

    @classmethod
    def load_from_disk(cls, output_prefix):
        """alternate constructor: reload all fit models (including partial results) from disk"""
        return cls(
            model_outputs=[
                joblib.load(metadata_fname)
                for metadata_fname in glob.glob(f"{output_prefix}*.metadata_joblib")
            ]
        )

    def summarize(
        self,
        abstain_label: str = "Unknown",
        global_evaluation_column_name: Optional[
            Union[str, sentinels.Sentinel, List[Union[str, sentinels.Sentinel]]]
        ] = None,
        remove_incomplete_strategy: RemoveIncompleteStrategy = RemoveIncompleteStrategy.DROP_INCOMPLETE_MODELS,
    ) -> ExperimentSetGlobalPerformance:
        """
        Summarize classification performance with all models across all folds (ignoring any incomplete models trained on some but not all folds).
        To override default confusion matrix ground truth values, pass global_evaluation_column_name to evaluate on a specific metadata column or combination of columns.
        (You can incorporate the default ground truth values in combination with metadata columns by using the special value `model_evaluation.Y_TRUE_VALUES`)

        The remove_incomplete_strategy parameters controls whether to ignore incomplete models (default) or incomplete folds (False). See RemoveIncompleteStrategy for more details.
        """
        # don't summarize incomplete models, because indices will be distorted by missing fold(s)
        # so clone self and remove incomplete
        self_without_incomplete_models = self.copy().remove_incomplete(
            remove_incomplete_strategy=remove_incomplete_strategy
        )

        return ExperimentSetGlobalPerformance(
            model_global_performances={
                model_name: self_without_incomplete_models._summarize_single_model_across_folds(
                    model_name=model_name,
                    abstain_label=abstain_label,
                    global_evaluation_column_name=global_evaluation_column_name,
                )
                for model_name in self_without_incomplete_models.model_outputs.keys(
                    dimensions=0
                )
            }
        )

    def _summarize_single_model_across_folds(
        self,
        model_name: str,
        abstain_label: str,
        global_evaluation_column_name: Optional[
            Union[str, sentinels.Sentinel, List[Union[str, sentinels.Sentinel]]]
        ] = None,
    ) -> ModelGlobalPerformance:
        return ModelGlobalPerformance(
            model_name=model_name,
            per_fold_outputs={
                fold_id: val
                for (model_name, fold_id), val in self.model_outputs[
                    model_name, :
                ].items()
            },
            abstain_label=abstain_label,
            global_evaluation_column_name=global_evaluation_column_name,
        )


@dataclass(eq=False)
class FeaturizedData:
    """Container for featurized data for models."""

    X: Union[np.ndarray, pd.DataFrame]
    y: Union[np.ndarray, pd.Series]  # Ground truth
    sample_names: Union[np.ndarray, pd.Series, pd.Index, List[str]]
    metadata: pd.DataFrame

    sample_weights: Optional[Union[np.ndarray, pd.Series]] = None

    # Optional fields:
    # Express this way to avoid mutable default value
    abstained_sample_names: Union[np.ndarray, pd.Series, pd.Index, List[str]] = field(
        default_factory=lambda: np.empty(0, dtype="object")
    )
    abstained_sample_y: Union[np.ndarray, pd.Series] = field(
        default_factory=lambda: np.empty(0)
    )
    abstained_sample_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    extras: Dict[str, Any] = field(default_factory=dict)

    # TODO: Post-init: validate that if any abstain field is provided, then all are provided
    # TODO: Post-init: if y or sample_names were python lists, convert to numpy arrays, so we can index.
    # TODO: post-init: validate that X.shape[0] == y.shape[0] == sample_names.shape[0] == metadata.shape[0]
    # TODO: should metadata be optional?
    # TODO: add abstained_sample_weights?

    def copy(self) -> Self:
        return dataclasses.replace(self)

    def apply_abstention_mask(self, mask: np.ndarray) -> Self:
        """Pass a boolean mask. Returns a copy of self with the mask samples turned into abstentions."""
        if mask.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"Must supply boolean mask, but got mask.shape[0] ({mask.shape[0]}) != self.X.shape[0] ({self.X.shape[0]})"
            )
        return dataclasses.replace(
            self,
            X=self.X[~mask],
            y=self.y[~mask],
            sample_names=self.sample_names[~mask],
            metadata=self.metadata[~mask],
            sample_weights=self.sample_weights[~mask]
            if self.sample_weights is not None
            else None,
            abstained_sample_names=np.hstack(
                [self.abstained_sample_names, self.sample_names[mask]]
            ),
            abstained_sample_y=np.hstack([self.abstained_sample_y, self.y[mask]]),
            abstained_sample_metadata=pd.concat(
                [self.abstained_sample_metadata, self.metadata[mask]],
                axis=0,
            ),
            # All other fields (i.e. "extras") stay the same
        )

    @classmethod
    def concat(cls: Type[Self], lst: List[Self]) -> Self:
        """Concatenate multiple FeaturizedData objects into one. Extras are dropped."""
        # sample_weights must be None or available for all
        if len(lst) == 0:
            raise ValueError("Cannot concatenate empty list of FeaturizedData objects")
        is_sample_weights_available = [
            featurized_data.sample_weights is not None for featurized_data in lst
        ]
        if any(is_sample_weights_available) and not all(is_sample_weights_available):
            raise ValueError(
                "sample_weights must be None or available for all FeaturizedData objects"
            )
        return cls(
            X=pd.concat(
                [pd.DataFrame(featurized_data.X) for featurized_data in lst], axis=0
            )
            if isinstance(lst[0].X, pd.DataFrame)
            else np.vstack([featurized_data.X for featurized_data in lst]),
            y=np.hstack([featurized_data.y for featurized_data in lst]),
            sample_names=np.hstack(
                [featurized_data.sample_names for featurized_data in lst]
            ),
            metadata=pd.concat(
                [featurized_data.metadata for featurized_data in lst], axis=0
            ),
            sample_weights=np.hstack(
                [featurized_data.sample_weights for featurized_data in lst]
            )
            if any(is_sample_weights_available)
            else None,
            abstained_sample_names=np.hstack(
                [featurized_data.abstained_sample_names for featurized_data in lst]
            ),
            abstained_sample_y=np.hstack(
                [featurized_data.abstained_sample_y for featurized_data in lst]
            ),
            abstained_sample_metadata=pd.concat(
                [featurized_data.abstained_sample_metadata for featurized_data in lst],
                axis=0,
            ),
        )
