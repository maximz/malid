from collections import defaultdict
from pathlib import Path
from typing import Type, Union, List, Optional, Dict, Mapping, Tuple

import dataclasses
import joblib
import numpy as np
from pandas._typing import ArrayLike
import pandas as pd
import sklearn.base
from IPython.display import display
from kdict import kdict
from sklearn.pipeline import Pipeline, make_pipeline
from wrap_glmnet import GlmnetLogitNetWrapper
from malid.external.standard_scaler_that_preserves_input_type import (
    StandardScalerThatPreservesInputType,
)

from malid import config
from malid import io
from malid.datamodels import (
    GeneLocus,
    GeneralAnndataType,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    CVScorer,
)
import choosegpu
import crosseval
from crosseval import (
    Classifier,
    ModelSingleFoldPerformance,
    ExperimentSet,
)
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from malid.train.one_vs_rest_except_negative_class_classifier import OneVsRestClassifier
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)
from StratifiedGroupKFoldRequiresGroups import (
    StratifiedGroupKFoldRequiresGroups,
)

from malid.external.logging_context_for_warnings import ContextLoggerWrapper
from log_with_context import add_logging_context

logger = ContextLoggerWrapper(name=__name__)


def make_internal_cv():
    # Higher number of internal cross validation folds helps with stability because we train each model on more samples and have more runs to average over.
    # But the tradeoff is that each run is evaluated on fewer held-out samples.
    internal_cv_nfolds = 5
    internal_cv = StratifiedGroupKFoldRequiresGroups(
        n_splits=internal_cv_nfolds, shuffle=True, random_state=0
    )
    return internal_cv


def prepare_models_to_train(
    # TODO: Remove these unused parameters
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    fold_id: int,
    target_obs_column: TargetObsColumnEnum,
    chosen_models: List[str],
    use_gpu: bool,
    output_prefix: Path,
    train_metadata: Optional[pd.DataFrame] = None,
    n_jobs=1,
    cv_scoring=CVScorer.Deviance,
    n_lambdas_to_tune: Optional[int] = None,
):
    # import models in child process to avoid infinite hang if running with multiprocessing
    # see https://stackoverflow.com/a/42506478/130164
    # and https://stackoverflow.com/a/44300201/130164
    # and https://github.com/yuanyuanli85/Keras-Multiple-Process-Prediction
    from malid.train.model_definitions import (
        make_models,
    )

    # Configure nested cross validation.
    if train_metadata is None:
        raise ValueError(
            "train_metadata must be supplied for nested (internal) cross validation."
        )
    # Configure training procedure's internal cross-validation for hyperparameters:
    # Ensure that each patient's entire set of sequences is always in either the training set or the test set, but never in both in the same fold.
    # Note that this will stratify internal CV by the y values for a particular target_obs_column,
    # but global CV is not necessarily stratified by that key (always using the one stratified by disease).
    # We use a wrapper of StratifiedGroupKFold that requires groups to be passed in, otherwise it throws error.
    internal_cv = make_internal_cv()
    train_participant_labels = train_metadata["participant_label"].values

    # Configure GPU if requested.
    choosegpu.configure_gpu(enable=use_gpu)

    models = make_models(
        chosen_models=chosen_models,
        n_jobs=n_jobs,
        internal_cv=internal_cv,
        cv_scoring=cv_scoring,
        n_lambdas_to_tune=n_lambdas_to_tune,
    )

    return models, train_participant_labels


def prepend_scaler_if_not_present(clf: Classifier) -> Pipeline:
    """
    Given an estimator or a pipeline, return a pipeline with a StandardScaler prepended if not already present.
    Always returns a new (cloned) copy of the classifier, even if no changes are made.
    Note: we actually use StandardScalerThatPreservesInputType instead of StandardScaler, which allows for "pandas in, pandas out" behavior.
    """
    # model_clf may be an individual estimator, or it may already be a pipeline
    is_pipeline = type(clf) == Pipeline
    if is_pipeline:
        # If already a pipeline, prepend a StandardScalerThatPreservesInputType
        clf = sklearn.base.clone(clf)
        if (
            "standardscaler" in clf.named_steps.keys()
            or "standardscalerthatpreservesinputtype" in clf.named_steps.keys()
        ):
            # The pipeline already has a StandardScaler or StandardScalerThatPreservesInputType step.
            # Nothing more to do.

            # But to be consistent with all other paths through this function that return a new classifier object,
            # we are still returning the cloned version of clf.
            return clf

        # Insert StandardScalerThatPreservesInputType into existing pipeline
        clf.steps.insert(
            0,
            (
                "standardscalerthatpreservesinputtype",
                StandardScalerThatPreservesInputType(),
            ),
        )
        return clf
    else:
        # Not yet a pipeline.
        # Convert to pipeline, with StandardScalerThatPreservesInputType included
        return make_pipeline(StandardScalerThatPreservesInputType(), clf)


def run_model_multiclass(
    model_name: str,
    model_clf: Classifier,
    X_train: Union[ArrayLike, pd.DataFrame],
    X_test: Optional[Union[ArrayLike, pd.DataFrame]],
    y_train: Union[ArrayLike, pd.Series],
    y_test: Optional[Union[ArrayLike, pd.Series]],
    fold_id: int,
    output_prefix: Optional[Union[str, Path]],
    fold_label_train: Optional[str],
    fold_label_test: Optional[str],
    train_sample_weights: Optional[Union[ArrayLike, pd.Series]] = None,
    train_groups: Optional[Union[ArrayLike, pd.Series]] = None,
    test_sample_weights: Optional[Union[ArrayLike, pd.Series]] = None,
    test_metadata: Optional[pd.DataFrame] = None,
    test_abstention_ground_truth_labels: Optional[Union[ArrayLike, pd.Series]] = None,
    test_abstention_metadata: Optional[pd.DataFrame] = None,
    fail_on_error=False,
    export=True,
) -> Tuple[Union[Classifier, None], Union[ModelSingleFoldPerformance, None]]:
    """
    Train a model.

    If exporting model or performance: output_prefix must be provided

    If evaluating performance: X_test, y_test, fold_label_train, and fold_label_test must all be provided to evaluate the trained model and produce a ModelSingleFoldPerformance object.

    Output type depends on inputs:
    - If X_test, y_test, and fold_label_test are not provided: -> Union[Tuple[Classifier, None], Tuple[None, None]].
        Returns (clf, None). Or returns (None, None) if training fails but fail_on_error is False.
        Explanation: We train but do not evaluate. Or we fail to train and still don't evaluate.

    -  If X_test, y_test, fold_label_train, and fold_label_test ARE provided: -> Union[Tuple[Classifier, ModelSingleFoldPerformance], Tuple[None, None]].
        Returns (clf, performance). Or returns (None, None) if training fails but fail_on_error is False.
        Explanation: We train and evaluate. Or we fail to train and therefore don't evaluate either.
    """
    # TODO: move this to crosseval?

    # Validation
    if export:
        if output_prefix is None:
            raise ValueError("output_prefix must be provided if export=True")

    if X_test is not None or y_test is not None:
        if (
            X_test is None
            or y_test is None
            or fold_label_train is None
            or fold_label_test is None
        ):
            raise ValueError(
                "X_test, y_test, fold_label_train, and fold_label_test must all be provided to evaluate the trained model and produce a ModelSingleFoldPerformance object."
            )

    try:
        clf = sklearn.base.clone(model_clf)
        export_clf_fname = (
            f"{output_prefix}.{model_name}.{fold_id}.joblib" if export else None
        )
        # Add extra context parameters to all child logs and warnings:
        with add_logging_context(model_name=model_name, fold_id=fold_id):
            logger.info(f"Training {model_name} on fold {fold_id}...")
            clf, elapsed_time = crosseval.train_classifier(
                clf=clf,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                train_sample_weights=train_sample_weights,
                train_groups=train_groups,
                export_clf_fname=export_clf_fname,
            )
            logger.info(
                f"Finished training {model_name} on fold {fold_id} (input shape = {X_train.shape}) in {int(elapsed_time)} seconds."
            )

        if X_test is not None:
            # also evaluate
            performance = ModelSingleFoldPerformance(
                model_name=model_name,
                fold_id=fold_id,
                y_true=y_test,
                clf=clf,
                X_test=X_test,
                fold_label_train=fold_label_train,
                fold_label_test=fold_label_test,
                train_time=elapsed_time,
                test_metadata=test_metadata,
                test_sample_weights=test_sample_weights,
                test_abstentions=test_abstention_ground_truth_labels,
                test_abstention_metadata=test_abstention_metadata,
                export_clf_fname=export_clf_fname,
            )
            if export:
                performance.export(
                    metadata_fname=f"{output_prefix}.{model_name}.{fold_id}.metadata_joblib"
                )
            return clf, performance
        return clf, None

    except Exception as err:
        # TODO: fail_on_error should be the responsibility of the caller.
        if fail_on_error:
            raise err
        logger.error(f"Failed to fit {model_name} on fold {fold_id}: {err}")
        return None, None


def does_fitted_model_support_lambda_setting_change(
    fitted_clf: Classifier,
) -> bool:
    """Can we run modify_fitted_model_lambda_setting() on this fitted_clf?"""
    final_clf = crosseval._get_final_estimator_if_pipeline(fitted_clf)
    return isinstance(final_clf, GlmnetLogitNetWrapper)


def does_fitted_model_support_conversion_from_glmnet_to_sklearn(
    fitted_clf: Classifier,
) -> bool:
    """
    Can we run convert_trained_glmnet_pipeline_to_untrained_sklearn_pipeline_at_best_lambda() on this fitted clf?
    Slightly different from does_fitted_model_support_lambda_setting_change(): also supports our special wrapper models like OneVsRestClassifier and its child classes.
    """
    final_clf = crosseval._get_final_estimator_if_pipeline(fitted_clf)
    if isinstance(final_clf, GlmnetLogitNetWrapper):
        if not hasattr(final_clf, "lambda_max_"):
            raise ValueError("The model must be fitted before conversion.")
        return True
    if isinstance(final_clf, OneVsRestClassifier):
        if not hasattr(final_clf, "estimators_"):
            raise ValueError("The model must be fitted before conversion.")
        return does_fitted_model_support_conversion_from_glmnet_to_sklearn(
            final_clf.estimators_[0].clf
        )
    return False


def modify_fitted_model_lambda_setting(
    fitted_clf: Classifier,
    performance: ModelSingleFoldPerformance,
    X_test: np.ndarray,
    output_prefix: Union[str, Path],
    export: bool = True,
) -> Tuple[Classifier, ModelSingleFoldPerformance,]:
    """
    If we fitted a GlmnetLogitNetWrapper, also evaluate performance with alternate use_lambda_1se setting (without refitting from scratch).
    Supports Pipelines where the final estimator is a GlmnetLogitNetWrapper."""
    # Support pipelines:
    final_clf = crosseval._get_final_estimator_if_pipeline(fitted_clf)

    if not isinstance(final_clf, GlmnetLogitNetWrapper):
        # only supports GlmnetLogitNetWrapper
        raise ValueError(
            f"Expected fitted_clf's final estimator to be a GlmnetLogitNetWrapper, but got {type(final_clf)}"
        )
    if X_test is None:
        # extra validation, because run_model_multiclass() allows X_test to be None, but we don't allow that here.
        raise ValueError("X_test must be provided")

    new_lambda_setting = not final_clf.use_lambda_1se
    model_name_modified = (
        f"{performance.model_name}_{'lambda1se' if new_lambda_setting else 'lambdamax'}"
    )
    # TODO: Switch to deepcopy() and set_params()? https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.set_params
    final_clf_modified = final_clf.switch_lambda(use_lambda_1se=new_lambda_setting)

    # Repackage final_clf_modified as a Pipeline if fitted_clf was a Pipeline
    if crosseval.is_clf_a_sklearn_pipeline(fitted_clf):
        from copy import deepcopy

        fitted_clf_modified = deepcopy(fitted_clf)
        # steps is a list of tuples, so can't modify the final tuple directly: .steps[-1][1] is immutable
        step_name, _ = fitted_clf_modified.steps.pop()
        fitted_clf_modified.steps.append((step_name, final_clf_modified))
    else:
        fitted_clf_modified = final_clf_modified

    export_clf_fname_modified = (
        f"{output_prefix}.{model_name_modified}.{performance.fold_id}.joblib"
        if export
        else None
    )
    if export_clf_fname_modified is not None:
        # Save clf (or pipeline) to disk
        try:
            joblib.dump(fitted_clf_modified, export_clf_fname_modified)
        except Exception as err:
            logger.error(
                f"Error in saving {model_name_modified} clf to disk for fold {performance.fold_id}: {err}"
            )

    performance = dataclasses.replace(
        performance,
        model_name=model_name_modified,
        clf=fitted_clf_modified,
        X_test=X_test,
        export_clf_fname=export_clf_fname_modified,
    )
    if export:
        performance.export(
            metadata_fname=f"{output_prefix}.{model_name_modified}.{performance.fold_id}.metadata_joblib"
        )

    return fitted_clf_modified, performance


def convert_trained_glmnet_pipeline_to_untrained_sklearn_pipeline_at_best_lambda(
    # This can be a pipeline or a single estimator:
    fitted_glmnet_clf: Classifier,
) -> Classifier:
    """
    Prepare to train a sklearn model using the best lambda from the glmnet model. This is to enable an extra sanity check: the results should be identical.
    This functino accepts a fitted GlmnetLogitNetWrapper estimator, or a Pipeline containing such an estimator.
    It returns an unfitted scikit-learn LogisticRegression estimator, or a Pipeline containing such an estimator.
    The sklearn LogisticRegression is configured to match the best lambda from the glmnet model, but is otherwise unfit.
    """

    from malid.train.model_definitions import (
        _make_sklearn_linear_model,
        _convert_glmnet_lambda_to_sklearn_C,
    )

    final_clf = crosseval._get_final_estimator_if_pipeline(fitted_glmnet_clf)
    if not does_fitted_model_support_conversion_from_glmnet_to_sklearn(final_clf):
        # Validation
        raise ValueError(
            f"Expected fitted_glmnet_clf's final estimator to be a GlmnetLogitNetWrapper, but got {type(final_clf)}"
        )

    if isinstance(final_clf, OneVsRestClassifier):
        # Special case for our custom wrapper
        converted_estimators_dict = {
            inner_estimator.positive_class: convert_trained_glmnet_pipeline_to_untrained_sklearn_pipeline_at_best_lambda(
                inner_estimator.clf
            )
            for inner_estimator in final_clf.estimators_
        }
        sklearn_final_clf = sklearn.base.clone(final_clf)
        sklearn_final_clf.estimator = converted_estimators_dict
    elif isinstance(final_clf, GlmnetLogitNetWrapper):
        sklearn_final_clf = _make_sklearn_linear_model(
            l1_ratio=final_clf.alpha,
            C=_convert_glmnet_lambda_to_sklearn_C(
                lamb=final_clf.lambda_max_, n_train=final_clf.n_train_samples_
            ),
        )
    else:
        raise ValueError(f"Unexpected type of final_clf: {type(final_clf)}")

    # If fitted_glmnet_clf was a Pipeline, we need to repackage sklearn_final_clf in a Pipeline with the same steps as fitted_glmnet_clf
    if not crosseval.is_clf_a_sklearn_pipeline(fitted_glmnet_clf):
        # Was not a pipeline. Return as is.
        return sklearn_final_clf

    # We are refitting, so clone the pipeline instead of deepcopy
    new_pipeline = sklearn.base.clone(fitted_glmnet_clf)

    # steps is a list of tuples, so can't modify the final tuple directly: .steps[-1][1] is immutable
    new_pipeline.steps.pop()  # remove last step
    new_pipeline.steps.append(
        (sklearn_final_clf.__class__.__name__.lower(), sklearn_final_clf)
    )

    return new_pipeline


def disable_glmnet_storage_of_internal_cv_predicted_probabilities(
    model_clf: Classifier,
) -> Classifier:
    """
    If this is a Glmnet model, disable storage of internal CV predicted probabilities -- they're huge! Pass through otherwise.

    This function should be executed before fitting the model. It will return a cloned/unfit copy of the input (except if the input has no glmnet model, in which case we pass through without cloning).

    Supports Pipelines where the final estimator is a Glmnet model.
    Supports OneVsRestClassifier and other wrappers of Glmnet models.
    """
    # Support pipelines:
    final_clf = crosseval._get_final_estimator_if_pipeline(model_clf)
    final_clf_modified = sklearn.base.clone(final_clf)
    if hasattr(final_clf, "store_cv_predicted_probabilities"):
        final_clf_modified.store_cv_predicted_probabilities = False
    elif hasattr(final_clf, "estimator") and (
        hasattr(final_clf.estimator, "store_cv_predicted_probabilities")
        or (
            # Support final_clf.estimator being a dict or Mapping, as is possible in OneVsRestClassifier: specialized estimators may be provided for each class
            # (Note: so far we only use that behavior in convert_trained_glmnet_pipeline_to_untrained_sklearn_pipeline_at_best_lambda() above. There, the specialized per-class estimators are all sklearn estimators, not glmnet estimators, so this check will be False)
            isinstance(final_clf.estimator, Mapping)
            and any(
                # Check if even one of the specialized estimators has this attribute
                hasattr(est, "store_cv_predicted_probabilities")
                for est in final_clf.estimator.values()
            )
        )
    ):
        if isinstance(final_clf.estimator, Mapping):
            # It's a dict of estimators by class name
            def _convert_inner(est):
                if hasattr(est, "store_cv_predicted_probabilities"):
                    est_clone = sklearn.base.clone(est)
                    est_clone.store_cv_predicted_probabilities = False
                    return est_clone
                else:
                    # pass through
                    return est

            final_clf_modified.estimator = {
                key: _convert_inner(value) for key, value in final_clf.estimator.items()
            }
        else:
            # It's a single estimator
            final_clf_modified.estimator.store_cv_predicted_probabilities = False
    else:
        # Not a glmnet model. Can't make any changes.
        # Return the original input (without any cloning)
        return model_clf

    # Repackage final_clf_modified as a Pipeline if fitted_clf was a Pipeline
    if crosseval.is_clf_a_sklearn_pipeline(model_clf):
        model_clf_modified = sklearn.base.clone(model_clf)  # clone the pipeline
        # steps is a list of tuples, so can't modify the final tuple directly: .steps[-1][1] is immutable
        step_name, _ = model_clf_modified.steps.pop()
        model_clf_modified.steps.append((step_name, final_clf_modified))
        return model_clf_modified
    else:
        # Input was not a pipeline.
        return final_clf_modified


## Model tuning:


def _get_combined_model_name(model_name: str, is_tuned: bool) -> str:
    return f"{model_name}{'.decision_thresholds_tuned' if is_tuned else ''}"


def evaluate_performance(
    model: Union[ImmuneClassifierMixin, AdjustedProbabilitiesDerivedModel],
    adata: GeneralAnndataType,
    fold_id: int,
    model_name: str,
    is_tuned: bool,
    fold_label_train: str,
    fold_label_test: str,
) -> ModelSingleFoldPerformance:
    """evaluates one (tuned or untuned) classifier on one fold's validation or test set"""

    # get supervised data from model (eject from wrapper if needed)
    featurized = model.featurize(adata)

    return ModelSingleFoldPerformance(
        clf=model,
        model_name=_get_combined_model_name(model_name, is_tuned),
        X_test=featurized.X,
        y_true=featurized.y,
        fold_id=fold_id,
        fold_label_train=fold_label_train,
        fold_label_test=fold_label_test,
        test_metadata=featurized.metadata,
        test_abstentions=featurized.abstained_sample_y,
        test_abstention_metadata=featurized.abstained_sample_metadata,
    )


# Consider a TypeVar? https://stackoverflow.com/a/60366318/130164
def tune_on_validation_set(
    gene_locus: GeneLocus,
    targets: Dict[TargetObsColumnEnum, Path],
    model_names: List[str],
    model_class: Type[ImmuneClassifierMixin],
    # Some models require sample_weight_strategy; if so, it will be provided here
    sample_weight_strategy: Optional[SampleWeightStrategy] = None,
    fold_label_train: str = "train_smaller",
    # load_obs_only: fast path; doesn't load .X. Useful for some models that don't require language model embeddings. Passed to io.load_fold_embeddings()
    load_obs_only=False,
) -> Mapping[
    Tuple[TargetObsColumnEnum, int, str],
    Tuple[ImmuneClassifierMixin, AdjustedProbabilitiesDerivedModel],
]:
    """tune on validation set, and evaluate tuned vs original on validation set"""
    # TODO: add kdict return type
    GeneLocus.validate_single_value(gene_locus)

    # mapping of (target_obs_column, fold_id, model_name) -> (clf_original, clf_tuned) tuple
    clfs: Mapping[
        Tuple[TargetObsColumnEnum, int, str],
        Tuple[ImmuneClassifierMixin, AdjustedProbabilitiesDerivedModel],
    ] = kdict()

    # map of target_obs_column to list of per-fold,per-model ModelSingleFoldPerformance performance objects
    validation_performance = defaultdict(list)

    # for each fold:
    for fold_id in config.all_fold_ids:
        for target_obs_column in targets.keys():
            # go one target_obs_column at a time

            # load validation set.
            validation_set = io.load_fold_embeddings(
                fold_id=fold_id,
                fold_label="validation",
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
                load_obs_only=load_obs_only,
            )
            for model_name in model_names:
                # load original model trained on train-smaller set
                try:
                    clf = model_class(
                        fold_id=fold_id,
                        # TODO: add model_name to ImmuneClassifierMixin?
                        model_name=model_name,
                        fold_label_train=fold_label_train,
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_column,
                        sample_weight_strategy=sample_weight_strategy,
                    )
                except FileNotFoundError as err:
                    logger.info(
                        f"Skipping {model_name} from {fold_id} because file not found: {err}"
                    )
                    continue

                # tune on validation set
                clf_tuned = clf.tune_model_decision_thresholds_to_validation_set(
                    validation_set=validation_set
                )

                # save out tuned model
                joblib.dump(
                    clf_tuned,
                    clf_tuned.models_base_dir
                    / f"{fold_label_train}_model.{model_name}.decision_thresholds_tuned.{fold_id}.joblib",
                )

                if fold_id != -1:
                    # do not add global fold performance to cross-validation ExperimentSet
                    for model, is_tuned in zip([clf, clf_tuned], [False, True]):
                        validation_performance[target_obs_column].append(
                            evaluate_performance(
                                model=model,
                                adata=validation_set,
                                fold_id=fold_id,
                                model_name=model_name,
                                is_tuned=is_tuned,
                                fold_label_train=fold_label_train,
                                fold_label_test="validation",
                            )
                        )

                clfs[target_obs_column, fold_id, model_name] = (clf, clf_tuned)

        # clear cache after every fold
        io.clear_cached_fold_embeddings()

    ## evaluate original vs tuned on validation set

    # summarize performance
    for (
        target_obs_col,
        validation_performance_objects_for_target,
    ) in validation_performance.items():
        results_output_dir = targets[target_obs_col]
        print(
            "Validation set performance:",
            gene_locus,
            target_obs_col,
            "-->",
            results_output_dir,
        )

        validation_classification_experiment_set = ExperimentSet(
            model_outputs=validation_performance_objects_for_target
        )
        validation_classification_experiment_set_global_performance = validation_classification_experiment_set.summarize(
            # Specify custom ground truth column name for global scores and confusion matrix
            global_evaluation_column_name=target_obs_col.value.confusion_matrix_expanded_column_name
        )

        combined_stats = (
            validation_classification_experiment_set_global_performance.get_model_comparison_stats()
        )
        display(combined_stats)

        fname = (
            results_output_dir
            / f"{fold_label_train}_model.compare_model_scores.validation_set_performance.tsv"
        )
        combined_stats.to_csv(fname, sep="\t")
        print(fname)

        validation_classification_experiment_set_global_performance.export_all_models(
            func_generate_classification_report_fname=lambda model_name: results_output_dir
            / f"{fold_label_train}_model.validation_set_performance.{model_name}.classification_report.txt",
            func_generate_confusion_matrix_fname=lambda model_name: results_output_dir
            / f"{fold_label_train}_model.validation_set_performance.{model_name}.confusion_matrix.png",
            dpi=72,
        )

    return clfs


def evaluate_original_and_tuned_on_test_set(
    clfs: Mapping[
        Tuple[TargetObsColumnEnum, int, str],
        Tuple[ImmuneClassifierMixin, AdjustedProbabilitiesDerivedModel],
    ],
    gene_locus: GeneLocus,
    targets: Dict[TargetObsColumnEnum, Path],
    # Some models require sample_weight_strategy; if so, it will be provided here
    sample_weight_strategy: Optional[SampleWeightStrategy] = None,
    fold_label_train: str = "train_smaller",
    # load_obs_only: fast path; doesn't load .X. Useful for some models that don't require language model embeddings. Passed to io.load_fold_embeddings()
    load_obs_only=False,
):
    """evaluate original vs tuned on test set"""
    # TODO: add kdict input type
    # map of target_obs_column to list of per-fold,per-model ModelSingleFoldPerformance performance objects
    test_performance = defaultdict(list)

    # clfs is mapping of (target_obs_column, fold_id, model_name) -> (clf_original, clf_tuned) tuple
    for fold_id in clfs.keys(dimensions=1):
        # go one fold_id at a time
        if fold_id == -1:
            # skip global fold, because it does not have a test set, and should not be included in cross-validation ExperimentSet
            continue
        for target_obs_column in clfs[:, fold_id, :].keys(dimensions=0):
            # go one target_obs_column at a time

            # load test set for this target_obs_column and fold_id:
            test_set = io.load_fold_embeddings(
                fold_id=fold_id,
                fold_label="test",
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
                load_obs_only=load_obs_only,
            )

            # get every model (both untuned and tuned versions) for this target_obs_column + fold_id
            for (_, _, model_name), (clf, clf_tuned) in clfs[
                target_obs_column, fold_id, :
            ].items():
                for model, is_tuned in zip([clf, clf_tuned], [False, True]):
                    test_set_performance = evaluate_performance(
                        model=model,
                        adata=test_set,
                        fold_id=fold_id,
                        model_name=model_name,
                        is_tuned=is_tuned,
                        fold_label_train=fold_label_train,
                        fold_label_test="test",
                    )
                    test_performance[target_obs_column].append(test_set_performance)
                    # export test set performance
                    # "test_performance_of_train_smaller_model" prefix avoids clash with "train_smaller_model" globbing pattern
                    # note that test_set_performance.model_name has been altered based on is_tuned
                    test_set_performance.export(
                        metadata_fname=model.models_base_dir
                        / f"test_performance_of_{fold_label_train}_model.{test_set_performance.model_name}.{fold_id}.metadata_joblib",
                    )

        # clear cache after every fold
        io.clear_cached_fold_embeddings()

    # summarize performance
    for target_obs_col, test_performance_objects_for_target in test_performance.items():
        results_output_dir = targets[target_obs_col]
        print(
            "Test set performance:",
            gene_locus,
            target_obs_col,
            "-->",
            results_output_dir,
        )

        test_classification_experiment_set = ExperimentSet(
            model_outputs=test_performance_objects_for_target
        )
        test_classification_experiment_set_global_performance = test_classification_experiment_set.summarize(
            # Specify custom ground truth column name for global scores and confusion matrix
            global_evaluation_column_name=target_obs_col.value.confusion_matrix_expanded_column_name
        )

        combined_stats = (
            test_classification_experiment_set_global_performance.get_model_comparison_stats()
        )
        display(combined_stats)

        fname = (
            results_output_dir
            / f"{fold_label_train}_model.compare_model_scores.test_set_performance.tsv"
        )
        combined_stats.to_csv(fname, sep="\t")
        print(fname)

        test_classification_experiment_set_global_performance.export_all_models(
            func_generate_classification_report_fname=lambda model_name: results_output_dir
            / f"{fold_label_train}_model.test_set_performance.{model_name}.classification_report.txt",
            func_generate_confusion_matrix_fname=lambda model_name: results_output_dir
            / f"{fold_label_train}_model.test_set_performance.{model_name}.confusion_matrix.png",
            dpi=72,
        )

        ## Review binary misclassifications: Binary prediction vs ground truth
        # For binary case, make new confusion matrix of actual disease label (y) vs predicted y_binary
        # (But this changes global score metrics)
        # TODO: Should we do this for all models?
        if target_obs_col.value.is_target_binary_for_repertoire_composition_classifier:
            # this is a binary healthy/sick classifier
            # re-summarize with different ground truth label
            test_classification_experiment_set.summarize(
                # Specify custom ground truth column name for global scores and confusion matrix
                global_evaluation_column_name=target_obs_col.value.confusion_matrix_expanded_column_name
            ).export_all_models(
                func_generate_classification_report_fname=lambda model_name: results_output_dir
                / f"{fold_label_train}_model.test_set_performance.{model_name}.classification_report.binary_vs_ground_truth.txt",
                func_generate_confusion_matrix_fname=lambda model_name: results_output_dir
                / f"{fold_label_train}_model.test_set_performance.{model_name}.confusion_matrix.binary_vs_ground_truth.png",
                confusion_matrix_pred_label="Predicted binary label",
                dpi=72,
            )
