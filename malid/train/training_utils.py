import logging
from collections import defaultdict
from pathlib import Path
from typing import Type, Union, List, Optional, Dict, Mapping, Tuple

import anndata
import joblib
import numpy as np
import pandas as pd
import sklearn.base
from IPython.display import display
from kdict import kdict
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from malid import config
from malid import io
from malid.datamodels import GeneLocus, TargetObsColumnEnum, CVScorer
import choosegpu
from malid.external import model_evaluation
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)
from StratifiedGroupKFoldRequiresGroups import (
    StratifiedGroupKFoldRequiresGroups,
)

logger = logging.getLogger(__name__)


def prepare_models_to_train(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    fold_id: int,
    target_obs_column: TargetObsColumnEnum,
    chosen_models: List[str],
    use_gpu: bool,
    output_prefix: Path,
    train_metadata: Optional[pd.DataFrame] = None,
    n_jobs=1,
    cv_scoring=CVScorer.AUC,
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
    internal_cv_nfolds = 3
    internal_cv = StratifiedGroupKFoldRequiresGroups(
        n_splits=internal_cv_nfolds, shuffle=True, random_state=0
    )
    train_participant_labels = train_metadata["participant_label"].values

    # Configure GPU if requested.
    choosegpu.configure_gpu(enable=use_gpu)

    models = make_models(
        chosen_models=chosen_models,
        n_jobs=n_jobs,
        internal_cv=internal_cv,
        cv_scoring=cv_scoring,
    )

    return models, train_participant_labels


@ignore_warnings(category=ConvergenceWarning)
def run_model_multiclass(
    model_name: str,
    model_clf: model_evaluation.Classifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fold_id: int,
    output_prefix: Union[str, Path],
    fold_label_train: str,
    fold_label_test: str,
    train_sample_weights: Optional[np.ndarray] = None,
    train_groups: Optional[np.ndarray] = None,
    test_sample_weights: Optional[np.ndarray] = None,
    test_metadata: Optional[pd.DataFrame] = None,
    test_abstention_ground_truth_labels: Optional[np.ndarray] = None,
    test_abstention_metadata: Optional[pd.DataFrame] = None,
    fail_on_error=False,
    export=True,
) -> Tuple[
    Union[model_evaluation.Classifier, None],
    Union[model_evaluation.ModelSingleFoldPerformance, None],
]:
    # TODO: move this to model_evaluation?
    try:
        clf = sklearn.base.clone(model_clf)
        export_clf_fname = (
            f"{output_prefix}.{model_name}.{fold_id}.joblib" if export else None
        )
        clf, elapsed_time = model_evaluation.train_classifier(
            clf=clf,
            model_name=model_name,
            fold_id=fold_id,
            X_train=X_train,
            y_train=y_train,
            train_sample_weights=train_sample_weights,
            train_groups=train_groups,
            export_clf_fname=export_clf_fname,
        )

        if X_test is not None:
            # also evaluate
            performance = model_evaluation.ModelSingleFoldPerformance(
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
        if fail_on_error:
            raise err
        logger.error(f"Failed to fit {model_name} on fold {fold_id}: {err}")
        return None, None


## Model tuning:


def _get_combined_model_name(model_name: str, is_tuned: bool) -> str:
    return f"{model_name}{'.decision_thresholds_tuned' if is_tuned else ''}"


def evaluate_performance(
    model: Union[ImmuneClassifierMixin, AdjustedProbabilitiesDerivedModel],
    adata: anndata.AnnData,
    fold_id: int,
    model_name: str,
    is_tuned: bool,
    fold_label_train: str,
    fold_label_test: str,
) -> model_evaluation.ModelSingleFoldPerformance:
    """evaluates one (tuned or untuned) classifier on one fold's validation or test set"""

    # get supervised data from model (eject from wrapper if needed)
    featurized = model.featurize(adata)

    return model_evaluation.ModelSingleFoldPerformance(
        clf=model,
        model_name=_get_combined_model_name(model_name, is_tuned),
        X_test=featurized.X,
        y_true=featurized.y,
        fold_id=fold_id,
        fold_label_train=fold_label_train,
        fold_label_test=fold_label_test,
        test_metadata=featurized.metadata,
        # Set these to null - different convention between FeaturizedData and ModelSingleFoldPerformance
        test_abstentions=featurized.abstained_sample_y
        if featurized.abstained_sample_y.shape[0] > 0
        else None,
        test_abstention_metadata=featurized.abstained_sample_metadata
        if featurized.abstained_sample_metadata.shape[0] > 0
        else None,
    )


# Consider a TypeVar? https://stackoverflow.com/a/60366318/130164
def tune_on_validation_set(
    gene_locus: GeneLocus,
    targets: Dict[TargetObsColumnEnum, Path],
    model_names: List[str],
    model_class: Type[ImmuneClassifierMixin],
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
            )
            for model_name in model_names:
                # load original model trained on train-smaller set
                try:
                    clf = model_class(
                        fold_id=fold_id,
                        model_name=model_name,
                        fold_label_train="train_smaller",
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_column,
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
                    / f"train_smaller_model.{model_name}.decision_thresholds_tuned.{fold_id}.joblib",
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
                                fold_label_train="train_smaller",
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
        print(gene_locus, target_obs_col)

        validation_classification_experiment_set = model_evaluation.ExperimentSet(
            model_outputs=validation_performance_objects_for_target
        )
        validation_classification_experiment_set_global_performance = (
            validation_classification_experiment_set.summarize()
        )

        display(
            validation_classification_experiment_set_global_performance.get_model_comparison_stats()
        )

    return clfs


def evaluate_original_and_tuned_on_test_set(
    clfs: Mapping[
        Tuple[TargetObsColumnEnum, int, str],
        Tuple[ImmuneClassifierMixin, AdjustedProbabilitiesDerivedModel],
    ],
    gene_locus: GeneLocus,
    targets: Dict[TargetObsColumnEnum, Path],
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
                        fold_label_train="train_smaller",
                        fold_label_test="test",
                    )
                    test_performance[target_obs_column].append(test_set_performance)
                    # export test set performance
                    # "test_performance_of_train_smaller_model" prefix avoids clash with "train_smaller_model" globbing pattern
                    # note that test_set_performance.model_name has been altered based on is_tuned
                    test_set_performance.export(
                        metadata_fname=model.models_base_dir
                        / f"test_performance_of_train_smaller_model.{test_set_performance.model_name}.{fold_id}.metadata_joblib",
                    )

        # clear cache after every fold
        io.clear_cached_fold_embeddings()

    # summarize performance
    for target_obs_col, test_performance_objects_for_target in test_performance.items():
        results_output_dir = targets[target_obs_col]
        print(gene_locus, target_obs_col, "-->", results_output_dir)

        test_classification_experiment_set = model_evaluation.ExperimentSet(
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
            / f"train_smaller_model.compare_model_scores.test_set_performance.tsv"
        )
        combined_stats.to_csv(fname, sep="\t")
        print(fname)

        test_classification_experiment_set_global_performance.export_all_models(
            func_generate_classification_report_fname=lambda model_name: results_output_dir
            / f"train_smaller_model.test_set_performance.{model_name}.classification_report.txt",
            func_generate_confusion_matrix_fname=lambda model_name: results_output_dir
            / f"train_smaller_model.test_set_performance.{model_name}.confusion_matrix.png",
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
                / f"train_smaller_model.test_set_performance.{model_name}.classification_report.binary_vs_ground_truth.txt",
                func_generate_confusion_matrix_fname=lambda model_name: results_output_dir
                / f"train_smaller_model.test_set_performance.{model_name}.confusion_matrix.binary_vs_ground_truth.png",
                confusion_matrix_pred_label="Predicted binary label",
                dpi=72,
            )
