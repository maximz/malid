"""
Train CDR3 amino acid exact sequence match models.
"""

import gc
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn.base
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from feature_engine.preprocessing import MatchVariables

from malid import config
from malid import io
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
)
from malid.external import model_evaluation
from malid.train import training_utils
from malid.trained_model_wrappers import ExactMatchesClassifier

logger = logging.getLogger(__name__)


def _get_fold_data(
    fold_id: int,
    fold_label: str,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
) -> pd.DataFrame:
    df = io.load_fold_embeddings(
        fold_id=fold_id,
        fold_label=fold_label,
        target_obs_column=target_obs_column,
        gene_locus=gene_locus,
    ).obs
    return df


def _try_a_p_value(
    p_value: float,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sequence_pvalues_per_disease: pd.DataFrame,
    feature_names_order: List[str],
    target_obs_column: TargetObsColumnEnum,
    fold_id: int,
    gene_locus: GeneLocus,
    fold_label_train: str,
    fold_label_test: str,
    chosen_models: List[str],
    use_gpu: bool,
    output_prefix: Path,
    n_jobs: int,
    fail_on_error: bool,
) -> Optional[list]:
    # Apply these scores to train sequences and any matching test sequences.
    # Then given a p-value, featurize at a p-value -> specimen-level feature vectors.
    featurized_train = ExactMatchesClassifier._featurize(
        df=train_df,
        sequences_with_fisher_result=sequence_pvalues_per_disease,
        p_value_threshold=p_value,
        feature_order=feature_names_order,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
    )
    logger.info(
        f"Fold {fold_id}-{fold_label_train} (target {target_obs_column}): abstained on {featurized_train.abstained_sample_metadata.shape[0]} train specimens at p-value {p_value}"
    )

    featurized_test = ExactMatchesClassifier._featurize(
        df=test_df,
        sequences_with_fisher_result=sequence_pvalues_per_disease,
        p_value_threshold=p_value,
        feature_order=feature_names_order,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
    )
    logger.info(
        f"Fold {fold_id}-{fold_label_test} (target {target_obs_column}): abstained on {featurized_test.abstained_sample_metadata.shape[0]} test specimens at p-value {p_value}"
    )

    # Sanity checks
    if featurized_train.X.shape[0] == 0:
        logger.warning(
            f"Skipping p-value {p_value} for fold {fold_id}, {gene_locus} because no {fold_label_train} (train) sequences were featurized."
        )
        return None
    if featurized_test.X.shape[0] == 0:
        logger.warning(
            f"Skipping p-value {p_value} for fold {fold_id}, {gene_locus} because no {fold_label_test} (test) sequences were featurized."
        )
        return None
    assert np.array_equal(
        featurized_train.X.columns, feature_names_order
    ) and np.array_equal(
        featurized_test.X.columns, feature_names_order
    ), "Feature order does not match"
    if np.unique(featurized_train.y).shape[0] == 1:
        logger.warning(
            f"Skipping p-value {p_value} for fold {fold_id}, {gene_locus} because all featurized {fold_label_train} (train) specimens have the same {target_obs_column} label (can't train a classifier with y nunique = 1)."
        )
        return None
    if (featurized_train.X.values == 0).all():
        # Check if any rows are all 0s -- these abstentions should have been removed.
        raise ValueError(
            "Some rows in featurized_train.X are all 0s. Abstentions should be removed."
        )
    if (featurized_test.X.values == 0).all():
        # Check if any rows are all 0s -- these abstentions should have been removed.
        raise ValueError(
            "Some rows in featurized_test.X are all 0s. Abstentions should be removed."
        )

    logger.info(
        f"Training exact matches classifier on fold {fold_id}, {gene_locus}: Featurized {fold_label_train} (train) and {fold_label_test} (test) at p-value {p_value}"
    )
    models, train_participant_labels = training_utils.prepare_models_to_train(
        X_train=featurized_train.X,
        y_train=featurized_train.y,
        train_metadata=featurized_train.metadata,
        fold_id=fold_id,
        target_obs_column=target_obs_column,
        chosen_models=chosen_models,
        use_gpu=use_gpu,
        output_prefix=output_prefix,
        n_jobs=n_jobs,
    )

    results = []

    for model_name, model_clf in models.items():
        is_pipeline = type(model_clf) == Pipeline
        if is_pipeline:
            # If already a pipeline:
            model_pipeline = sklearn.base.clone(model_clf)

            # Scale columns.
            # Prepend a StandardScaler if it doesn't exist already
            if "standardscaler" in model_pipeline.named_steps.keys():
                logger.warning(
                    f"The pipeline already has a StandardScaler step already! Not inserting for model {model_name}"
                )
            else:
                logger.info(
                    f"Inserting StandardScaler into existing pipeline for model {model_name}"
                )
                model_pipeline.steps.insert(
                    0, ("standardscaler", preprocessing.StandardScaler())
                )

            # Prepend a MatchVariables if it doesn't exist already
            # Confirms features are in same order.
            # Puts in same order if they're not.
            # Throws error if any train column missing.
            # Drops any test column not found in train column list.
            # It's like saving a feature_order and doing X.loc[feature_order]
            if "matchvariables" not in model_pipeline.named_steps.keys():
                model_pipeline.steps.insert(
                    0,
                    ("matchvariables", MatchVariables(missing_values="raise")),
                )
        else:  # Not yet a pipeline.
            model_pipeline = make_pipeline(
                MatchVariables(missing_values="raise"),
                preprocessing.StandardScaler(),
                sklearn.base.clone(model_clf),
            )

        # train model and evaluate on test set
        logger.debug(
            f"Training ExactMatchesClassifier {model_name} on fold {fold_id}-{fold_label_train}, {gene_locus} data for p_value={p_value}"
        )
        model_pipeline, test_performance = training_utils.run_model_multiclass(
            model_name=model_name,
            model_clf=model_pipeline,
            X_train=featurized_train.X,
            X_test=featurized_test.X,
            y_train=featurized_train.y,
            y_test=featurized_test.y,
            fold_id=fold_id,
            output_prefix=output_prefix,
            fold_label_train=fold_label_train,
            fold_label_test=fold_label_test,
            train_groups=train_participant_labels,
            test_metadata=featurized_test.metadata,
            test_abstention_ground_truth_labels=featurized_test.abstained_sample_y,
            test_abstention_metadata=featurized_test.abstained_sample_metadata,
            fail_on_error=fail_on_error,
            # don't export until later when we choose best p value
            export=False,
        )
        # record one clf and performance at a particular p_value for a particular model name
        results.append((model_name, (model_pipeline, p_value, test_performance)))
    return results


def _run_models_on_fold(
    fold_id: int,
    gene_locus: GeneLocus,
    fold_label_train: str,
    fold_label_test: str,
    chosen_models: List[str],
    target_obs_column: TargetObsColumnEnum,
    n_jobs=1,
    use_gpu=False,
    clear_cache=True,
    p_values: Optional[Union[List[float], np.ndarray]] = None,
    fail_on_error=False,
):
    # n_jobs is the number of parallel jobs to run for p-value threshold tuning.
    models_base_dir = ExactMatchesClassifier._get_model_base_dir(
        gene_locus=gene_locus, target_obs_column=target_obs_column
    )
    models_base_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = models_base_dir / f"{fold_label_train}_model"
    train_df = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_train,
        target_obs_column=target_obs_column,
        gene_locus=gene_locus,
    )
    test_df = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_test,
        target_obs_column=target_obs_column,
        gene_locus=gene_locus,
    )

    # Compute fisher scores for all train sequences
    sequence_pvalues_per_disease = (
        ExactMatchesClassifier._compute_fisher_scores_for_sequences(
            train_df=train_df, target_obs_column=target_obs_column
        )
    )
    # Feature names order is sorted list of diseases (or other target labels) with predictive sequences
    feature_names_order = sequence_pvalues_per_disease.columns
    logger.info(
        f"Training exact matches classifier on fold {fold_id}-{fold_label_train}, {gene_locus}, {target_obs_column}: Computed Fisher p-value scores for {len(feature_names_order)} diseases"
    )

    # Loop over p-values
    # Later, consider different p-values for each disease class.
    if p_values is None:
        p_values = [0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0]
    # Run in parallel
    # ("loky" backend required; "multiprocessing" backend can deadlock with xgboost, see https://github.com/dmlc/xgboost/issues/7044#issuecomment-1039912899 , https://github.com/dmlc/xgboost/issues/2163 , and https://github.com/dmlc/xgboost/issues/4246 )
    p_value_job_outputs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_try_a_p_value)(
            p_value=p_value,
            train_df=train_df,
            test_df=test_df,
            sequence_pvalues_per_disease=sequence_pvalues_per_disease,
            feature_names_order=feature_names_order,
            target_obs_column=target_obs_column,
            fold_id=fold_id,
            gene_locus=gene_locus,
            fold_label_train=fold_label_train,
            fold_label_test=fold_label_test,
            chosen_models=chosen_models,
            use_gpu=use_gpu,
            output_prefix=output_prefix,
            # This controls nested multiprocessing:
            n_jobs=n_jobs,
            fail_on_error=fail_on_error,
        )
        for p_value in p_values
    )

    # Unwrap each job's outputs
    model_results = defaultdict(list)  # model_name -> (clf, p_value, auc)
    for p_value_job_output in p_value_job_outputs:
        if p_value_job_output is None:
            # this p-value was skipped
            continue
        # this p-value was run and we have a list of results to unwrap
        for (
            model_name,
            (model_pipeline, p_value, test_performance),
        ) in p_value_job_output:
            if test_performance is None:
                # training_utils.run_model_multiclass had an issue with this model
                logger.warning(
                    f"Model {model_name} on fold {fold_id}, {gene_locus}, {target_obs_column} was skipped for p-value {p_value}"
                )
                continue
            model_results[model_name].append(
                (model_pipeline, p_value, test_performance)
            )

    # For each model: choose p-value with best test performance for this fold
    model_results_best = {}
    for model_name, model_results_all_p_values in model_results.items():
        if len(model_results_all_p_values) == 0:
            logger.warning(
                f"No results for model {model_name} on fold {fold_id}. Skipping."
            )
            continue
        # sort by score
        best_result = max(
            # Use MCC with abstentions counted (penalized)
            model_results_all_p_values,
            key=lambda tup: tup[2].scores(with_abstention=True)["mcc"],
        )
        logger.info(
            f"ExactMatchesClassifier {model_name} on fold {fold_id}-{fold_label_train}, {gene_locus} data: best p_value={best_result[1]}"
        )
        # select that clf and that p-value
        model_results_best[model_name] = best_result
        # TODO: save all p-values and their performance for future analysis of metrics across p value choices?

    if len(model_results_best) == 0:
        raise ValueError(
            f"ExactMatchesClassifier: no results for any model on fold {fold_id}, {gene_locus} - perhaps all p-value thresholds failed featurization?"
        )

    # Save out those models (and p-values)
    results = []
    for model_name, (clf, p_value, test_performance) in model_results_best.items():
        export_clf_fname = f"{output_prefix}.{model_name}.{fold_id}.joblib"
        test_performance.export_clf_fname = export_clf_fname
        joblib.dump(clf, export_clf_fname)
        joblib.dump(p_value, f"{output_prefix}.{model_name}.{fold_id}.p_value.joblib")
        test_performance.export(
            metadata_fname=f"{output_prefix}.{model_name}.{fold_id}.metadata_joblib"
        )
        results.append(test_performance)

    # save that set of sequences with associated p values
    joblib.dump(
        sequence_pvalues_per_disease,
        f"{output_prefix}.{fold_id}.{fold_label_train}.sequences_joblib",
    )

    # Clear RAM
    if clear_cache:
        io.clear_cached_fold_embeddings()
        del train_df, test_df
    gc.collect()

    return results


def run_classify_with_all_models(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    fold_label_train: str,
    fold_label_test: str,
    chosen_models: List[str],
    n_jobs: int,
    fold_ids: Optional[List[int]] = None,
    use_gpu=False,
    clear_cache=True,
    p_values: Optional[Union[List[float], np.ndarray]] = None,
    fail_on_error=False,
) -> model_evaluation.ExperimentSet:
    """Run classification. n_jobs is passed to `_run_models_on_fold` to parallelize the p-value threshold tuning."""
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    TargetObsColumnEnum.validate(target_obs_column)

    if fold_ids is None:
        fold_ids = config.all_fold_ids
    logger.info(f"Starting train on folds: {fold_ids}")

    return model_evaluation.ExperimentSet(
        model_outputs=[
            _run_models_on_fold(
                fold_id=fold_id,
                gene_locus=gene_locus,
                fold_label_train=fold_label_train,
                fold_label_test=fold_label_test,
                chosen_models=chosen_models,
                target_obs_column=target_obs_column,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
                clear_cache=clear_cache,
                p_values=p_values,
                fail_on_error=fail_on_error,
            )
            for fold_id in fold_ids
        ]
    )
