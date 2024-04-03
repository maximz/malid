"""
Train CDR3 amino acid convergent sequence classifier models.
"""

from collections import defaultdict
import gc
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import sklearn.base
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from feature_engine.preprocessing import MatchVariables

from malid import config
from malid import io
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
)
import crosseval
from malid.train import training_utils
from malid.trained_model_wrappers import ConvergentClusterClassifier

from malid.external.logging_context_for_warnings import ContextLoggerWrapper
from log_with_context import add_logging_context

logger = ContextLoggerWrapper(name=__name__)


def _get_fold_data(
    fold_id: int,
    fold_label: str,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    # Model 2 does not require the embedding .X, so take the fast path and just load .obs:
    load_obs_only=True,
) -> pd.DataFrame:
    df = io.load_fold_embeddings(
        fold_id=fold_id,
        fold_label=fold_label,
        target_obs_column=target_obs_column,
        gene_locus=gene_locus,
        # load_obs_only defaults to True: Model 2 does not require the embedding .X, so take the fast path and just load .obs:
        load_obs_only=load_obs_only,
    ).obs
    return df


def _try_a_p_value(
    p_value: float,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cluster_centroids_scored: pd.DataFrame,
    featurize_sequence_identity_threshold: float,
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
    # Featurize train
    featurized_train = ConvergentClusterClassifier._featurize(
        # this is not df_train_clustered, so it does not have cluster assignments. we will rematch to predictive cluster subset only
        df=df_train,
        cluster_centroids_with_class_specific_p_values=cluster_centroids_scored,
        p_value_threshold=p_value,
        sequence_identity_threshold=featurize_sequence_identity_threshold,
        feature_order=feature_names_order,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
    )
    logger.info(
        f"Fold {fold_id}-{fold_label_train} (target {target_obs_column}): abstained on {featurized_train.abstained_sample_metadata.shape[0]} train specimens at p-value {p_value}"
    )

    # Featurize test
    featurized_test = ConvergentClusterClassifier._featurize(
        df=df_test,
        cluster_centroids_with_class_specific_p_values=cluster_centroids_scored,
        p_value_threshold=p_value,
        sequence_identity_threshold=featurize_sequence_identity_threshold,
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
    if (featurized_train.X.values == 0).all(axis=1).any():
        # Check if any rows are all 0s -- these abstentions should have been removed.
        raise ValueError(
            "Some rows in featurized_train.X are all 0s. Abstentions should be removed."
        )
    if (featurized_test.X.values == 0).all(axis=1).any():
        # Check if any rows are all 0s -- these abstentions should have been removed.
        raise ValueError(
            "Some rows in featurized_test.X are all 0s. Abstentions should be removed."
        )

    ## Build and run classifiers.
    logger.info(
        f"Training ConvergentClusterClassifier on fold {fold_id}, {gene_locus}: Featurized {fold_label_train} (train) and {fold_label_test} (test) at p-value {p_value}"
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

    # Pipeline order is: MatchVariables -> standardscaler -> clf.
    for model_name, model_clf in models.items():
        # Convert the model into a pipeline that starts with a StandardScaler
        model_pipeline = training_utils.prepend_scaler_if_not_present(model_clf)

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

        # train model and evaluate on test set
        logger.debug(
            f"Training ConvergentClusterClassifier {model_name} on fold {fold_id}-{fold_label_train}, {gene_locus} data for p_value={p_value}"
        )
        with add_logging_context(model_name=model_name, p_value=p_value):
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

        # For Glmnet models, also record performance with lambda_1se flag flipped.
        if (
            model_pipeline is not None
            and test_performance is not None
            and training_utils.does_fitted_model_support_lambda_setting_change(
                model_pipeline
            )
        ):
            (
                model_pipeline2,
                test_performance2,
            ) = training_utils.modify_fitted_model_lambda_setting(
                fitted_clf=model_pipeline,
                performance=test_performance,
                X_test=featurized_test.X,
                output_prefix=output_prefix,
                # don't export until later when we choose best p value
                export=False,
            )
            results.append(
                (
                    test_performance2.model_name,
                    (model_pipeline2, p_value, test_performance2),
                )
            )

        if (
            model_pipeline is not None
            and training_utils.does_fitted_model_support_conversion_from_glmnet_to_sklearn(
                model_pipeline
            )
        ):
            # Also train a sklearn model using the best lambda from the glmnet model, as an extra sanity check. Results should be identical.
            sklearn_model = training_utils.convert_trained_glmnet_pipeline_to_untrained_sklearn_pipeline_at_best_lambda(
                model_pipeline
            )
            sklearn_model_name = (
                model_name.replace("_cv", "") + "_sklearn_with_lambdamax"
            )
            sklearn_model, result_sklearn = training_utils.run_model_multiclass(
                model_name=sklearn_model_name,
                model_clf=sklearn_model,
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
            results.append(
                (
                    sklearn_model_name,
                    (sklearn_model, p_value, result_sklearn),
                )
            )
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
    # Model 2 does not require the embedding .X, so take the fast path and just load .obs:
    load_obs_only=True,
):
    # n_jobs is the number of parallel jobs to run for p-value threshold tuning.
    if p_values is None:
        # If default p values have not been overriden in this training run, then use defaults:

        # First check if there are different defaults for this TargetObsColumnEnum.
        if target_obs_column.value.convergent_clustering_p_values is not None:
            p_values = target_obs_column.value.convergent_clustering_p_values
        # Otherwise use these global defaults
        p_values = [0.0005, 0.001, 0.005, 0.01, 0.05]

    models_base_dir = ConvergentClusterClassifier._get_model_base_dir(
        gene_locus=gene_locus, target_obs_column=target_obs_column
    )
    models_base_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = models_base_dir / f"{fold_label_train}_model"

    df_train = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_train,
        target_obs_column=target_obs_column,
        gene_locus=gene_locus,
        load_obs_only=load_obs_only,
    )

    # Cluster training set
    df_train_clustered = ConvergentClusterClassifier._cluster_training_set(
        df=df_train,
        sequence_identity_threshold=config.sequence_identity_thresholds.cluster_amino_acids_across_patients[
            gene_locus
        ],
        # Force a copy - don't modify df_train.
        # Keep that original version around without any cluster assignments. Later, we rematch df_train to only the subset of clusters determined to be "predictive".
        inplace=False,
    )

    # Get cluster enrichment for each disease (Fisher's exact test p values),
    # based on how many unique participants from each disease type fall into each cluster.
    cluster_pvalues_per_disease = (
        ConvergentClusterClassifier._compute_fisher_scores_for_clusters(
            train_df=df_train_clustered, target_obs_column=target_obs_column
        )
    )
    # Feature names order is sorted list of diseases (or other target labels) with predictive clusters
    feature_names_order = cluster_pvalues_per_disease.columns  # these are the classes
    logger.info(
        f"Training ConvergentClusterClassifier on fold {fold_id}-{fold_label_train}, {gene_locus}, {target_obs_column}: Computed Fisher p-value scores for {len(feature_names_order)} classes ({feature_names_order}) and {cluster_pvalues_per_disease.shape[0]} clusters"
    )

    # Make cluster centroids, weighed by number of clone members (number of unique VDJ sequences)
    # Consensus sequence series is indexed by cluster v_gene,j_gene,cdr3len,cluster_id_within_group

    # This is expensive, so first reduce to only the clusters that are significant for at least one disease at the largest p-value
    cluster_pvalues_per_disease = cluster_pvalues_per_disease.loc[
        cluster_pvalues_per_disease[feature_names_order].min(axis=1) <= max(p_values)
    ]
    # filter down df_train_clustered too
    # df_train_clustered = pd.merge(
    #     df_train_clustered,
    #     cluster_pvalues_per_disease[[]],
    #     left_on=[
    #         "v_gene",
    #         "j_gene",
    #         "cdr3_aa_sequence_trim_len",
    #         "cluster_id_within_clustering_group",
    #     ],
    #     right_index=True,
    #     how="inner",
    # )
    # Faster approach, since global_resulting_cluster_ID is a tuple, as is the multiindex
    df_train_clustered = df_train_clustered[
        df_train_clustered["global_resulting_cluster_ID"].isin(
            cluster_pvalues_per_disease.index
        )
    ]
    logger.info(
        f"Reduced starting cluster list to {cluster_pvalues_per_disease.shape[0]} clusters, eliminating any that are not significant for any class label at the loosest p-value threshold"
    )

    cluster_centroids: pd.Series = ConvergentClusterClassifier._get_cluster_centroids(
        clustered_df=df_train_clustered,
    )
    logger.info(f"Computed centroids for {cluster_centroids.shape[0]} clusters")
    del df_train_clustered

    # Merge score columns in. Resulting columns are "consensus_sequence" and class-specific score columns (whose names are in feature_names_order)
    cluster_centroids_scored = ConvergentClusterClassifier._merge_cluster_centroids_with_cluster_class_association_scores(
        cluster_centroids, cluster_pvalues_per_disease
    )
    if cluster_centroids_scored.shape[0] != cluster_centroids.shape[0]:
        raise ValueError(
            f"Expected {cluster_centroids.shape[0]} rows after merge, but got {cluster_centroids_scored.shape[0]}"
        )

    df_test = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_test,
        target_obs_column=target_obs_column,
        gene_locus=gene_locus,
        load_obs_only=load_obs_only,
    )

    featurize_sequence_identity_threshold = (
        config.sequence_identity_thresholds.assign_test_sequences_to_clusters[
            gene_locus
        ]
    )

    # Loop over p-values
    # Later, consider different p-values for each disease class.
    # Run in parallel
    # ("loky" backend required; "multiprocessing" backend can deadlock with xgboost, see https://github.com/dmlc/xgboost/issues/7044#issuecomment-1039912899 , https://github.com/dmlc/xgboost/issues/2163 , and https://github.com/dmlc/xgboost/issues/4246 )
    # Wrap in tqdm for progress bar (see https://stackoverflow.com/a/76726101/130164)
    p_value_job_outputs = list(
        tqdm(
            Parallel(return_as="generator", n_jobs=n_jobs, backend="loky")(
                delayed(_try_a_p_value)(
                    p_value=p_value,
                    df_train=df_train,
                    df_test=df_test,
                    cluster_centroids_scored=cluster_centroids_scored,
                    featurize_sequence_identity_threshold=featurize_sequence_identity_threshold,
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
            ),
            total=len(p_values),
        )
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
            f"ConvergentClusterClassifier {model_name} on fold {fold_id}-{fold_label_train}, {gene_locus} data: best p_value={best_result[1]}"
        )
        # select that clf and that p-value
        model_results_best[model_name] = best_result
        # TODO: save all p-values and their performance for future analysis of metrics across p value choices?

    if len(model_results_best) == 0:
        raise ValueError(
            f"ConvergentClusterClassifier: no results for any model on fold {fold_id}, {gene_locus} - perhaps all p-value thresholds failed featurization?"
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

    # Save the set of clusters with associated p values
    joblib.dump(
        cluster_centroids_scored,
        f"{output_prefix}.{fold_id}.{fold_label_train}.clusters_joblib",
    )

    # Clear RAM
    if clear_cache:
        io.clear_cached_fold_embeddings()
        del df_train, df_test
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
    # Model 2 does not require the embedding .X, so take the fast path and just load .obs:
    load_obs_only=True,
) -> crosseval.ExperimentSet:
    """Run classification. n_jobs is passed to `_run_models_on_fold` to parallelize the p-value threshold tuning."""
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    TargetObsColumnEnum.validate(target_obs_column)
    target_obs_column.confirm_compatibility_with_gene_locus(gene_locus)
    target_obs_column.confirm_compatibility_with_cross_validation_split_strategy(
        config.cross_validation_split_strategy
    )

    if fold_ids is None:
        fold_ids = config.all_fold_ids
    logger.info(f"Starting train on folds: {fold_ids}")

    return crosseval.ExperimentSet(
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
                load_obs_only=load_obs_only,
            )
            for fold_id in fold_ids
        ]
    )
