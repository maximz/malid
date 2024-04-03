"""
Train sequence models. Keep past exposures mixed in with acute infections.
"""

from collections import defaultdict
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import gc
from typing import List, Optional, Type
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import anndata

from malid import config, io, helpers
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
import crosseval
from malid.train import training_utils
from malid.trained_model_wrappers import (
    VJGeneSpecificSequenceClassifier,
)

from malid.external.logging_context_for_warnings import ContextLoggerWrapper
from log_with_context import add_logging_context

from malid.trained_model_wrappers.vj_gene_specific_sequence_classifier import (
    SplitKeyType,
    SequenceSubsetStrategy,
)

logger = ContextLoggerWrapper(name=__name__)

# Configure whether we train sklearn versions of the glmnet models.
# TODO: Remove this from the main code path. Convert it to an automated test in a sensible n>p regime to make sure training results are the same.
enable_training_sklearn_versions_of_glmnet = False


def _extend_model_list_with_sklearn_versions_of_glmnet_models(
    model_names: List[str],
) -> List[str]:
    """
    Extend with sklearn versions of the glmnet models:
    For each model in the list, if it is a glmnet model, add a sklearn version of the model.
    """
    if not enable_training_sklearn_versions_of_glmnet:
        # no-op
        return model_names

    return model_names + [
        model_name.replace("_cv", "") + "_sklearn_with_lambdamax"
        for model_name in model_names
        if "_cv" in model_name
    ]


def _get_fold_data(
    fold_id: int,
    fold_label: str,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
) -> anndata.AnnData:
    # for each fold:
    # load anndata without any filtering at all
    return io.load_fold_embeddings(
        fold_id=fold_id,
        fold_label=fold_label,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )


def _run_models_on_fold(
    fold_id: int,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
    fold_label_train: str,
    fold_label_test: str,
    chosen_models: List[str],
    n_jobs: int,
    use_gpu=False,
    clear_cache=True,
    fail_on_error=False,
    sequence_subset_strategy: SequenceSubsetStrategy = SequenceSubsetStrategy.split_Vgene_and_isotype,
    exclude_rare_v_genes: bool = True,
    resume: bool = False,
):
    base_classifier = sequence_subset_strategy.base_model
    models_base_dir = base_classifier._get_model_base_dir(
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )
    models_base_dir.mkdir(parents=True, exist_ok=True)

    adata_train = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_train,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )
    # If exclude_rare_v_genes is enabled AND the split parameters include v_gene: remove rare V genes.
    # (Note, we don't do this when splitting by V family; keep the rare V genes in that case.)
    if exclude_rare_v_genes and "v_gene" in base_classifier.split_on:
        v_genes_to_keep = helpers.find_non_rare_v_genes(adata_train)
        adata_train = adata_train[adata_train.obs["v_gene"].isin(v_genes_to_keep)]

    if "v_family" in base_classifier.split_on:
        # If we are splitting by V family, always remove some very rare V families. Not making this configurable.
        # BCR contains rare V families whose names start with "VH".
        # The important ones - "IGHV*" and "TRBV*" - are left alone by this filter.
        adata_train = adata_train[~adata_train.obs["v_family"].str.startswith("VH")]

    classes = np.unique(adata_train.obs[target_obs_column.value.obs_column_name])

    # Create temporary directory for saving models during training
    temporary_save_dir = (
        models_base_dir / f"{fold_label_train}_model.{fold_id}.split_models.tmp"
    )
    if temporary_save_dir.exists():
        if not resume:
            # Delete any existing temporary files from previous runs. Remove the whole folder.
            logger.info("Overwriting a previous incomplete run.")
            shutil.rmtree(temporary_save_dir)
        else:
            logger.info("Resuming previous incomplete run.")
    temporary_save_dir.mkdir(parents=True, exist_ok=True)

    # Make filename for every split key. We will store models there as we train
    training_chunk_hashes = base_classifier.generate_training_chunk_hashes(adata_train)
    temporary_save_files = {
        split_key: temporary_save_dir / f"{split_key_hash}.joblib"
        for split_key, split_key_hash in training_chunk_hashes.items()
    }

    parallel_job_outputs = []
    split_keys_to_skip = []  # Won't be retrained
    if resume:
        # Look for any existing temporary files from previous runs, and load them if they exist.
        # Do this before we start parallelization, so that we don't spin up wasteful processes that are passed an anndata copy, load the results from disk, and then pass back a result copy.
        for split_key, temporary_save_file in temporary_save_files.items():
            if temporary_save_file.exists():
                # Load from disk
                results = joblib.load(temporary_save_file)

                # Catch problematic situations and don't allow resuming if they are the case:

                # 1) Check if the loaded models match the chosen_models list.
                # If not, the model list has changed and we need to retrain the missing models and remove unnecessary models.
                # Here we actually allow the model list to change, but only if it expands: we require that the loaded model names are a superset of the desired model names.
                # If that's not the case, we will ignore anything that was reloaded and just retrain all (currently desired) models, overwriting the existing files.
                model_names_desired_set = set(
                    # Extend with sklearn versions of the glmnet models
                    _extend_model_list_with_sklearn_versions_of_glmnet_models(
                        chosen_models
                    )
                )
                # Filter results down to models in the desired list, in case we reloaded a superset of the desired models:
                results = [r for r in results if r[0] in model_names_desired_set]
                model_names_loaded_set = set(model_name for model_name, _, _ in results)
                if model_names_loaded_set != model_names_desired_set:
                    # The model list has changed since the last run,
                    # and the loaded model names are not a superset of desired model names.
                    # Therefore we will retrain all models.
                    logger.info(
                        f"Model list has changed since last run. Loaded model names {model_names_loaded_set} do not match desired model names {model_names_desired_set}. Retraining all models.",
                        extra={"split_key": split_key},
                    )

                # 2) Sanity check that the reloaded split keys are all what we expect them to be.
                elif any(
                    split_key_loaded != split_key for _, split_key_loaded, _ in results
                ):
                    logger.info(
                        f"Reloaded split key from temporary files does not match expected split key; retraining all models.",
                        extra={"split_key": split_key},
                    )

                else:
                    # All models are already trained. Skip retraining.
                    logger.info(
                        f"Reloaded models from previous incomplete training run. Skipping retraining.",
                        extra={"split_key": split_key},
                    )
                    split_keys_to_skip.append(split_key)
                    parallel_job_outputs.append(results)

    # Parallelize over V-J gene pair subsets of the data
    # ("loky" backend required; "multiprocessing" backend can deadlock with xgboost, see https://github.com/dmlc/xgboost/issues/7044#issuecomment-1039912899 , https://github.com/dmlc/xgboost/issues/2163 , and https://github.com/dmlc/xgboost/issues/4246 )
    # Wrap in tqdm for progress bar (see https://stackoverflow.com/a/76726101/130164)
    parallel_job_outputs.extend(
        list(
            tqdm(
                Parallel(return_as="generator", n_jobs=n_jobs, backend="loky")(
                    delayed(_fit_all_on_a_sequence_subset)(
                        split_key=split_key,
                        adata_split=adata_split,
                        base_classifier=base_classifier,
                        fold_id=fold_id,
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_column,
                        sample_weight_strategy=sample_weight_strategy,
                        chosen_models=chosen_models,
                        use_gpu=use_gpu,
                        n_jobs=n_jobs,  # This controls nested multiprocessing
                        temporary_save_file=temporary_save_files[split_key],
                    )
                    # split_key could be (v_gene, j_gene) for example
                    for split_key, adata_split in base_classifier.generate_training_chunks(
                        adata_train,
                        # When splitting anndata into chunks, return chunks as copies rather than as views.
                        # Otherwise, when we pass views into Joblib, to achieve views in the child processes, the entire source anndata (adata_split_as_view._adata_ref) has to get copied to each child process.
                        # With return_copies=True, we convert the adata_split views into standalone anndatas before we enter joblib parallelization land.
                        # This avoids the issue of copying a giant full andata to each child process.
                        return_copies=True,
                        # Do not generate jobs for these chunks:
                        skip_chunks=split_keys_to_skip,
                    )
                ),
                # API note: len(training_chunk_hashes) == base_classifier.n_training_chunks(adata_train)
                total=len(training_chunk_hashes) - len(split_keys_to_skip),
            )
        )
    )

    # Now we have all the results from parallelization plus any results that were reloaded from disk.
    if len(parallel_job_outputs) != len(training_chunk_hashes):
        # Even failing jobs should return None, so we should always have the same number of results as splits.
        # This is a sanity check that we didn't both reload AND retrain the same models accidentally.
        raise ValueError(
            f"Expected {len(training_chunk_hashes)} splits, but got {len(parallel_job_outputs)} results. Will not save any final models or eliminate the temporary directory."
        )

    # Unwrap each job's outputs
    # Resulting nested dict structure: model_name -> split_key such as (v_gene, j_gene) -> clf
    models = defaultdict(dict)
    for job_output in parallel_job_outputs:
        if job_output is None:
            # Skip failed jobs: these are ones where we rejected a V-J gene pair because it had too few sequences
            continue
        for (
            model_name,
            split_key,
            clf,
        ) in job_output:
            models[model_name][split_key] = clf

    for model_name, clfs in models.items():
        joblib.dump(
            {"models": clfs, "classes": classes},
            models_base_dir
            / f"{fold_label_train}_model.{model_name}.{fold_id}.split_models.joblib",
        )

    # After successful completion, delete temporary files. Remove the whole folder.
    shutil.rmtree(temporary_save_dir)

    if clear_cache:
        io.clear_cached_fold_embeddings()
    del adata_train
    gc.collect()


def _fit_all_on_a_sequence_subset(
    split_key: SplitKeyType,
    adata_split: anndata.AnnData,
    base_classifier: Type[VJGeneSpecificSequenceClassifier],
    fold_id: int,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
    chosen_models: List[str],
    use_gpu: bool,
    n_jobs: int,
    temporary_save_file: Path,
) -> Optional[list]:
    """
    This function will be run in parallel on each V-J gene pair subset of the data. Returns list of model fits.
    If there are very few sequences in this V-J gene pair subset, returns None instead.
    """
    # Check if there are enough sequences to train a model
    min_sequences_per_split = 10
    if adata_split.shape[0] < min_sequences_per_split:
        logger.info(
            f"Skipping {split_key}: only {adata_split.shape[0]} sequences, less than the {min_sequences_per_split} required to train a model"
        )
        return None

    results = []

    # Deferred to here, so that we don't waste time if we are reloading from disk.
    adata_split_featurized = base_classifier._featurize_split(
        adata_split,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )

    with add_logging_context(split_key=split_key):
        for model_name, clf in fit(
            split_key=split_key,
            X_train=adata_split_featurized.X,
            y_train=adata_split_featurized.y,
            train_sample_weights=adata_split_featurized.sample_weights,
            train_metadata=adata_split_featurized.metadata,
            is_train_raw=True,  # see is_raw comments above
            fold_id=fold_id,
            target_obs_column=target_obs_column,
            chosen_models=chosen_models,
            use_gpu=use_gpu,
            n_jobs=n_jobs,
        ):
            results.append((model_name, split_key, clf))

    # Save to temporary directory
    joblib.dump(results, temporary_save_file)

    return results


def fit(
    split_key: SplitKeyType,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_weights: np.ndarray,
    train_metadata: pd.DataFrame,
    is_train_raw: bool,
    fold_id: int,
    target_obs_column: TargetObsColumnEnum,
    chosen_models: List[str],
    use_gpu: bool,
    n_jobs: int,
):
    ## Build and run classifiers.
    models, train_participant_labels = training_utils.prepare_models_to_train(
        X_train=X_train,
        y_train=y_train,
        fold_id=fold_id,
        target_obs_column=target_obs_column,
        chosen_models=chosen_models,
        use_gpu=use_gpu,
        output_prefix=None,
        train_metadata=train_metadata,
        n_jobs=n_jobs,
    )

    for model_name, model_clf in models.items():
        try:
            if not is_train_raw:
                patched_model = model_clf
            else:
                # If we are using raw data, we need to convert the model into a pipeline that starts with a StandardScaler.
                patched_model = training_utils.prepend_scaler_if_not_present(model_clf)

            # If this is a Glmnet model, don't store internal CV predicted probabilities. They're huge.
            patched_model = training_utils.disable_glmnet_storage_of_internal_cv_predicted_probabilities(
                patched_model
            )

            clf, _ = training_utils.run_model_multiclass(
                model_name=model_name,
                model_clf=patched_model,
                X_train=X_train,
                y_train=y_train,
                fold_id=fold_id,
                train_sample_weights=train_sample_weights,
                train_groups=train_participant_labels,
                # disable export and evaluation:
                export=False,
                X_test=None,
                y_test=None,
                output_prefix=None,
                fold_label_train=None,
                fold_label_test=None,
            )
            yield model_name, clf

            # For Glmnet models, also retrain with sklearn
            if (
                enable_training_sklearn_versions_of_glmnet
                and clf is not None
                and training_utils.does_fitted_model_support_conversion_from_glmnet_to_sklearn(
                    clf
                )
            ):
                # Also train a sklearn model using the best lambda from the glmnet model, as an extra sanity check. Results should be identical.
                sklearn_clf = training_utils.convert_trained_glmnet_pipeline_to_untrained_sklearn_pipeline_at_best_lambda(
                    clf
                )
                sklearn_model_name = (
                    model_name.replace("_cv", "") + "_sklearn_with_lambdamax"
                )
                sklearn_clf, _ = training_utils.run_model_multiclass(
                    model_name=sklearn_model_name,
                    model_clf=sklearn_clf,
                    X_train=X_train,
                    y_train=y_train,
                    fold_id=fold_id,
                    train_sample_weights=train_sample_weights,
                    train_groups=train_participant_labels,
                    # disable export and evaluation:
                    export=False,
                    X_test=None,
                    y_test=None,
                    output_prefix=None,
                    fold_label_train=None,
                    fold_label_test=None,
                )
                yield sklearn_model_name, sklearn_clf

        except Exception as err:
            logger.warning(f"Failed to fit model {model_name} for {split_key}: {err}")


def run_classify_with_all_models(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
    fold_label_train: str,
    fold_label_test: str,
    chosen_models: List[str],
    # n_jobs used for internal parallelism, not for parallelizing over folds (because sequence model fit is very expensive)
    n_jobs: int,
    fold_ids: Optional[List[int]] = None,
    use_gpu=False,
    clear_cache=True,
    fail_on_error=False,
    sequence_subset_strategy: SequenceSubsetStrategy = SequenceSubsetStrategy.split_Vgene_and_isotype,
    exclude_rare_v_genes: bool = True,
    resume: bool = False,
) -> crosseval.ExperimentSet:
    """
    Train sequence models. Sequence data is re-scaled from original raw data for each split.

    sequence_subset_strategy: Sequence splitting strategy used when training sequence models, e.g. "train a separate model for each V gene and J gene combination".

    n_jobs is used to parallelize fits over data subsets.

    exclude_rare_v_genes: Don't fit models for splits corresponding to rare V genes. This parameter is ignored and this functionality is disabled (i.e. no extra filtering applied) if the sequence subset strategy does not include splitting by V gene.

    resume: bool, default False
        If resume is True, we will reuse any models that have already been trained and saved to disk.
        Otherwise will immediately overwrite all previous temporary models.
        The temporary directory is always removed after a full successful train.
    """
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    TargetObsColumnEnum.validate(target_obs_column)
    SampleWeightStrategy.validate(sample_weight_strategy)
    target_obs_column.confirm_compatibility_with_gene_locus(gene_locus)
    target_obs_column.confirm_compatibility_with_cross_validation_split_strategy(
        config.cross_validation_split_strategy
    )

    if fold_ids is None:
        fold_ids = config.all_fold_ids
    logger.info(f"Starting train on folds: {fold_ids}")

    job_outputs = [
        _run_models_on_fold(
            fold_id=fold_id,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
            fold_label_train=fold_label_train,
            fold_label_test=fold_label_test,
            chosen_models=chosen_models,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            clear_cache=clear_cache,
            fail_on_error=fail_on_error,
            sequence_subset_strategy=sequence_subset_strategy,
            exclude_rare_v_genes=exclude_rare_v_genes,
            resume=resume,
        )
        for fold_id in fold_ids
    ]

    return crosseval.ExperimentSet(model_outputs=job_outputs)
