"""
Train sequence models. Keep past exposures mixed in with acute infections.
"""

import gc
import logging
from typing import List, Optional

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from malid import config, helpers
from malid import io
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
from malid.external import model_evaluation
from malid.train import training_utils
from malid.trained_model_wrappers import SequenceClassifier

logger = logging.getLogger(__name__)


def _get_fold_data(
    fold_id: int,
    fold_label: str,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
):
    # for each fold:
    # load anndata without any filtering at all
    adata = io.load_fold_embeddings(
        fold_id=fold_id,
        fold_label=fold_label,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )
    is_raw = False

    if helpers.should_switch_to_raw(adata):
        # Use raw data
        logger.info(f"Switching to raw data for alternative target {target_obs_column}")
        is_raw = True

    featurized = SequenceClassifier._featurize(
        repertoire=adata,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
        allow_missing_isotypes=False,
        is_raw=is_raw,
    )

    return (
        featurized.X,
        featurized.y,
        featurized.metadata,
        featurized.sample_weights,
        is_raw,
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
):
    models_base_dir = SequenceClassifier._get_model_base_dir(
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )
    models_base_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = models_base_dir / f"{fold_label_train}_model"

    (
        X_train,
        y_train,
        train_metadata,
        train_sample_weights,
        is_train_raw,
    ) = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_train,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )
    X_test, y_test, test_metadata, test_sample_weights, is_test_raw = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_test,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )
    if is_train_raw != is_test_raw:
        raise ValueError(
            f"The raw state of the train and test data is different: {is_train_raw} vs {is_test_raw}"
        )

    ## Build and run classifiers.

    models, train_participant_labels = training_utils.prepare_models_to_train(
        X_train=X_train,
        y_train=y_train,
        fold_id=fold_id,
        target_obs_column=target_obs_column,
        chosen_models=chosen_models,
        use_gpu=use_gpu,
        output_prefix=output_prefix,
        train_metadata=train_metadata,
        n_jobs=n_jobs,
    )

    results = []
    for model_name, model_clf in models.items():
        if not is_train_raw:
            patched_model = model_clf
        else:
            # If we are using raw data, we need to convert the model into a pipeline that starts with a StandardScaler.
            # model_clf may be an individual estimator, or it may already be a pipeline
            is_pipeline = type(model_clf) == Pipeline
            if is_pipeline:
                # If already a pipeline, prepend a StandardScaler
                patched_model = model_clf
                if "standardscaler" in patched_model.named_steps.keys():
                    raise ValueError("The pipeline already has a StandardScaler step")
                logger.info(
                    f"Inserting StandardScaler into existing pipeline for model {model_name}"
                )
                patched_model.steps.insert(
                    0, ("standardscaler", preprocessing.StandardScaler())
                )
            else:
                # Not yet a pipeline.
                logger.info(
                    f"Converting to pipeline with StandardScaler included for model {model_name}"
                )
                patched_model = make_pipeline(preprocessing.StandardScaler(), model_clf)

        patched_model, result = training_utils.run_model_multiclass(
            model_name=model_name,
            model_clf=patched_model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_groups=train_participant_labels,
            train_sample_weights=train_sample_weights,
            test_sample_weights=test_sample_weights,
            fold_id=fold_id,
            output_prefix=output_prefix,
            fold_label_train=fold_label_train,
            fold_label_test=fold_label_test,
            fail_on_error=fail_on_error,
        )
        results.append(result)

    if clear_cache:
        io.clear_cached_fold_embeddings()
    del X_train, y_train, train_metadata, train_sample_weights
    del X_test, y_test, test_metadata, test_sample_weights
    gc.collect()

    return results


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
) -> model_evaluation.ExperimentSet:
    """Train sequence models. Sequence data is scaled automatically (in this method or pre-scaled if target_obs_column = 'disease')."""
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    TargetObsColumnEnum.validate(target_obs_column)
    SampleWeightStrategy.validate(sample_weight_strategy)

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
        )
        for fold_id in fold_ids
    ]

    return model_evaluation.ExperimentSet(model_outputs=job_outputs)
