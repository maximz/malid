"""
Train repertoire stats models.
"""

import logging
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn.base
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer

from malid import config, helpers
from malid import io
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
)
from malid.external import model_evaluation
from malid.train import training_utils
from malid.trained_model_wrappers import RepertoireClassifier

logger = logging.getLogger(__name__)


def _get_fold_data(
    fold_id: int,
    fold_label: str,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    vj_count_matrix_columns_by_isotype: Optional[Dict[str, pd.Index]] = None,
):
    # for each fold:
    # load anndata without any filtering at all
    adata = io.load_fold_embeddings(
        fold_id=fold_id,
        fold_label=fold_label,
        target_obs_column=target_obs_column,
        gene_locus=gene_locus,
    )

    if vj_count_matrix_columns_by_isotype is None:
        # This means we are training the model
        # Filter V genes to remove rare V genes, so their minute differences in proportions don't affect our model

        # Get observed V gene frequencies
        v_gene_frequencies = (
            adata.obs["v_gene"]
            .astype("category")
            .cat.remove_unused_categories()
            .value_counts(normalize=True)
        )

        # Filtering criteria: median. Removing bottom half of genes, the ones that make up this fraction or less of the training data
        v_gene_frequency_threshold = v_gene_frequencies.quantile(0.5)

        # Which V genes does this leave?
        v_gene_frequency_filter = v_gene_frequencies > v_gene_frequency_threshold
        kept_v_genes = v_gene_frequencies[v_gene_frequency_filter].index.tolist()
        removed_v_genes = v_gene_frequencies[~v_gene_frequency_filter].index.tolist()

        # Apply filter
        orig_shape = adata.shape[0]
        orig_nunique = adata.obs["v_gene"].nunique()
        adata = adata[adata.obs["v_gene"].isin(kept_v_genes)]
        adata.obs["v_gene"] = (
            adata.obs["v_gene"].astype("category").cat.remove_unused_categories()
        )
        new_shape = adata.shape[0]
        new_nunique = adata.obs["v_gene"].nunique()

        # Report
        logger.info(
            f"Filtered out V genes for fold {fold_id}-{fold_label} ({gene_locus}, {target_obs_column}): {removed_v_genes}. Shape {orig_shape}->{new_shape}, V gene nunique {orig_nunique}->{new_nunique}."
        )

        # These V genes will be removed from test set too by virtue of vj_count_matrix_columns_by_isotype not including those V genes.

    ## Featurize repertoire
    # Make features for each isotype-supergroup (amplified separately, so don't compare across them) -- aggregate to specimen level (within each fold)
    # vj_count_matrix_columns_by_isotype is None if training fold
    featurized = RepertoireClassifier._featurize(
        dataset=adata,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        allow_missing_isotypes=False,
        vj_count_matrix_column_order=vj_count_matrix_columns_by_isotype,
    )

    # add to metadata
    featurized.metadata = featurized.metadata.assign(
        fold_id=fold_id, fold_label=fold_label
    )

    return (
        featurized.X,
        featurized.y,
        featurized.metadata,
        featurized.extras["vj_count_matrix_columns_by_isotype"],
    )


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
    fail_on_error=False,
):
    models_base_dir = RepertoireClassifier._get_model_base_dir(
        gene_locus=gene_locus, target_obs_column=target_obs_column
    )
    models_base_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = models_base_dir / f"{fold_label_train}_model"

    (
        X_train,
        y_train,
        train_metadata,
        train_vj_count_matrix_columns_by_isotype,
    ) = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_train,
        target_obs_column=target_obs_column,
        gene_locus=gene_locus,
        vj_count_matrix_columns_by_isotype=None,
    )

    (
        X_test,
        y_test,
        test_metadata,
        test_vj_count_matrix_columns_by_isotype,
    ) = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_test,
        target_obs_column=target_obs_column,
        gene_locus=gene_locus,
        vj_count_matrix_columns_by_isotype=train_vj_count_matrix_columns_by_isotype,
    )

    # sanity checks
    assert set(train_vj_count_matrix_columns_by_isotype.keys()) == set(
        test_vj_count_matrix_columns_by_isotype.keys()
    )
    for isotype_supergroup in train_vj_count_matrix_columns_by_isotype.keys():
        assert np.array_equal(
            train_vj_count_matrix_columns_by_isotype[isotype_supergroup],
            test_vj_count_matrix_columns_by_isotype[isotype_supergroup],
        )

    # Prep design matrix
    # don't log1p+scale+PCA the counts matrix up front. instead we will log1p+scale+PCA in the pipeline. so that we run PCA only on training data, and then project test data into same embedding
    def make_column_transformer(n_pcs):
        # See column transformer docs above -- match the order of feature_names
        return ColumnTransformer(
            [
                (
                    f"log1p-scale-PCA_{isotype_group}",
                    Pipeline(
                        steps=[
                            (
                                "log1p",
                                FunctionTransformer(
                                    # Runs this function as a transformation.
                                    np.log1p,
                                    validate=True,
                                    # Preserve feature names, so that get_feature_names_out() is supported
                                    feature_names_out="one-to-one",
                                ),
                            ),
                            ("scale", preprocessing.StandardScaler()),
                            ("pca", PCA(n_pcs, random_state=0)),
                        ]
                    ),
                    make_column_selector(pattern=f"{isotype_group}:pca"),
                )
                for isotype_group in helpers.isotype_groups_kept[gene_locus]
            ],
            remainder="passthrough",
        )

    n_samples = X_train.shape[0]
    n_pcs_effective = min(n_samples, RepertoireClassifier.n_pcs)
    if n_pcs_effective != RepertoireClassifier.n_pcs:
        logger.warning(
            f"Using only {n_pcs_effective} PCs, instead of desired {RepertoireClassifier.n_pcs} PCs, for fold {fold_id}-{fold_label_train} because n_samples={n_samples}"
        )

    # sanity check PCA transformation
    # https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data
    # make the PCA transformation be on a certain set of columns only, so we can hstack to adata.X with other features.
    column_trans = make_column_transformer(n_pcs=n_pcs_effective)
    column_trans = column_trans.fit(X_train)

    # sanity check resulting dimensions with an example transform
    assert column_trans.transform(X_train).shape == (
        X_train.shape[0],
        # f"{isotype_group}:pca_{n_pc}" from 1 to n_pcs+1 for each isotype_group, then other features from obs
        n_pcs_effective * len(helpers.isotype_groups_kept[gene_locus])
        + len(RepertoireClassifier._features_from_obs[gene_locus]),
    )

    ## Build and run classifiers.

    models, train_participant_labels = training_utils.prepare_models_to_train(
        X_train=X_train,
        y_train=y_train,
        train_metadata=train_metadata,
        fold_id=fold_id,
        target_obs_column=target_obs_column,
        chosen_models=chosen_models,
        use_gpu=use_gpu,
        output_prefix=output_prefix,
        n_jobs=n_jobs,
    )

    results = []
    for model_name, model_clf in models.items():
        # log1p, scale, PCA the count matrices only. then scale everything. then run classifier.
        # model_clf may be an individual estimator, or it may already be a pipeline
        is_pipeline = type(model_clf) == Pipeline
        if is_pipeline:
            # If already a pipeline:
            # Prepend a StandardScaler if it doesn't exist already
            model_pipeline = sklearn.base.clone(model_clf)
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
            # Prepend make_column_transformer
            model_pipeline.steps.insert(
                0, ("columntransformer", make_column_transformer(n_pcs=n_pcs_effective))
            )
        else:
            # Not yet a pipeline.
            model_pipeline = make_pipeline(
                make_column_transformer(n_pcs=n_pcs_effective),
                preprocessing.StandardScaler(),
                sklearn.base.clone(model_clf),
            )

        model_pipeline, result = training_utils.run_model_multiclass(
            model_name=model_name,
            model_clf=model_pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            fold_id=fold_id,
            output_prefix=output_prefix,
            fold_label_train=fold_label_train,
            fold_label_test=fold_label_test,
            train_groups=train_participant_labels,
            test_metadata=test_metadata,
            fail_on_error=fail_on_error,
        )
        results.append(result)

    # Save out column order so we can create features for new test sets later
    joblib.dump(
        train_vj_count_matrix_columns_by_isotype,
        f"{output_prefix}.{fold_id}.{fold_label_train}.specimen_vj_gene_counts_columns_joblib",
    )

    # Clear RAM
    if clear_cache:
        io.clear_cached_fold_embeddings()

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
    fail_on_error=False,
) -> model_evaluation.ExperimentSet:
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    TargetObsColumnEnum.validate(target_obs_column)

    if fold_ids is None:
        fold_ids = config.all_fold_ids
    logger.info(f"Starting train on folds: {fold_ids}")

    # Run in parallel
    # ("loky" backend required; "multiprocessing" backend can deadlock with xgboost, see https://github.com/dmlc/xgboost/issues/7044#issuecomment-1039912899 , https://github.com/dmlc/xgboost/issues/2163 , and https://github.com/dmlc/xgboost/issues/4246 )
    job_outputs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_models_on_fold)(
            fold_id=fold_id,
            gene_locus=gene_locus,
            fold_label_train=fold_label_train,
            fold_label_test=fold_label_test,
            chosen_models=chosen_models,
            target_obs_column=target_obs_column,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            clear_cache=clear_cache,
            fail_on_error=fail_on_error,
        )
        for fold_id in fold_ids
    )

    return model_evaluation.ExperimentSet(model_outputs=job_outputs)
