"""
Train blending metamodels.
"""

import dataclasses
import gc
import logging
from typing import Dict, List, Optional, Tuple, Union

import anndata
import joblib
import numpy as np
import sklearn.base
from feature_engine.preprocessing import MatchVariables
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline

from malid import config, helpers, io
from malid.datamodels import (
    CVScorer,
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    DEMOGRAPHICS_COLUMNS,
)
from malid.external import model_evaluation
from malid.train import training_utils
from malid.trained_model_wrappers import BlendingMetamodel
from malid.trained_model_wrappers.blending_metamodel import (
    DemographicsFeaturizer,
    MetamodelConfig,
)
from malid.trained_model_wrappers import ConvergentClusterClassifier
from malid.trained_model_wrappers import RepertoireClassifier
from malid.trained_model_wrappers.rollup_sequence_classifier import (
    RollupSequenceClassifier,
)


logger = logging.getLogger(__name__)


def get_metamodel_flavors(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    fold_id: int,
    base_model_train_fold_name: str,
) -> Dict[str, MetamodelConfig]:
    """Define metamodel flavors (i.e. different combinations of options).
    These will be trained for each fold, for each gene locus and then for all gene loci together.
    Each metamodel flavor has an identifier (gene locus, target, name) and a MetamodelConfig.
    Usually, we have one default metamodel flavor per TargetObsColumn. These get a default name equal to the TargetObsColumn's name.
    Some TargetObsColumns have multiple metamodel flavors, which get a name equal to the TargetObsColumn's name plus a suffix.

    Parameters:
        gene_locus: gene locus or combination of gene loci (composite flag)
        target_obs_column: TargetObsColumnEnum
        fold_id: fold ID (used to load the correct base models)
        base_model_train_fold_name: name of the fold used to train the base models (used to load the correct base models)

    """
    sample_weight_strategy = SampleWeightStrategy.ISOTYPE_USAGE
    default_config = MetamodelConfig(
        sample_weight_strategy=sample_weight_strategy,
        submodels={
            single_gene_locus: {
                # Loads models 1+2+3 from disk, but generates model3-rollup from scratch (rollup is just wrapper logic, not something we've trained)
                # Model 1. Should run on all specimens. So this will have the authoritative metadata.
                "repertoire_stats": RepertoireClassifier(
                    fold_id=fold_id,
                    model_name=config.metamodel_base_model_names.model_name_overall_repertoire_composition,
                    fold_label_train=base_model_train_fold_name,  # what base models were trained on
                    gene_locus=single_gene_locus,
                    target_obs_column=target_obs_column,
                ),
                # Model 2. May have abstentions for specimens that had no sequences fall in a predictive cluster
                "convergent_cluster_model": ConvergentClusterClassifier(
                    fold_id=fold_id,
                    model_name=config.metamodel_base_model_names.model_name_convergent_clustering,
                    fold_label_train=base_model_train_fold_name,  # what base models were trained on
                    gene_locus=single_gene_locus,
                    target_obs_column=target_obs_column,
                ),
                # Model 3. Runs on all specimens.
                "sequence_model": RollupSequenceClassifier(
                    fold_id=fold_id,
                    model_name_sequence_disease=config.metamodel_base_model_names.model_name_sequence_disease,
                    fold_label_train=base_model_train_fold_name,  # what base models were trained on
                    gene_locus=single_gene_locus,
                    target_obs_column=target_obs_column,
                    sample_weight_strategy=sample_weight_strategy,
                ),
            }
            for single_gene_locus in gene_locus
        },
        extra_metadata_featurizers=None,
        regress_out_featurizers=None,
        interaction_terms=None,
    )

    metamodel_flavors = {"default": default_config}

    if target_obs_column == TargetObsColumnEnum.disease and GeneLocus.BCR in gene_locus:
        # Predict disease from specimen isotype counts/proportions alone
        # (on full dataset, not on a subset with defined age/sex/ethnicity)
        # NOTE: other models above might have abstentions; this one won't

        metamodel_flavors["isotype_counts_only"] = dataclasses.replace(
            default_config,
            # Remove main models
            submodels=None,
            # Use isotype count features only
            extra_metadata_featurizers={
                "isotype_counts": DemographicsFeaturizer(
                    covariate_columns=[
                        f"isotype_proportion:{isotype}"
                        for isotype in helpers.isotype_groups_kept[GeneLocus.BCR]
                    ]
                )
            },
        )

    # Control for demographics.
    extra_covariate_columns = {
        # Look at demographics effect on disease prediction
        TargetObsColumnEnum.disease_all_demographics_present: DEMOGRAPHICS_COLUMNS,
    }
    if target_obs_column in extra_covariate_columns.keys():
        # Repeat this target, additionally regressing out demographics OR adding demographic columns to the metamodel featurization
        # Could also add other columns in addition to demographics, or on their own
        # Make sure they are also listed in helpers.extract_specimen_metadata_from_obs_df() so that this metadata is available when training metamodel.

        metamodel_flavors["with_demographics_columns"] = dataclasses.replace(
            default_config,
            # Add as extra columns
            extra_metadata_featurizers={
                "demographics": DemographicsFeaturizer(
                    covariate_columns=extra_covariate_columns[target_obs_column]
                )
            },
            interaction_terms=(
                # Left
                [
                    "BCR:*",
                    "TCR:*",
                ],
                # Right
                ["demographics:*"],
            ),
        )

        metamodel_flavors["demographics_regressed_out"] = dataclasses.replace(
            default_config,
            # Regress out these columns
            regress_out_featurizers={
                "demographics": DemographicsFeaturizer(
                    covariate_columns=extra_covariate_columns[target_obs_column]
                )
            },
        )

        # And add versions that are demographics only.
        # NOTE: other models above might have abstentions; these ones won't
        # - Use all demographic features
        metamodel_flavors["demographics_only"] = dataclasses.replace(
            default_config,
            # Remove main models
            submodels=None,
            # Keep demographics model only
            extra_metadata_featurizers={
                "demographics": DemographicsFeaturizer(
                    covariate_columns=extra_covariate_columns[target_obs_column]
                )
            },
        )
        # - Use only a single demographics feature at a time (e.g. only sex, only age, etc.)
        for covariate_column in extra_covariate_columns[target_obs_column]:
            metamodel_flavors[
                f"demographics_only_{covariate_column}"
            ] = dataclasses.replace(
                default_config,
                # Remove main models
                submodels=None,
                # Keep demographics model only
                extra_metadata_featurizers={
                    "demographics": DemographicsFeaturizer(
                        covariate_columns=[covariate_column]
                    )
                },
            )

    return metamodel_flavors


def _get_fold_data(
    fold_id: int,
    fold_label: str,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
) -> Dict[GeneLocus, anndata.AnnData]:
    # gene_locus may have multiple flags, so we need to get the data for each locus
    return {
        single_gene_locus: io.load_fold_embeddings(
            fold_id=fold_id,
            fold_label=fold_label,
            gene_locus=single_gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=sample_weight_strategy,
        )
        for single_gene_locus in gene_locus
    }


def run_classify_with_all_models(
    # metamodel_config is specific to a particular fold ID (uses those fold's models)
    fold_id: int,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    metamodel_flavor: str,
    metamodel_config: MetamodelConfig,
    base_model_train_fold_name: str,
    metamodel_fold_label_train: str,
    # if metamodel_fold_label_test is None, then we don't evaluate the metamodel on any test set
    metamodel_fold_label_test: Optional[str],
    chosen_models: List[str],
    n_jobs=1,
    use_gpu=False,
    clear_cache=True,
    fail_on_error=False,
) -> Tuple[model_evaluation.ExperimentSet, MetamodelConfig]:
    """Train metamodel using all active flags in gene_locus object."""
    logger.info(f"Starting train on fold {fold_id}")

    GeneLocus.validate(gene_locus)  # Multi-flag allowed here!
    TargetObsColumnEnum.validate(target_obs_column)
    SampleWeightStrategy.validate(metamodel_config.sample_weight_strategy)

    # The metamodel base dir captures the gene locus (or loci), the classification target, and the name of the metamodel flavor (corresponds to all the metamodel settings)
    metamodels_base_dir = BlendingMetamodel._get_metamodel_base_dir(
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        metamodel_flavor=metamodel_flavor,
    )
    metamodels_base_dir.mkdir(exist_ok=True, parents=True)

    # Within the metamodel base dir, there are still further variants of the metamodel, based on:
    # the base-model training fold, the meta-model training fold, the meta-model name, and the fold ID.
    # e.g. the base models are trained on fold 0's "train_smaller" set, then a "rf_multiclass" metamodel is trained on fold 0's "validation" set.
    # (The fold ID and metamodel name are added later, not to this output_prefix starting point.)
    output_prefix = (
        metamodels_base_dir
        / f"{base_model_train_fold_name}_applied_to_{metamodel_fold_label_train}_model"
    )

    # construct probability vectors for each specimen in validation set (called "train" here) and test set
    # feature vector for each specimen includes `P(specimen is healthy)` from healthy/sick repertoire stats model and trimmed mean `P(sequence is disease D)` across all sequences from a specimen
    adatas_train: Dict[GeneLocus, anndata.AnnData] = _get_fold_data(
        fold_id=fold_id,
        fold_label=metamodel_fold_label_train,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=metamodel_config.sample_weight_strategy,
    )
    featurized_train = BlendingMetamodel._featurize(
        data=adatas_train,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        metamodel_config=metamodel_config,
    )
    # Featurizing the training set can update metamodel_config:
    # It can provide a regress_out_pipeline to be used for downstream test sets, and can fit the one-hot encoders in the featurizers
    # Retrieve the latest metamodel_config.
    metamodel_config = featurized_train.extras["metamodel_config"]

    adatas_test, featurized_test = None, None
    if metamodel_fold_label_test is not None:
        # metamodel_fold_label_test can be None for global fold (no test set exists to evaluate the metamodel)
        adatas_test = _get_fold_data(
            fold_id=fold_id,
            fold_label=metamodel_fold_label_test,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            sample_weight_strategy=metamodel_config.sample_weight_strategy,
        )
        featurized_test = BlendingMetamodel._featurize(
            data=adatas_test,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            metamodel_config=metamodel_config,
        )

    X_train = featurized_train.X
    y_train = featurized_train.y
    feature_order = X_train.columns

    (
        X_test,
        y_test,
        metadata_test,
        abstentions_test_ground_truth_labels,
        abstentions_test_metadata,
    ) = (None, None, None, None, None)
    if featurized_test is not None:
        # featurized_test can be None for global fold (no test set exists to evaluate the metamodel)
        X_test = featurized_test.X
        y_test = featurized_test.y
        metadata_test = featurized_test.metadata
        abstentions_test_ground_truth_labels = featurized_test.abstained_sample_y
        abstentions_test_metadata = featurized_test.abstained_sample_metadata

        # Make feature order consistent
        X_test = X_test[feature_order]
        assert np.array_equal(X_train.columns, X_test.columns)

    ## Build and run classifiers.

    models, train_participant_labels = training_utils.prepare_models_to_train(
        X_train=X_train,
        y_train=y_train,
        train_metadata=featurized_train.metadata,
        fold_id=fold_id,
        target_obs_column=target_obs_column,
        chosen_models=chosen_models,
        use_gpu=use_gpu,
        output_prefix=output_prefix,
        n_jobs=n_jobs,
        # Use MCC scoring for internal/nested cross-validation,
        # because we intend for the metamodel to be used out of the box for label prediction (we won't do any more decision threshold tuning).
        # This is in contrast to AUC (default) internal-CV optimization for the base models, which we use to generated predicted probabilities, not labels (though we can do optional decision threshold tuning with MCC for those to get label predictions too)
        cv_scoring=CVScorer.MCC,
    )

    results = []

    # Pipeline order is: matchvariables -> standardscaler -> clf.
    for model_name, model_clf in models.items():
        # model_clf may be an individual estimator, or it may already be a pipeline
        is_pipeline = type(model_clf) == Pipeline
        if is_pipeline:
            # If already a pipeline, we will prepend steps, in reverse order.
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

        else:
            # Not yet a pipeline.
            # See notes above on these pipeline steps.
            model_pipeline = make_pipeline(
                MatchVariables(missing_values="raise"),
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
            fold_label_train=metamodel_fold_label_train,
            fold_label_test=metamodel_fold_label_test,
            train_groups=train_participant_labels,
            test_metadata=metadata_test,
            test_abstention_ground_truth_labels=abstentions_test_ground_truth_labels,
            test_abstention_metadata=abstentions_test_metadata,
            fail_on_error=fail_on_error,
        )
        if result is not None:
            # result can be None for global fold (no test set exists to evaluate the metamodel)
            results.append(result)

    # Export model components
    # This is common across all metamodel names (since they all have the same input featurization)
    metamodel_components = {
        "gene_locus": gene_locus,
        "target_obs_column": target_obs_column,
        "metamodel_flavor": metamodel_flavor,
        "metamodel_config": metamodel_config,  # may include RegressOutCovariates pipeline
        # feature order not necessary to export here (already stored in clf.feature_names_in_), but helpful for debugging
        "feature_order": feature_order,
    }
    joblib.dump(
        metamodel_components,
        f"{output_prefix}.{fold_id}.metamodel_components.joblib",
    )

    # Clear RAM
    if clear_cache:
        io.clear_cached_fold_embeddings()
    del adatas_train, adatas_test
    gc.collect()

    return model_evaluation.ExperimentSet(model_outputs=results), metamodel_config
