"""
Train blending metamodels.
"""

import dataclasses
import gc
import logging
from typing import Dict, List, Optional, Tuple, Union
import itertools

import anndata
import joblib
import numpy as np
import sklearn.base
from feature_engine.preprocessing import MatchVariables
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline

from malid import config, helpers, io
from malid.datamodels import (
    CVScorer,
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    map_cross_validation_split_strategy_to_default_target_obs_column,
    DEMOGRAPHICS_COLUMNS,
)
import crosseval
from malid.train import training_utils
from malid.trained_model_wrappers import BlendingMetamodel
from malid.trained_model_wrappers.blending_metamodel import (
    DemographicsFeaturizer,
    MetamodelConfig,
    STUB_SUBMODEL,
)
from malid.trained_model_wrappers import ConvergentClusterClassifier
from malid.trained_model_wrappers import RepertoireClassifier
from malid.trained_model_wrappers import (
    VJGeneSpecificSequenceModelRollupClassifier,
)
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)
from malid.trained_model_wrappers.rollup_sequence_classifier import (
    RollupSequenceClassifier,
)


logger = logging.getLogger(__name__)


def get_metamodel_flavors(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    fold_id: int,
    base_model_train_fold_name: str,
    # If not provided, these will be set to base_model_train_fold_name followed by suffix "1" and "2":
    base_model_train_fold_name_for_sequence_model: Optional[str] = None,
    base_model_train_fold_name_for_aggregation_model: Optional[str] = None,
    # Special flag to skip loading submodels from disk (slow), and instead return stubs inside MetamodelConfig.submodels. Assumes all submodels are trained and available.
    use_stubs_instead_of_submodels: bool = False,
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
    target_obs_column.confirm_compatibility_with_gene_locus(gene_locus)
    target_obs_column.confirm_compatibility_with_cross_validation_split_strategy(
        config.cross_validation_split_strategy
    )

    if base_model_train_fold_name_for_sequence_model is None:
        # e.g. train_smaller1
        base_model_train_fold_name_for_sequence_model = f"{base_model_train_fold_name}1"
    if base_model_train_fold_name_for_aggregation_model is None:
        # e.g. train_smaller2
        base_model_train_fold_name_for_aggregation_model = (
            f"{base_model_train_fold_name}2"
        )

    # Which submodels are available? Some may have failed to train or not been trained yet.
    # Try to load each one in.
    # (Alternative considered: load them all for "default" configuration below, but if that fails, then we can't recover and create subsets.)
    all_available_submodels: Dict[GeneLocus, Dict[str, ImmuneClassifierMixin]] = {}
    for single_gene_locus in gene_locus:
        all_available_submodels[single_gene_locus] = {}
        # Load models 1+2+3 from disk, but generate model3-rollup from scratch (rollup is just wrapper logic, not something we've trained)
        try:
            # Model 1. Should run on all specimens. So this will have the authoritative metadata.
            all_available_submodels[single_gene_locus]["repertoire_stats"] = (
                STUB_SUBMODEL
                if use_stubs_instead_of_submodels
                else RepertoireClassifier(
                    fold_id=fold_id,
                    model_name=config.metamodel_base_model_names.model_name_overall_repertoire_composition[
                        single_gene_locus
                    ],
                    fold_label_train=base_model_train_fold_name,  # what base models were trained on
                    gene_locus=single_gene_locus,
                    target_obs_column=target_obs_column,
                    sample_weight_strategy=config.sample_weight_strategy,
                )
            )
        except Exception as err:
            logger.warning(
                f"Failed to load repertoire_stats model for {single_gene_locus}: {err}"
            )

        try:
            # Model 2. May have abstentions for specimens that had no sequences fall in a predictive cluster
            all_available_submodels[single_gene_locus]["convergent_cluster_model"] = (
                STUB_SUBMODEL
                if use_stubs_instead_of_submodels
                else ConvergentClusterClassifier(
                    fold_id=fold_id,
                    model_name=config.metamodel_base_model_names.model_name_convergent_clustering[
                        single_gene_locus
                    ],
                    fold_label_train=base_model_train_fold_name_for_sequence_model,  # what base models were trained on
                    gene_locus=single_gene_locus,
                    target_obs_column=target_obs_column,
                )
            )
        except Exception as err:
            logger.warning(
                f"Failed to load convergent_cluster_model model for {single_gene_locus}: {err}"
            )

        try:
            # Model 3. Runs on all specimens.
            all_available_submodels[single_gene_locus]["sequence_model"] = (
                STUB_SUBMODEL
                if use_stubs_instead_of_submodels
                else VJGeneSpecificSequenceModelRollupClassifier(
                    fold_id=fold_id,
                    # Provide sequence model name here (rollup model will be trained on top of this model):
                    base_sequence_model_name=config.metamodel_base_model_names.base_sequence_model_name[
                        single_gene_locus
                    ],
                    base_model_train_fold_label=base_model_train_fold_name_for_sequence_model,
                    # Rollup model details:
                    rollup_model_name=config.metamodel_base_model_names.aggregation_sequence_model_name[
                        single_gene_locus
                    ],
                    fold_label_train=base_model_train_fold_name_for_aggregation_model,
                    gene_locus=single_gene_locus,
                    target_obs_column=target_obs_column,
                    sample_weight_strategy=config.sample_weight_strategy,
                    sequence_subset_strategy=config.metamodel_base_model_names.base_sequence_model_subset_strategy,
                )
            )
        except Exception as err:
            logger.warning(
                f"Failed to load sequence_model model for {single_gene_locus}: {err}"
            )

    # Subset to submodels available in all gene loci.
    # (We want the submodel choice to be consistent across gene loci used in a metamodel.
    # E.g. if training a BCR+TCR metamodel, and model2 is not available for TCR, then don't include the BCR model2 either.)

    # Find which submodels are available in each gene locus,
    # and take the intersection across gene loci,
    # while keeping the sort order of the submodels consistent with the model 1/2/3 designation:
    submodel_name_sort_order = {
        "repertoire_stats": 1,
        "convergent_cluster_model": 2,
        "sequence_model": 3,
    }
    all_available_submodel_names: List[str] = sorted(
        set.intersection(
            *[
                set(submodels.keys())
                for single_gene_locus, submodels in all_available_submodels.items()
            ]
        ),
        key=lambda val: submodel_name_sort_order[val],
    )
    # Apply the filter:
    for single_gene_locus, submodels in all_available_submodels.items():
        all_available_submodels[single_gene_locus] = {
            submodel_name: submodel
            for submodel_name, submodel in submodels.items()
            if submodel_name in all_available_submodel_names
        }

    def _create_config_with_submodels(
        submodel_names: List[str], **kwargs
    ) -> MetamodelConfig:
        """
        Attempt to form a MetamodelConfig with the given submodels.
        If any requested submodels are missing, throw an error.
        kwargs are passed to the MetamodelConfig constructor.
        """
        return MetamodelConfig(
            sample_weight_strategy=config.sample_weight_strategy,
            submodels={
                single_gene_locus: {
                    name: all_available_submodels[single_gene_locus][name]
                    for name in submodel_names
                }
                for single_gene_locus in gene_locus
            },
            **kwargs,
        )

    # Define the metamodel flavors. Each flavor has a name and a MetamodelConfig.
    metamodel_flavors: Dict[str, MetamodelConfig] = {}

    # Generate "default" flavor (Models 1 + 2 + 3) and all dependent flavors.
    try:
        default_config = _create_config_with_submodels(
            ["repertoire_stats", "convergent_cluster_model", "sequence_model"]
        )
        metamodel_flavors["default"] = default_config

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

    except Exception as err:
        logger.warning(
            f"Failed to create 'default' or dependent metamodel flavor: {err}"
        )

    ## Try Model 1, 2, 3, 1+2, 1+3, 2+3 combinations (but only for the default classification target, e.g. TargetObsColumnEnum.disease)
    # Use the available submodel names to form all combinations of submodels.
    if (
        target_obs_column
        == map_cross_validation_split_strategy_to_default_target_obs_column[
            config.cross_validation_split_strategy
        ]
    ):
        # How many submodels to combine: between 1 and n-1 inclusive, where n is the total number of submodels. The n-combination case is already created (default config above)
        for number_of_included_submodels in range(1, len(all_available_submodel_names)):
            for chosen_submodel_names in itertools.combinations(
                all_available_submodel_names, number_of_included_submodels
            ):
                # Name this combination of submodels.
                combination_name = (
                    f"subset_of_submodels_{'_'.join(chosen_submodel_names)}"
                )

                # Generate a config for this combination of submodels.
                metamodel_flavors[combination_name] = _create_config_with_submodels(
                    chosen_submodel_names
                )

    if (
        target_obs_column
        == map_cross_validation_split_strategy_to_default_target_obs_column[
            config.cross_validation_split_strategy
        ]
        and GeneLocus.BCR in gene_locus
    ):
        # Predict disease (or other default classification target) from specimen isotype counts/proportions alone
        # (on full dataset, not on a subset with defined age/sex/ethnicity)
        # NOTE: other models above might have abstentions; this one won't

        metamodel_flavors["isotype_counts_only"] = MetamodelConfig(
            sample_weight_strategy=config.sample_weight_strategy,
            # No main models
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
) -> Tuple[crosseval.ExperimentSet, MetamodelConfig]:
    """Train metamodel using all active flags in gene_locus object."""
    logger.info(f"Starting train on fold {fold_id}")

    GeneLocus.validate(gene_locus)  # Multi-flag allowed here!
    TargetObsColumnEnum.validate(target_obs_column)
    SampleWeightStrategy.validate(metamodel_config.sample_weight_strategy)
    target_obs_column.confirm_compatibility_with_gene_locus(gene_locus)
    target_obs_column.confirm_compatibility_with_cross_validation_split_strategy(
        config.cross_validation_split_strategy
    )

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
    output_prefix = metamodels_base_dir / BlendingMetamodel._get_model_file_prefix(
        base_model_train_fold_name, metamodel_fold_label_train
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

    X_train = featurized_train.X
    y_train = featurized_train.y
    feature_order = X_train.columns

    # Clear RAM
    if clear_cache:
        io.clear_cached_fold_embeddings()
    del adatas_train
    gc.collect()

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

        X_test = featurized_test.X
        y_test = featurized_test.y
        metadata_test = featurized_test.metadata
        abstentions_test_ground_truth_labels = featurized_test.abstained_sample_y
        abstentions_test_metadata = featurized_test.abstained_sample_metadata

        # Make feature order consistent
        X_test = X_test[feature_order]
        assert np.array_equal(X_train.columns, X_test.columns)

        # Clear RAM
        if clear_cache:
            io.clear_cached_fold_embeddings()
        del adatas_test
        gc.collect()
    else:
        # edge case: global fold has no test set exists to evaluate the metamodel
        (
            X_test,
            y_test,
            metadata_test,
            abstentions_test_ground_truth_labels,
            abstentions_test_metadata,
        ) = (None, None, None, None, None)

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
        # Scale columns.
        # Convert the model into a pipeline that starts with a StandardScaler, if it doesn't exist already.
        model_clf = training_utils.prepend_scaler_if_not_present(model_clf)

        # Prepend a MatchVariables if it doesn't exist already
        # Confirms features are in same order.
        # Puts in same order if they're not.
        # Throws error if any train column missing.
        # Drops any test column not found in train column list.
        # It's like saving a feature_order and doing X.loc[feature_order]
        if "matchvariables" not in model_clf.named_steps.keys():
            model_clf.steps.insert(
                0,
                ("matchvariables", MatchVariables(missing_values="raise")),
            )

        model_clf, result = training_utils.run_model_multiclass(
            model_name=model_name,
            model_clf=model_clf,
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

        # For Glmnet models, also record performance with lambda_1se flag flipped.
        if (
            model_clf is not None
            and result is not None
            and training_utils.does_fitted_model_support_lambda_setting_change(
                model_clf
            )
        ):
            model_clf2, result2 = training_utils.modify_fitted_model_lambda_setting(
                fitted_clf=model_clf,
                performance=result,
                X_test=X_test,
                output_prefix=output_prefix,
            )
            results.append(result2)

        if (
            model_clf is not None
            and training_utils.does_fitted_model_support_conversion_from_glmnet_to_sklearn(
                model_clf
            )
        ):
            # Also train a sklearn model using the best lambda from the glmnet model, as an extra sanity check. Results should be identical.
            sklearn_model = training_utils.convert_trained_glmnet_pipeline_to_untrained_sklearn_pipeline_at_best_lambda(
                model_clf
            )
            sklearn_model, result_sklearn = training_utils.run_model_multiclass(
                model_name=model_name.replace("_cv", "") + "_sklearn_with_lambdamax",
                model_clf=sklearn_model,
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
            results.append(result_sklearn)

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

    return crosseval.ExperimentSet(model_outputs=results), metamodel_config
