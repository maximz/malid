from typing import List, Optional, Union
import numpy as np
import pandas as pd
import crosseval
from crosseval import ExperimentSet, FeaturizedData, Classifier
from feature_engine.preprocessing import MatchVariables
from malid.external.standard_scaler_that_preserves_input_type import (
    StandardScalerThatPreservesInputType,
)
from malid.train import training_utils
from malid.train.one_vs_rest_except_negative_class_classifier import (
    OneVsNegativeClassifier,
)
from malid.train.vj_gene_specific_sequence_model_rollup_classifier_as_binary_ovr import (
    OneVsNegativeClassifierWithFeatureSubsettingByClass,
    BinaryOvRClassifierWithFeatureSubsettingByClass,
)
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
from malid import io, config
from malid.trained_model_wrappers import (
    VJGeneSpecificSequenceModelRollupClassifier,
    VJGeneSpecificSequenceClassifier,
)
import xgboost as xgb
from enum import Enum
import joblib
import anndata
import dataclasses

from malid.trained_model_wrappers.vj_gene_specific_sequence_classifier import (
    SequenceSubsetStrategy,
)
from malid.trained_model_wrappers.vj_gene_specific_sequence_model_rollup_classifier import (
    AggregationStrategy,
    SubsetRollupClassifierFeaturizedData,
)

from malid.external.logging_context_for_warnings import ContextLoggerWrapper
from log_with_context import add_logging_context

logger = ContextLoggerWrapper(name=__name__)


# Configure whether we train sklearn versions of the glmnet models.
# TODO: Remove this from the main code path. Convert it to an automated test in a sensible n>p regime to make sure training results are the same.
enable_training_sklearn_versions_of_glmnet = False


class ModelNameSuffixes(Enum):
    binary_ovr = "_as_binary_ovr"
    with_nans = "_with_nans"
    reweighed_by_subset_frequencies = "_reweighed_by_subset_frequencies"


@dataclasses.dataclass(eq=False)
class ContainerOfSeveralFeaturizedDatas:
    train_featurized_with_nans: FeaturizedData
    test_featurized_with_nans: FeaturizedData
    train_featurized_fillna: FeaturizedData
    test_featurized_fillna: FeaturizedData
    train_featurized_reweighed_by_subset_frequencies: FeaturizedData
    test_featurized_reweighed_by_subset_frequencies: FeaturizedData


def _load_base_model(
    fold_id: int,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
    # Provide sequence model name here (rollup model will be trained on top of this model)
    base_model_name: str,
    base_model_fold_label_train: str,
    sequence_subset_strategy: SequenceSubsetStrategy = SequenceSubsetStrategy.split_Vgene_and_isotype,
):
    base_classifier = sequence_subset_strategy.base_model
    return base_classifier(
        fold_id=fold_id,
        model_name_sequence_disease=base_model_name,
        fold_label_train=base_model_fold_label_train,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )


def _get_fold_data(
    fold_id: int,
    fold_label: str,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
) -> anndata.AnnData:
    return io.load_fold_embeddings(
        fold_id=fold_id,
        fold_label=fold_label,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )


def _featurize_for_fold(
    base_model: VJGeneSpecificSequenceClassifier,  # or a child class
    aggregation_strategy: AggregationStrategy,
    adata: anndata.AnnData,
    pregenerated_sequence_predictions: pd.DataFrame,
    feature_order: Optional[List[str]] = None,
) -> SubsetRollupClassifierFeaturizedData:
    return VJGeneSpecificSequenceModelRollupClassifier._featurize(
        data=adata,
        gene_subset_specific_sequence_model=base_model,
        aggregation_strategy=aggregation_strategy,
        feature_order=feature_order,
        pregenerated_sequence_predictions=pregenerated_sequence_predictions,
        # Keep NaNs for now - see VJGeneSpecificSequenceModelRollupClassifier._fill_missing_later below
        fill_missing=False,
    )


def _featurize_for_fold_reweighed_by_subset_frequencies(
    base_model: VJGeneSpecificSequenceClassifier,  # or a child class
    aggregation_strategy: AggregationStrategy,
    adata: anndata.AnnData,
    pregenerated_sequence_predictions: pd.DataFrame,
    feature_order: Optional[List[str]] = None,
    standardization_transformer_before_reweighing_by_subset_frequencies: Optional[
        StandardScalerThatPreservesInputType
    ] = None,
) -> SubsetRollupClassifierFeaturizedData:
    """Alternative version passing in reweigh_by_subset_frequencies and standardization_transformer_before_reweighing_by_subset_frequencies"""
    return VJGeneSpecificSequenceModelRollupClassifier._featurize(
        data=adata,
        gene_subset_specific_sequence_model=base_model,
        aggregation_strategy=aggregation_strategy,
        feature_order=feature_order,
        pregenerated_sequence_predictions=pregenerated_sequence_predictions,
        reweigh_by_subset_frequencies=True,
        standardization_transformer_before_reweighing_by_subset_frequencies=standardization_transformer_before_reweighing_by_subset_frequencies,
        # fill_missing must be True for this code path, for now
        fill_missing=True,
    )


def _run_models_on_fold(
    fold_id: int,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
    fold_label_train: str,
    fold_label_test: str,
    # Which rollup models to train:
    chosen_models: List[str],
    n_jobs: int,
    # Provide sequence model name here (rollup model will be trained on top of this model):
    base_model_name: str,
    base_model_fold_label_train: str,
    aggregation_strategies: List[AggregationStrategy],
    sequence_subset_strategy: SequenceSubsetStrategy = SequenceSubsetStrategy.split_Vgene_and_isotype,
    use_gpu=False,
    clear_cache=True,
    fail_on_error=False,
):
    base_model = _load_base_model(
        fold_id=fold_id,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
        base_model_name=base_model_name,
        base_model_fold_label_train=base_model_fold_label_train,
        sequence_subset_strategy=sequence_subset_strategy,
    )
    models_base_dir = VJGeneSpecificSequenceModelRollupClassifier._get_model_base_dir(
        sequence_models_base_dir=base_model.models_base_dir,
        base_sequence_model_name=base_model_name,
        base_model_train_fold_label=base_model_fold_label_train,
        split_short_name=base_model.split_short_name,
    )
    models_base_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = models_base_dir / f"{fold_label_train}_model"

    def featurize_for_aggregation_strategy(
        aggregation_strategy: AggregationStrategy,
        adata_train: anndata.AnnData,
        adata_test: anndata.AnnData,
        pregenerated_sequence_predictions_train: pd.DataFrame,
        pregenerated_sequence_predictions_test: pd.DataFrame,
    ):
        train_featurized_with_nans = _featurize_for_fold(
            base_model=base_model,
            aggregation_strategy=aggregation_strategy,
            adata=adata_train,
            # pregenerated_sequence_predictions is precomputed for efficiency
            pregenerated_sequence_predictions=pregenerated_sequence_predictions_train,
        )
        test_featurized_with_nans = _featurize_for_fold(
            base_model=base_model,
            aggregation_strategy=aggregation_strategy,
            adata=adata_test,
            # pregenerated_sequence_predictions is precomputed for efficiency
            pregenerated_sequence_predictions=pregenerated_sequence_predictions_test,
            # Pass feature order to test featurize step.
            # We don't want the model to generate features for all possible V-J gene pairs it has submodels for,
            # just the gene pairs that are actually in the training data.
            # And by passing feature order to test featurize step, we can apply consistent fill NA logic.
            feature_order=train_featurized_with_nans.X.columns.tolist(),
        )
        # Sanity check: confirm feature order is consistent
        assert np.array_equal(
            train_featurized_with_nans.X.columns, test_featurized_with_nans.X.columns
        )

        # So far we've kept NaNs in featurized data .X in cases when a specimen has no sequences with a particular V and J gene.
        # We will fill those NaNs for all but xgboost, which supports missing values.
        train_featurized_fillna = (
            VJGeneSpecificSequenceModelRollupClassifier._fill_missing_later(
                train_featurized_with_nans
            )
        )
        test_featurized_fillna = (
            VJGeneSpecificSequenceModelRollupClassifier._fill_missing_later(
                test_featurized_with_nans
            )
        )

        # Alternative: Reweigh by subset frequencies. (fill_missing must be enabled for now)
        train_featurized_reweighed_by_subset_frequencies = _featurize_for_fold_reweighed_by_subset_frequencies(
            base_model=base_model,
            aggregation_strategy=aggregation_strategy,
            adata=adata_train,
            # pregenerated_sequence_predictions is precomputed for efficiency
            pregenerated_sequence_predictions=pregenerated_sequence_predictions_train,
        )
        test_featurized_reweighed_by_subset_frequencies = _featurize_for_fold_reweighed_by_subset_frequencies(
            base_model=base_model,
            aggregation_strategy=aggregation_strategy,
            adata=adata_test,
            # pregenerated_sequence_predictions is precomputed for efficiency
            pregenerated_sequence_predictions=pregenerated_sequence_predictions_test,
            #
            # Pass feature order to test featurize step.
            # We don't want the model to generate features for all possible V-J gene pairs it has submodels for,
            # just the gene pairs that are actually in the training data.
            # And by passing feature order to test featurize step, we can apply consistent fill NA logic.
            feature_order=train_featurized_reweighed_by_subset_frequencies.X.columns.tolist(),
            #
            # Pass the trained standardizer
            standardization_transformer_before_reweighing_by_subset_frequencies=train_featurized_reweighed_by_subset_frequencies.extras[
                "standardization_transformer_before_reweighing_by_subset_frequencies"
            ],
        )
        # Sanity check: confirm feature order is consistent
        assert np.array_equal(
            train_featurized_reweighed_by_subset_frequencies.X.columns,
            test_featurized_reweighed_by_subset_frequencies.X.columns,
        )
        assert (
            train_featurized_reweighed_by_subset_frequencies.extras[
                "reweigh_by_subset_frequencies"
            ]
            == test_featurized_reweighed_by_subset_frequencies.extras[
                "reweigh_by_subset_frequencies"
            ]
            == True
        )
        return ContainerOfSeveralFeaturizedDatas(
            train_featurized_with_nans=train_featurized_with_nans,
            test_featurized_with_nans=test_featurized_with_nans,
            train_featurized_fillna=train_featurized_fillna,
            test_featurized_fillna=test_featurized_fillna,
            train_featurized_reweighed_by_subset_frequencies=train_featurized_reweighed_by_subset_frequencies,
            test_featurized_reweighed_by_subset_frequencies=test_featurized_reweighed_by_subset_frequencies,
        )

    # Featurize for each aggregation strategy.
    # But make featurization more efficient by precomputing and sharing sequence probabilities.
    adata_train = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_train,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )
    adata_test = _get_fold_data(
        fold_id=fold_id,
        fold_label=fold_label_test,
        gene_locus=gene_locus,
        target_obs_column=target_obs_column,
        sample_weight_strategy=sample_weight_strategy,
    )
    pregenerated_sequence_predictions_train = (
        VJGeneSpecificSequenceModelRollupClassifier._generate_sequence_predictions(
            gene_subset_specific_sequence_model=base_model, adata=adata_train
        )
    )
    pregenerated_sequence_predictions_test = (
        VJGeneSpecificSequenceModelRollupClassifier._generate_sequence_predictions(
            gene_subset_specific_sequence_model=base_model, adata=adata_test
        )
    )
    featurized_datas = {
        aggregation_strategy: featurize_for_aggregation_strategy(
            aggregation_strategy=aggregation_strategy,
            adata_train=adata_train,
            adata_test=adata_test,
            pregenerated_sequence_predictions_train=pregenerated_sequence_predictions_train,
            pregenerated_sequence_predictions_test=pregenerated_sequence_predictions_test,
        )
        for aggregation_strategy in aggregation_strategies
    }

    results = []

    def _run_single(
        model_name: str,
        model_clf: Classifier,
        train_featurized: FeaturizedData,
        test_featurized: FeaturizedData,
        are_nans_present: bool,
        aggregation_strategy: AggregationStrategy,
    ) -> List[Optional[crosseval.ModelSingleFoldPerformance]]:
        # Convert the model into a pipeline that starts with a StandardScaler
        patched_model = training_utils.prepend_scaler_if_not_present(model_clf)

        # Prepend a MatchVariables if it doesn't exist already
        # Confirms features are in same order.
        # Puts in same order if they're not.
        # Throws error if any train column missing.
        # Drops any test column not found in train column list.
        # It's like saving a feature_order and doing X.loc[feature_order]
        if "matchvariables" not in patched_model.named_steps.keys():
            patched_model.steps.insert(
                0,
                (
                    "matchvariables",
                    MatchVariables(
                        missing_values="ignore" if are_nans_present else "raise"
                    ),
                ),
            )

        patched_model, result = training_utils.run_model_multiclass(
            model_name=model_name,
            model_clf=patched_model,
            X_train=train_featurized.X,
            X_test=test_featurized.X,
            y_train=train_featurized.y,
            y_test=test_featurized.y,
            train_groups=train_participant_labels,
            fold_id=fold_id,
            output_prefix=output_prefix,
            fold_label_train=fold_label_train,
            fold_label_test=fold_label_test,
            fail_on_error=fail_on_error,
        )

        results_from_run_single = [result]

        # For Glmnet models, also record performance with lambda_1se flag flipped.
        if (
            patched_model is not None
            and result is not None
            and training_utils.does_fitted_model_support_lambda_setting_change(
                patched_model
            )
        ):
            patched_model2, result2 = training_utils.modify_fitted_model_lambda_setting(
                fitted_clf=patched_model,
                performance=result,
                X_test=test_featurized.X,
                output_prefix=output_prefix,
            )
            results_from_run_single.append(result2)

        if (
            enable_training_sklearn_versions_of_glmnet
            and patched_model is not None
            and training_utils.does_fitted_model_support_conversion_from_glmnet_to_sklearn(
                patched_model
            )
        ):
            # Also train a sklearn model using the best lambda from the glmnet model, as an extra sanity check. Results should be identical.
            sklearn_model = training_utils.convert_trained_glmnet_pipeline_to_untrained_sklearn_pipeline_at_best_lambda(
                patched_model
            )
            sklearn_model, result_sklearn = training_utils.run_model_multiclass(
                model_name=model_name.replace("_cv", "") + "_sklearn_with_lambdamax",
                model_clf=sklearn_model,
                X_train=train_featurized.X,
                X_test=test_featurized.X,
                y_train=train_featurized.y,
                y_test=test_featurized.y,
                train_groups=train_participant_labels,
                fold_id=fold_id,
                output_prefix=output_prefix,
                fold_label_train=fold_label_train,
                fold_label_test=fold_label_test,
                fail_on_error=fail_on_error,
            )
            results_from_run_single.append(result_sklearn)

        # Save model settings to disk for each entry in results_from_run_single
        for result in results_from_run_single:
            if result is None:
                # Skip this entry:
                # run_model_multiclass can return None if fail_on_error is False and the model fails to train.
                continue
            model_settings = VJGeneSpecificSequenceModelRollupClassifier.make_model_settings(
                are_nans_present=are_nans_present,
                reweigh_by_subset_frequencies=train_featurized.extras[
                    "reweigh_by_subset_frequencies"
                ],
                standardization_transformer_before_reweighing_by_subset_frequencies=train_featurized.extras[
                    "standardization_transformer_before_reweighing_by_subset_frequencies"
                ],
                aggregation_strategy=aggregation_strategy,
            )
            joblib.dump(
                model_settings,
                f"{output_prefix}.{result.model_name}.{fold_id}.settings_joblib",
            )

        return results_from_run_single

    def convert_to_binary_ovr(
        inner_clf: Classifier,
    ) -> Classifier:
        # By default, our models are usually multinomial.
        # For this specific problem, we will train a set of binary one-vs-rest model_clf's to avoid feature leakage.
        # Each will have access to the subset of features related to its positive class only.

        # For example, the Covid-vs-rest model will have access to [average P(Covid) from V1-J1, average P(Covid) from V1-J2, ...],
        # but not to [average P(HIV) from V1-J1, average P(HIV) from V1-J2, ...].

        # The binary OvR classifiers will not have their output probabilities sum to 1. We have no calibration guarantees.
        # So we can't evaluate this model the same way as we do other models.
        # Confusion matrix/label-based scores are not going to be meaningful, because choosing the highest predicted class probability is no longer valid. (We will revisit this when we add multilabel classification.)
        # However, ROC AUC will still be a valid metric, because it only uses the ordering of a single class's predicted probabilities across examples.
        # Any metric is fine as long as it’s evaluated one model at a time (i.e. using one column of the n_examples x n_classes probabilities matrix).
        # Multiclass AUC is done that way: it’s a bunch of binary AUCs under the hood, using one column of that probabilities matrix at a time.
        # So AUC is still valid even if a hypothetical extreme case where the model for classA always gives predicted probabilities between 0 and 0.1 while the model for classB always gives predicted probabilities between 0.9 and 1.0.
        # (Inside crosseval, we are careful to use a custom ROC AUC implementation that doesn't require the original probabilities to sum to 1, which is a pesky requirement of sklearn's built-in version.)
        # Similarly, we could define a valid accuracy measure, but it would have to be evaluated for one model (one class) at a time.

        # The other advantage of this design is it dramatically reduce the number of features.
        # (Lasso does not have a unique solution if p > n: https://www.stat.cmu.edu/~ryantibs/papers/lassounique.pdf)

        # Note that the StandardScaler will run outside this, not in the inner submodel pipelines.
        # (Standardizing out front is no problem, because standardizing is column wise, and all rows are used in all submodels.)
        if isinstance(inner_clf, OneVsNegativeClassifier):
            # Special case: Detect OneVsNegativeClassifier base model.
            # In this case, don't wrap in BinaryOvRClassifierWithFeatureSubsettingByClass.
            # The double wrapper is broken, because the inner OneVsNegativeClassifier will receive class labels [0, 1] from the outer binary OvR wrapper. Therefore the special negative class label will be missing.
            # Instead of wrapping the OneVsNegativeClassifier, replace it with a OneVsNegativeClassifierWithFeatureSubsettingByClass, which combines the functionality of OneVsNegativeClassifier and BinaryOvRClassifierWithFeatureSubsettingByClass.
            # (We are not going to change the model name; this is a silent fix.)
            return OneVsNegativeClassifierWithFeatureSubsettingByClass(
                estimator=inner_clf.estimator,
                negative_class=inner_clf.negative_class,
                # Our feature names are e.g. "HIV_IGHV4-34_IGHJ3".
                feature_name_class_delimeter="_",
                n_jobs=n_jobs,
                # Allow some classes to fail to train, e.g. if they have too few samples and therefore internal cross validation for some binary OvR problems fails.
                allow_some_classes_to_fail_to_train=True,
            )

        # See model_definitions.py for other examples of these wrappers.
        # But we're defining this one only here, because it's specific to this one problem.
        return BinaryOvRClassifierWithFeatureSubsettingByClass(
            estimator=inner_clf,
            # Our feature names are e.g. "HIV_IGHV4-34_IGHJ3".
            feature_name_class_delimeter="_",
            n_jobs=n_jobs,
            # Allow some classes to fail to train, e.g. if they have too few samples and therefore internal cross validation for some binary OvR problems fails.
            allow_some_classes_to_fail_to_train=True,
        )

    for aggregation_strategy, featurized_data_collection in featurized_datas.items():
        models, train_participant_labels = training_utils.prepare_models_to_train(
            X_train=featurized_data_collection.train_featurized_with_nans.X,
            y_train=featurized_data_collection.train_featurized_with_nans.y,
            fold_id=fold_id,
            target_obs_column=target_obs_column,
            chosen_models=chosen_models,
            use_gpu=use_gpu,
            output_prefix=output_prefix,
            train_metadata=featurized_data_collection.train_featurized_with_nans.metadata,
            n_jobs=n_jobs,
        )
        for model_name, model_clf in models.items():
            # Run the model in binary OvR mode.
            results.extend(
                _run_single(
                    # e.g. "xgboost_mean_aggregated_as_binary_ovr"
                    model_name="".join(
                        [
                            model_name,
                            aggregation_strategy.model_name_suffix,
                            ModelNameSuffixes.binary_ovr.value,
                        ]
                    ),
                    model_clf=convert_to_binary_ovr(model_clf),
                    train_featurized=featurized_data_collection.train_featurized_fillna,
                    test_featurized=featurized_data_collection.test_featurized_fillna,
                    are_nans_present=False,
                    aggregation_strategy=aggregation_strategy,
                )
            )

            if isinstance(model_clf, xgb.XGBClassifier):
                # Xgboost supports missing values
                # Fit another copy (with slightly different name) that does not have the NaNs filled.
                # run_single will store the with-nans behavior in a settings object so that this choice properly affects future featurize() calls after the trained model is reloaded from disk.

                # Run the model in binary OvR mode.
                results.extend(
                    _run_single(
                        # e.g. "xgboost_mean_aggregated_as_binary_ovr_with_nans"
                        model_name="".join(
                            [
                                model_name,
                                aggregation_strategy.model_name_suffix,
                                ModelNameSuffixes.binary_ovr.value,
                                ModelNameSuffixes.with_nans.value,
                            ]
                        ),
                        model_clf=convert_to_binary_ovr(model_clf),
                        train_featurized=featurized_data_collection.train_featurized_with_nans,
                        test_featurized=featurized_data_collection.test_featurized_with_nans,
                        are_nans_present=True,
                        aggregation_strategy=aggregation_strategy,
                    )
                )

            # Alternative:
            # Add reweighed-by-subset-frequencies models. Use binary OvR design.
            results.extend(
                _run_single(
                    # e.g. "xgboost_mean_aggregated_as_binary_ovr_reweighed_by_subset_frequencies"
                    model_name="".join(
                        [
                            model_name,
                            aggregation_strategy.model_name_suffix,
                            ModelNameSuffixes.binary_ovr.value,
                            ModelNameSuffixes.reweighed_by_subset_frequencies.value,
                        ]
                    ),
                    model_clf=convert_to_binary_ovr(model_clf),
                    train_featurized=featurized_data_collection.train_featurized_reweighed_by_subset_frequencies,
                    test_featurized=featurized_data_collection.test_featurized_reweighed_by_subset_frequencies,
                    are_nans_present=False,  # NaNs were filled
                    aggregation_strategy=aggregation_strategy,
                )
            )

    if clear_cache:
        io.clear_cached_fold_embeddings()

    return results


def run_classify_with_all_models(
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    sample_weight_strategy: SampleWeightStrategy,
    fold_label_train: str,
    fold_label_test: str,
    # Which rollup models to train:
    chosen_models: List[str],
    # n_jobs used for internal parallelism, not for parallelizing over folds (because sequence model fit is very expensive)
    n_jobs: int,
    # Provide sequence model name here (rollup model will be trained on top of this model):
    base_model_name: str,
    base_model_fold_label_train: str,
    aggregation_strategies: List[AggregationStrategy],
    sequence_subset_strategy: SequenceSubsetStrategy = SequenceSubsetStrategy.split_Vgene_and_isotype,
    fold_ids: Optional[List[int]] = None,
    use_gpu=False,
    clear_cache=True,
    fail_on_error=False,
) -> ExperimentSet:
    """Train models"""
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

    with add_logging_context(base_model_name=base_model_name):
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
                base_model_name=base_model_name,
                base_model_fold_label_train=base_model_fold_label_train,
                aggregation_strategies=aggregation_strategies,
                sequence_subset_strategy=sequence_subset_strategy,
                use_gpu=use_gpu,
                clear_cache=clear_cache,
                fail_on_error=fail_on_error,
            )
            for fold_id in fold_ids
        ]

        return ExperimentSet(model_outputs=job_outputs)
