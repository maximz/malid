import gc
import logging
from pathlib import Path
from typing import List, Tuple, Mapping, Union, Optional

import anndata
import joblib
import pandas as pd
from kdict import kdict

from malid import io
from malid.datamodels import (
    TargetObsColumnEnum,
    SampleWeightStrategy,
    GeneLocus,
)
from malid.external import model_evaluation
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from malid.trained_model_wrappers import (
    SequenceClassifier,
    RollupSequenceClassifier,
)

logger = logging.getLogger(__name__)


def _run_rollup_on_test_set(
    rollup_clf: Union[RollupSequenceClassifier, AdjustedProbabilitiesDerivedModel],
    test_set: anndata.AnnData,
    fold_id: int,
    fold_label_train: str,
    gene_locus: GeneLocus,
    target_obs_column: TargetObsColumnEnum,
    fold_label_test: Optional[str] = None,
    model_name_out: Optional[str] = None,  # defaults to model_name_sequence_disease
    export_clf_fname: Optional[Union[str, Path]] = None,
) -> model_evaluation.ModelSingleFoldPerformance:
    """
    Run rollup on test set:
    For each specimen in test_set, get rollup class probability vector (from sequence classifier only) and winning label.
    """
    # We cannot make this a normal instance method on RollupSequenceClassifier,
    # because rollup_clf can also be an AdjustedProbabilitiesDerivedModel,
    # and logic defined in the base class can't access the derived class's overloaded predict methods.
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    TargetObsColumnEnum.validate(target_obs_column)

    if model_name_out is None:
        model_name_out = rollup_clf.clf_sequence_disease.model_name_sequence_disease

    # Note that featurized.X will be an anndata - unconventional nature of RollupSequenceClassifier
    featurized = rollup_clf.featurize(test_set)
    specimen_rollup_probas: pd.DataFrame = rollup_clf.predict_proba(featurized.X)

    # Instead of calling rollup_clf.predict(), which re-executes predict_proba(),
    # just perform the idxmax ourselves.
    specimen_rollup_predicted_labels: pd.Series = specimen_rollup_probas.idxmax(axis=1)

    # Note that metadata in featurized may contain more specimens than survived the rollup process
    # (i.e. some may be abstained on)
    # so we filter below with .loc[specimen_order]

    # Arrange all returned objects in a consistent specimen order
    specimen_order = specimen_rollup_probas.index
    specimen_rollup_predicted_labels = specimen_rollup_predicted_labels.loc[
        specimen_order
    ]
    specimen_metadata = featurized.metadata.loc[specimen_order]

    return model_evaluation.ModelSingleFoldPerformance(
        model_name=model_name_out,
        fold_id=fold_id,
        y_true=specimen_metadata[target_obs_column.value.obs_column_name],
        fold_label_train=fold_label_train,
        fold_label_test=fold_label_test,
        y_pred=specimen_rollup_predicted_labels,
        class_names=rollup_clf.classes_,
        y_preds_proba=specimen_rollup_probas,
        test_metadata=specimen_metadata,
        export_clf_fname=export_clf_fname,
    )


def generate_rollups_on_all_classification_targets(
    fold_ids: List[int],
    targets: List[Tuple[TargetObsColumnEnum, SampleWeightStrategy]],
    gene_locus: GeneLocus,
    chosen_models: List[str],
    fold_label_train="train_smaller",  # What base models were trained on
    also_tune_decision_thresholds=True,
    fold_label_test="test",
    clear_cache=True,
    fail_on_error=False,
) -> Mapping[
    Tuple[TargetObsColumnEnum, SampleWeightStrategy],
    model_evaluation.ExperimentSet,
]:
    # Run each test set on each model before moving to next test set, so that we minimize how often data must be reloaded from disk.
    results: Mapping[
        Tuple[TargetObsColumnEnum, SampleWeightStrategy],
        model_evaluation.ExperimentSet,
    ] = kdict()  # indexed by (target_obs_col, sample_weight_strategy)

    # Control fold_id and cache manually (go fold by fold) so that we limit repetitive I/O
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    for fold_id in fold_ids:
        for target_obs_column, sample_weight_strategy in targets:
            TargetObsColumnEnum.validate(target_obs_column)
            SampleWeightStrategy.validate(sample_weight_strategy)

            # should already exist
            sequence_models_base_dir = SequenceClassifier._get_model_base_dir(
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
            )
            if not sequence_models_base_dir.exists():
                err = f"Sequence models base dir for gene_locus={gene_locus}, target={target_obs_column}, sample_weight_strategy={sample_weight_strategy} does not exist: {sequence_models_base_dir}"
                if fail_on_error:
                    raise ValueError(err)
                logger.warning(f"{err}. Skipping.")
                continue

            rollup_models_save_dir = RollupSequenceClassifier._get_model_base_dir(
                sequence_models_base_dir=sequence_models_base_dir
            )
            rollup_models_save_dir.mkdir(exist_ok=True)

            logger.info(
                f"Rolling up models for fold {fold_id} - gene_locus={gene_locus}, target={target_obs_column}, sample_weight_strategy={sample_weight_strategy} -> {rollup_models_save_dir}"
            )

            test_set = None
            if fold_id != -1:
                # Skip evaluation on global fold because it has no test set, and because we should only include cross-validation folds in the ExperimentSet
                # Loading here will leverage caching
                # Don't necessarily need to load full test set because we are going to be using sequence classifier on high quality test set only.
                test_set = io.load_fold_embeddings(
                    fold_id=fold_id,
                    fold_label=fold_label_test,
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_column,
                    sample_weight_strategy=sample_weight_strategy,
                )

            # Run the rollup
            # This gives results for each model, given one fold test set and one target_obs_column.
            # returns: (specimen_rollup_probas_all, specimen_rollup_predicted_labels_all, specimen_metadata_all, rollup_clfs_all);
            # each one is a dict indexed by model name
            if (target_obs_column, sample_weight_strategy) not in results:
                results[
                    target_obs_column, sample_weight_strategy
                ] = model_evaluation.ExperimentSet()

            if also_tune_decision_thresholds and fold_label_train != "train_smaller":
                raise ValueError(
                    f"If tuning decision thresholds, must use fold_label_train='train_smaller'"
                )

            for (
                model_name_to_load
            ) in chosen_models:  # sequence-classifier model name to load
                # Optionally repeat rollup with tuning the decision thresholds of the rollup model on validation set
                for tune_model_decision_thresholds_on_validation_set in (
                    [False, True] if also_tune_decision_thresholds else [False]
                ):
                    # Create output model name
                    model_name_out = (
                        f"{model_name_to_load}-rollup_tuned"
                        if tune_model_decision_thresholds_on_validation_set
                        else model_name_to_load
                    )

                    try:
                        # Rollup one model on one test set for one target_obs_column.
                        logger.info(
                            f"Rolling up {model_name_out}. Input model name {model_name_to_load}, tune_model_decision_thresholds_on_validation_set={tune_model_decision_thresholds_on_validation_set}, target_obs_column={target_obs_column}, sample_weight_strategy={sample_weight_strategy}."
                        )

                        rollup_clf = RollupSequenceClassifier(
                            fold_id=fold_id,
                            model_name_sequence_disease=model_name_to_load,
                            fold_label_train=fold_label_train,
                            gene_locus=gene_locus,
                            target_obs_column=target_obs_column,
                            sample_weight_strategy=sample_weight_strategy,
                            sequence_models_base_dir=sequence_models_base_dir,
                        )
                        if tune_model_decision_thresholds_on_validation_set:
                            rollup_clf = (
                                rollup_clf.tune_model_decision_thresholds_to_validation_set()
                            )

                        # Generate filename where we'll save rollup clf to disk
                        # This is useful when tune_model_decision_thresholds_on_validation_set is True
                        # Because we have just created a tuned wrapper that we can reuse later
                        fname_out = (
                            rollup_models_save_dir
                            / f"{fold_label_train}_model.{model_name_out}.{fold_id}.joblib"
                        )

                        # Save clf (or pipeline) to disk
                        try:
                            joblib.dump(rollup_clf, fname_out)
                            logger.info(
                                f"For fold {fold_id} - {target_obs_column} - wrote rollup_clf {model_name_out} -> {fname_out}"
                            )
                        except Exception as err:
                            fname_out = (
                                None  # don't pass it on to metadata object below
                            )
                            logger.error(
                                f"Error in saving rollup_clf {model_name_out} clf to disk for fold {fold_id}: {err}"
                            )

                        if fold_id != -1:
                            # skip evaluation on global fold because it has no test set, and because we should only include cross-validation folds in the ExperimentSet
                            rollup_output = _run_rollup_on_test_set(
                                rollup_clf=rollup_clf,
                                test_set=test_set,
                                fold_id=fold_id,
                                fold_label_train=fold_label_train,
                                gene_locus=gene_locus,
                                target_obs_column=target_obs_column,
                                fold_label_test=fold_label_test,
                                model_name_out=model_name_out,
                                export_clf_fname=fname_out,
                            )
                            try:
                                rollup_output.export(
                                    metadata_fname=rollup_models_save_dir
                                    / f"{fold_label_train}_model.{model_name_out}.{fold_id}.metadata_joblib"
                                )
                            except Exception as err:
                                logger.exception(
                                    f"Error in saving rollup_clf {model_name_out} clf to disk for fold {fold_id}: {err}"
                                )
                            results[target_obs_column, sample_weight_strategy].add(
                                rollup_output
                            )

                    except Exception as err:
                        if fail_on_error:
                            raise err
                        logger.error(f"Failed to rollup {model_name_out}: {err}")

        if clear_cache:
            # when done with all target_obs_columns for this fold,
            # manually clear cache before moving to next fold
            io.clear_cached_fold_embeddings()
            gc.collect()

    return results
