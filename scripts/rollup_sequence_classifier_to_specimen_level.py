"""
Roll up the sequence predictions to patient level, WITHOUT blending with healthy/sick model's predictions.
Using trimmed mean.
Defaults to all folds and all TargetObsColumns, one at a time.
"""

from typing import List, Optional, Mapping, Tuple
import gc
import os

from malid import io, config, cli_utils
from malid.external import model_evaluation
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
    combine_classification_option_names,
)
from malid.train.train_rollup_sequence_classifier import (
    generate_rollups_on_all_classification_targets,
)
import click
import time

import logging

logger = logging.getLogger(__name__)


def run_on_targets(
    gene_locus: GeneLocus,
    fold_ids: List[int],
    targets: List[Tuple[TargetObsColumnEnum, SampleWeightStrategy]],
    fold_label_train="train_smaller",
):
    GeneLocus.validate_single_value(gene_locus)

    # this is a kdict
    results: Mapping[
        Tuple[TargetObsColumnEnum, SampleWeightStrategy],
        model_evaluation.ExperimentSet,
    ] = generate_rollups_on_all_classification_targets(
        fold_ids=fold_ids,
        targets=targets,
        gene_locus=gene_locus,
        fold_label_test="test",
        chosen_models=[
            "lasso_multiclass",
            # Roll up the dummy sequence classifiers (this is not the same as a synthetic dummy classifier made from the patient label frequencies)
            "dummy_most_frequent",
            "dummy_stratified",
        ],
        fold_label_train=fold_label_train,
        also_tune_decision_thresholds=True,
    )

    # Analyze models across folds for each (target_obs_column, sample_weight_strategy)
    for (target_obs_column, sample_weight_strategy), experiment_set in results.items():
        output_dir = (
            config.paths.sequence_models_output_dir
            / gene_locus.name
            / "rollup_models"
            / combine_classification_option_names(
                target_obs_column=target_obs_column,
                sample_weight_strategy=sample_weight_strategy,
            )
        )  # might not yet exist
        output_dir.mkdir(exist_ok=True, parents=True)  # create if needed

        summary = experiment_set.summarize(
            global_evaluation_column_name=target_obs_column.value.confusion_matrix_expanded_column_name
        )
        summary.export_all_models(
            # Rollup of sequence predictions to patient level (without blending)
            func_generate_classification_report_fname=lambda model_name: output_dir
            / f"sequence_prediction_rollup.{model_name}.{fold_label_train}_model.report.txt",
            func_generate_confusion_matrix_fname=lambda model_name: output_dir
            / f"sequence_prediction_rollup.{model_name}.{fold_label_train}_model.confusion_matrix.png",
            confusion_matrix_pred_label="Rollup of sequence predictions",
            dpi=300,
        )

        combined_stats = summary.get_model_comparison_stats()
        combined_stats.to_csv(
            output_dir
            / f"sequence_prediction_rollup.{fold_label_train}_model.compare_model_scores.tsv",
            sep="\t",
        )

        logger.info(
            f"{gene_locus}, {target_obs_column}, {sample_weight_strategy} -> {output_dir}"
        )


@click.command()
@cli_utils.accepts_gene_loci
@cli_utils.accepts_target_obs_columns
@cli_utils.accepts_fold_ids
def run(
    gene_locus: List[GeneLocus],
    target_obs_column: List[TargetObsColumnEnum],
    fold_ids: List[int],
):
    # input arguments are lists.
    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )

    click.echo(f"Selected gene loci: {gene_locus}")
    click.echo(f"Selected target_obs_columns: {target_obs_column}")
    click.echo(f"Selected fold_ids: {fold_ids}")

    for single_gene_locus in gene_locus:
        run_on_targets(
            gene_locus=single_gene_locus,
            fold_ids=fold_ids,
            targets=[
                (single_target_obs_column, SampleWeightStrategy.ISOTYPE_USAGE)
                for single_target_obs_column in target_obs_column
            ],
        )


if __name__ == "__main__":
    run()
