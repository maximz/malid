"""
Train sequence model.
Defaults to all folds and all TargetObsColumns, one at a time.

Usage examples:
    > python scripts/train_vj_gene_specific_sequence_model_rollup.py;
    > python scripts/train_vj_gene_specific_sequence_model_rollup.py --target_obs_column disease --target_obs_column disease_all_demographics_present;
    > python scripts/train_vj_gene_specific_sequence_model_rollup.py --target_obs_column disease --fold_id 0 --fold_id 1;
    > python scripts/train_vj_gene_specific_sequence_model_rollup.py --target_obs_column disease --fold_id 0;
"""

from typing import List
import gc

from malid import io
from malid import config, cli_utils
from malid.train import train_vj_gene_specific_sequence_model_rollup
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
import click
import time
from malid.trained_model_wrappers.vj_gene_specific_sequence_model_rollup_classifier import (
    AggregationStrategy,
)

# Load base sequence model names from other script.
# (Separate rollup models will be trained on top of these base models.)
from train_vj_gene_specific_sequence_model import (
    resulting_trained_models as available_base_model_names,
    filter_sequence_level_model_names,
)

from enumchoice import EnumChoice
from malid.trained_model_wrappers.vj_gene_specific_sequence_classifier import (
    SequenceSubsetStrategy,
)

import logging

logger = logging.getLogger(__name__)


def run_standard_train(
    gene_loci_used: GeneLocus,
    target_obs_columns: List[TargetObsColumnEnum],
    fold_ids: List[int],
    composite_sample_weight_strategy: SampleWeightStrategy,
    base_model_names: List[str],
    aggregation_strategies: List[AggregationStrategy],
    sequence_subset_strategy: SequenceSubsetStrategy,
    n_jobs=1,
    use_preferred_base_model_only: bool = False,
):
    """
    For each fold ID, for each targetobscolumn, for each sample weight strategy, run one train cycle.
    """

    ## Model 3, separated by V and J genes, rollup classifier:
    # A VJGeneSpecificSequenceClassifier was trained on portion of train_smaller used to train base sequence model:
    base_model_training_fold_name = "train_smaller1"
    # Now we train a rollup model on top of that sequence model:
    training_fold_name = "train_smaller2"  # Train on portion of train_smaller used to train rollup model on top of sequence model
    testing_fold_name = "validation"  # Evaluate on validation set

    # map target_obs_column to model_output_prefix
    # Predict on "disease" (standard target) or some alternative target. This means we load an extra obs column, subset to specimens with that obs column defined, and restart from raw and rescale X.

    GeneLocus.validate(gene_loci_used)  # packed (by OR) into single item here
    SampleWeightStrategy.validate(
        composite_sample_weight_strategy
    )  # packed (by OR) into single item here

    for gene_locus in gene_loci_used:  # One gene locus at a time
        chosen_base_model_names = filter_sequence_level_model_names(
            sequence_model_names=base_model_names,
            gene_locus=gene_locus,
            preferred_only=use_preferred_base_model_only,
        )

        # Control fold_id and cache manually so that we limit repetitive I/O
        for fold_id in fold_ids:
            # Predict on "disease" (standard target) or some alternative target. This means we load an extra obs column, subset to specimens with that obs column defined, and restart from raw and rescale X.
            for target_obs_col in target_obs_columns:
                GeneLocus.validate_single_value(gene_locus)
                TargetObsColumnEnum.validate(target_obs_col)

                for base_model_name in chosen_base_model_names:
                    try:
                        logger.info(
                            f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={composite_sample_weight_strategy}, base_model_name={base_model_name}"
                        )
                        train_itime = time.time()
                        # note that sequence data is already scaled automatically
                        train_vj_gene_specific_sequence_model_rollup.run_classify_with_all_models(
                            gene_locus=gene_locus,
                            target_obs_column=target_obs_col,
                            sample_weight_strategy=composite_sample_weight_strategy,
                            fold_label_train=training_fold_name,
                            fold_label_test=testing_fold_name,
                            # Which rollup models to train:
                            # (We think elasticnet is justified for this aggregate-to-specimen-level task:
                            # the features have colinearity (correlated within a V gene, for example) and represent many weak signals (no sequence model is that great).
                            # Ridge helps them all "speak", and lasso helps with the collinearity.)
                            chosen_models=config.model_names_to_train,
                            n_jobs=n_jobs,
                            # Provide sequence model name here (rollup model will be trained on top of this model):
                            base_model_name=base_model_name,
                            base_model_fold_label_train=base_model_training_fold_name,
                            aggregation_strategies=aggregation_strategies,
                            sequence_subset_strategy=sequence_subset_strategy,
                            # Control fold_id and cache manually so that we limit repetitive I/O:
                            fold_ids=[fold_id],
                            clear_cache=False,
                        )
                        train_etime = time.time()
                        logger.info(
                            f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={composite_sample_weight_strategy}, base_model_name={base_model_name} completed in {train_etime - train_itime}"
                        )

                    except Exception as err:
                        logger.exception(
                            f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={composite_sample_weight_strategy}, base_model_name={base_model_name} failed with error: {err}"
                        )

            # manage fold ID and cache manually: now that we are done with this fold, clear cache
            io.clear_cached_fold_embeddings()
            gc.collect()


@click.command()
@cli_utils.accepts_gene_loci
@cli_utils.accepts_target_obs_columns
@cli_utils.accepts_fold_ids
@cli_utils.accepts_sample_weight_strategies
@cli_utils.accepts_n_jobs
@click.option(
    "--aggregation-strategy",
    "aggregation_strategies",
    multiple=True,
    default=list(AggregationStrategy),
    show_default=True,
    type=EnumChoice(AggregationStrategy, case_sensitive=False),
    help="Sequence-level prediction aggregation strategies used in rollup models.",
)
@click.option(
    "--sequence-subset-strategy",
    type=EnumChoice(SequenceSubsetStrategy, case_sensitive=False),
    default=config.metamodel_base_model_names.base_sequence_model_subset_strategy,
    show_default=True,
    help="Sequence splitting strategy used in trained sequence models.",
)
@click.option(
    "--use-preferred-base-model-only",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optionally use only the base sequence model name that is used downstream in the metamodel. By default, we will train against many base sequence models for performance comparison.",
)
def run(
    gene_locus: List[GeneLocus],
    target_obs_column: List[TargetObsColumnEnum],
    fold_ids: List[int],
    sample_weight_strategy: List[SampleWeightStrategy],
    n_jobs: int,
    aggregation_strategies: List[AggregationStrategy],
    sequence_subset_strategy: SequenceSubsetStrategy,
    use_preferred_base_model_only: bool = False,
):
    # Input arguments are lists. Combine them into multipacked flag values.
    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )
    sample_weight_strategy: SampleWeightStrategy = (
        SampleWeightStrategy.combine_flags_list_into_single_multiflag_value(
            sample_weight_strategy
        )
    )

    click.echo(f"Selected gene_locus: {gene_locus}")
    click.echo(f"Selected target_obs_columns: {target_obs_column}")
    click.echo(f"Selected fold_ids: {fold_ids}")
    click.echo(f"Selected sample_weight_strategies: {sample_weight_strategy}")
    click.echo(f"Available base_model_names: {available_base_model_names}")
    click.echo(
        f"use_preferred_base_model_only setting: {use_preferred_base_model_only}"
    )
    click.echo(f"Selected aggregation_strategies: {aggregation_strategies}")
    click.echo(f"Selected sequence_subset_strategy: {sequence_subset_strategy}")

    run_standard_train(
        gene_loci_used=gene_locus,
        target_obs_columns=target_obs_column,
        fold_ids=fold_ids,
        composite_sample_weight_strategy=sample_weight_strategy,
        base_model_names=available_base_model_names,
        aggregation_strategies=aggregation_strategies,
        sequence_subset_strategy=sequence_subset_strategy,
        n_jobs=n_jobs,  # Inner jobs to run in parallel, within a specific targetobscolumn-foldID combination.
        use_preferred_base_model_only=use_preferred_base_model_only,
    )


if __name__ == "__main__":
    run()
