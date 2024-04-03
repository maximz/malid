import gc
import os
import time
import click
from typing import List

from malid import config, cli_utils, io
from malid.train import train_repertoire_stats_model
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)

import logging

logger = logging.getLogger(__name__)


def run_standard_train(
    gene_loci_used: GeneLocus,
    target_obs_columns: List[TargetObsColumnEnum],
    fold_ids: List[int],
    composite_sample_weight_strategy: SampleWeightStrategy,
    n_jobs=1,
):
    training_fold_name, testing_fold_name = config.get_fold_split_labels()
    logger.info(f"Training on {training_fold_name}, evaluating on {testing_fold_name}")

    GeneLocus.validate(gene_loci_used)  # packed (by OR) into single item here
    SampleWeightStrategy.validate(
        composite_sample_weight_strategy
    )  # packed (by OR) into single item here

    for gene_locus in gene_loci_used:  # One gene locus at a time
        # Control fold_id and cache manually so that we limit repetitive I/O
        for fold_id in fold_ids:
            for target_obs_col in target_obs_columns:
                GeneLocus.validate_single_value(gene_locus)
                TargetObsColumnEnum.validate(target_obs_col)

                try:
                    logger.info(
                        f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={composite_sample_weight_strategy}"
                    )
                    train_itime = time.time()

                    train_repertoire_stats_model.run_classify_with_all_models(
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_col,
                        sample_weight_strategy=composite_sample_weight_strategy,
                        fold_label_train=training_fold_name,
                        fold_label_test=testing_fold_name,
                        chosen_models=config.model_names_to_train,
                        n_jobs=n_jobs,
                        # control fold_id and cache manually so that we limit repetitive I/O
                        fold_ids=[fold_id],
                        clear_cache=False,
                    )

                    train_etime = time.time()
                    logger.info(
                        f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={composite_sample_weight_strategy} completed in {train_etime - train_itime}"
                    )
                except Exception as err:
                    logger.exception(
                        f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={composite_sample_weight_strategy} failed with error: {err}"
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
def run(
    gene_locus: List[GeneLocus],
    target_obs_column: List[TargetObsColumnEnum],
    fold_ids: List[int],
    sample_weight_strategy: List[SampleWeightStrategy],
    n_jobs: int,
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

    run_standard_train(
        gene_loci_used=gene_locus,
        target_obs_columns=target_obs_column,
        fold_ids=fold_ids,
        composite_sample_weight_strategy=sample_weight_strategy,
        n_jobs=n_jobs,  # Inner jobs to run in parallel, within a specific targetobscolumn-foldID combination.
    )


if __name__ == "__main__":
    run()
