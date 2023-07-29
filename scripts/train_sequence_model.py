"""
Train sequence model.
Defaults to all folds and all TargetObsColumns, one at a time.

Usage examples:
    > python scripts/train_sequence_model.py;
    > python scripts/train_sequence_model.py --target_obs_column disease --target_obs_column disease_all_demographics_present;
    > python scripts/train_sequence_model.py --target_obs_column disease --fold_id 0 --fold_id 1;
    > python scripts/train_sequence_model.py --target_obs_column disease --fold_id 0;
"""

from typing import List, Optional
import gc

from malid import io
from malid import config, cli_utils
from malid.train import train_sequence_model
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
import click
import os
import time

import logging

logger = logging.getLogger(__name__)


def run_standard_train(
    gene_loci_used: GeneLocus,
    target_obs_columns: List[TargetObsColumnEnum],
    fold_ids: List[int],
    sample_weight_strategies: Optional[List[SampleWeightStrategy]] = None,
    n_jobs=1,
):
    """
    Train on training set (after a validation set has been split out of the training set), and evaluate on validation set
    These will become the inner classifiers for the meta-classifier.

    For each fold ID, for each targetobscolumn, for each sample weight strategy, run one train cycle.
    """

    if sample_weight_strategies is None:
        sample_weight_strategies = [SampleWeightStrategy.ISOTYPE_USAGE]
    training_fold_name, testing_fold_name = config.get_fold_split_labels()

    # map target_obs_column to model_output_prefix
    # Predict on "disease" (standard target) or some alternative target. This means we load an extra obs column, subset to specimens with that obs column defined, and restart from raw and rescale X.

    GeneLocus.validate(gene_loci_used)  # packed (by OR) into single item here
    for gene_locus in gene_loci_used:
        # Control fold_id and cache manually so that we limit repetitive I/O
        for fold_id in fold_ids:
            # Predict on "disease" (standard target) or some alternative target. This means we load an extra obs column, subset to specimens with that obs column defined, and restart from raw and rescale X.
            for target_obs_col in target_obs_columns:
                for sample_weight_strategy in sample_weight_strategies:
                    GeneLocus.validate_single_value(gene_locus)
                    TargetObsColumnEnum.validate(target_obs_col)
                    SampleWeightStrategy.validate(sample_weight_strategy)

                    try:
                        logger.info(
                            f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={sample_weight_strategy}"
                        )
                        train_itime = time.time()
                        # note that sequence data is already scaled automatically
                        train_sequence_model.run_classify_with_all_models(
                            gene_locus=gene_locus,
                            target_obs_column=target_obs_col,
                            sample_weight_strategy=sample_weight_strategy,
                            fold_label_train=training_fold_name,
                            fold_label_test=testing_fold_name,
                            chosen_models=[
                                "dummy_most_frequent",
                                "dummy_stratified",
                                "lasso_multiclass",
                            ],
                            n_jobs=n_jobs,
                            # control fold_id and cache manually so that we limit repetitive I/O
                            fold_ids=[fold_id],
                            clear_cache=False,
                        )
                        train_etime = time.time()
                        logger.info(
                            f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={sample_weight_strategy} completed in {train_etime - train_itime}"
                        )

                    except Exception as err:
                        logger.exception(
                            f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={sample_weight_strategy} failed with error: {err}"
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
    # input arguments are lists.

    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )

    click.echo(f"Selected gene_locus: {gene_locus}")
    click.echo(f"Selected target_obs_columns: {target_obs_column}")
    click.echo(f"Selected fold_ids: {fold_ids}")
    click.echo(f"Selected sample_weight_strategies: {sample_weight_strategy}")

    run_standard_train(
        gene_loci_used=gene_locus,
        target_obs_columns=target_obs_column,
        fold_ids=fold_ids,
        sample_weight_strategies=sample_weight_strategy,
        n_jobs=n_jobs,  # Inner jobs to run in parallel, within a specific targetobscolumn-foldID combination.
    )


if __name__ == "__main__":
    run()
