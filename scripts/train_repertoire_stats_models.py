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
    n_jobs=1,
):
    training_fold_name, testing_fold_name = config.get_fold_split_labels()
    logger.info(f"Training on {training_fold_name}, evaluating on {testing_fold_name}")

    for gene_locus in gene_loci_used:
        # Control fold_id and cache manually so that we limit repetitive I/O
        for fold_id in fold_ids:
            for target_obs_col in target_obs_columns:
                try:
                    GeneLocus.validate_single_value(gene_locus)
                    TargetObsColumnEnum.validate(target_obs_col)
                    logger.info(
                        f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}"
                    )
                    train_itime = time.time()

                    train_repertoire_stats_model.run_classify_with_all_models(
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_col,
                        fold_label_train=training_fold_name,
                        fold_label_test=testing_fold_name,
                        chosen_models=[
                            "dummy_most_frequent",
                            "dummy_stratified",
                            "lasso_cv",
                            "ridge_cv",
                            "elasticnet_cv",
                            # non-CV lasso with a fixed default lambda:
                            "lasso_multiclass",
                            "xgboost",
                            "rf_multiclass",
                            "linearsvm_ovr",
                        ],
                        n_jobs=n_jobs,
                        # control fold_id and cache manually so that we limit repetitive I/O
                        fold_ids=[fold_id],
                        clear_cache=False,
                    )

                    train_etime = time.time()
                    logger.info(
                        f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus} completed in {train_etime - train_itime}"
                    )
                except Exception as err:
                    logger.exception(
                        f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus} failed with error: {err}"
                    )

            # manage fold ID and cache manually: now that we are done with this fold, clear cache
            io.clear_cached_fold_embeddings()
            gc.collect()


@click.command()
@cli_utils.accepts_gene_loci
@cli_utils.accepts_target_obs_columns
@cli_utils.accepts_fold_ids
@cli_utils.accepts_n_jobs
def run(
    gene_locus: List[GeneLocus],
    target_obs_column: List[TargetObsColumnEnum],
    fold_ids: List[int],
    n_jobs: int,
):
    # input arguments are lists.

    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )

    click.echo(f"Selected gene_locus: {gene_locus}")
    click.echo(f"Selected target_obs_columns: {target_obs_column}")
    click.echo(f"Selected fold_ids: {fold_ids}")

    run_standard_train(
        gene_loci_used=gene_locus,
        target_obs_columns=target_obs_column,
        fold_ids=fold_ids,
        n_jobs=n_jobs,  # Inner jobs to run in parallel, within a specific targetobscolumn-foldID combination.
    )


if __name__ == "__main__":
    run()
