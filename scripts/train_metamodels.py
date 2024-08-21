"""
Train specimen-level ensemble metamodel using existing base models trained on train-smaller set.
"""

from typing import List, Optional
import gc

from malid import io, config, cli_utils
from malid.train import train_metamodel
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
)
import click
import time

import logging

logger = logging.getLogger(__name__)


def run_standard_train(
    gene_locus: GeneLocus,
    target_obs_columns: List[TargetObsColumnEnum],
    fold_ids: List[int],
    n_jobs=1,
    metamodel_flavors: Optional[List[str]] = None,
    base_model_train_fold_name="train_smaller",
    metamodel_fold_label_train="validation",
    metamodel_fold_label_test="test",
    # If not provided, these will be set to base_model_train_fold_name followed by suffix "1" and "2":
    base_model_train_fold_name_for_sequence_model: Optional[str] = None,
    base_model_train_fold_name_for_aggregation_model: Optional[str] = None,
    # Which model definitions to train (e.g. random forest):
    chosen_models: Optional[List[str]] = None,
):
    GeneLocus.validate(gene_locus)  # may be a single or a composite gene locus flag

    if chosen_models is None:
        chosen_models = config.model_names_to_train

    # Control fold_id and cache manually so that we limit repetitive I/O
    for fold_id in fold_ids:
        for target_obs_column in target_obs_columns:
            TargetObsColumnEnum.validate(target_obs_column)
            try:
                flavors = train_metamodel.get_metamodel_flavors(
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_column,
                    fold_id=fold_id,
                    base_model_train_fold_name=base_model_train_fold_name,
                    base_model_train_fold_name_for_sequence_model=base_model_train_fold_name_for_sequence_model,
                    base_model_train_fold_name_for_aggregation_model=base_model_train_fold_name_for_aggregation_model,
                )
            except Exception as err:
                logger.warning(
                    f"Failed to generate metamodel flavors for fold {fold_id}, {gene_locus}, {target_obs_column}: {err}"
                )
                continue
            for (
                metamodel_flavor,
                metamodel_config,
            ) in flavors.items():
                # Defines each train operation
                if (
                    metamodel_flavors is not None
                    and len(metamodel_flavors) > 0
                    and metamodel_flavor not in metamodel_flavors
                ):
                    # Not a whitelisted metamodel flavor. Skip.
                    continue
                try:
                    logger.info(
                        f"fold {fold_id}, {gene_locus}, target {target_obs_column}, metamodel flavor {metamodel_flavor}: {metamodel_config}"
                    )
                    train_itime = time.time()

                    train_metamodel.run_classify_with_all_models(
                        fold_id=fold_id,
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_column,
                        metamodel_flavor=metamodel_flavor,
                        metamodel_config=metamodel_config,
                        base_model_train_fold_name=base_model_train_fold_name,
                        metamodel_fold_label_train=metamodel_fold_label_train,
                        # Disable evaluation because global fold doesn't have a test set
                        metamodel_fold_label_test=(
                            metamodel_fold_label_test if fold_id != -1 else None
                        ),
                        chosen_models=chosen_models,
                        n_jobs=n_jobs,
                        # control fold_id and cache manually so that we limit repetitive I/O
                        clear_cache=False,
                    )

                    train_etime = time.time()
                    logger.info(
                        f"fold {fold_id} - gene_locus={gene_locus}, fold_id={fold_id}, target {target_obs_column}, metamodel flavor {metamodel_flavor}, {metamodel_config} completed in {train_etime - train_itime}"
                    )
                except Exception as err:
                    logger.exception(
                        f"fold {fold_id} - gene_locus={gene_locus}, fold_id={fold_id}, target {target_obs_column}, metamodel flavor {metamodel_flavor}, {metamodel_config} failed with error: {err}"
                    )

        # manage fold ID and cache manually: now that we are done with this fold, clear cache
        io.clear_cached_fold_embeddings()
        gc.collect()


@click.command()
@cli_utils.accepts_gene_loci
@cli_utils.accepts_target_obs_columns
@cli_utils.accepts_fold_ids
@cli_utils.accepts_n_jobs
@click.option(
    "--metamodel_flavor",
    "metamodel_flavors",
    multiple=True,
    default=None,
    show_default=True,
    type=str,
    help="Optional filter of metamodel flavors to train",
)
@click.option(
    "--base_model_train_fold_name",
    default="train_smaller",
    show_default=True,
    type=str,
)
@click.option(
    "--metamodel_fold_label_train",
    default="validation",
    show_default=True,
    type=str,
)
@click.option(
    "--metamodel_fold_label_test",
    default="test",
    show_default=True,
    type=str,
)
@click.option(
    "--base_model_train_fold_name_for_sequence_model",
    default=None,
    show_default=True,
    type=str,
)
@click.option(
    "--base_model_train_fold_name_for_aggregation_model",
    default=None,
    show_default=True,
    type=str,
)
def run(
    gene_locus: List[GeneLocus],
    target_obs_column: List[TargetObsColumnEnum],
    fold_ids: List[int],
    n_jobs: int,
    metamodel_flavors: Optional[List[str]],
    base_model_train_fold_name: str,
    metamodel_fold_label_train: str,
    metamodel_fold_label_test: str,
    # If not provided, these will be set to base_model_train_fold_name followed by suffix "1" and "2":
    base_model_train_fold_name_for_sequence_model: Optional[str] = None,
    base_model_train_fold_name_for_aggregation_model: Optional[str] = None,
):
    # input arguments are lists.

    # gene loci are packed (by OR) into a single composite flag here
    gene_loci_used: GeneLocus = (
        GeneLocus.combine_flags_list_into_single_multiflag_value(gene_locus)
    )

    click.echo(f"Selected gene_locus: {gene_loci_used}")
    click.echo(f"Selected target_obs_columns: {target_obs_column}")
    click.echo(f"Selected fold_ids: {fold_ids}")
    click.echo(f"Optional metamodel flavor filter: {metamodel_flavors}")
    click.echo(f"base_model_train_fold_name: {base_model_train_fold_name}")
    click.echo(f"metamodel_fold_label_train: {metamodel_fold_label_train}")
    click.echo(f"metamodel_fold_label_test: {metamodel_fold_label_test}")

    # Individual gene locus
    for single_gene_locus in gene_loci_used:
        print(single_gene_locus)
        GeneLocus.validate_single_value(single_gene_locus)
        run_standard_train(
            gene_locus=single_gene_locus,
            target_obs_columns=target_obs_column,
            fold_ids=fold_ids,
            n_jobs=n_jobs,  # Inner jobs to run in parallel, within a specific targetobscolumn-foldID combination.
            metamodel_flavors=metamodel_flavors,
            base_model_train_fold_name=base_model_train_fold_name,
            metamodel_fold_label_train=metamodel_fold_label_train,
            metamodel_fold_label_test=metamodel_fold_label_test,
            base_model_train_fold_name_for_sequence_model=base_model_train_fold_name_for_sequence_model,
            base_model_train_fold_name_for_aggregation_model=base_model_train_fold_name_for_aggregation_model,
        )

    # Together in combined metamodel
    if len(gene_loci_used) > 1:
        print(gene_loci_used)
        run_standard_train(
            gene_locus=gene_loci_used,
            target_obs_columns=target_obs_column,
            fold_ids=fold_ids,
            n_jobs=n_jobs,  # Inner jobs to run in parallel, within a specific targetobscolumn-foldID combination.
            metamodel_flavors=metamodel_flavors,
            base_model_train_fold_name=base_model_train_fold_name,
            metamodel_fold_label_train=metamodel_fold_label_train,
            metamodel_fold_label_test=metamodel_fold_label_test,
        )


if __name__ == "__main__":
    run()
