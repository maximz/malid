"""
After fine-tuning the language model on each fold (aka separate embedder per fold), embed our data with the embedder.
We must run this _after_ cross-validation divisions since this is fold-specific

Or if using an off-the-shelf embedder, we can proceed to this step immediately, and the embeddings are not actually fold-specific though this script is structured that way.

This writes to config.paths.anndatas_dir — not scaled_anndatas_dir.
The next script will scale the anndatas produced by this script, and write them to scaled_anndatas_dir.

Usage examples:
    > python scripts/run_embedding.py --help;
    > python scripts/run_embedding.py;               # all folds, all loci
    > python scripts/run_embedding.py --fold_id 0;   # single fold
    > python scripts/run_embedding.py --fold_id 0 --fold_id 1;
    > python scripts/run_embedding.py --fold_id 0 --locus BCR; # fold 0 BCR only
"""

from typing import List
import time
import gc
import logging
import click

from malid import config, helpers, io, cli_utils
from malid.apply_embedding import load_embedding_model, run_embedding_model
from malid.datamodels import DataSource, GeneLocus
import choosegpu

logger = logging.getLogger(__name__)


def run_on_single_locus(
    gene_locus: GeneLocus,
    fold_ids: List[int],
    external_cohort: bool = False,
):
    itime = time.time()
    GeneLocus.validate_single_value(gene_locus)

    output_dir = config.paths.anndatas_dir / gene_locus.name
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Writing to: {output_dir}")

    for fold_id in fold_ids:
        if external_cohort and fold_id != -1:
            raise ValueError(
                "We only support external cohort embeddings for the global fold (fold -1)."
            )

        embedder = load_embedding_model(gene_locus=gene_locus, fold_id=fold_id)

        if external_cohort:
            # Special case
            fold_labels = ["external"]
        else:
            # Default
            fold_labels = ["train_smaller", "validation", "test"]

        for fold_label in fold_labels:
            if fold_id == -1 and fold_label == "test":
                # skip: global fold does not have a test set
                continue
            fname_out = output_dir / f"fold.{fold_id}.{fold_label}.h5ad"

            if external_cohort:
                # Special case: get external specimen labels
                external_specimens = helpers.get_all_specimen_info()
                # All are "peak timepoints'.
                # Some may be filtered out for now having enough sequences in the sample_sequences step. This is captured in the survived_filters column.
                # (is_selected_for_cv_strategy will be False for all, so in_training_set will be False for all)
                external_specimens = external_specimens[
                    (external_specimens["data_source"] == DataSource.external_cdna)
                    & (external_specimens["survived_filters"])
                ]
                df = io.load_raw_parquet_sequences_for_specimens(
                    specimen_labels=external_specimens["specimen_label"].tolist(),
                    gene_locus=gene_locus,
                )
            else:
                # Default
                df = io.load_raw_parquet_sequences_for_fold(
                    fold_id=fold_id, fold_label=fold_label, gene_locus=gene_locus
                )

            click.echo(
                f"Processing: fold {fold_id} {fold_label} (input = {df.shape[0]} sequences) -> {fname_out}"
            )

            # Make and save adata
            adata = run_embedding_model(embedder=embedder, df=df)
            adata.write(fname_out)

            click.echo(
                f"Wrote fold {fold_id} {fold_label}: {adata.shape[0]} sequences -> {fname_out}"
            )

            del adata
            del df
            gc.collect()

    etime = time.time()
    click.echo(f"Time elapsed: {etime - itime}")


@click.command()
@cli_utils.accepts_gene_loci
@cli_utils.accepts_fold_ids
@click.option(
    "--external-cohort",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optionally embed external cohort instead of main data.",
)
def run(
    gene_locus: List[GeneLocus],
    fold_ids: List[int],
    external_cohort: bool = False,
):
    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )
    click.echo(f"Selected gene_locus: {gene_locus}")
    click.echo(f"Selected fold_ids: {fold_ids}")
    click.echo(f"Selected external_cohort: {external_cohort}")
    GeneLocus.validate(gene_locus)  # packed (by OR) into single item here
    choosegpu.configure_gpu(enable=True)  # Embed with GPU
    for single_gene_locus in gene_locus:
        click.echo(f"Running on {single_gene_locus}...")
        run_on_single_locus(
            gene_locus=single_gene_locus,
            fold_ids=fold_ids,
            external_cohort=external_cohort,
        )


if __name__ == "__main__":
    run()
