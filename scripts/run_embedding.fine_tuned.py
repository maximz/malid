"""
After fine-tuning the language model on each fold (aka separate embedder per fold), embed our data again.
We must run this _after_ cross-validation divisions since this is fold-specific

This writes to config.paths.anndatas_dir — not scaled_anndatas_dir.
The next notebook will scale the anndatas produced by this script, and write them to scaled_anndatas_dir.

Usage examples:
    > python scripts/run_embedding.fine_tuned.py --help;
    > python scripts/run_embedding.fine_tuned.py;               # all folds, all loci
    > python scripts/run_embedding.fine_tuned.py --fold_id 0;   # single fold
    > python scripts/run_embedding.fine_tuned.py --fold_id 0 --fold_id 1;
    > python scripts/run_embedding.fine_tuned.py --fold_id 0 --locus BCR; # fold 0 BCR only
"""

from typing import List
import time
import gc
import logging
import click

from malid import config, io, cli_utils
from malid.apply_embedding import load_embedding_model, run_embedding_model
from malid.datamodels import GeneLocus
import choosegpu

logger = logging.getLogger(__name__)


def run_on_single_locus(
    gene_locus: GeneLocus,
    fold_ids: List[int],
):
    itime = time.time()
    GeneLocus.validate_single_value(gene_locus)

    output_dir = config.paths.anndatas_dir / gene_locus.name
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Writing to: {output_dir}")

    for fold_id in fold_ids:
        embedder = load_embedding_model(gene_locus=gene_locus, fold_id=fold_id)
        for fold_label in ["train_smaller", "validation", "test"]:
            if fold_id == -1 and fold_label == "test":
                # skip: global fold does not have a test set
                continue
            df = io.load_raw_parquet_sequences_for_fold(
                fold_id=fold_id, fold_label=fold_label, gene_locus=gene_locus
            )

            fname_out = output_dir / f"fold.{fold_id}.{fold_label}.h5ad"
            logger.info(
                f"Processing: fold {fold_id} {fold_label} (input = {df.shape[0]} sequences) -> {fname_out}"
            )

            # Make and save adata
            adata = run_embedding_model(
                embedder=embedder, df=df, gene_locus=gene_locus, fold_id=fold_id
            )
            adata.write(fname_out)

            logger.info(
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
def run(
    gene_locus: List[GeneLocus],
    fold_ids: List[int],
):
    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )
    click.echo(f"Selected gene_locus: {gene_locus}")
    click.echo(f"Selected fold_ids: {fold_ids}")
    GeneLocus.validate(gene_locus)  # packed (by OR) into single item here
    choosegpu.configure_gpu(enable=True)  # Embed with GPU
    for single_gene_locus in gene_locus:
        click.echo(f"Running on {single_gene_locus}...")
        run_on_single_locus(gene_locus=single_gene_locus, fold_ids=fold_ids)


if __name__ == "__main__":
    run()
