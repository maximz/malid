"""
After embedding with a fine-tuned embedder (scripts/run_embedding.fine_tuned.py), scale and PCA the anndatas.
Already divided into train-smaller, validation, test.

This reads from config.paths.anndatas_dir and writes to config.paths.scaled_anndatas_dir.

Usage examples:
    > python scripts/scale_anndatas_created_with_finetuned_embedding.py --help;
    > python scripts/scale_anndatas_created_with_finetuned_embedding.py;               # all folds
    > python scripts/scale_anndatas_created_with_finetuned_embedding.py --fold_id 0;   # single fold
    > python scripts/scale_anndatas_created_with_finetuned_embedding.py --fold_id 0 --fold_id 1;
"""

from typing import List
import scanpy as sc
import gc
import joblib

import choosegpu

import malid.external.genetools_scanpy_helpers
from malid import config, cli_utils
from malid.datamodels import GeneLocus
from pathlib import Path
import anndata

choosegpu.configure_gpu(enable=False)
import logging

logger = logging.getLogger(__name__)
import click

n_components = 50  # Number of principal components to compute


def load_original_anndata(fold_id: int, fold_label: str, input_dir: Path):
    adata = sc.read(input_dir / f"fold.{fold_id}.{fold_label}.h5ad")
    # Sanity check:
    if adata.obs_names.duplicated().any():
        raise ValueError("Obs names duplicated")
    return adata


def write_part_to_disk(
    fold_id: int, adata_part: anndata.AnnData, part_name: str, output_dir: Path
):
    fname_out = output_dir / f"fold.{fold_id}.{part_name}.h5ad"
    adata_part.write(fname_out)
    print(fname_out, adata_part.shape)


def run_on_single_locus(gene_locus: GeneLocus, fold_ids: List[int]):
    GeneLocus.validate_single_value(gene_locus)
    input_dir = config.paths.anndatas_dir / gene_locus.name
    output_dir = config.paths.scaled_anndatas_dir / gene_locus.name
    output_dir.mkdir(exist_ok=True, parents=True)

    for fold_id in fold_ids:
        # Fit scaling and PCA on train-smaller anndata and apply to validation and test anndata.
        # At this point these are all unscaled.

        # Run scale and PCA operations on CPU because matrices are too big for GPU.
        # Do this on one at a time to conserve RAM.

        logger.info(f"Processing: fold {fold_id}, train_smaller")
        adata_train_smaller = load_original_anndata(
            fold_id=fold_id, fold_label="train_smaller", input_dir=input_dir
        )
        # Sanity checks:
        if adata_train_smaller.uns["embedded"] != "unirep_fine_tuned":
            raise ValueError(
                f"Expected anndata to be embedded with 'unirep_fine_tuned', got {adata_train_smaller.uns['embedded']}"
            )
        if adata_train_smaller.uns["embedded_fine_tuned_on_fold_id"] != fold_id:
            raise ValueError(
                f"Expected anndata to be embedded with fold_id {fold_id}, got {adata_train_smaller.uns['embedded_fine_tuned_on_fold_id']}"
            )

        # Scale inplace and set raw
        (
            adata_train_smaller,
            scale_transformer,
        ) = malid.external.genetools_scanpy_helpers.scale_anndata(
            adata_train_smaller, scale_transformer=None, inplace=True, set_raw=True
        )
        # PCA inplace
        (
            adata_train_smaller,
            pca_transformer,
        ) = malid.external.genetools_scanpy_helpers.pca_anndata(
            adata_train_smaller,
            pca_transformer=None,
            n_components=n_components,
            inplace=True,
        )

        write_part_to_disk(
            fold_id=fold_id,
            adata_part=adata_train_smaller,
            part_name="train_smaller",
            output_dir=output_dir,
        )
        del adata_train_smaller
        gc.collect()

        # Save the transformations so we can apply to future new test sets
        joblib.dump(
            {"scale": scale_transformer, "pca": pca_transformer},
            output_dir / f"fold.{fold_id}.train_smaller.transformations.joblib",
        )

        # Apply those same transformations to validation and test sets:
        for fold_label in ["validation", "test"]:
            if fold_id == -1 and fold_label == "test":
                # skip: global fold does not have a test set
                continue
            logger.info(f"Processing: fold {fold_id}, {fold_label}")
            adata_dependent = load_original_anndata(
                fold_id=fold_id, fold_label=fold_label, input_dir=input_dir
            )

            # Scale inplace - using existing transformer - and set raw
            adata_dependent, _ = malid.external.genetools_scanpy_helpers.scale_anndata(
                adata_dependent,
                scale_transformer=scale_transformer,
                inplace=True,
                set_raw=True,
            )
            # PCA inplace - no more copying - using existing transformer
            adata_dependent, _ = malid.external.genetools_scanpy_helpers.pca_anndata(
                adata_dependent,
                pca_transformer=pca_transformer,
                inplace=True,
            )

            write_part_to_disk(
                fold_id=fold_id,
                adata_part=adata_dependent,
                part_name=fold_label,
                output_dir=output_dir,
            )
            del adata_dependent
            gc.collect()


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
    for single_gene_locus in gene_locus:
        click.echo(f"Running on {single_gene_locus}...")
        run_on_single_locus(gene_locus=single_gene_locus, fold_ids=fold_ids)


if __name__ == "__main__":
    run()
