"""
After creating embedder (scripts/run_embedding.py), scale and PCA the anndatas.
Already divided into train-smaller, validation, test.

This reads from config.paths.anndatas_dir and writes to config.paths.scaled_anndatas_dir.

Usage examples:
    > python scripts/scale_embedding_anndatas.py --help;
    > python scripts/scale_embedding_anndatas.py;               # all folds
    > python scripts/scale_embedding_anndatas.py --fold_id 0;   # single fold
    > python scripts/scale_embedding_anndatas.py --fold_id 0 --fold_id 1;
"""

from typing import List
import scanpy as sc
import gc
import joblib

import choosegpu

import genetools.scanpy_helpers
from malid import config, cli_utils, apply_embedding
from malid.datamodels import GeneLocus
from pathlib import Path
import anndata
import logging
import click

choosegpu.configure_gpu(enable=False)

logger = logging.getLogger(__name__)

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


def run_on_single_locus(
    gene_locus: GeneLocus, fold_ids: List[int], external_cohort: bool = False
):
    GeneLocus.validate_single_value(gene_locus)
    input_dir = config.paths.anndatas_dir / gene_locus.name
    output_dir = config.paths.scaled_anndatas_dir / gene_locus.name
    output_dir.mkdir(exist_ok=True, parents=True)
    click.echo(f"Writing to: {output_dir}")

    def fit_for_train_set(fold_id):
        # Fit scaling and PCA on train-smaller anndata and apply to validation and test anndata.
        # At this point these are all unscaled.

        # Run scale and PCA operations on CPU because matrices are too big for GPU.
        # Do this on one at a time to conserve RAM.

        click.echo(f"Processing: fold {fold_id}, train_smaller")

        click.echo("Loading unscaled anndata...")
        adata_train_smaller = load_original_anndata(
            fold_id=fold_id, fold_label="train_smaller", input_dir=input_dir
        )
        click.echo(
            f"Unscaled anndata has shape {adata_train_smaller.X.shape}, dtype {adata_train_smaller.X.dtype}"
        )
        # Sanity checks
        apply_embedding.verify_right_embedder_used(
            embedder=config.embedder,
            adata=adata_train_smaller,
            gene_locus=gene_locus,
            fold_id=fold_id,
        )

        # Scale inplace and set raw
        click.echo("Scaling...")
        (
            adata_train_smaller,
            scale_transformer,
        ) = genetools.scanpy_helpers.scale_anndata(
            adata_train_smaller, scale_transformer=None, inplace=True, set_raw=True
        )

        # PCA inplace
        click.echo("Running PCA...")
        adata_train_smaller, pca_transformer = genetools.scanpy_helpers.pca_anndata(
            adata_train_smaller,
            pca_transformer=None,
            n_components=n_components,
            inplace=True,
        )

        click.echo("Writing to disk...")
        write_part_to_disk(
            fold_id=fold_id,
            adata_part=adata_train_smaller,
            part_name="train_smaller",
            output_dir=output_dir,
        )
        del adata_train_smaller
        gc.collect()

        # Save the transformations so we can apply to future new test sets
        transformations_to_apply = {"scale": scale_transformer, "pca": pca_transformer}
        joblib.dump(
            transformations_to_apply,
            output_dir / f"fold.{fold_id}.train_smaller.transformations.joblib",
        )

        return transformations_to_apply

    for fold_id in fold_ids:
        if external_cohort and fold_id != -1:
            raise ValueError(
                "We only support external cohort embeddings for the global fold (fold -1)."
            )

        if external_cohort:
            # Special case
            transformations_to_apply = apply_embedding.load_transformations(
                gene_locus=gene_locus, fold_id=fold_id
            )
            fold_labels_dependent = ["external"]
        else:
            # Default
            transformations_to_apply = fit_for_train_set(fold_id)
            fold_labels_dependent = ["validation", "test"]

        # Apply those same transformations to validation and test sets:
        for fold_label in fold_labels_dependent:
            if fold_id == -1 and fold_label == "test":
                # skip: global fold does not have a test set
                continue
            click.echo(f"Processing: fold {fold_id}, {fold_label}")

            click.echo("Loading unscaled anndata...")
            adata_dependent = load_original_anndata(
                fold_id=fold_id, fold_label=fold_label, input_dir=input_dir
            )
            click.echo(
                f"Unscaled anndata has shape {adata_dependent.X.shape}, dtype {adata_dependent.X.dtype}"
            )

            # Scale and PCA inplace, avoiding copies
            adata_dependent = apply_embedding.transform_embedded_anndata(
                transformations_to_apply=transformations_to_apply,
                adata=adata_dependent,
            )

            click.echo("Writing to disk...")
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
@click.option(
    "--external-cohort",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optionally scale external cohort embedding instead of main data.",
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
    for single_gene_locus in gene_locus:
        click.echo(f"Running on {single_gene_locus}...")
        run_on_single_locus(
            gene_locus=single_gene_locus,
            fold_ids=fold_ids,
            external_cohort=external_cohort,
        )


if __name__ == "__main__":
    run()
