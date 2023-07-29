"""
Run fine-tuned language model (different for each fold) on off-peak data, and apply existing scaling and PCA transformations.

Recall that we have a separate fine-tuned language model for each train-smaller set.
So treat this as an extension of the test set.
Get each participant's corresponding test fold ID, and apply the language model, scaling, and PCA transformations trained on that fold's train-smaller set.

This writes to config.paths.scaled_anndatas_dir.

Usage examples:
    > python scripts/off_peak.run_embedding_fine_tuned.and_scale.py --help;
    > python scripts/off_peak.run_embedding_fine_tuned.and_scale.py;               # all folds, all loci
    > python scripts/off_peak.run_embedding_fine_tuned.and_scale.py --fold_id 0;   # single fold
    > python scripts/off_peak.run_embedding_fine_tuned.and_scale.py --fold_id 0 --fold_id 1;
    > python scripts/off_peak.run_embedding_fine_tuned.and_scale.py --fold_id 0 --locus BCR; # fold 0 BCR only
"""

from typing import List
import time
import gc
import pandas as pd

import choosegpu
from malid import config, io, cli_utils, helpers, apply_embedding
import logging

from malid.datamodels import GeneLocus

logger = logging.getLogger(__name__)
import click


def find_offpeak_samples():
    # load data - figure out which embedder fold ID to use for each participant
    specimen_metadata = helpers.get_all_specimen_info(add_cv_fold_information=False)
    specimen_metadata = specimen_metadata[specimen_metadata["cohort"] == "Boydlab"]

    # Choose all samples that were not included in the peak-timepoints-only training set,
    # but are valid specimens (i.e. survived sanity filters)
    specimens_filtered = specimen_metadata[
        (specimen_metadata["survived_filters"]) & (~specimen_metadata["is_peak"])
    ]

    # Merge in test fold ID for each participant (based on where their peak timepoint specimens were placed)
    participants_in_test_folds = helpers.get_test_fold_id_for_each_participant()
    specimens_filtered = pd.merge(
        specimens_filtered,
        participants_in_test_folds,
        how="left",
        left_on="participant_label",
        right_index=True,
        validate="m:1",
    )

    # N/As mean this patient had no peak timepoint included in any training sets
    # Assign them to test fold -1 to use the global model trained on all data
    specimens_filtered["test_fold_id"].fillna(-1, inplace=True)
    if specimens_filtered["test_fold_id"].isna().any():
        raise ValueError("Test fold ID was not set for some specimens")

    # Convert to int so we display fold IDs as expected
    specimens_filtered["test_fold_id"] = specimens_filtered["test_fold_id"].astype(int)

    # specimens_filtered["test_fold_id"].value_counts()
    # specimens_filtered[
    #     ["participant_label", "disease_subtype", "test_fold_id"]
    # ].drop_duplicates()

    return specimens_filtered


def run_on_single_locus(
    gene_locus: GeneLocus,
    fold_ids: List[int],
):
    GeneLocus.validate_single_value(gene_locus)

    output_dir = (
        config.paths.scaled_anndatas_dir / gene_locus.name
    )  # should already exist
    click.echo(f"Writing to: {output_dir}")

    offpeak_specimens = find_offpeak_samples()

    itime = time.time()
    for fold_id in fold_ids:
        offpeak_specimens_in_fold = offpeak_specimens[
            offpeak_specimens["test_fold_id"] == fold_id
        ]["specimen_label"].values
        logger.info(
            f"Fold {fold_id} has {len(offpeak_specimens_in_fold)} off-peak specimens: {offpeak_specimens_in_fold}"
        )

        df = io.load_raw_parquet_sequences_for_specimens(
            specimen_labels=offpeak_specimens_in_fold, gene_locus=gene_locus
        )

        if df.shape[0] == 0:
            logger.warning(f"Fold {fold_id} had no offpeak sequences. Skipping.")
            continue

        fname_out = output_dir / f"off_peak_timepoints.fold.{fold_id}.h5ad"
        logger.info(
            f"Processing: fold {fold_id} (input = {df.shape[0]} sequences) -> {fname_out}"
        )

        # Make adata
        adata = apply_embedding.run_embedding_model(
            embedder=apply_embedding.load_embedding_model(
                gene_locus=gene_locus, fold_id=fold_id
            ),
            df=df,
            gene_locus=gene_locus,
            fold_id=fold_id,
        )
        adata = apply_embedding.transform_embedded_anndata(
            transformations_to_apply=apply_embedding.load_transformations(
                gene_locus=gene_locus, fold_id=fold_id
            ),
            adata=adata,
        )

        # Save
        adata.write(fname_out)
        logger.info(
            f"Wrote fold {fold_id}-offpeak: {adata.shape[0]} sequences, with {adata.obs['specimen_label'].nunique()} specimens -> {fname_out}"
        )

        # Garbage collect
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
