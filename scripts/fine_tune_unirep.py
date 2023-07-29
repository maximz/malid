"""
Fine tune Unirep.
Defaults to all folds and embedding all gene loci.

Usage examples:
    > python scripts/fine_tune_unirep.py --help;
    > python scripts/fine_tune_unirep.py;
    > python scripts/fine_tune_unirep.py --locus BCR;
    > python scripts/fine_tune_unirep.py --fold_id 0 --fold_id 1;
    > python scripts/fine_tune_unirep.py --fold_id 0 --locus BCR --locus TCR;
"""

from typing import List
import numpy as np
import gc
import shutil
import logging
import click

from malid import config, io, cli_utils
from malid.apply_embedding import load_sequences_from_fold
from malid.datamodels import GeneLocus
import choosegpu

choosegpu.configure_gpu(enable=True)  # Configure GPU
from jax_unirep.evotuning import fit

logger = logging.getLogger(__name__)


learning_rate = 1e-5
batch_size = 100  # consumes 11GB of GPU RAM
backend = "gpu"  # or cpu. requires jax-GPU
epochs_per_print = 1  # also controls how often weights are dumped.


def get_and_sample_sequences(fold_id: int, gene_locus: GeneLocus):
    train_seqs = load_sequences_from_fold(
        fold_id=fold_id,
        fold_label="train_smaller",
        gene_locus=gene_locus,
    )
    validation_seqs = load_sequences_from_fold(
        fold_id=fold_id,
        fold_label="validation",
        gene_locus=gene_locus,
    )

    logger.info(
        f"Fold {fold_id}, before sampling: train_seqs.shape={train_seqs.shape}, validation_seqs.shape={validation_seqs.shape}"
    )

    rng = np.random.default_rng(0)

    # sample from train set once
    # TODO: sample from all patients and all isotypes
    num_train_to_choose = 500000
    train_seqs = rng.choice(train_seqs, size=num_train_to_choose, replace=False)
    # train_seqs.shape

    # sample from validation set once (keep consistent for all evaluations)
    # TODO: sample from all patients and all isotypes
    num_validation_to_choose = 20000
    validation_seqs = rng.choice(
        validation_seqs, size=num_validation_to_choose, replace=False
    )
    # validation_seqs.shape
    # train_seqs[0], validation_seqs[0]

    logger.info(
        f"Fold {fold_id}, after sampling: train_seqs.shape={train_seqs.shape}, validation_seqs.shape={validation_seqs.shape}"
    )

    # Free up RAM
    gc.collect()

    return train_seqs, validation_seqs


@click.command()
@cli_utils.accepts_fold_ids
@cli_utils.accepts_gene_loci
@click.option("--num_epochs", type=int, default=40)
def run(
    gene_locus: List[GeneLocus],
    fold_ids: List[int],
    num_epochs: int = 40,
):
    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )

    click.echo(f"Selected gene_locus: {gene_locus}")
    click.echo(f"Selected fold_ids: {fold_ids}")
    click.echo(f"Selected number of epochs: {num_epochs}")

    GeneLocus.validate(gene_locus)  # packed (by OR) into single item here
    for single_gene_locus in gene_locus:
        GeneLocus.validate_single_value(single_gene_locus)
        base_output_dir = config.paths.fine_tuned_embedding_dir / single_gene_locus.name

        for fold_id in fold_ids:
            output_dir = base_output_dir / f"fold_{fold_id}"

            # Clear out and remove folder if it already exists
            if output_dir.exists():
                if not output_dir.is_dir():
                    raise ValueError(
                        f"Output directory {output_dir} already xists but is not a directory."
                    )
                shutil.rmtree(output_dir)

            # Recreate folder
            output_dir.mkdir(parents=True, exist_ok=False)

            logger.info(
                f"Fine-tuning fold {fold_id}, gene_locus={gene_locus} -> {output_dir}"
            )

            train_seqs, validation_seqs = get_and_sample_sequences(
                fold_id=fold_id,
                gene_locus=single_gene_locus,
            )

            # Run fine-tuning
            # Returns params at final epoch - not necessarily params with best validation set loss.
            finetuned_params = fit(
                sequences=train_seqs,
                n_epochs=num_epochs,
                params=None,  # set to None if you want to use the published weights as the starting point.
                batch_method="random",
                batch_size=batch_size,
                step_size=learning_rate,
                holdout_seqs=validation_seqs,
                output_dir=output_dir,
                epochs_per_print=epochs_per_print,
                backend=backend,
            )


if __name__ == "__main__":
    run()
