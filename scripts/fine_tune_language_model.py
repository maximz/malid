"""
Fine tune a language model embedding.
Defaults to all folds and embedding all gene loci.
Uses config.embedder choice of language model.

Usage examples:
    > python scripts/fine_tune_language_model.py --help;
    > python scripts/fine_tune_language_model.py;
    > python scripts/fine_tune_language_model.py --locus BCR;
    > python scripts/fine_tune_language_model.py --fold_id 0 --fold_id 1;
    > python scripts/fine_tune_language_model.py --fold_id 0 --locus BCR --locus TCR;
"""

from typing import List, Tuple, Optional, Type
import numpy as np
import gc
import shutil
import logging
import click
import choosegpu
import itertools

from malid import config, cli_utils
from malid.datamodels import GeneLocus
from malid.apply_embedding import load_sequence_embedding_content_for_fold
from malid.embedders.base_embedder import BaseFineTunedEmbedder

logger = logging.getLogger(__name__)


def get_and_sample_sequences(
    fold_id: int,
    gene_locus: GeneLocus,
    embedder_class: Type[BaseFineTunedEmbedder],
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray]]:
    train_seqs, validation_seqs = [
        load_sequence_embedding_content_for_fold(
            fold_id=fold_id,
            fold_label=fold_label,
            gene_locus=gene_locus,
            embedder_class=embedder_class,
        )
        for fold_label in ["train_smaller", "validation"]
    ]
    # Unpack
    train_seqs, train_seqs_cdr3_start, train_seqs_cdr3_end = train_seqs
    (
        validation_seqs,
        validation_seqs_cdr3_start,
        validation_seqs_cdr3_end,
    ) = validation_seqs

    logger.info(
        f"Fold {fold_id}, before sampling: train_seqs.shape={train_seqs.shape}, validation_seqs.shape={validation_seqs.shape}"
    )

    # We want to use the same subset of sequences for fine tuning anytime we fine tune a particular fold ID+gene locus,
    # so that we can compare different language models apples-to-apples, or just so we can rerun the fine tuning process without fear of changes.
    # So far we are relying on this random seed to do the job - and it seems to work.
    # (Later, consider saving out the chosen train and validation sequence IDs the first time fine tuning is run for a particular fold ID and gene locus (scoped to a data version but not to a particular embedder),
    # and then reload from disk for subsequent runs.)
    rng = np.random.default_rng(0)

    # sample from train set once
    # TODO: sample from all patients and all isotypes
    train_seq_idx = rng.choice(
        train_seqs.shape[0],
        size=min(embedder_class.num_train_to_choose, train_seqs.shape[0]),
        replace=False,
    )
    train_seqs = train_seqs[train_seq_idx]
    train_seqs_cdr3_start = (
        train_seqs_cdr3_start[train_seq_idx]
        if train_seqs_cdr3_start is not None
        else None
    )
    train_seqs_cdr3_end = (
        train_seqs_cdr3_end[train_seq_idx] if train_seqs_cdr3_end is not None else None
    )
    train_seq_weights = embedder_class._make_weights_for_sequence_positions(
        train_seqs,
        train_seqs_cdr3_start,
        train_seqs_cdr3_end,
        embedder_sequence_content=embedder_class.embedder_sequence_content,
    )

    # sample from validation set once (keep consistent for all evaluations)
    # TODO: sample from all patients and all isotypes
    validation_seq_idx = rng.choice(
        validation_seqs.shape[0],
        size=min(embedder_class.num_validation_to_choose, validation_seqs.shape[0]),
        replace=False,
    )
    validation_seqs = validation_seqs[validation_seq_idx]
    validation_seqs_cdr3_start = (
        validation_seqs_cdr3_start[validation_seq_idx]
        if validation_seqs_cdr3_start is not None
        else None
    )
    validation_seqs_cdr3_end = (
        validation_seqs_cdr3_end[validation_seq_idx]
        if validation_seqs_cdr3_end is not None
        else None
    )
    validation_seq_weights = embedder_class._make_weights_for_sequence_positions(
        validation_seqs,
        validation_seqs_cdr3_start,
        validation_seqs_cdr3_end,
        embedder_sequence_content=embedder_class.embedder_sequence_content,
    )

    logger.info(
        f"Fold {fold_id}, after sampling: train_seqs.shape={train_seqs.shape}, validation_seqs.shape={validation_seqs.shape}"
    )

    # Free up RAM
    gc.collect()

    return train_seqs, train_seq_weights, validation_seqs, validation_seq_weights


@click.command()
@cli_utils.accepts_fold_ids
@cli_utils.accepts_gene_loci
@click.option(
    "--num_epochs",
    type=int,
    default=None,
    help="Number of epochs to train for. Defaults to a specific value for each type of config.embedder.",
)
def run(
    gene_locus: List[GeneLocus],
    fold_ids: List[int],
    num_epochs: Optional[int] = None,
    emit_every_n_epochs: int = 1,  # controls how often weights are dumped
):
    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )
    GeneLocus.validate(gene_locus)  # packed (by OR) into single item here

    click.echo(f"Selected gene_locus: {gene_locus}")
    click.echo(f"Selected fold_ids: {fold_ids}")
    click.echo(f"Selected embedder class: {config.embedder}")

    choosegpu.configure_gpu(enable=True)  # Configure GPU

    from malid.embedders.base_embedder import BaseFineTunedEmbedder
    from malid.embedders.biotransformers import ProtBertFineTunedEmbedder

    if not issubclass(config.embedder, BaseFineTunedEmbedder):
        raise ValueError(
            f"config.embedder must be a subclass of BaseFineTunedEmbedder, but is {config.embedder}."
        )

    if num_epochs is None:
        # Set default
        if issubclass(config.embedder, ProtBertFineTunedEmbedder):
            # Special case for ProtBert: we know we don't need to train as long
            num_epochs = 12
        else:
            num_epochs = 25

    click.echo(f"Selected number of epochs: {num_epochs}")

    for single_gene_locus in gene_locus:
        GeneLocus.validate_single_value(single_gene_locus)
        base_output_dir = config.paths.fine_tuned_embedding_dir / single_gene_locus.name

        for fold_id in fold_ids:
            output_dir = base_output_dir / f"fold_{fold_id}"

            # Clear out and remove folder if it already exists
            if output_dir.exists():
                if not output_dir.is_dir():
                    raise ValueError(
                        f"Output directory {output_dir} already exists but is not a directory."
                    )
                shutil.rmtree(output_dir)

            # Recreate folder
            output_dir.mkdir(parents=True, exist_ok=False)

            logger.info(
                f"Fine-tuning fold {fold_id}, gene_locus={gene_locus} -> {output_dir}"
            )

            (
                train_sequences,
                train_sequence_position_weights,
                validation_sequences,
                validation_sequence_position_weights,
            ) = get_and_sample_sequences(
                fold_id=fold_id,
                gene_locus=single_gene_locus,
                embedder_class=config.embedder,
            )

            # Run fine-tuning
            config.embedder.finetune(
                train_sequences=train_sequences,
                train_sequence_position_weights=train_sequence_position_weights,
                validation_sequences=validation_sequences,
                validation_sequence_position_weights=validation_sequence_position_weights,
                num_epochs=num_epochs,
                output_dir=output_dir,
                emit_every_n_epochs=emit_every_n_epochs,
            )


if __name__ == "__main__":
    run()
