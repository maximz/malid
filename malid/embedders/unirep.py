from __future__ import annotations
from typing_extensions import Self
import logging
from typing import List, Optional, Tuple, Type, Union
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from malid import config
from malid.datamodels import EmbedderSequenceContent, GeneLocus
import choosegpu
from static_class_property import classproperty
from .base_embedder import (
    BaseEmbedder,
    BaseFineTunedEmbedder,
    run_in_batches,
)
from collections import defaultdict

logger = logging.getLogger(__name__)


class UnirepEmbedder(BaseEmbedder):
    """
    UniRep embeddings.
    """

    name = "unirep"
    friendly_name = "CDR123 (UniRep off the shelf)"
    dimensionality = 1900

    def __init__(
        self, fold_id: Optional[int] = None, gene_locus: Optional[GeneLocus] = None
    ) -> None:
        # BaseEmbedder ignores the parameters. just including for consistency with BaseFineTunedEmbedder

        # Import jax_unirep here rather than at module import time,
        # because by the time we get to this point, the user may have properly called config.configure_gpu().
        # Otherwise jax will be imported and by default will use all GPUs.

        # Just in case, force defaulting to CPU if the user has not explicitly called configure_gpu(enable=True) before calling this method:
        choosegpu.ensure_gpu_settings_configured()

        self.params = self._get_pretrained_params()

    @staticmethod
    def _get_pretrained_params() -> Tuple:
        """
        Get default pretrained UniRep parameters.
        Ensure GPU settings are configured before calling this function.
        """
        from jax_unirep.utils import load_params

        return load_params()

    # Run in batches since GPU memory is limited to ~12GB.
    # TODO: Make jax-unirep accept a batch size parameter used within its batches-by-sequence-length.
    @run_in_batches(num_sequences_per_batch=10000)
    def embed(
        self, sequences: pd.DataFrame, dtype: Optional[Union[np.dtype, str]] = None
    ) -> np.ndarray:
        from jax_unirep import get_reps

        sequence_vectors, sequence_cut_start, sequence_cut_end = self._get_sequences(
            df=sequences,
            embedder_sequence_content=self.embedder_sequence_content,
        )
        if sequence_cut_start is not None or sequence_cut_end is not None:
            # TODO: implement use of sequence_cut_start and sequence_cut_end by returning intermediate hidden states from jax-unirep
            raise ValueError(
                f"{self.embedder_sequence_content} not supported by UniRep"
            )
        h_avg, h_final, c_final = get_reps(sequence_vectors, params=self.params[1])
        if dtype is not None:
            h_avg = h_avg.astype(dtype)
        return h_avg

    def calculate_cross_entropy_loss(
        self, sequences: Union[List[str], np.ndarray], batch_size: Optional[int] = None
    ) -> float:
        """
        Calculate the average cross entropy loss for a set of sequences. Sequences do not need to be the same length.

        This function calculates the cross-entropy loss for a batch of protein sequences of arbitrary length,
        by batching togethger all sequences of the same length and passing them through the mLSTM.
        We want to calculate a single average for the sequences, so we take the weighted average of the
        per-length losses (weighted by number of sequences in a given length)

        Note: batch_size parameter is ignored.
        """
        return self._calculate_cross_entropy_loss(sequences, self.params)

    @classmethod
    def _calculate_cross_entropy_loss(
        cls, sequences: Union[List[str], np.ndarray], params: Tuple
    ) -> float:
        """
        Calculate the average cross entropy loss for a set of sequences. Sequences do not need to be the same length.

        This function calculates the cross-entropy loss for a batch of protein sequences of arbitrary length,
        by batching togethger all sequences of the same length and passing them through the mLSTM.
        We want to calculate a single average for the sequences, so we take the weighted average of the
        per-length losses (weighted by number of sequences in a given length)

        Ensure GPU settings are configured before calling this function.
        """
        from jax_unirep.evotuning_models import mlstm1900_apply_fun
        from jax_unirep.utils import input_output_pairs
        from jax_unirep.evotuning import avg_loss

        def _calculate_for_equal_length_sequences(
            sequences_equal_length: Union[List[str], np.ndarray]
        ) -> float:
            x, y = input_output_pairs(sequences_equal_length)
            return float(avg_loss([x], [y], params, mlstm1900_apply_fun))

        # Compute weighted average loss
        length_indexed_sequences = defaultdict(list)
        for seq in sequences:
            length_indexed_sequences[len(seq)].append(seq)
        losses = [
            # From each group, we return: [len(sequence_subset[0]), len(sequence_subset), loss]
            [
                len(length_indexed_sequences[i][0]),
                len(length_indexed_sequences[i]),
                _calculate_for_equal_length_sequences(length_indexed_sequences[i]),
            ]
            for i in sorted(length_indexed_sequences.keys())
        ]
        summary = pd.DataFrame(losses, columns=["length", "n", "avg_loss"])
        weighted_avg_loss = sum(summary["avg_loss"] * summary["n"]) / sum(summary["n"])
        return weighted_avg_loss


class UnirepFineTunedEmbedder(BaseFineTunedEmbedder, UnirepEmbedder):
    name = "unirep_fine_tuned"
    friendly_name = "CDR123 (UniRep)"

    @classproperty
    def non_fine_tuned_version(cls) -> Type[BaseEmbedder]:
        return UnirepEmbedder

    def _load_fine_tuned_model(self: Self, fold_id: int, gene_locus: GeneLocus):
        # load best params
        self.params: Tuple = joblib.load(
            config.paths.fine_tuned_embedding_dir
            / gene_locus.name
            / f"fold_{fold_id}"
            / "best_params.joblib"
        )

    @classmethod
    def finetune(
        cls,
        train_sequences: np.ndarray,
        train_sequence_position_weights: List[np.ndarray],
        validation_sequences: np.ndarray,
        validation_sequence_position_weights: List[np.ndarray],
        num_epochs: int,
        output_dir: Union[Path, str],
        emit_every_n_epochs: int,
    ) -> None:
        """
        Fine tune.
        GPU required. Call choosegpu.configure_gpu(enable=True) first before running this method (which will cause neural net library import).
        """

        # Sanity check
        for position_weights in (
            train_sequence_position_weights,
            validation_sequence_position_weights,
        ):
            for seq_weight_array in position_weights:
                if not all(seq_weight_array == 1):
                    raise ValueError(
                        "Passed-in weights were not 1 at all sequence positions, but our UniRep implementation does not support reweighting certain sequence positions in the fine-tuning loss function."
                    )

        # Just in case, force defaulting to CPU if the user has not explicitly called configure_gpu(enable=True) before calling this method.
        # This will prevent accidentally using all GPUs.
        choosegpu.ensure_gpu_settings_configured()

        # Calculate and save validation set loss before fine tuning.
        loss_before_finetuning = cls._calculate_cross_entropy_loss(
            sequences=validation_sequences,
            params=cls._get_pretrained_params(),
        )
        np.savetxt(
            Path(output_dir) / "validation_loss_before_finetuning.txt",
            np.atleast_1d(loss_before_finetuning),
            fmt="%g",
        )
        # Reload with np.loadtxt("filename.txt").item(), gives you the float back

        from jax_unirep.evotuning import fit

        # Run fine-tuning
        # Returns params at final epoch - not necessarily params with best validation set loss.
        finetuned_params = fit(
            sequences=train_sequences,
            n_epochs=num_epochs,
            params=None,  # set to None if you want to use the published weights as the starting point.
            batch_method="random",
            batch_size=100,  # consumes 11GB of GPU RAM
            step_size=1e-5,  # learning rate
            holdout_seqs=validation_sequences,
            output_dir=output_dir,
            epochs_per_print=emit_every_n_epochs,
            backend="gpu",  # or cpu. requires jax-GPU
        )
        del finetuned_params


class UnirepEmbedderCDR3Only(UnirepEmbedder):
    name = "unirep_cdr3"
    friendly_name = "CDR3 (UniRep off the shelf)"

    @classproperty
    def embedder_sequence_content(cls) -> EmbedderSequenceContent:
        return EmbedderSequenceContent.CDR3


class UnirepFineTunedEmbedderCDR3Only(UnirepFineTunedEmbedder):
    name = "unirep_fine_tuned_cdr3"
    friendly_name = "CDR3 (UniRep)"

    @classproperty
    def non_fine_tuned_version(cls) -> Type[BaseEmbedder]:
        return UnirepEmbedderCDR3Only

    @classproperty
    def embedder_sequence_content(cls) -> EmbedderSequenceContent:
        return EmbedderSequenceContent.CDR3
