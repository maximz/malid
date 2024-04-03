from __future__ import annotations
from typing import List, Union, Optional

import logging

import numpy as np
import pandas as pd
import choosegpu
from static_class_property import classproperty

from malid.datamodels import EmbedderSequenceContent, GeneLocus
from .base_embedder import (
    BaseEmbedder,
    run_in_batches,
)

logger = logging.getLogger(__name__)

"""When adding a new embedder, see the instructions in embedders/__init__.py."""


class AbLangEmbeddder(BaseEmbedder):
    """
    AbLang (BCR only).
    Expects full VDJ sequences as input.
    Already trained on BCRs only; does not need fine tuning like the off-the-shelf general protein LMs.
    """

    name = "ablang"
    friendly_name = "Full VDJ (AbLang, off the shelf)"
    dimensionality = 768
    gene_loci_supported: GeneLocus = GeneLocus.BCR  # supports BCR only

    @classproperty
    def embedder_sequence_content(cls) -> EmbedderSequenceContent:
        return EmbedderSequenceContent.FR1ThruFR4

    def __init__(
        self, fold_id: Optional[int] = None, gene_locus: Optional[GeneLocus] = None
    ) -> None:
        # BaseEmbedder ignores the parameters. just including for consistency with BaseFineTunedEmbedder

        # Just in case, force defaulting to CPU if the user has not explicitly called configure_gpu(enable=True) before calling this method:
        choosegpu.ensure_gpu_settings_configured()

        # Import here rather than at module import time,
        # because by the time we get to this point, the user may have properly called config.configure_gpu().
        # Otherwise the early import will by default cause using all GPUs.
        import torch
        import ablang

        # load into GPU if available, otherwise CPU
        self.is_gpu = torch.cuda.is_available()
        self.model = ablang.pretrained(
            chain="heavy",
            device="cuda:0" if self.is_gpu else "cpu",
        )
        self.model.freeze()  # pytorch model.eval() - prepares for inference mode

    # Wrap in our own batching decorator to run mean-of-final-hidden-state operation for every batch,
    # rather than storing full embeddings in RAM until we finish all the batches and then taking means.
    # But don't decorate out here so we can customize depending on CPU or GPU mode:
    # @run_in_batches(num_sequences_per_batch=900)
    def embed(
        self, sequences: pd.DataFrame, dtype: Optional[Union[np.dtype, str]] = None
    ) -> np.ndarray:
        # Instead we apply the decorator inside, so we can programmatically pass in the batch size.
        wrapping_decorator = run_in_batches(
            num_sequences_per_batch=900 if self.is_gpu else 2000
        )

        def _embed(sequence_subset: pd.DataFrame) -> np.ndarray:
            (
                sequence_vectors,
                sequence_cut_start,
                sequence_cut_end,
            ) = self._get_sequences(
                df=sequence_subset,
                embedder_sequence_content=self.embedder_sequence_content,
            )

            # Get embedding of each sequence position
            # note that special tokens (class/start-of-sequence, end-of-sequence, and padding) are automatically removed
            # result is a list of numpy arrays, one for each sequence, each with shape (sequence_length, 768)
            embeddings: Union[List[np.ndarray], np.ndarray] = self.model(
                sequence_vectors, mode="rescoding"
            )
            if len(sequence_vectors) == 1:
                # Special case: single sequence --> np.ndarray
                # Wrap in a list to be consistent
                embeddings = [embeddings]

            # Return mean of a selected portion of each embedding
            pooled = self._partial_pooling(
                embeddings, sequence_cut_start, sequence_cut_end
            )
            if dtype is not None:
                pooled = pooled.astype(dtype)
            return pooled

        wrapped_function = wrapping_decorator(_embed)

        return wrapped_function(sequences)

    def calculate_cross_entropy_loss(
        self, sequences: List[str], batch_size: Optional[int] = None
    ) -> float:
        """
        Calculate the average cross entropy loss for a set of sequences. Sequences do not need to be the same length.
        """
        # TODO: Implement. Then remove the logic to skip this test in tests/test_embeddings.py.
        raise NotImplementedError(
            "We have not yet implemented cross entropy loss calculation for AbLang."
        )
