from __future__ import annotations
from typing_extensions import Self
from typing import List, Optional, Type, Union, ClassVar, TYPE_CHECKING
from pathlib import Path
import gc
import logging

import numpy as np
import pandas as pd
import choosegpu
from static_class_property import classproperty

from malid import config
from malid.datamodels import EmbedderSequenceContent, GeneLocus
from .base_embedder import (
    BaseEmbedder,
    BaseFineTunedEmbedder,
    run_in_batches,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biotransformers.wrappers.transformers_wrappers import TransformersWrapper

"""When adding a new embedder, see the instructions in embedders/__init__.py."""


class BiotransformerEmbedding(BaseEmbedder):
    """
    Embeddings with any BioTransformers supported backend.
    """

    # TODO: this is still downloading checkpoints to $HOME/.cache/torch/hub/checkpoints/, not to TRANSFORMERS_CACHE.

    # ClassVars: shared between all instances of a class. Doesn't vary from one instance to another. But can be overriden and changed in a subclass.
    _backend: ClassVar[str]
    _number_of_sequences_per_embedding_batch: ClassVar[int] = 900  # based on ProtBert

    def __init__(
        self, fold_id: Optional[int] = None, gene_locus: Optional[GeneLocus] = None
    ) -> None:
        # BaseEmbedder ignores the parameters. just including for consistency with BaseFineTunedEmbedder

        # Just in case, force defaulting to CPU if the user has not explicitly called configure_gpu(enable=True) before calling this method:
        choosegpu.ensure_gpu_settings_configured()

        # Import biotransformers here rather than at module import time,
        # because by the time we get to this point, the user may have properly called config.configure_gpu().
        # Otherwise the early import will by default cause using all GPUs.
        import torch
        from biotransformers import BioTransformers

        # load into GPU if available, otherwise CPU
        self.is_gpu = torch.cuda.is_available()

        self.model = BioTransformers(
            backend=self._backend, num_gpus=1 if self.is_gpu else 0
        )

    # Wrap in our own batching decorator to run mean-of-final-hidden-state operation for every batch,
    # rather than storing full embeddings in RAM until we finish all the batches and then taking means.
    #
    # For example, for Protbert:
    # - batch size of 1500 sequences works well with 20GB of GPU RAM, but fails with 12GB.
    # - batch size of 1000 sometimes works with 12GB of GPU RAM.
    # - we settled on a batch size of 900 that works reliably with 12GB of GPU RAM.
    #
    # Because this parameter may vary depending on the "backend" (e.g. Protbert vs. ESM), we make it a class variable,
    # and rather than hardcoding it in the decorator, we pass it in as a parameter.
    # To do so, we actually don't apply the decorator out front like normal:
    # @run_in_batches(num_sequences_per_batch=900)
    def embed(
        self, sequences: pd.DataFrame, dtype: Optional[Union[np.dtype, str]] = None
    ) -> np.ndarray:
        # Instead we apply the decorator inside, so we can programmatically pass in the batch size.
        wrapping_decorator = run_in_batches(
            num_sequences_per_batch=self._number_of_sequences_per_embedding_batch
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
            # note that special tokens (class/start-of-sequence, end-of-sequence, and padding) are automatically removed in compute_embeddings()
            embeddings = self.model.compute_embeddings(
                sequence_vectors,
                pool_mode=("full"),
                batch_size=self._number_of_sequences_per_embedding_batch,
            )["full"]

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
        self,
        sequences: Union[List[str], np.ndarray],
        batch_size: Optional[int] = None,
        silent: bool = True,
    ) -> float:
        """
        Calculate the average cross entropy loss for a set of sequences.
        Sequences do not need to be the same length.
        If silent is True (default), disable tqdm progress bar.
        """
        return self._calculate_cross_entropy_loss(
            model=self.model,
            sequences=sequences,
            batch_size=batch_size,
            silent=silent,
        )

    @classmethod
    def _calculate_cross_entropy_loss(
        cls,
        model: TransformersWrapper,
        sequences: Union[List[str], np.ndarray],
        batch_size: Optional[int] = None,
        silent: bool = True,
    ) -> float:
        """
        Calculate the average cross entropy loss for a set of sequences.
        Sequences do not need to be the same length.
        If silent is True (default), disable tqdm progress bar.
        """
        # Static method underpinning calculate_cross_entropy_loss()
        # Since model already loaded, we assume choosegpu.ensure_gpu_settings_configured() has already been run.
        if batch_size is None:
            # Set batch size to the default for this backend
            batch_size = cls._number_of_sequences_per_embedding_batch
        result = model.compute_cross_entropy_loss(
            sequences=sequences,
            batch_size=batch_size,
            silent=silent,
        )

        # Clear GPU memory
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        return result


class BiotransformerFineTunedEmbedding(BaseFineTunedEmbedder, BiotransformerEmbedding):
    """
    Fine-tuned embeddings with any BioTransformers supported backend.
    """

    _lr: ClassVar[float] = 1.0e-6  # learning rate
    _warmup_updates: ClassVar[int] = 1024
    _warmup_init_lr: ClassVar[float] = 1e-7

    def _load_fine_tuned_model(self: Self, fold_id: int, gene_locus: GeneLocus):
        self.model.load_model(
            config.paths.fine_tuned_embedding_dir
            / gene_locus.name
            / f"fold_{fold_id}"
            / "finetuned.ckpt"
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
        # Just in case, force defaulting to CPU if the user has not explicitly called configure_gpu(enable=True) before calling this method.
        # This will prevent accidentally using all GPUs. But will cause a crash because bio-trans.finetune requires a GPU.
        choosegpu.ensure_gpu_settings_configured()

        from biotransformers import BioTransformers
        from pytorch_lightning.callbacks import ModelCheckpoint

        # Start with original model
        bio_trans = BioTransformers(backend=cls._backend, num_gpus=1)

        # Calculate and save validation set loss before fine tuning.
        # Alternative considered: override on_train_start() in LightningModule to perform a validation step at the very beginning of the training process. It would be a one liner function: self.trainer.validate(self)
        loss_before_finetuning = cls._calculate_cross_entropy_loss(
            model=bio_trans,
            sequences=validation_sequences,
        )
        np.savetxt(
            Path(output_dir) / "validation_loss_before_finetuning.txt",
            np.atleast_1d(loss_before_finetuning),
            fmt="%g",
        )
        # Reload with np.loadtxt("filename.txt").item(), gives you the float back

        # Run fine-tuning
        # Returns params at final epoch - not necessarily params with best validation set loss.
        bio_trans.finetune(
            train_sequences=train_sequences,
            train_sequence_position_weights=train_sequence_position_weights,
            validation_sequences=validation_sequences,
            validation_sequence_position_weights=validation_sequence_position_weights,
            epochs=num_epochs,
            lr=cls._lr,  # learning rate
            warmup_updates=cls._warmup_updates,
            warmup_init_lr=cls._warmup_init_lr,
            # the docs suggested these overrides of the defaults, but we haven't tried them:
            # toks_per_batch=2000,
            # batch_size=16, # unrecognized parameter
            # acc_batch_size=256,
            accelerator="ddp",  # Consider switching to "gpu" since single GPU?
            amp_level=None,  # Error with default: "You have asked for `amp_level='O2'` but it's only supported with `amp_backend='apex'`."
            checkpoint=None,  # use the published weights as the starting point
            # How to save checkpoints:
            checkpoint_callbacks=[
                # Save best model according to validation loss. Seems to run mid-epoch and at end of epoch.
                ModelCheckpoint(
                    save_last=False,
                    save_top_k=1,
                    # Goal is to decrease validation loss:
                    mode="min",
                    monitor="val_loss",
                    every_n_epochs=emit_every_n_epochs,  # Run validation check at end of every N epochs
                    save_on_train_epoch_end=False,  # Run check at the end of validation phase, not at the end of train phase
                ),
            ],
            ## Training checks:
            log_every_n_steps=25,  # How often to log training loss: print after every N steps (not batches)
            ## Validation checks:
            check_val_every_n_epoch=emit_every_n_epochs,  # Run validation check at end of every N epochs
            val_check_interval=1 / 5,  # Check validation 5 times per epoch as well
            # val_check_interval=5000, # Set to check validation after every 5000 training batches (not steps - there are a lot more batches than steps) so that it matches how often we check train
            ## Output:
            logs_save_dir=output_dir,
            logs_name_exp="finetune_masked",  # default
        )

        del bio_trans


#####


class Esm2Embedder(BiotransformerEmbedding):
    """
    ESM-2 (30 layers, 150M parameters) embeddings.
    """

    # esm2_t36_3B_UR50D (36 layers, 3B parameters) is too big to fine tune with 20GB of GPU RAM,
    # but does work for a pretrained embedding.

    # esm2_t33_650M_UR50D (33 layers, 650M parameters) is also too big to fine tune with 20GB of GPU RAM.

    _backend: ClassVar[str] = "esm2_t30_150M_UR50D"
    name = "esm2"
    friendly_name = "CDR123 (ESM2 off the shelf)"
    dimensionality = 640


class Esm2FineTunedEmbedder(BiotransformerFineTunedEmbedding, Esm2Embedder):
    """
    ESM-2 (30 layers, 150M parameters) embeddings (fine tuned).
    """

    name = "esm2_fine_tuned"
    friendly_name = "CDR123 (ESM2)"

    @classproperty
    def non_fine_tuned_version(cls) -> Type[BaseEmbedder]:
        return Esm2Embedder


class Esm2EmbedderCDR3Only(Esm2Embedder):
    name = "esm2_cdr3"
    friendly_name = "CDR3 (ESM2 off the shelf)"

    # In practice, we saw that a batch size of 4000 sequences of average length 15 amino acids fits just under 10GB of GPU RAM with esm2_t30_150M_UR50D.
    _number_of_sequences_per_embedding_batch: ClassVar[int] = 4000

    @classproperty
    def embedder_sequence_content(cls) -> EmbedderSequenceContent:
        return EmbedderSequenceContent.CDR3


class Esm2FineTunedEmbedderCDR3Only(Esm2FineTunedEmbedder):
    name = "esm2_fine_tuned_cdr3"
    friendly_name = "CDR3 (ESM2)"

    @classproperty
    def non_fine_tuned_version(cls) -> Type[BaseEmbedder]:
        return Esm2EmbedderCDR3Only

    @classproperty
    def embedder_sequence_content(cls) -> EmbedderSequenceContent:
        return EmbedderSequenceContent.CDR3
