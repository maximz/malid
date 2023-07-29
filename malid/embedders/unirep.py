import logging

import joblib
import numpy as np

from malid import config
from malid.datamodels import GeneLocus
import choosegpu
from .base_embedder import BaseEmbedder, BaseFineTunedEmbedder, run_in_batches

logger = logging.getLogger(__name__)


class UnirepEmbedder(BaseEmbedder):
    """
    UniRep embeddings.
    """

    name = "unirep"

    def __init__(self):
        self.params = None

    # Run in batches since GPU memory is limited to ~12GB.
    # TODO: Make jax-unirep accept a batch size parameter used within its batches-by-sequence-length.
    @run_in_batches(num_sequences_per_batch=10000)
    def embed(self, arr: np.ndarray) -> np.ndarray:
        # Import jax_unirep here rather than at module import time,
        # because by the time we get to this point, the user may have properly called config.configure_gpu().
        # Otherwise jax will be imported and by default will use all GPUs.

        # Just in case, force defaulting to CPU if the user has not explicitly called configure_gpu(enable=True) before calling this method:
        choosegpu.ensure_gpu_settings_configured()

        from jax_unirep import get_reps

        h_avg, h_final, c_final = get_reps(arr, params=self.params)
        logger.info(f"Finished batch ({self.name})")
        return h_avg


class UnirepFineTunedEmbedder(BaseFineTunedEmbedder, UnirepEmbedder):
    name = "unirep_fine_tuned"
    fold_id = None
    gene_locus: GeneLocus

    def load_fine_tuned_parameters(
        self, fold_id: int, gene_locus: GeneLocus
    ) -> "UnirepFineTunedEmbedder":
        GeneLocus.validate_single_value(gene_locus)
        self.fold_id = fold_id
        self.gene_locus = gene_locus

        # load best params
        self.params = joblib.load(
            config.paths.fine_tuned_embedding_dir
            / gene_locus.name
            / f"fold_{fold_id}"
            / "best_params.joblib"
        )[1]

        return self

    def embed(self, arr: np.ndarray) -> np.ndarray:
        if self.params is None or self.fold_id is None:
            raise ValueError(
                "Must call load_fine_tuned_parameters() first to use unirep_fine_tuned."
            )
        return super().embed(arr)
