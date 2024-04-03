from typing import Dict, Optional, Tuple, Type, Union
import numpy as np
import pandas as pd
import anndata
import joblib
import logging
from sklearn.base import TransformerMixin  # describes transformation objects

import genetools.scanpy_helpers
from malid import config, embedders, io
from malid.datamodels import GeneLocus
from malid.embedders.base_embedder import BaseEmbedder, BaseFineTunedEmbedder

logger = logging.getLogger(__name__)


def load_sequence_embedding_content_for_fold(
    fold_id: int,
    fold_label: str,
    gene_locus: GeneLocus,
    embedder_class: Type[BaseEmbedder],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load sequences and extract correct sequence region for a particular fold and gene locus."""
    df = io.load_raw_parquet_sequences_for_fold(
        fold_id=fold_id, fold_label=fold_label, gene_locus=gene_locus
    )
    if df["specimen_label"].nunique() < 2:
        raise ValueError("Parquet load did not return multiple specimens.")

    return embedder_class._get_sequences(
        df=df, embedder_sequence_content=embedder_class.embedder_sequence_content
    )


def load_embedding_model(
    gene_locus: GeneLocus, fold_id: int
) -> Union[BaseEmbedder, BaseFineTunedEmbedder]:
    # if config.embedder is a fine-tuned embedder, this will load the fine-tuned parameters
    # otherwise the general-purpose embedder will be loaded and the arguments are ignored.
    embedder = config.embedder(gene_locus=gene_locus, fold_id=fold_id)
    return embedder


def run_embedding_model(
    embedder: BaseEmbedder,
    df: pd.DataFrame,  # usually df is output of io.load_raw_parquet_sequences_for_fold()
) -> anndata.AnnData:
    # For specimens in a certain test fold:
    # apply the embedder fine-tuned on that fold's training set,
    # or a general-purpose non-fine-tuned embedder.

    # Make adata.
    # Use a lower dtype:
    # Default is float32. Example precision: [ 0.10056811,  0.43847042,  0.36644596 ]
    # We can use float16 instead. Example precision: [ 0.8955 , 0.969  , -0.936 ]
    dtype = np.float16
    adata = anndata.AnnData(
        X=embedder.embed(sequences=df, dtype=dtype),
        obs=df,
        dtype=dtype,
    )
    adata.uns["embedded"] = embedder.name
    if isinstance(embedder, BaseFineTunedEmbedder):
        adata.uns["embedded_fine_tuned_on_fold_id"] = embedder.fold_id
        adata.uns["embedded_fine_tuned_on_gene_locus"] = embedder.gene_locus.name

    if adata.obs["specimen_label"].isna().any():
        raise ValueError("adata contains null specimen_label(s)")

    # Fix obs names
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs_names_make_unique()

    return adata


def verify_right_embedder_used(
    embedder: BaseEmbedder,
    adata: anndata.AnnData,
    gene_locus: Optional[GeneLocus] = None,
    fold_id: Optional[int] = None,
) -> None:
    """Verify that the expected embedder was used to create the embedding, else raise error"""
    if adata.uns["embedded"] != embedder.name:
        raise ValueError(
            f"Expected anndata to be embedded with {embedder.name}, but got {adata.uns['embedded']}"
        )
    if embedder.is_fine_tuned:
        if fold_id is None or gene_locus is None:
            raise ValueError(
                "fold_id and gene_locus must be specified for fine-tuned embedder"
            )
        if adata.uns["embedded_fine_tuned_on_fold_id"] != fold_id:
            raise ValueError(
                f"Expected anndata to be embedded with fold_id {fold_id}, but got {adata.uns['embedded_fine_tuned_on_fold_id']}"
            )
        if adata.uns["embedded_fine_tuned_on_gene_locus"] != gene_locus.name:
            raise ValueError(
                f"Expected anndata to be embedded with GeneLocus {gene_locus.name}, but got {adata.uns['embedded_fine_tuned_on_gene_locus']}"
            )


def get_embedder_used_for_embedding_anndata(
    adata: anndata.AnnData,
) -> Union[Type[BaseEmbedder], Type[BaseFineTunedEmbedder]]:
    """Get the embedder used to create the embedding, else raise error"""
    return embedders.get_embedder_by_name(adata.uns["embedded"])


def load_transformations(
    gene_locus: GeneLocus, fold_id: int
) -> Dict[str, TransformerMixin]:
    """Load scale and PCA transformations that were created from train-smaller"""
    return joblib.load(
        config.paths.scaled_anndatas_dir
        / gene_locus.name
        / f"fold.{fold_id}.train_smaller.transformations.joblib"
    )


def transform_embedded_anndata(
    transformations_to_apply: Dict[str, TransformerMixin], adata: anndata.AnnData
):
    """Apply scale and PCA transformations that were created from train-smaller. Runs inplace."""

    # Scale inplace using existing transformer - and set raw
    logger.info("Scaling...")
    adata, _ = genetools.scanpy_helpers.scale_anndata(
        adata,
        scale_transformer=transformations_to_apply["scale"],
        inplace=True,
        set_raw=True,
    )

    # PCA inplace using existing transformer
    logger.info("Running PCA...")
    adata, _ = genetools.scanpy_helpers.pca_anndata(
        adata,
        pca_transformer=transformations_to_apply["pca"],
        inplace=True,
    )

    return adata
