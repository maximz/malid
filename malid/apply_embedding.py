from typing import Dict, Optional
import numpy as np
import pandas as pd
import anndata
import joblib
import logging
from sklearn.base import TransformerMixin  # describes transformation objects

import malid.external.genetools_scanpy_helpers
from malid import config, io
from malid.datamodels import GeneLocus
from malid.embedders.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


def _get_sequences(df: pd.DataFrame) -> np.ndarray:
    seqs_df = df[["cdr1_seq_aa_q_trim", "cdr2_seq_aa_q_trim", "cdr3_seq_aa_q_trim"]]

    # Detect N/As (whether np.nan or empty string) and error out
    if seqs_df.mask(seqs_df == "").isna().any().any():
        raise ValueError("Sequences contain N/As or empty stirngs")

    # These strings might be stored as categoricals, so cast each to str to avoid error: "ValueError: Cannot setitem on a Categorical with a new category, set the categories first"
    sequences = (
        seqs_df["cdr1_seq_aa_q_trim"].astype(str).fillna("")
        + seqs_df["cdr2_seq_aa_q_trim"].astype(str).fillna("")
        + seqs_df["cdr3_seq_aa_q_trim"].astype(str).fillna("")
    ).astype(str)

    return sequences.values


def load_sequences_from_fold(
    fold_id: int,
    fold_label: str,
    gene_locus: GeneLocus,
) -> np.ndarray:
    """Load embedding and extract CDR1+2+3 sequences."""
    df = io.load_raw_parquet_sequences_for_fold(
        fold_id=fold_id, fold_label=fold_label, gene_locus=gene_locus
    )
    if df["specimen_label"].nunique() < 2:
        raise ValueError("Parquet load did not return multiple specimens.")

    return _get_sequences(df)


def load_embedding_model(
    gene_locus: Optional[GeneLocus] = None, fold_id: Optional[int] = None
) -> BaseEmbedder:
    embedder = config.choose_embedder()()

    if embedder.is_fine_tuned:
        # get fold-specific fine-tuned embedder
        if fold_id is None or gene_locus is None:
            raise ValueError(
                "fold_id and gene_locus must be specified for fine-tuned embedder"
            )

        GeneLocus.validate_single_value(gene_locus)
        embedder = embedder.load_fine_tuned_parameters(
            fold_id=fold_id, gene_locus=gene_locus
        )

        # Validation
        if embedder.gene_locus != gene_locus:
            raise ValueError("Embedder locus differed")
        if embedder.fold_id != fold_id:
            raise ValueError(
                f"Embedder fold_id={embedder.fold_id} did not match fold_id={fold_id}"
            )

    if config.embedder.name != embedder.name:
        raise ValueError(
            f"Embedder names did not match config.embedder.name={config.embedder.name}"
        )

    return embedder


def run_embedding_model(
    embedder: BaseEmbedder,
    df: pd.DataFrame,
    gene_locus: Optional[GeneLocus] = None,
    fold_id: Optional[int] = None,
) -> anndata.AnnData:
    # For specimens in a certain test fold:
    # apply the embedder fine-tuned on that fold's training set,
    # or a general-purpose non-fine-tuned embedder.

    # Make adata.
    adata = anndata.AnnData(
        X=embedder.embed(_get_sequences(df=df)),
        obs=df,
        # Default is float32. Example precision: [ 0.10056811,  0.43847042,  0.36644596 ]
        # We can use float16 instead. Example precision: [ 0.8955 , 0.969  , -0.936 ]
        dtype=np.float16,
    )
    adata.uns["embedded"] = embedder.name
    if embedder.is_fine_tuned:
        if fold_id is None or gene_locus is None:
            raise ValueError(
                "fold_id and gene_locus must be specified for fine-tuned embedder"
            )
        adata.uns["embedded_fine_tuned_on_fold_id"] = fold_id
        adata.uns["embedded_fine_tuned_on_gene_locus"] = gene_locus.name

    if adata.obs["specimen_label"].isna().any():
        raise ValueError("adata contains null specimen_label(s)")

    # Fix obs names
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs_names_make_unique()

    return adata


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
    """Apply scale and PCA transformations that were created from train-smaller"""

    # Scale inplace using existing transformer - and set raw
    adata, _ = malid.external.genetools_scanpy_helpers.scale_anndata(
        adata,
        scale_transformer=transformations_to_apply["scale"],
        inplace=True,
        set_raw=True,
    )

    # PCA inplace using existing transformer
    adata, _ = malid.external.genetools_scanpy_helpers.pca_anndata(
        adata,
        pca_transformer=transformations_to_apply["pca"],
        inplace=True,
    )

    return adata
