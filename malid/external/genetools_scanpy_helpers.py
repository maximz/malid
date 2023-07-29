from typing import Optional, Tuple

import anndata
import sklearn.decomposition
import sklearn.preprocessing

import logging

logger = logging.getLogger(__name__)


def scale_anndata(
    adata: anndata.AnnData,
    scale_transformer: Optional[sklearn.preprocessing.StandardScaler] = None,
    inplace=False,
    set_raw=False,
    **kwargs,
) -> Tuple[anndata.AnnData, sklearn.preprocessing.StandardScaler]:
    """
    Scale anndata, like with scanpy.pp.scale.
    Accepts pre-computed StandardScaler preprocessing transformer, so you can apply the same scaling to multiple anndatas.

    Args:
    - scale_transformer: pre-defined preprocessing transformer to scale adata.X
    - inplace: whether to modify input adata in place
    - set_raw: whether to set adata.raw equal to input adata

    Returns: adata, scale_transformer
    """
    # TODO: set var and uns parameters too, and support max_value clipping like in in scanpy
    if scale_transformer is None:
        scale_transformer = sklearn.preprocessing.StandardScaler(**kwargs).fit(adata.X)

    if inplace:
        if set_raw:
            adata.raw = adata
        adata.X = scale_transformer.transform(adata.X).astype(adata.X.dtype)
    else:
        # Copy, but be very memory-frugal about it -- try not to allocate memory we won't need (i.e. don't waste RAM copying old adata.X)
        # TODO: consider anndata._mutated_copy(X=X)
        old_adata = adata
        adata = anndata.AnnData(
            X=scale_transformer.transform(adata.X).astype(adata.X.dtype),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
            obsm=adata.obsm.copy(),
            varm=adata.varm.copy(),
            layers=adata.layers.copy(),
            raw=(adata.raw.copy() if (adata.raw is not None and not set_raw) else None),
            obsp=adata.obsp.copy(),
            varp=adata.varp.copy(),
        )
        if set_raw:
            adata.raw = old_adata

    return adata, scale_transformer


def scale_train_and_test_anndatas(
    adata_train, adata_test=None, inplace=False, set_raw=False, **kwargs
):
    """
    Scale train anndata (like with scanpy.pp.scale), then apply same scaling to test anndata -- as opposed to scaling them independently.
    If adata_test isn't supplied, this just scales adata_train indpendently.
    """
    adata_train_scaled, scale_transformer = scale_anndata(
        adata_train, scale_transformer=None, inplace=inplace, set_raw=set_raw, **kwargs
    )
    adata_test_scaled = None
    if adata_test is not None:
        adata_test_scaled, _ = scale_anndata(
            adata_test,
            scale_transformer=scale_transformer,
            inplace=inplace,
            set_raw=set_raw,
            **kwargs,
        )
    return adata_train_scaled, adata_test_scaled


def pca_anndata(
    adata: anndata.AnnData,
    pca_transformer: Optional[sklearn.decomposition.PCA] = None,
    n_components=None,
    inplace=True,
    **kwargs,
) -> Tuple[anndata.AnnData, sklearn.decomposition.PCA]:
    """
    PCA anndata, like with scanpy.pp.pca.
    Accepts pre-computed PCA transformer, so you can apply the same PCA to multiple anndatas.

    Args:
    - pca_transformer: pre-defined preprocessing transformer to run PCA on adata.X
    - n_components: number of PCA components
    - inplace: whether to modify input adata in place

    Returns: adata, pca_transformer
    """
    # TODO: set var and uns parameters too
    if pca_transformer is None:
        pca_transformer = sklearn.decomposition.PCA(
            n_components=n_components, **kwargs
        ).fit(adata.X)

    # Unlike scale_anndata, here we're not being careful to avoid unnecessary copying, since obsm is usually not that big
    adata = adata.copy() if not inplace else adata

    adata.obsm["X_pca"] = pca_transformer.transform(adata.X)

    return adata, pca_transformer


def pca_train_and_test_anndatas(
    adata_train, adata_test=None, n_components=None, inplace=True, **kwargs
):
    """
    PCA train anndata (like with scanpy.pp.pca), then apply same PCA to test anndata -- as opposed to PCAing them independently.
    If adata_test isn't supplied, this just scales adata_train independently.
    """
    adata_train_pcaed, pca_transformer = pca_anndata(
        adata_train,
        pca_transformer=None,
        n_components=n_components,
        inplace=inplace,
        **kwargs,
    )
    adata_test_pcaed = None
    if adata_test is not None:
        adata_test_pcaed, _ = pca_anndata(
            adata_test,
            pca_transformer=pca_transformer,
            n_components=n_components,
            inplace=inplace,
            **kwargs,
        )
    return adata_train_pcaed, adata_test_pcaed


def umap_anndata(
    adata,
    umap_transformer=None,
    n_neighbors: Optional[int] = None,
    n_components: Optional[int] = None,
    inplace=True,
    use_rapids=False,
    use_pca=False,
    **kwargs,
):
    """
    UMAP anndata, like with scanpy.tl.umap.
    Accepts pre-computed UMAP transformer, so you can apply the same UMAP to multiple anndatas.

    Args:
    - umap_transformer: pre-defined preprocessing transformer to run UMAP on adata.X
    - n_components: number of UMAP components
    - inplace: whether to modify input adata in place

    Anndata should already be scaled.

    Returns: adata, umap_transformer
    """
    if use_rapids:
        # GPU support
        from cuml import UMAP
    else:
        from umap import UMAP

    if use_pca:
        # Allow using adata.obsm["X_pca"] if it exists
        if "X_pca" not in adata.obsm:
            # PCA must be precomputed
            use_pca = False
            logger.warning(
                f"X_pca not found in adata.obsm, so not using PCA representation for UMAP despite use_pca=True"
            )

    if umap_transformer is None:
        umap_transformer = UMAP(
            n_neighbors=n_neighbors, n_components=n_components, **kwargs
        ).fit(adata.obsm["X_pca"] if use_pca else adata.X)

    # Unlike scale_anndata, here we're not being careful to avoid unnecessary copying, since obsm is usually not that big
    adata = adata.copy() if not inplace else adata

    # TODO: set var and uns parameters too
    adata.obsm["X_umap"] = umap_transformer.transform(
        adata.obsm["X_pca"] if use_pca else adata.X
    )

    return adata, umap_transformer


def umap_train_and_test_anndatas(
    adata_train,
    adata_test=None,
    n_neighbors: Optional[int] = None,
    n_components: Optional[int] = None,
    inplace=True,
    use_rapids=False,
    use_pca=False,
    **kwargs,
):
    """
    UMAP train anndata (like with scanpy.tl.umap), then apply same UMAP to test anndata -- as opposed to PCAing them independently.
    If adata_test isn't supplied, this just scales adata_train independently.
    """
    adata_train_umaped, umap_transformer = umap_anndata(
        adata_train,
        umap_transformer=None,
        n_neighbors=n_neighbors,
        n_components=n_components,
        inplace=inplace,
        use_rapids=use_rapids,
        use_pca=use_pca,
        **kwargs,
    )
    adata_test_umaped = None
    if adata_test is not None:
        adata_test_umaped, _ = umap_anndata(
            adata_test,
            umap_transformer=umap_transformer,
            n_neighbors=n_neighbors,
            n_components=n_components,
            inplace=inplace,
            use_rapids=use_rapids,
            use_pca=use_pca,
            **kwargs,
        )
    return adata_train_umaped, adata_test_umaped
