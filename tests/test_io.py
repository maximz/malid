import anndata
import numpy as np
import pandas as pd

from malid import io


def test_label_past_exposures_with_an_anndata_view():
    obsdf = pd.DataFrame(
        {"disease": ["Covid19", "HIV", "Covid19", "Covid19", "HIV"]},
        index=["cellA", "cellB", "cellC", "cellD", "cellE"],
    )
    obsdf["disease_subtype"] = obsdf["disease"].copy()
    adata = anndata.AnnData(np.random.randn(5, 3), obs=obsdf)

    subset_adata = adata[["cellA", "cellB"]]
    assert subset_adata.is_view

    subset_adata = io.label_past_exposures(subset_adata)
    assert not subset_adata.is_view
    assert not subset_adata.obs["past_exposure"].isna().any()
    assert not subset_adata.obs["disease.separate_past_exposures"].isna().any()

    assert "past_exposure" not in adata.obs.columns
    assert "disease.separate_past_exposures" not in adata.obs.columns
