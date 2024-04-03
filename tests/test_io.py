import anndata
import numpy as np
import pandas as pd
import pytest

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


def test_compute_clone_size_sample_weights_column():
    # one entry per clone per isotype per specimen
    obsdf = pd.DataFrame(
        {
            "specimen_label": [
                "specimenA",
                "specimenA",
                "specimenA",
                "specimenA",
                "specimenB",
            ],
            "isotype_supergroup": ["IGHG", "IGHG", "IGHG", "IGHA", "IGHA"],
            "num_clone_members": [1, 2, 3, 4, 5],
        },
        index=["clone1", "clone2", "clone3", "clone4", "clone5"],
    )
    adata = anndata.AnnData(np.random.randn(5, 3), obs=obsdf)
    computed = io.compute_clone_size_sample_weights_column(adata)
    expected = pd.Series(
        [1 / (1 + 2 + 3), 2 / (1 + 2 + 3), 3 / (1 + 2 + 3), 1, 1],
        index=["clone1", "clone2", "clone3", "clone4", "clone5"],
        name="num_clone_members",
    )
    pd.testing.assert_series_equal(computed, expected)


def test_convert_vgene_to_vfamily():
    df = pd.DataFrame(
        # Inputs and expcted outputs
        [
            ("TRBV2", "TRBV2"),
            ("IGHV1-18", "IGHV1"),
            ("IGHV3-30-3", "IGHV3"),
            ("IGHV1/OR15-1", "IGHV1"),
            ("TRBV29/OR9-2", "TRBV29"),
            ("VH1-67P", "VH1"),
        ],
        # Result series should be named v_family
        columns=["v_gene", "v_family"],
        # Index should be preserved
        index=["cell1", "cell2", "cell3", "cell4", "cell5", "cell6"],
    )
    pd.testing.assert_series_equal(
        io.convert_vgene_to_vfamily(df["v_gene"]), df["v_family"]
    )


@pytest.mark.parametrize("v_gene", [" ", "", np.nan])
def test_convert_vgene_to_vfamily_fails_on_empty_string_or_nan(v_gene):
    with pytest.raises(
        ValueError, match="Some v_genes could not be converted to v_family"
    ):
        io.convert_vgene_to_vfamily(pd.Series(["IGHV1-18", v_gene]))
