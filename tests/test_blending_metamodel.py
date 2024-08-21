from typing import Dict, List
import numpy as np
import pandas as pd
from malid.datamodels import GeneLocus
from crosseval import FeaturizedData
from malid.trained_model_wrappers import BlendingMetamodel
from malid.trained_model_wrappers.blending_metamodel import (
    DemographicsFeaturizer,
    _combine_dfs,
    cartesian_product,
)
import pytest


def test_demographics_featurizer_respects_binary_ordered_categorical_column_order():
    # Usually whichever value is seen first is the one that's chosen for binary columns.
    df = pd.DataFrame({"sex": ["F", "M"]})
    assert np.array_equal(
        DemographicsFeaturizer(["sex"]).featurize(df).columns, ["sex_F"]
    )
    df = pd.DataFrame({"sex": ["M", "F"]})
    assert np.array_equal(
        DemographicsFeaturizer(["sex"]).featurize(df).columns, ["sex_M"]
    )

    # But when the binary column has an ordered categorical dtype, the dtype's specified order should be respected.
    df = pd.DataFrame({"sex": ["F", "M"]})
    df["sex"] = df["sex"].astype(
        pd.CategoricalDtype(categories=["M", "F"], ordered=True)
    )
    assert np.array_equal(
        DemographicsFeaturizer(["sex"]).featurize(df).columns, ["sex_F"]
    )
    df = pd.DataFrame({"sex": ["M", "F"]})
    df["sex"] = df["sex"].astype(
        pd.CategoricalDtype(categories=["M", "F"], ordered=True)
    )
    assert np.array_equal(
        DemographicsFeaturizer(["sex"]).featurize(df).columns, ["sex_F"]
    )


def test_convert_feature_name_to_friendly_name():
    for input, expected in [
        # pass through any unmatched
        ("gibberish", "gibberish"),
        (
            "BCR:sequence_model:Healthy/Background",
            "Language embedding (BCR): P(Healthy/Background)",
        ),
        (
            "demographics:ethnicity_condensed_African",
            "Demographics: ethnicity_condensed_African",
        ),
        ("isotype_counts:isotype_proportion:IGHG", "Isotype proportion of IGHG"),
        (
            "interaction|BCR:convergent_cluster_model:HIV|demographics:age",
            "Interaction: [CDR3 clustering (BCR): P(HIV)] x [Demographics: age]",
        ),
        (
            "interaction|TCR:repertoire_stats:Covid19|demographics:ethnicity_condensed_Hispanic/Latino",
            "Interaction: [Repertoire composition (TCR): P(Covid19)] x [Demographics: ethnicity_condensed_Hispanic/Latino]",
        ),
    ]:
        observed = BlendingMetamodel.convert_feature_name_to_friendly_name(input)
        assert (
            observed == expected
        ), f"Expected {input} to be converted to '{expected}', but got '{observed}'"


def test_harmonize_submodels():
    # Different abstentions in different loci. Each loci has two submodels, one of which produces abstentions.
    featurized_by_single_locus: Dict[GeneLocus, List[FeaturizedData]] = {
        GeneLocus.BCR: [
            FeaturizedData(
                X=pd.DataFrame(
                    np.random.randn(4, 3),
                    index=["sample1", "sample2", "sample3", "sample4"],
                ),
                y=["Covid19", "Covid19", "HIV", "HIV"],
                sample_names=["sample1", "sample2", "sample3", "sample4"],
                metadata=pd.DataFrame(
                    {
                        "name": ["sample1", "sample2", "sample3", "sample4"],
                        "isotype_proportion:IGHG": [0.5, 0.5, 0.5, 0.5],
                    },
                    index=["sample1", "sample2", "sample3", "sample4"],
                ),
            ),
            FeaturizedData(
                X=pd.DataFrame(
                    np.random.randn(3, 3), index=["sample1", "sample2", "sample3"]
                ),
                y=["Covid19", "Covid19", "HIV"],
                sample_names=["sample1", "sample2", "sample3"],
                metadata=pd.DataFrame(
                    {
                        "name": ["sample1", "sample2", "sample3"],
                        "isotype_proportion:IGHG": [0.5, 0.5, 0.5],
                    },
                    index=["sample1", "sample2", "sample3"],
                ),
                abstained_sample_names=["sample4"],
                abstained_sample_y=["HIV"],
                abstained_sample_metadata=pd.DataFrame(
                    {"name": ["sample4"], "isotype_proportion:IGHG": [0.5]},
                    index=["sample4"],
                ),
            ),
        ],
        GeneLocus.TCR: [
            FeaturizedData(
                X=pd.DataFrame(
                    np.random.randn(4, 3),
                    index=["sample1", "sample2", "sample3", "sample4"],
                ),
                y=["Covid19", "Covid19", "HIV", "HIV"],
                sample_names=["sample1", "sample2", "sample3", "sample4"],
                metadata=pd.DataFrame(
                    {
                        "name": ["sample1", "sample2", "sample3", "sample4"],
                        "tcr_specific_metadata": [1.0, 1.0, 1.0, 1.0],
                    },
                    index=["sample1", "sample2", "sample3", "sample4"],
                ),
            ),
            FeaturizedData(
                X=pd.DataFrame(np.random.randn(2, 3), index=["sample1", "sample2"]),
                y=["Covid19", "Covid19"],
                sample_names=["sample1", "sample2"],
                metadata=pd.DataFrame(
                    {
                        "name": ["sample1", "sample2"],
                        "tcr_specific_metadata": [1.0, 1.0],
                    },
                    index=["sample1", "sample2"],
                ),
                abstained_sample_names=["sample3", "sample4"],
                abstained_sample_y=["HIV", "HIV"],
                abstained_sample_metadata=pd.DataFrame(
                    {
                        "name": ["sample3", "sample4"],
                        "tcr_specific_metadata": [1.0, 1.0],
                    },
                    index=["sample3", "sample4"],
                ),
            ),
        ],
    }
    (
        metadata_df,
        specimens_with_full_predictions,
        abstained_specimen_names,
        X,
    ) = BlendingMetamodel._harmonize_across_models_and_gene_loci(
        featurized_by_single_locus
    )

    assert metadata_df.shape[0] == 4
    assert set(metadata_df.index) == {"sample1", "sample2", "sample3", "sample4"}

    assert specimens_with_full_predictions == {"sample1", "sample2"}
    assert abstained_specimen_names == {"sample3", "sample4"}

    assert X.shape[0] == 4
    assert set(X.index) == {"sample1", "sample2", "sample3", "sample4"}

    # isotype_proportion:IGHG was only defined for BCR samples, but should be kept in the merged metadata.
    assert not metadata_df["isotype_proportion:IGHG"].isna().any()
    assert (metadata_df["isotype_proportion:IGHG"] == 0.5).all()
    # similarly, make sure the TCR-only metadata column is kept
    assert not metadata_df["tcr_specific_metadata"].isna().any()
    assert (metadata_df["tcr_specific_metadata"] == 1.0).all()


def test_harmonize_one_submodel_per_locus_with_different_abstentions():
    # Edge case: single submodel, so abstained samples will never get the chance to be seen in the main/not-abstained list produced by another submodel (because there are no other submodels).
    featurized_by_single_locus: Dict[GeneLocus, List[FeaturizedData]] = {
        GeneLocus.BCR: [
            FeaturizedData(
                X=pd.DataFrame(
                    np.random.randn(3, 3),
                    index=[
                        "sample1_always_valid",
                        "sample2_always_valid",
                        "sample3_tcr_abstains",
                    ],
                ),
                y=["Covid19", "Covid19", "HIV"],
                sample_names=[
                    "sample1_always_valid",
                    "sample2_always_valid",
                    "sample3_tcr_abstains",
                ],
                metadata=pd.DataFrame(
                    {
                        "name": [
                            "sample1_always_valid",
                            "sample2_always_valid",
                            "sample3_tcr_abstains",
                        ],
                        "isotype_proportion:IGHG": [0.5, 0.5, 0.5],
                    },
                    index=[
                        "sample1_always_valid",
                        "sample2_always_valid",
                        "sample3_tcr_abstains",
                    ],
                ),
                abstained_sample_names=[
                    "sample4_everyone_abstains",
                    "sample5_bcr_abstains",
                ],
                abstained_sample_y=["HIV", "HIV"],
                abstained_sample_metadata=pd.DataFrame(
                    {
                        "name": ["sample4_everyone_abstains", "sample5_bcr_abstains"],
                        "isotype_proportion:IGHG": [0.5, 0.5],
                    },
                    index=["sample4_everyone_abstains", "sample5_bcr_abstains"],
                ),
            )
        ],
        GeneLocus.TCR: [
            FeaturizedData(
                X=pd.DataFrame(
                    np.random.randn(3, 3),
                    index=[
                        "sample1_always_valid",
                        "sample2_always_valid",
                        "sample5_bcr_abstains",
                    ],
                ),
                y=["Covid19", "Covid19", "HIV"],
                sample_names=[
                    "sample1_always_valid",
                    "sample2_always_valid",
                    "sample5_bcr_abstains",
                ],
                metadata=pd.DataFrame(
                    {
                        "name": [
                            "sample1_always_valid",
                            "sample2_always_valid",
                            "sample5_bcr_abstains",
                        ],
                        "tcr_specific_metadata": [1.0, 1.0, 1.0],
                    },
                    index=[
                        "sample1_always_valid",
                        "sample2_always_valid",
                        "sample5_bcr_abstains",
                    ],
                ),
                abstained_sample_names=[
                    "sample3_tcr_abstains",
                    "sample4_everyone_abstains",
                ],
                abstained_sample_y=["HIV", "HIV"],
                abstained_sample_metadata=pd.DataFrame(
                    {
                        "name": ["sample3_tcr_abstains", "sample4_everyone_abstains"],
                        "tcr_specific_metadata": [1.0, 1.0],
                    },
                    index=["sample3_tcr_abstains", "sample4_everyone_abstains"],
                ),
            )
        ],
    }
    (
        metadata_df,
        specimens_with_full_predictions,
        abstained_specimen_names,
        X,
    ) = BlendingMetamodel._harmonize_across_models_and_gene_loci(
        featurized_by_single_locus
    )

    assert metadata_df.shape[0] == 5
    assert set(metadata_df.index) == {
        "sample1_always_valid",
        "sample2_always_valid",
        "sample3_tcr_abstains",
        "sample4_everyone_abstains",
        "sample5_bcr_abstains",
    }

    assert specimens_with_full_predictions == {
        "sample1_always_valid",
        "sample2_always_valid",
    }
    assert abstained_specimen_names == {
        "sample3_tcr_abstains",
        "sample4_everyone_abstains",
        "sample5_bcr_abstains",
    }

    assert X.shape[0] == 4
    assert set(X.index) == {
        "sample1_always_valid",
        "sample2_always_valid",
        "sample3_tcr_abstains",
        "sample5_bcr_abstains",
    }

    # isotype_proportion:IGHG was only defined for BCR samples, but should be kept in the merged metadata.
    assert not metadata_df["isotype_proportion:IGHG"].isna().any()
    assert (metadata_df["isotype_proportion:IGHG"] == 0.5).all()
    # similarly, make sure the TCR-only metadata column is kept
    assert not metadata_df["tcr_specific_metadata"].isna().any()
    assert (metadata_df["tcr_specific_metadata"] == 1.0).all()


def test_combine_dfs():
    combined = _combine_dfs(
        pd.DataFrame({"col1": [1, 2, 3], "col2": [1, 2, 3]}, index=["a", "b", "c"]),
        pd.DataFrame({"col1": [1, 2, 3], "col3": [1, 2, 1]}, index=["a", "b", "c"]),
    )
    pd.testing.assert_frame_equal(
        combined,
        pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [1, 2, 3], "col3": [1, 2, 1]},
            index=["a", "b", "c"],
        ),
    )


def test_combine_dfs_conflicting_column_values():
    df1 = pd.DataFrame({"col1": [1, 2, 3], "col2": [1, 2, 3]}, index=["a", "b", "c"])
    df2 = pd.DataFrame({"col1": [1, 2, 3], "col2": [1, 2, 1]}, index=["a", "b", "c"])
    with pytest.raises(
        ValueError,
        match="Index changed in merge but allow_nonequal_indexes was set to False, suggesting conflicting values",
    ):
        # Conflicting values with allow_nonequal_indexes set to False -->
        # inner merge -->
        # index becomes ['a', 'b'] --> error
        _combine_dfs(df1, df2, allow_nonequal_indexes=False)
    with pytest.raises(
        ValueError, match="Merged index has duplicates, suggesting conflicting values"
    ):
        # Conflicting values with allow_nonequal_indexes set to True -->
        # outer merge -->
        # index becomes ['a', 'b', 'c', 'c'] --> error
        _combine_dfs(df1, df2, allow_nonequal_indexes=True)


def test_cartesian_product():
    df = pd.DataFrame(
        np.random.randint(1, 5, size=(10, 4)), columns=["A", "B", "C", "D"]
    ).assign(other="E")
    transformed = cartesian_product(
        df, features_left=["A", "B"], features_right=["D", "C"]
    )
    assert np.array_equal(
        transformed.columns,
        [
            "A",
            "B",
            "C",
            "D",
            "other",
            "interaction|A|D",
            "interaction|A|C",
            "interaction|B|D",
            "interaction|B|C",
        ],
    )
    assert np.array_equal(transformed["interaction|B|D"], df["B"] * df["D"])


def test_cartesian_product_with_same_columns_on_both_sides():
    df = pd.DataFrame(
        np.random.randint(1, 5, size=(10, 3)), columns=["A", "B", "C"]
    ).assign(other="D")
    transformed = cartesian_product(
        df, features_left=["A", "B", "C"], features_right=["A", "B", "C"]
    )
    assert np.array_equal(
        transformed.columns,
        [
            "A",
            "B",
            "C",
            "other",
            "interaction|B|A",
            "interaction|C|A",
            "interaction|C|B",
        ],
    )
    assert np.array_equal(transformed["interaction|C|A"], df["C"] * df["A"])


@pytest.mark.parametrize("filter_func_enabled", [True, False])
def test_cartesian_product_with_filter_function(filter_func_enabled: bool):
    df = pd.DataFrame(
        np.random.randint(1, 5, size=(10, 4)),
        columns=["Age", "Sex_Female", "Ethnicity_A", "Ethnicity_B"],
    )
    expected_columns = [
        "Age",
        "Sex_Female",
        "Ethnicity_A",
        "Ethnicity_B",
        "interaction|Sex_Female|Age",
        "interaction|Ethnicity_A|Age",
        "interaction|Ethnicity_B|Age",
        "interaction|Ethnicity_A|Sex_Female",
        "interaction|Ethnicity_B|Sex_Female",
    ]
    if filter_func_enabled:
        filter_function = lambda left, right: not (
            "Ethnicity" in left and "Ethnicity" in right
        )  # noqa: E731
    else:
        filter_function = None
        expected_columns.append("interaction|Ethnicity_B|Ethnicity_A")
    transformed = cartesian_product(
        df,
        features_left=["Age", "Sex_Female", "Ethnicity_A", "Ethnicity_B"],
        features_right=["Age", "Sex_Female", "Ethnicity_A", "Ethnicity_B"],
        filter_function=filter_function,
    )
    assert np.array_equal(transformed.columns, expected_columns)
