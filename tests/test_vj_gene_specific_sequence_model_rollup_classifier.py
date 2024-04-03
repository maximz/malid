import numpy as np
import pandas as pd
import pytest
import anndata
import genetools

from malid import config, helpers
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
)
from malid.trained_model_wrappers import (
    VJGeneSpecificSequenceModelRollupClassifier,
    VJGeneSpecificSequenceClassifier,
)

# see https://github.com/pytest-dev/pytest/pull/4682/files for an alternative:
from contextlib import nullcontext as does_not_raise

from malid.trained_model_wrappers.vj_gene_specific_sequence_model_rollup_classifier import (
    AggregationStrategy,
)

from .test_vj_gene_specific_sequence_classifier import MockSequenceModel

classes = ["Covid19", "HIV", "Healthy"]

# Most of the tests in this file are for split_VJ, i.e. splitting by V gene and J gene.


def test_aggregation_strategy_model_name_suffix():
    assert AggregationStrategy.mean.model_name_suffix == "_mean_aggregated"


@pytest.fixture
def sequence_predicted_probas():
    size = 500
    return pd.DataFrame(
        {
            "v_gene": np.random.choice(["V1", "V2", "V3"], size=size, replace=True),
            "j_gene": np.random.choice(["J1", "J2"], size=size, replace=True),
            "specimen_label": np.random.choice(
                ["specimen1", "specimen2"], size=size, replace=True
            ),
            "weight": np.random.uniform(low=0, high=1, size=size),
        }
        | {class_name: np.random.random(size) for class_name in classes}
    )


@pytest.fixture
def expected_columns():
    cols = set()
    col_map = {}
    for class_name in classes:
        for v_gene in ["V1", "V2", "V3"]:
            for j_gene in ["J1", "J2"]:
                col_name = f"{class_name}_{v_gene}_{j_gene}"
                cols.add(col_name)
                col_map[col_name] = {
                    "class_name": class_name,
                    "v_gene": v_gene,
                    "j_gene": j_gene,
                }
    return cols, col_map


@pytest.mark.parametrize("aggregation_strategy", list(AggregationStrategy))
def test_featurize_sequence_predictions(
    sequence_predicted_probas,
    expected_columns,
    aggregation_strategy: AggregationStrategy,
):
    (
        feature_vectors,
        col_map,
    ) = VJGeneSpecificSequenceModelRollupClassifier._featurize_sequence_predictions(
        sequence_predicted_probas,
        split_on=["v_gene", "j_gene"],
        aggregation_strategy=aggregation_strategy,
    )
    assert np.array_equal(feature_vectors.index, ["specimen1", "specimen2"])
    assert set(feature_vectors.columns) == expected_columns[0]
    assert col_map == expected_columns[1]


@pytest.mark.parametrize("aggregation_strategy", list(AggregationStrategy))
def test_featurize_sequence_predictions_with_missings(
    sequence_predicted_probas,
    expected_columns,
    aggregation_strategy: AggregationStrategy,
):
    # Test NAs: drop (V1, J1) for specimen1, as if that person just did not have those sequences
    # Note: this will change the column order
    sequence_predicted_probas_subset = sequence_predicted_probas[
        ~(
            (sequence_predicted_probas["v_gene"] == "V1")
            & (sequence_predicted_probas["j_gene"] == "J1")
            & (sequence_predicted_probas["specimen_label"] == "specimen1")
        )
    ]
    (
        feature_vectors,
        col_map,
    ) = VJGeneSpecificSequenceModelRollupClassifier._featurize_sequence_predictions(
        sequence_predicted_probas_subset,
        split_on=["v_gene", "j_gene"],
        aggregation_strategy=aggregation_strategy,
    )
    assert np.array_equal(feature_vectors.index, ["specimen1", "specimen2"])
    assert set(feature_vectors.columns) == expected_columns[0]
    assert col_map == expected_columns[1]

    assert not feature_vectors.loc["specimen2"].isna().any()

    specimen1_na_at_which_entries = feature_vectors.loc["specimen1"].isna()
    specimen1_na_at_which_entries = specimen1_na_at_which_entries[
        specimen1_na_at_which_entries
    ].index
    assert set(specimen1_na_at_which_entries) == {
        f"{class_name}_V1_J1" for class_name in classes
    }


# Repeat the simple test for splitting on V gene only, not on V+J gene.
@pytest.fixture
def expected_columns_split_on_vgene_only():
    cols = set()
    col_map = {}
    for class_name in classes:
        for v_gene in ["V1", "V2", "V3"]:
            col_name = f"{class_name}_{v_gene}"
            cols.add(col_name)
            col_map[col_name] = {"class_name": class_name, "v_gene": v_gene}
    return cols, col_map


@pytest.fixture
def sequence_predicted_probas_split_on_vgene_only(sequence_predicted_probas):
    # Based on VJGeneSpecificSequenceModelRollupClassifier._featurize(): j_gene won't be included as column.
    return sequence_predicted_probas.drop(columns=["j_gene"])


@pytest.mark.parametrize("aggregation_strategy", list(AggregationStrategy))
def test_featurize_sequence_predictions_v_gene_only(
    sequence_predicted_probas_split_on_vgene_only,
    expected_columns_split_on_vgene_only,
    aggregation_strategy: AggregationStrategy,
):
    (
        feature_vectors,
        col_map,
    ) = VJGeneSpecificSequenceModelRollupClassifier._featurize_sequence_predictions(
        sequence_predicted_probas_split_on_vgene_only,
        split_on=["v_gene"],
        aggregation_strategy=aggregation_strategy,
    )
    assert np.array_equal(feature_vectors.index, ["specimen1", "specimen2"])
    assert set(feature_vectors.columns) == expected_columns_split_on_vgene_only[0]
    assert col_map == expected_columns_split_on_vgene_only[1]


# More elaborate test:
# Three diseases
# Three samples, one per disease. Each sample has three cells, except the last sample only has two cells.
# Three V-J gene pairs. All samples have all V-J gene pairs, except the last sample does not have one of the V-J gene pairs.
# There's a submodel for each V-J gene pair, but one of the submodels does not predict one of the classes (i.e. its training set did not have any examples of that class).


@pytest.fixture
def adata():
    ad = anndata.AnnData(
        X=np.random.randn(11, 10),
        obs=pd.DataFrame(
            {
                "v_gene": [
                    "IGHV1-24",
                    "IGHV1-24",
                    "IGHV1-24",
                    "IGHV1-18",
                    "IGHV1-18",
                    "IGHV1-18",
                    #
                    # Based on specimen label configuration below, specimen3 will not have any V4-34 sequences.
                    # This is useful as a test case where a gene subset is available in some but not all specimens, leading to partial NaNs in the post-aggregation matrix.
                    # The post-aggregation marix should have features for this gene, with values filled in for some specimen rows but not other specimen rows (NaN for specimen3, in this case):
                    "IGHV4-34",
                    "IGHV4-34",
                    #
                    # Based on specimen label configuration below, specimen3 will also not have any V1-9 sequences, which is used for another test below.
                    "IGHV1-9",
                    "IGHV1-9",
                    #
                    # Add sequences for which there is no corresponding model.
                    # This is useful as a test case where a gene subset is present in the anndatas but never has predicted sequence scores, because no sequence-level submodel was trained for this subset.
                    # Therefore features for this subset must be dropped from the post-aggregation matrix; there are no predicted probabilities for this subset to be aggregated:
                    "IGHVno-such-model",
                ],
                "j_gene": [
                    "IGHJ6",
                    "IGHJ6",
                    "IGHJ6",
                    "IGHJ1",
                    "IGHJ1",
                    "IGHJ1",
                    "IGHJ3",
                    "IGHJ3",
                    "IGHJ5",
                    "IGHJ5",
                    # Add sequences for which there is no corresponding model - see comment above:
                    "IGHJno-such-model",
                ],
                "specimen_label": [
                    "specimen1",
                    "specimen2",
                    "specimen3",
                    "specimen1",
                    "specimen2",
                    "specimen3",
                    "specimen1",
                    "specimen2",
                    "specimen1",
                    "specimen2",
                    "specimen1",
                ],
                "disease": [
                    "Covid19",
                    "HIV",
                    "Healthy",
                    "Covid19",
                    "HIV",
                    "Healthy",
                    "Covid19",
                    "HIV",
                    "Covid19",
                    "HIV",
                    "Covid19",
                ],
                "isotype_supergroup": "IGHG",
                "sample_weight_isotype_rebalance": 0.8,
                # generate v_mut between 0 and 0.2 for BCR (if TCR, should always be 0)
                "v_mut": np.random.uniform(low=0, high=0.2, size=11),
            },
            index=[f"cell{ix}" for ix in range(11)],
        ),
    )
    # Add other necessary columns for helpers.extract_specimen_metadata_from_anndata
    ad.obs["participant_label"] = ad.obs["specimen_label"]
    ad.obs["disease_subtype"] = ad.obs["disease"]
    ad.obs["disease.separate_past_exposures"] = ad.obs["disease"]
    ad.obs["disease.rollup"] = ad.obs["disease"]
    ad.obs["past_exposure"] = False
    # Set raw to be the same as X. When executing each submodel, we return to the raw data, and then standardize each subset independently.
    ad.raw = ad
    return ad


@pytest.fixture
def sequence_classifier():
    # Configure submodels. First one is the happy path; rest are various edge cases:
    models = {
        # Normal submodel, and our "test set" anndata is designed so all specimens have sequences for this V-J group.
        # So this represents the happy path:
        ("IGHV1-24", "IGHJ6"): MockSequenceModel(classes),
        #
        # The ("IGHV1-18", "IGHJ1") submodel is missing a class in what it can predict: it cannot predict Healthy.
        # This situation can happen if the training set did not have any Healthy sequences in that V-J group.
        # Therefore the aggregation feature Healthy_IGHV1-18_IGHJ1 should be 0 for all specimens:
        ("IGHV1-18", "IGHJ1"): MockSequenceModel(classes[:-1]),
        #
        # Normal submodel - but specimen3 does not have any sequences for this V-J group:
        ("IGHV4-34", "IGHJ3"): MockSequenceModel(classes),
        #
        # Mix of the last two situations:
        # This submodel cannot predict Healthy, and specimen3 does not have any sequences for this V-J group:
        ("IGHV1-9", "IGHJ5"): MockSequenceModel(classes[:-1]),
        #
        # Add a submodel for another V-J gene pair that is not in the data.
        # The test is that it does not turn into any features.
        # We don't want the model to generate features for all possible V-J gene pairs it has submodels for,
        # just the gene pairs that are actually in the training data.
        ("IGHVnonexistent", "IGHJnonexistent"): MockSequenceModel(classes),
    }
    # Make sequence classifier from these submodels
    # Note: VJGeneSpecificSequenceClassifier implies SequenceSubsetStrategy.split_VJ
    clf = VJGeneSpecificSequenceClassifier(
        fold_id=1,
        # In this test, we don't care about the model name,
        # but generally we will set this to the name of the model class that is used.
        model_name_sequence_disease="elasticnet_cv",
        fold_label_train="train_smaller1",
        gene_locus=GeneLocus.BCR,
        target_obs_column=TargetObsColumnEnum.disease,
        sample_weight_strategy=config.sample_weight_strategy,
        models=models,
        classes=classes,
    )
    assert np.array_equal(clf.classes_, classes)
    return clf


@pytest.mark.parametrize("fill_missing", [False, True])
@pytest.mark.parametrize("reweigh_by_subset_frequencies", [False, True])
@pytest.mark.parametrize("aggregation_strategy", list(AggregationStrategy))
@pytest.mark.parametrize("pass_in_sample_weights", [False, True])
def test_custom_rollup_featurize(
    adata,
    sequence_classifier,
    fill_missing: bool,
    reweigh_by_subset_frequencies: bool,
    aggregation_strategy: AggregationStrategy,
    pass_in_sample_weights: bool,
):
    if not pass_in_sample_weights:
        adata = adata.copy()
        adata.obs["sample_weight_isotype_rebalance"] = np.nan

    # this code path is not supported:
    is_expected_to_raise_error = reweigh_by_subset_frequencies and not fill_missing

    with does_not_raise() if not is_expected_to_raise_error else pytest.raises(
        ValueError, match="reweigh_by_subset_frequencies requires fill_missing"
    ):
        fd = VJGeneSpecificSequenceModelRollupClassifier._featurize(
            data=adata,
            # VJGeneSpecificSequenceModelRollupClassifier does not imply SequenceSubsetStrategy.split_VJ
            # In this case, it uses the sequence_subset_strategy that was used to train the sequence_classifier.
            gene_subset_specific_sequence_model=sequence_classifier,
            aggregation_strategy=aggregation_strategy,
            fill_missing=fill_missing,
            reweigh_by_subset_frequencies=reweigh_by_subset_frequencies,
        )
    if is_expected_to_raise_error:
        # stop here if fd was not created
        return
    assert np.array_equal(fd.sample_names, ["specimen1", "specimen2", "specimen3"])
    assert np.array_equal(fd.metadata.index, ["specimen1", "specimen2", "specimen3"])
    assert np.array_equal(fd.X.index, ["specimen1", "specimen2", "specimen3"])
    cols_expected = set()
    col_map = {}
    for class_name in classes:
        for (v_gene, j_gene) in [
            ("IGHV1-24", "IGHJ6"),
            ("IGHV1-18", "IGHJ1"),
            ("IGHV4-34", "IGHJ3"),
            ("IGHV1-9", "IGHJ5"),
            # this is intentionally missing ("IGHVno-such-model", "IGHJno-such-model") because there is no such submodel
        ]:
            # this will also test that only real V-J pairs added, not all combinations
            col_name = f"{class_name}_{v_gene}_{j_gene}"
            cols_expected.add(col_name)
            col_map[col_name] = {
                "class_name": class_name,
                "v_gene": v_gene,
                "j_gene": j_gene,
            }
    assert set(fd.X.columns) == cols_expected
    assert fd.extras["column_map"] == col_map

    # (IGHV4-34, IGHJ3) and (IGHV1-9, IGHJ5) are absent for specimen3. See comments in the adata fixture above.
    # Therefore, for specimen3's entries for the (IGHV4-34, IGHJ3) and (IGHV1-9, IGHJ5) sequence subsets, the predicted probability for all classes should be:
    # * 1/n_classes if fill_missing is True and reweigh_by_subset_frequencies is False
    # * 1/n_classes if fill_missing is True and reweigh_by_subset_frequencies is True, because reweigh_by_subset_frequencies will multiply by the subset's frequency which is 0
    # * NaN if fill_missing is False.
    cols_to_check_nans_were_filled = [
        f"{class_name}_IGHV4-34_IGHJ3" for class_name in classes
    ] + [f"{class_name}_IGHV1-9_IGHJ5" for class_name in classes]
    expected_fill_value = 0 if reweigh_by_subset_frequencies else 1.0 / len(classes)
    if fill_missing:
        assert not fd.X.isna().any().any(), "Should not have any NaNs"

        assert (
            fd.X.loc["specimen3"][cols_to_check_nans_were_filled] == expected_fill_value
        ).all(), "Should have filled in NaNs with 1/num_classes (or with 0 if reweigh_by_subset_frequencies is enabled)"

        assert fd.X.equals(
            VJGeneSpecificSequenceModelRollupClassifier._fill_missing_later(fd).X
        ), "Sanity check: filling missings later should have no effect since they're all already filled"
    else:
        assert fd.X.isna().any().any(), "Should have NaNs"
        assert (
            fd.X.loc["specimen3"][cols_to_check_nans_were_filled].isna().all()
        ), f"Should have NaNs for specimen3 at {cols_to_check_nans_were_filled}"
        filled_later = VJGeneSpecificSequenceModelRollupClassifier._fill_missing_later(
            fd
        )
        assert (
            not filled_later.X.isna().any().any()
        ), "Filling missings later should remove all NaNs"
        assert (
            filled_later.X.loc["specimen3"][cols_to_check_nans_were_filled]
            == expected_fill_value
        ).all(), "Filling missings later should replace all NaNs with 1/num_classes (or with 0 if reweigh_by_subset_frequencies is enabled)"

    # Another edge case:
    # The (IGHV1-18, IGHJ1) submodel was never able to predict Healthy - the training set did not have any Healthy sequences in that V-J group. (See comments in sequence_classifier fixture above.)
    # Therefore Healthy_IGHV1-18_IGHJ1 should be 0 for all specimens (this logic is handled by the VJGeneSpecificSequenceClassifier itself, not the rollup classifier).
    # But the submodel did emit predicted probabilities for the other classes.
    assert (
        fd.X["Healthy_IGHV1-18_IGHJ1"] == 0
    ).all(), "Should have kept the 0 for Healthy_IGHV1-18_IGHJ1 provided by VJGeneSpecificSequenceClassifier.predict_proba()"

    # The same is true for the (IGHV1-9, IGHJ5) submodel: it also is never able to predict Healthy. So we would expect "Healthy_IGHV1-9_IGHJ5" to be 0 for specimen1 and specimen2.
    # But not so fast: that sequence subset is also not present in one of the specimens (specimen3), so all aggregation matrix values for IGHV1-9_IGHJ5 (all *_IGHV1-9_IGHJ5 columns, not just Healthy) would start as NaN for specimen3 in particular.
    #
    # Then to make things more complicated, if fill_missing is enabled, that logic would set the *_IGHV1-9_IGHJ5 column values to 1/n_classes for specimen3 in particular.
    #
    # One more complexity: if reweigh_by_subset_frequencies is True in addition, then the column values would be standardized, making "Healthy_IGHV1-9_IGHJ5" no longer 0 for the other specimens!
    # All the IGHV1-9_IGHJ5 columns return to 0 for specimen3 because they are multiplied by the subset's frequency in that specimen, which is 0.
    # However, "Healthy_IGHV1-9_IGHJ5" is still no longer 0 for specimen1 and specimen2.
    #
    # So to summarize, we expect the following specimen1,2,3 values for Healthy_IGHV1-9_IGHJ5 feature in the different cases:
    # * fill_missing=False, reweigh_by_subset_frequencies=False: 0, 0, NaN
    # * fill_missing=False, reweigh_by_subset_frequencies=True: this code path is not supported, we are already guaranteed to not end up in this situation
    # * fill_missing=True, reweigh_by_subset_frequencies=False: 0, 0, 1/n_classes
    # * fill_missing=True, reweigh_by_subset_frequencies=True: nonzero-standardized-value, nonzero-standardized-value, 0
    unaffected_features_for_this_submodel = [
        "Covid19_IGHV1-9_IGHJ5",
        "HIV_IGHV1-9_IGHJ5",
    ]
    affected_feature_for_this_submodel = "Healthy_IGHV1-9_IGHJ5"
    all_features_for_this_submodel = unaffected_features_for_this_submodel + [
        affected_feature_for_this_submodel
    ]
    if not fill_missing:
        assert fd.X.loc["specimen3"][all_features_for_this_submodel].isna().all()
        assert (
            fd.X.loc[fd.X.index != "specimen3"][all_features_for_this_submodel]
            .notna()
            .all()
            .all()
        )
        assert (
            (
                fd.X.loc[fd.X.index != "specimen3"][
                    unaffected_features_for_this_submodel
                ]
                != 0
            )
            .all()
            .all()
        )
        assert (
            fd.X.loc[fd.X.index != "specimen3"][affected_feature_for_this_submodel] == 0
        ).all(), "For all except specimen3, should have kept the 0 for Healthy_IGHV1-9_IGHJ5 provided by VJGeneSpecificSequenceClassifier"

        filled_later = VJGeneSpecificSequenceModelRollupClassifier._fill_missing_later(
            fd
        )
        assert not filled_later.X.isna().any().any()
        assert (
            filled_later.X.loc["specimen3"][all_features_for_this_submodel]
            == expected_fill_value
        ).all()
    else:
        assert not fd.X.isna().any().any()
        assert (
            (
                fd.X.loc[fd.X.index != "specimen3"][
                    unaffected_features_for_this_submodel
                ]
                != 0
            )
            .all()
            .all()
        )
        assert (
            fd.X.loc["specimen3"][all_features_for_this_submodel]
            == expected_fill_value  # which depends on reweigh_by_subset_frequencies -- see definition above
        ).all()
        if not reweigh_by_subset_frequencies:
            assert (
                fd.X.loc[fd.X.index != "specimen3"][affected_feature_for_this_submodel]
                == 0
            ).all(), "For all except specimen3, should have kept the 0 for Healthy_IGHV1-9_IGHJ5 provided by VJGeneSpecificSequenceClassifier"
        else:
            # The edge cases combine:
            # * Healthy_IGHV1-9_IGHJ5 should be 0 for specimen3
            # * Healthy_IGHV1-9_IGHJ5 should be nonzero for specimen1 and specimen2 (values were standardized while they were 0,0,1/n_classes)
            assert (
                fd.X.loc[fd.X.index != "specimen3"][affected_feature_for_this_submodel]
                != 0
            ).all(), "For all except specimen3, should have standardized the zero value for Healthy_IGHV1-9_IGHJ5 provided by VJGeneSpecificSequenceClassifier, making it nonzero"


@pytest.mark.parametrize("gene_locus", list(GeneLocus))
def test_normalize_row_sum_within_each_isotype(gene_locus: GeneLocus):
    # Test the special normalization logic for models that split by isotype and have reweigh_by_subset_frequencies enabled.
    # For example, if we have split by V gene and isotype: The specimen counts used to reweigh should be normalized within each isotype, rather than across all isotypes.

    # 1 isotype for TCR, 3 isotypes for BCR
    isotypes = helpers.isotype_groups_kept[gene_locus]

    # Generate in ranges 0-10, 10-20, 20-30, etc.
    min_bounds = dict(zip(isotypes, np.arange(len(isotypes)) * 10))
    max_bounds = dict(zip(isotypes, np.arange(1, len(isotypes) + 1) * 10))

    # Create a df of counts
    # It should have format:
    # Index: specimen_label
    # Columns: multi-index of (v_gene, isotype_supergroup)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        [
            {
                "v_gene": v_gene,
                "isotype_supergroup": isotype,
                "specimen_label": specimen_label,
                "count": rng.integers(min_bounds[isotype], max_bounds[isotype]),
            }
            for v_gene in ["V1", "V2", "V3"]
            for isotype in isotypes
            for specimen_label in ["specimen1", "specimen2"]
        ]
    ).pivot(
        index="specimen_label", columns=["v_gene", "isotype_supergroup"], values="count"
    )

    # Run the normalization
    df_norm = VJGeneSpecificSequenceModelRollupClassifier._normalize_row_sum_within_each_isotype(
        df
    )

    # Check overall row sum == number of isotypes
    assert np.allclose(df_norm.sum(axis=1), len(isotypes))

    # Extract dataframe for each isotype.
    for isotype in isotypes:
        df_isotype = df_norm[
            df_norm.columns[
                df_norm.columns.get_level_values("isotype_supergroup") == isotype
            ]
        ]
        # Check that the row sum within each isotype == 1
        assert np.allclose(df_isotype.sum(axis=1), 1)
        # Check that the values do not change after running genetools.stats.normalize_rows(df_one_isotype)
        pd.testing.assert_frame_equal(
            df_isotype, genetools.stats.normalize_rows(df_isotype)
        )
