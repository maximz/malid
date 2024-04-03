from malid.datamodels import (
    DataSource,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    CrossValidationSplitStrategy,
    combine_classification_option_names,
    ObsOnlyAnndata,
    map_cross_validation_split_strategy_to_default_target_obs_column,
    diseases,
)
import numpy as np
import pandas as pd

# TODO: Test that disease subtype filters in CrossValidationSplitStrategy or in TargetObsColumnEnum are valid subtypes present in the all-specimens metadata file.
# TODO: Test that obs_column_name in TargetObsColumnEnum is a valid column name in a typical anndata object.


def test_target_obs_column_disease_restrictions_match_known_disease_names():
    # confirm that each limited_to_disease entry matches a known disease name
    for target_obs_column in TargetObsColumnEnum:
        if target_obs_column.value.limited_to_disease is not None:
            for disease_name in target_obs_column.value.limited_to_disease:
                assert disease_name in diseases


def test_cross_validation_split_strategy_disease_restrictions_match_known_disease_names():
    # confirm that each diseases_to_keep_all_subtypes entry matches a known disease name
    for cross_validation_split_strategy in CrossValidationSplitStrategy:
        for (
            disease_name
        ) in cross_validation_split_strategy.value.diseases_to_keep_all_subtypes:
            assert disease_name in diseases


def test_cross_validation_split_strategy_data_source_restrictions_match_known_data_sources():
    # confirm that each data_sources_keep entry is a valid DataSource enum value
    for cross_validation_split_strategy in CrossValidationSplitStrategy:
        for data_source in cross_validation_split_strategy.value.data_sources_keep:
            DataSource.validate(data_source)


def test_map_cross_validation_split_strategy_to_default_target_obs_column():
    # confirm that each cross_validation_split_strategy has a default target_obs_column
    for cross_validation_split_strategy in CrossValidationSplitStrategy:
        # - is present
        assert (
            cross_validation_split_strategy
            in map_cross_validation_split_strategy_to_default_target_obs_column
        )

        chosen_default = (
            map_cross_validation_split_strategy_to_default_target_obs_column[
                cross_validation_split_strategy
            ]
        )

        # - has a valid default target obs column
        TargetObsColumnEnum.validate(chosen_default)

        # - they are compatible
        chosen_default.confirm_compatibility_with_cross_validation_split_strategy(
            cross_validation_split_strategy
        )


def test_combine_classification_option_names():
    assert (
        combine_classification_option_names(TargetObsColumnEnum.age_group_healthy_only)
        == "age_group_healthy_only"
    )
    assert (
        combine_classification_option_names(
            TargetObsColumnEnum.age_group_healthy_only,
            sample_weight_strategy=SampleWeightStrategy.ISOTYPE_USAGE,
        )
        == "age_group_healthy_only_sample_weight_strategy_ISOTYPE_USAGE"
    )


def test_obs_only_anndata():
    adata = ObsOnlyAnndata(
        obs=pd.DataFrame(
            {"disease": ["Covid19", "HIV", "Covid19", "Covid19", "HIV"]},
            index=["cellA", "cellB", "cellC", "cellD", "cellE"],
        )
    )
    assert len(adata) == 5
    assert adata.shape == (5, 0)
    assert np.array_equal(
        adata.obs_names, ["cellA", "cellB", "cellC", "cellD", "cellE"]
    )
    assert not adata.is_view

    # test slicing
    adata = adata[adata.obs["disease"] == "Covid19"]
    assert len(adata) == 3
    assert adata.shape == (3, 0)
    assert np.array_equal(adata.obs_names, ["cellA", "cellC", "cellD"])
    assert adata.is_view

    # test copy
    adata = adata.copy()
    assert len(adata) == 3
    assert adata.shape == (3, 0)
    assert np.array_equal(adata.obs_names, ["cellA", "cellC", "cellD"])
    assert not adata.is_view


def test_sample_weight_multipack_flag():
    # can multipack
    SampleWeightStrategy.validate(
        SampleWeightStrategy.ISOTYPE_USAGE | SampleWeightStrategy.CLONE_SIZE
    )
    # name has _ separator and doesn't rely on flag order
    assert (
        (SampleWeightStrategy.ISOTYPE_USAGE | SampleWeightStrategy.CLONE_SIZE).name
        == (SampleWeightStrategy.CLONE_SIZE | SampleWeightStrategy.ISOTYPE_USAGE).name
        == "ISOTYPE_USAGE_CLONE_SIZE"
    )
    # nullable flag:
    assert (
        SampleWeightStrategy.NONE | SampleWeightStrategy.ISOTYPE_USAGE
        == SampleWeightStrategy.ISOTYPE_USAGE
    )
